use crate::services::aws::retry::retry_transient;
use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use eyre::Result;
use iris_mpc_common::{
    config::Config, helpers::aws::construct_message_attributes, job::BatchMetadata,
};

use std::collections::HashMap;
use std::time::Duration;

const SNS_RETRY_MAX_ATTEMPTS: u32 = 8;
const SNS_RETRY_INITIAL_BACKOFF: Duration = Duration::from_millis(200);

async fn send_message(
    results_topic_arn: String,
    message: String,
    sns_client: &SNSClient,
    message_group_id: String,
    message_attributes: HashMap<String, MessageAttributeValue>,
    metrics_message_type: String,
) -> Result<()> {
    // Retry transient SNS failures rather than letting them bubble to the
    // result-sender's `exit(1)`. The results topic is FIFO with content-based
    // deduplication (no MessageDeduplicationId is set), so re-publishing an
    // identical message is dedup-safe within the SNS dedup window.
    retry_transient(
        "SNS publish",
        SNS_RETRY_MAX_ATTEMPTS,
        SNS_RETRY_INITIAL_BACKOFF,
        || {
            let sns_client = sns_client.clone();
            let results_topic_arn = results_topic_arn.clone();
            let message = message.clone();
            let message_group_id = message_group_id.clone();
            let message_attributes = message_attributes.clone();
            async move {
                sns_client
                    .publish()
                    .topic_arn(results_topic_arn)
                    .message(message)
                    .message_group_id(message_group_id)
                    .set_message_attributes(Some(message_attributes))
                    .send()
                    .await
            }
        },
    )
    .await?;
    metrics::counter!("result.sent", "type" => metrics_message_type).increment(1);
    Ok(())
}

pub async fn send_error_results_to_sns(
    serialised_json_message: String,
    metadata: &BatchMetadata,
    sns_client: &SNSClient,
    config: &Config,
    base_message_attributes: &HashMap<String, MessageAttributeValue>,
    message_type: &str,
) -> Result<()> {
    let mut message_attributes = base_message_attributes.clone();
    let trace_attributes = construct_message_attributes(&metadata.trace_id, &metadata.span_id)?;
    message_attributes.extend(trace_attributes);
    send_message(
        config.results_topic_arn.clone(),
        serialised_json_message,
        sns_client,
        format!("party-id-{}", config.party_id),
        message_attributes,
        message_type.to_owned() + "_error",
    )
    .await?;
    Ok(())
}

pub async fn send_results_to_sns(
    result_events: Vec<String>,
    metadata: &[BatchMetadata],
    sns_client: &SNSClient,
    config: &Config,
    base_message_attributes: &HashMap<String, MessageAttributeValue>,
    message_type: &str,
) -> Result<()> {
    for (i, result_event) in result_events.iter().enumerate() {
        let mut message_attributes = base_message_attributes.clone();
        if metadata.len() > i {
            let trace_attributes =
                construct_message_attributes(&metadata[i].trace_id, &metadata[i].span_id)?;
            message_attributes.extend(trace_attributes);
        }
        send_message(
            config.results_topic_arn.clone(),
            result_event.clone(),
            sns_client,
            format!("party-id-{}", config.party_id),
            message_attributes,
            message_type.to_owned(),
        )
        .await?;
    }
    Ok(())
}
