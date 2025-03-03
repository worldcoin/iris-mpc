use aws_sdk_sns::{types::MessageAttributeValue, Client as SNSClient};
use eyre::Result;
use iris_mpc_common::{
    config::Config, helpers::aws::construct_message_attributes, job::BatchMetadata,
};
use std::collections::HashMap;

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
    sns_client
        .publish()
        .topic_arn(&config.results_topic_arn)
        .message(serialised_json_message)
        .message_group_id(format!("party-id-{}", config.party_id))
        .set_message_attributes(Some(message_attributes))
        .send()
        .await?;
    metrics::counter!("result.sent", "type" => message_type.to_owned()+"_error").increment(1);

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
        sns_client
            .publish()
            .topic_arn(&config.results_topic_arn)
            .message(result_event)
            .message_group_id(format!("party-id-{}", config.party_id))
            .set_message_attributes(Some(message_attributes))
            .send()
            .await?;
        metrics::counter!("result.sent", "type" => message_type.to_owned()).increment(1);
    }
    Ok(())
}
