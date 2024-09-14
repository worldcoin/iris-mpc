use aws_sdk_s3::{operation::put_object::PutObjectOutput, Client as S3Client, Error as S3Error};
use aws_sdk_secretsmanager::{
    operation::{
        get_random_password::GetRandomPasswordOutput, get_secret_value::GetSecretValueOutput,
        put_secret_value::PutSecretValueOutput,
    },
    Client as SecretsManagerClient, Error as SecretsManagerError,
};
use aws_sdk_sns::types::MessageAttributeValue;
use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceId, TraceState};
use reqwest::Client;
use serde_json::json;
use std::collections::HashMap;

pub const TRACE_ID_MESSAGE_ATTRIBUTE_NAME: &str = "TraceID";
pub const SPAN_ID_MESSAGE_ATTRIBUTE_NAME: &str = "SpanID";
pub const NODE_ID_MESSAGE_ATTRIBUTE_NAME: &str = "NodeID";

pub fn construct_message_attributes() -> eyre::Result<HashMap<String, MessageAttributeValue>> {
    let (trace_id, span_id) = telemetry_batteries::tracing::extract_span_ids();

    let mut message_attributes = HashMap::new();

    let trace_id_message_attribute = MessageAttributeValue::builder()
        .data_type("String")
        .string_value(trace_id.to_string())
        .build()?;

    message_attributes.insert(
        TRACE_ID_MESSAGE_ATTRIBUTE_NAME.to_string(),
        trace_id_message_attribute,
    );

    let span_id_message_attribute = MessageAttributeValue::builder()
        .data_type("String")
        .string_value(span_id.to_string())
        .build()?;

    message_attributes.insert(
        SPAN_ID_MESSAGE_ATTRIBUTE_NAME.to_string(),
        span_id_message_attribute,
    );

    Ok(message_attributes)
}

// This would only ever be leveraged if the code had isolated flows for every
// message, leaving for now, maybe it will happen in the future
pub fn trace_from_message_attributes(
    message_attributes: &HashMap<String, MessageAttributeValue>,
    receipt_handle: &str,
) -> eyre::Result<()> {
    if let Some(trace_id) = message_attributes.get(TRACE_ID_MESSAGE_ATTRIBUTE_NAME) {
        if let Some(span_id) = message_attributes.get(SPAN_ID_MESSAGE_ATTRIBUTE_NAME) {
            let trace_id = trace_id
                .string_value()
                .expect("Could not parse TraceID")
                .parse::<u128>()?;

            let span_id = span_id
                .string_value()
                .expect("Could not parse SpanID")
                .parse::<u64>()?;

            // Create and set the span parent context
            let parent_ctx = SpanContext::new(
                TraceId::from(trace_id),
                SpanId::from(span_id),
                TraceFlags::default(),
                true,
                TraceState::default(),
            );

            telemetry_batteries::tracing::trace_from_ctx(parent_ctx);
        } else {
            tracing::warn!(?receipt_handle, "SQS message missing SpanID");
        }
    } else {
        tracing::warn!(?receipt_handle, "SQS message missing TraceID");
    }

    Ok(())
}

pub async fn download_key_from_s3(env: &str, node_id: &str) -> Result<String, reqwest::Error> {
    let suffix = match env {
        "prod" => ".worldcoin.org",
        "stage" => "-stage.worldcoin.org",
        _ => "dev.worldcoin.org",
    };
    let s3_access_uri = format!("https://pki-smpc{}/public-key-{}", suffix, node_id);
    print!("Downloading key from endpoint: {}", s3_access_uri);
    let client = Client::new();
    let response = client.get(&s3_access_uri).send().await?.text().await?;
    Ok(response)
}

pub async fn download_key_from_asm(
    client: &SecretsManagerClient,
    secret_id: &str,
    version_stage: &str,
) -> Result<GetSecretValueOutput, SecretsManagerError> {
    Ok(client
        .get_secret_value()
        .secret_id(secret_id)
        .version_stage(version_stage)
        .send()
        .await?)
}

pub async fn upload_private_key_to_asm(
    client: &SecretsManagerClient,
    secret_id: &str,
    content: &str,
) -> Result<PutSecretValueOutput, SecretsManagerError> {
    Ok(client
        .put_secret_value()
        .secret_string(content)
        .secret_id(secret_id)
        .send()
        .await?)
}

pub async fn get_secret_string_from_asm(
    client: &SecretsManagerClient,
    secret_id: &str,
) -> Result<GetSecretValueOutput, SecretsManagerError> {
    Ok(client
        .get_secret_value()
        .secret_id(secret_id)
        .send()
        .await?)
}

pub async fn get_random_password(
    client: &SecretsManagerClient,
) -> Result<GetRandomPasswordOutput, SecretsManagerError> {
    Ok(client
        .get_random_password()
        .password_length(128)
        .send()
        .await?)
}

pub async fn create_secret_string_with_asm(
    sm_client: &SecretsManagerClient,
    private_key_seed_secret_id: &str,
) -> Result<String, SecretsManagerError> {
    let new_secret_string = get_random_password(sm_client)
        .await?
        .random_password
        .unwrap();

    // Serialize the secret into a JSON format
    let secret_json = json!({ "seed": new_secret_string }).to_string();

    // Create the secret in AWS Secrets Manager
    sm_client
        .create_secret()
        .name(private_key_seed_secret_id)
        .secret_string(&secret_json)
        .send()
        .await?;

    Ok(secret_json)
}

pub async fn upload_public_key_to_s3(
    client: &S3Client,
    bucket: &str,
    key: &str,
    content: &str,
) -> Result<PutObjectOutput, S3Error> {
    Ok(client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(content.to_string().into_bytes().into())
        .send()
        .await?)
}
