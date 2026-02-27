use super::{key_pair::SharesDecodingError, sha256::sha256_as_hex_string};
use crate::helpers::key_pair::SharesEncryptionKeyPairs;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sqs::{
    error::SdkError,
    operation::{delete_message::DeleteMessageError, receive_message::ReceiveMessageError},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use eyre::Report;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

#[derive(Serialize, Deserialize, Debug)]
pub struct SQSMessage {
    #[serde(rename = "Type")]
    pub notification_type: String,
    #[serde(rename = "MessageId")]
    pub message_id: String,
    #[serde(rename = "SequenceNumber")]
    pub sequence_number: String,
    #[serde(rename = "TopicArn")]
    pub topic_arn: String,
    #[serde(rename = "Message")]
    pub message: String,
    #[serde(rename = "Timestamp")]
    pub timestamp: String,
    #[serde(rename = "UnsubscribeURL")]
    pub unsubscribe_url: String,
    #[serde(
        rename = "MessageAttributes",
        serialize_with = "serialize_message_attributes",
        deserialize_with = "deserialize_message_attributes"
    )]
    pub message_attributes: HashMap<String, MessageAttributeValue>,
}

// Deserialize message attributes map from SQS body.
// For simplicity, it only deserializes attributes of type String.
// Update this function if other types are needed (String.Array, Number, and
// Binary).
fn deserialize_message_attributes<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, MessageAttributeValue>, D::Error>
where
    D: Deserializer<'de>,
{
    let attributes: HashMap<String, Value> = HashMap::deserialize(deserializer)?;
    let mut result: HashMap<String, MessageAttributeValue> = HashMap::new();

    for (key, value) in attributes.into_iter() {
        let attr_type = value.get("Type").and_then(|v| v.as_str());
        let attr_value = value.get("Value").and_then(|v| v.as_str());

        if let Some(message_attr_value) = process_attribute(&key, attr_type, attr_value) {
            result.insert(key, message_attr_value);
        }
    }

    Ok(result)
}

fn process_attribute(
    key: &String,
    attr_type: Option<&str>,
    attr_value: Option<&str>,
) -> Option<MessageAttributeValue> {
    let attr_type = attr_type?;
    let attr_value = attr_value?;

    if attr_type != "String" {
        tracing::warn!("Skipped deserializing attribute of type {}", attr_type);
        return None;
    }

    match MessageAttributeValue::builder()
        .data_type(attr_type.to_string())
        .string_value(attr_value.to_string())
        .build()
    {
        Ok(message_attr_value) => Some(message_attr_value),
        Err(e) => {
            tracing::warn!("Failed to build MessageAttributeValue {}: {:?}", key, e);
            None
        }
    }
}

// MessageAttributes serialization placeholder. It's left empty as we do not
// send messages to SQS.
fn serialize_message_attributes<S>(
    _: &HashMap<String, MessageAttributeValue>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let map = serde_json::map::Map::new();
    map.serialize(serializer)
}

pub const IDENTITY_DELETION_MESSAGE_TYPE: &str = "identity_deletion";
pub const ANONYMIZED_STATISTICS_MESSAGE_TYPE: &str = "anonymized_statistics";
pub const ANONYMIZED_STATISTICS_2D_MESSAGE_TYPE: &str = "anonymized_statistics_2d";
pub const CIRCUIT_BREAKER_MESSAGE_TYPE: &str = "circuit_breaker";
pub const UNIQUENESS_MESSAGE_TYPE: &str = "uniqueness";
pub const REAUTH_MESSAGE_TYPE: &str = "reauth";
pub const RESET_CHECK_MESSAGE_TYPE: &str = "reset_check";
pub const RECOVERY_CHECK_MESSAGE_TYPE: &str = "recovery_check";
pub const RESET_UPDATE_MESSAGE_TYPE: &str = "reset_update";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UniquenessRequest {
    pub batch_size: Option<usize>,
    pub signup_id: String,
    pub s3_key: String,
    pub or_rule_serial_ids: Option<Vec<u32>>,
    pub skip_persistence: Option<bool>,
    pub full_face_mirror_attacks_detection_enabled: Option<bool>,
    // If true, do not collect or compute anonymized statistics for this batch.
    pub disable_anonymized_stats: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CircuitBreakerRequest {
    pub batch_size: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IdentityDeletionRequest {
    pub serial_id: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReAuthRequest {
    pub reauth_id: String,
    pub batch_size: Option<usize>,
    pub s3_key: String,
    pub serial_id: u32,
    pub skip_persistence: Option<bool>,
    pub use_or_rule: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IdentityMatchCheckRequest {
    pub request_id: String,
    pub batch_size: Option<usize>,
    pub s3_key: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResetUpdateRequest {
    pub reset_id: String,
    pub serial_id: u32,
    pub s3_key: String,
}

#[derive(Error, Debug)]
pub enum ReceiveRequestError {
    #[error("Failed to read from request SQS: {0}")]
    FailedToReadFromSQS(#[from] Box<SdkError<ReceiveMessageError>>),

    #[error("Failed to delete request from SQS: {0}")]
    FailedToDeleteFromSQS(#[from] Box<SdkError<DeleteMessageError>>),

    #[error("Failed to mark request as deleted in the database: {0}")]
    FailedToMarkRequestAsDeleted(#[from] Report),

    #[error("Failed to parse {json_name} JSON: {err}")]
    JsonParseError {
        json_name: String,
        err: serde_json::Error,
    },

    #[error("Request does not contain a message type attribute")]
    NoMessageTypeAttribute,

    #[error("Request does not contain a string message type attribute")]
    NoStringMessageTypeAttribute,

    #[error("Message type attribute is not valid")]
    InvalidMessageType,

    #[error("Failed to join receive handle: {0}")]
    FailedToJoinHandle(#[from] tokio::task::JoinError),

    #[error("Failed to synchronize batch states: {0}")]
    BatchSyncError(Report),
    #[error("Batch polling timeout reached after {0} seconds")]
    BatchPollingTimeout(i32),
    #[error("Failed to parse shares: {0}")]
    FailedToProcessIrisShares(Report),
}

impl From<SdkError<ReceiveMessageError>> for ReceiveRequestError {
    fn from(value: SdkError<ReceiveMessageError>) -> Self {
        Self::FailedToReadFromSQS(Box::new(value))
    }
}

impl From<SdkError<DeleteMessageError>> for ReceiveRequestError {
    fn from(value: SdkError<DeleteMessageError>) -> Self {
        Self::FailedToDeleteFromSQS(Box::new(value))
    }
}

impl ReceiveRequestError {
    pub fn json_parse_error(json_name: &str, err: serde_json::error::Error) -> Self {
        ReceiveRequestError::JsonParseError {
            json_name: json_name.to_string(),
            err,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharesS3Object {
    pub iris_share_0: String,
    pub iris_share_1: String,
    pub iris_share_2: String,
    pub iris_hashes_0: String,
    pub iris_hashes_1: String,
    pub iris_hashes_2: String,
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
pub struct IrisCodeSharesJSON {
    #[serde(rename = "IRIS_version")]
    pub iris_version: String,
    #[serde(rename = "IRIS_shares_version")]
    pub iris_shares_version: String,
    pub left_iris_code_shares: String, // these are base64 encoded strings
    pub right_iris_code_shares: String, // these are base64 encoded strings
    pub left_mask_code_shares: String, // these are base64 encoded strings
    pub right_mask_code_shares: String, // these are base64 encoded strings
}

impl SharesS3Object {
    pub fn get(&self, party_id: usize) -> Option<(&str, &str)> {
        match party_id {
            0 => Some((&self.iris_share_0, &self.iris_hashes_0)),
            1 => Some((&self.iris_share_1, &self.iris_hashes_1)),
            2 => Some((&self.iris_share_2, &self.iris_hashes_2)),
            _ => None,
        }
    }
}

pub async fn get_iris_data_by_party_id(
    s3_key: &str,
    party_id: usize,
    bucket_name: &String,
    s3_client: &S3Client,
) -> Result<(String, String), SharesDecodingError> {
    let response = s3_client
        .get_object()
        .bucket(bucket_name)
        .key(s3_key)
        .send()
        .await
        .map_err(|err| {
            tracing::error!("Failed to download file: {}", err);
            SharesDecodingError::S3ResponseContent {
                key: s3_key.to_string(),
                message: err.to_string(),
            }
        })?;

    let object_body = response.body.collect().await.map_err(|e| {
        tracing::error!("Failed to get object body: {}", e);
        SharesDecodingError::S3ResponseContent {
            key: s3_key.to_string(),
            message: e.to_string(),
        }
    })?;

    let bytes = object_body.into_bytes();

    let shares_file: SharesS3Object = serde_json::from_slice(&bytes)?;

    let field_name = format!("iris_share_{}", party_id);

    let share_and_hash_opt = shares_file.get(party_id);
    match share_and_hash_opt {
        Some(share_and_hash) => Ok((share_and_hash.0.to_string(), share_and_hash.1.to_string())),
        _ => {
            tracing::error!("Failed to find field: {}", field_name);
            Err(SharesDecodingError::SecretStringNotFound)
        }
    }
}

pub fn decrypt_iris_share(
    share: String,
    key_pairs: SharesEncryptionKeyPairs,
) -> Result<IrisCodeSharesJSON, SharesDecodingError> {
    let share_bytes = STANDARD
        .decode(share.as_bytes())
        .map_err(|_| SharesDecodingError::Base64DecodeError)?;

    // try decrypting with key_pairs.current_key_pair, if it fails, try decrypting
    // with key_pairs.previous_key_pair (if it exists, otherwise, return an error)
    let decrypted = match key_pairs
        .current_key_pair
        .open_sealed_box(share_bytes.clone())
    {
        Ok(bytes) => Ok(bytes),
        Err(_) => {
            match if let Some(key_pair) = key_pairs.previous_key_pair.clone() {
                key_pair.open_sealed_box(share_bytes)
            } else {
                Err(SharesDecodingError::PreviousKeyNotFound)
            } {
                Ok(bytes) => Ok(bytes),
                Err(_) => Err(SharesDecodingError::SealedBoxOpenError),
            }
        }
    };

    let iris_share = match decrypted {
        Ok(bytes) => {
            let json_string = String::from_utf8(bytes)
                .map_err(SharesDecodingError::DecodedShareParsingToUTF8Error)?;

            let iris_share: IrisCodeSharesJSON =
                serde_json::from_str(&json_string).map_err(SharesDecodingError::SerdeError)?;
            iris_share
        }
        Err(e) => return Err(e),
    };

    Ok(iris_share)
}

pub fn validate_iris_share(
    hash: String,
    share: IrisCodeSharesJSON,
) -> Result<bool, SharesDecodingError> {
    let stringified_share = serde_json::to_string(&share)
        .map_err(SharesDecodingError::SerdeError)?
        .into_bytes();

    Ok(hash == sha256_as_hex_string(stringified_share))
}
