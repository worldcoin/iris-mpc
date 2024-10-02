use super::{key_pair::SharesDecodingError, sha256::calculate_sha256};
use crate::helpers::key_pair::SharesEncryptionKeyPairs;
use aws_sdk_sns::types::MessageAttributeValue;
use aws_sdk_sqs::{
    error::SdkError,
    operation::{delete_message::DeleteMessageError, receive_message::ReceiveMessageError},
};
use base64::{engine::general_purpose::STANDARD, Engine};
use eyre::Report;
use reqwest::Client;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::{collections::HashMap, sync::LazyLock};
use thiserror::Error;
use tokio_retry::{
    strategy::{jitter, FixedInterval},
    Retry,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct SQSMessage {
    #[serde(rename = "Type")]
    pub notification_type:  String,
    #[serde(rename = "MessageId")]
    pub message_id:         String,
    #[serde(rename = "SequenceNumber")]
    pub sequence_number:    String,
    #[serde(rename = "TopicArn")]
    pub topic_arn:          String,
    #[serde(rename = "Message")]
    pub message:            String,
    #[serde(rename = "Timestamp")]
    pub timestamp:          String,
    #[serde(rename = "UnsubscribeURL")]
    pub unsubscribe_url:    String,
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

pub const SMPC_MESSAGE_TYPE_ATTRIBUTE: &str = "message_type";
pub const IDENTITY_DELETION_MESSAGE_TYPE: &str = "identity_deletion";
pub const CIRCUIT_BREAKER_MESSAGE_TYPE: &str = "circuit_breaker";
pub const UNIQUENESS_MESSAGE_TYPE: &str = "uniqueness";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UniquenessRequest {
    pub batch_size:              Option<usize>,
    pub signup_id:               String,
    pub s3_presigned_url:        String,
    pub iris_shares_file_hashes: [String; 3],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CircuitBreakerRequest {
    pub batch_size: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IdentityDeletionRequest {
    pub serial_id: u32,
}

#[derive(Error, Debug)]
pub enum ReceiveRequestError {
    #[error("Failed to read from request SQS: {0}")]
    FailedToReadFromSQS(#[from] SdkError<ReceiveMessageError>),

    #[error("Failed to delete request from SQS: {0}")]
    FailedToDeleteFromSQS(#[from] SdkError<DeleteMessageError>),

    #[error("Failed to mark request as deleted in the database: {0}")]
    FailedToMarkRequestAsDeleted(#[from] Report),

    #[error("Failed to parse {json_name} JSON: {err}")]
    JsonParseError {
        json_name: String,
        err:       serde_json::Error,
    },

    #[error("Request does not contain a message type attribute")]
    NoMessageTypeAttribute,

    #[error("Request does not contain a string message type attribute")]
    NoStringMessageTypeAttribute,

    #[error("Message type attribute is not valid")]
    InvalidMessageType,

    #[error("Failed to join receive handle: {0}")]
    FailedToJoinHandle(#[from] tokio::task::JoinError),
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
}

#[derive(PartialEq, Serialize, Deserialize, Debug, Clone)]
pub struct IrisCodesJSON {
    #[serde(rename = "IRIS_version")]
    pub iris_version:           String,
    #[serde(rename = "IRIS_shares_version")]
    pub iris_shares_version:    String,
    pub left_iris_code_shares:  String, // these are base64 encoded strings
    pub right_iris_code_shares: String, // these are base64 encoded strings
    pub left_mask_code_shares:  String, // these are base64 encoded strings
    pub right_mask_code_shares: String, // these are base64 encoded strings
}

impl SharesS3Object {
    pub fn get(&self, party_id: usize) -> Option<&String> {
        match party_id {
            0 => Some(&self.iris_share_0),
            1 => Some(&self.iris_share_1),
            2 => Some(&self.iris_share_2),
            _ => None,
        }
    }
}

static S3_HTTP_CLIENT: LazyLock<Client> = LazyLock::new(Client::new);

impl UniquenessRequest {
    pub async fn get_iris_data_by_party_id(
        &self,
        party_id: usize,
    ) -> Result<String, SharesDecodingError> {
        // Send a GET request to the presigned URL
        let retry_strategy = FixedInterval::from_millis(200).map(jitter).take(5);
        let response = Retry::spawn(retry_strategy, || async {
            S3_HTTP_CLIENT
                .get(self.s3_presigned_url.clone())
                .send()
                .await
        })
        .await?;

        // Ensure the request was successful
        if response.status().is_success() {
            // Parse the JSON response into the SharesS3Object struct
            let shares_file: SharesS3Object = match response.json().await {
                Ok(file) => file,
                Err(e) => {
                    tracing::error!("Failed to parse JSON: {}", e);
                    return Err(SharesDecodingError::RequestError(e));
                }
            };

            // Construct the field name dynamically
            let field_name = format!("iris_share_{}", party_id);
            // Access the field dynamically
            if let Some(value) = shares_file.get(party_id) {
                Ok(value.to_string())
            } else {
                tracing::error!("Failed to find field: {}", field_name);
                Err(SharesDecodingError::SecretStringNotFound)
            }
        } else {
            tracing::error!("Failed to download file: {}", response.status());
            Err(SharesDecodingError::ResponseContent {
                status:  response.status(),
                url:     self.s3_presigned_url.clone(),
                message: response.text().await.unwrap_or_default(),
            })
        }
    }

    pub fn decrypt_iris_share(
        &self,
        share: String,
        key_pairs: SharesEncryptionKeyPairs,
    ) -> Result<IrisCodesJSON, SharesDecodingError> {
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

                let iris_share: IrisCodesJSON =
                    serde_json::from_str(&json_string).map_err(SharesDecodingError::SerdeError)?;
                iris_share
            }
            Err(e) => return Err(e),
        };

        Ok(iris_share)
    }

    pub fn validate_iris_share(
        &self,
        party_id: usize,
        share: IrisCodesJSON,
    ) -> Result<bool, SharesDecodingError> {
        let stringified_share = serde_json::to_string(&share)
            .map_err(SharesDecodingError::SerdeError)?
            .into_bytes();

        Ok(self.iris_shares_file_hashes[party_id] == calculate_sha256(stringified_share))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UniquenessResult {
    pub node_id:            usize,
    pub serial_id:          Option<u32>,
    pub is_match:           bool,
    pub signup_id:          String,
    pub matched_serial_ids: Option<Vec<u32>>,
}

impl UniquenessResult {
    pub fn new(
        node_id: usize,
        serial_id: Option<u32>,
        is_match: bool,
        signup_id: String,
        matched_serial_ids: Option<Vec<u32>>,
    ) -> Self {
        Self {
            node_id,
            serial_id,
            is_match,
            signup_id,
            matched_serial_ids,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IdentityDeletionResult {
    pub node_id:   usize,
    pub serial_id: u32,
    pub success:   bool,
}

impl IdentityDeletionResult {
    pub fn new(node_id: usize, serial_id: u32, success: bool) -> Self {
        Self {
            node_id,
            serial_id,
            success,
        }
    }
}

pub fn create_message_type_attribute_map(
    message_type: &str,
) -> HashMap<String, MessageAttributeValue> {
    let mut message_attributes_map = HashMap::new();
    let message_type_value = MessageAttributeValue::builder()
        .data_type("String")
        .string_value(message_type)
        .build()
        .unwrap();
    message_attributes_map.insert(SMPC_MESSAGE_TYPE_ATTRIBUTE.to_string(), message_type_value);
    message_attributes_map
}
