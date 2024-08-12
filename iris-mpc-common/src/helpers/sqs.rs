use base64::{Engine, engine::general_purpose};
use serde::{Deserialize, Serialize};

use crate::iris_db::iris::IrisCodeArray;

use super::key_pair::{SharesDecodingError, SharesEncryptionKeyPair};

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
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SMPCRequest {
    pub signup_id: String,
    pub s3_presigned_url: String,
    pub iris_shares_file_hashes: [String; 3],
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharesS3Object {
    pub iris_share_0: String,
    pub iris_share_1: String,
    pub iris_share_2: String,
}

impl SMPCRequest {
    pub async fn retrieve_iris_shares_from_s3(
        &self,
        node_id: String,
        presign_url: String,
    ) -> Result<String, SharesDecodingError> {
        // Send a GET request to the presigned URL
        let response = match reqwest::get(presign_url.clone()).await {
            Ok(response) => response,
            Err(e) => {
                eprintln!("Failed to send request: {}", e);
                return Err(SharesDecodingError::RequestError(e));
            }
        };
        // Ensure the request was successful
        return if response.status().is_success() {
            // Parse the JSON response into the SharesS3Object struct
            let shares_file: SharesS3Object = match response.json().await {
                Ok(file) => file,
                Err(e) => {
                    eprintln!("Failed to parse JSON: {}", e);
                    return Err(SharesDecodingError::RequestError(e));
                }
            };
            // Convert the struct to serde_json::Value
            let shares_value = match serde_json::to_value(&shares_file) {
                Ok(value) => value,
                Err(e) => {
                    eprintln!("Failed to convert struct to value: {}", e);
                    return Err(SharesDecodingError::SerdeError(e));
                }
            };

            // Construct the field name dynamically
            let field_name = format!("iris_share_{}", node_id);
            // Access the field dynamically
            if let Some(value) = shares_value.get(&field_name) {
                return Ok(value.to_string());
            } else {
                eprintln!("Failed to find field: {}", field_name);
                Err(SharesDecodingError::SecretStringNotFound)
            }
        } else {
            eprintln!("Failed to download file: {}", response.status());
            Err(SharesDecodingError::ResponseContent {
                status: response.status(),
                url: presign_url,
                message: response.text().await.unwrap_or_default(),
            })
        };
    }

    fn validate_hashes(&self, hashes: [String; 3]) -> bool {
        self.iris_shares_file_hashes == hashes
    }

    async fn decode_bytes(
        &self,
        node_id: String,
        presign_url: String,
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> Result<[u16; IrisCodeArray::IRIS_CODE_SIZE], SharesDecodingError> {
        let retrieved_codes = match self.retrieve_iris_shares_from_s3(node_id, presign_url).await {
            Ok(codes) => codes,
            Err(e) => {
                eprintln!("Failed to retrieve iris shares: {}", e);
                return Err(e);
            }
        };

        let code_and_masks = general_purpose::STANDARD.decode(retrieved_codes).unwrap();
        // TODO: convert this to {signup_id, codes, masks, etc.}

        // let mut buffer = [0u16; IrisCodeArray::IRIS_CODE_SIZE];
        // if encrypted_shares {
        //     match decryption_key_pair.open_sealed_box(code) {
        //         Ok(decrypted_code) => {
        //             buffer.copy_from_slice(bytemuck::cast_slice(&decrypted_code));
        //             Ok(buffer)
        //         }
        //         Err(e) => Err(e),
        //     }
        // } else {
        //     buffer.copy_from_slice(bytemuck::cast_slice(&code));
        //     Ok(buffer)
        // }
    }
    // TODO: probably not needed
    pub fn get_codes_and_shares(
        &self,
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> Result<[u16; IrisCodeArray::IRIS_CODE_SIZE], SharesDecodingError> {
        Self::decode_bytes(
            self.iris_code.as_bytes(),
            encrypted_shares,
            decryption_key_pair,
        )
    }
    pub fn get_mask_shares(
        &self,
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> Result<[u16; IrisCodeArray::IRIS_CODE_SIZE], SharesDecodingError> {
        Self::decode_bytes(
            self.mask_code.as_bytes(),
            encrypted_shares,
            decryption_key_pair,
        )
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResultEvent {
    pub node_id: usize,
    pub db_index: u32,
    pub is_match: bool,
    pub request_id: String,
}

impl ResultEvent {
    pub fn new(node_id: usize, db_index: u32, is_match: bool, request_id: String) -> Self {
        Self {
            node_id,
            db_index,
            is_match,
            request_id,
        }
    }
}
