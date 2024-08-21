use super::{
    key_pair::{SharesDecodingError, SharesEncryptionKeyPair},
    serialize_with_sorted_keys::SerializeWithSortedKeys,
    sha256::calculate_sha256,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SQSMessage {
    #[serde(rename = "Type")]
    pub notification_type: String,
    #[serde(rename = "MessageId")]
    pub message_id:        String,
    #[serde(rename = "SequenceNumber")]
    pub sequence_number:   String,
    #[serde(rename = "TopicArn")]
    pub topic_arn:         String,
    #[serde(rename = "Message")]
    pub message:           String,
    #[serde(rename = "Timestamp")]
    pub timestamp:         String,
    #[serde(rename = "UnsubscribeURL")]
    pub unsubscribe_url:   String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SMPCRequest {
    pub batch_size:              Option<usize>,
    pub signup_id:               String,
    pub s3_presigned_url:        String,
    pub iris_shares_file_hashes: [String; 3],
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
    pub left_iris_code_shares:  String, // these are base64 encoded strings
    pub right_iris_code_shares: String, // these are base64 encoded strings
    pub left_iris_mask_shares:  String, // these are base64 encoded strings
    pub right_iris_mask_shares: String, // these are base64 encoded strings
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

impl SMPCRequest {
    pub async fn get_iris_data_by_party_id(
        &self,
        party_id: usize,
    ) -> Result<String, SharesDecodingError> {
        // Send a GET request to the presigned URL
        let response = match reqwest::get(self.s3_presigned_url.clone()).await {
            Ok(response) => response,
            Err(e) => {
                tracing::error!("Failed to send request: {}", e);
                return Err(SharesDecodingError::RequestError(e));
            }
        };

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
        key_pair: SharesEncryptionKeyPair,
    ) -> Result<IrisCodesJSON, SharesDecodingError> {
        let share_bytes = STANDARD
            .decode(share.as_bytes())
            .map_err(|_| SharesDecodingError::Base64DecodeError)?;

        let decrypted = key_pair.open_sealed_box(share_bytes);

        let iris_share = match decrypted {
            Ok(bytes) => {
                let json_string = String::from_utf8(bytes)
                    .map_err(SharesDecodingError::DecodedShareParsingToUTF8Error)?;

                tracing::info!("shares_json_string: {:?}", json_string);
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
        let stringified_share = serde_json::to_string(&SerializeWithSortedKeys(&share))
            .map_err(SharesDecodingError::SerdeError)?
            .into_bytes();

        Ok(self.iris_shares_file_hashes[party_id] == calculate_sha256(stringified_share))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResultEvent {
    pub node_id:            usize,
    pub serial_id:          Option<u32>,
    pub is_match:           bool,
    pub signup_id:          String,
    pub matched_serial_ids: Option<Vec<u32>>,
}

impl ResultEvent {
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
