use super::key_pair::{SecretError, SharesEncryptionKeyPair};
use crate::iris_db::iris::IrisCodeArray;
use base64::{engine::general_purpose, Engine};
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
    pub request_id: String,
    pub iris_code:  String,
    pub mask_code:  String,
}

impl SMPCRequest {
    fn decode_bytes(
        bytes: &[u8],
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> Result<[u16; IrisCodeArray::IRIS_CODE_SIZE], SecretError> {
        let code = general_purpose::STANDARD.decode(bytes).unwrap();
        let mut buffer = [0u16; IrisCodeArray::IRIS_CODE_SIZE];
        if encrypted_shares {
            match decryption_key_pair.open_sealed_box(code) {
                Ok(decrypted_code) => {
                    buffer.copy_from_slice(bytemuck::cast_slice(&decrypted_code));
                    Ok(buffer)
                }
                Err(e) => Err(e),
            }
        } else {
            buffer.copy_from_slice(bytemuck::cast_slice(&code));
            Ok(buffer)
        }
    }
    pub fn get_iris_shares(
        &self,
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> Result<[u16; IrisCodeArray::IRIS_CODE_SIZE], SecretError> {
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
    ) -> Result<[u16; IrisCodeArray::IRIS_CODE_SIZE], SecretError> {
        Self::decode_bytes(
            self.mask_code.as_bytes(),
            encrypted_shares,
            decryption_key_pair,
        )
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResultEvent {
    pub node_id:    usize,
    pub db_index:   u32,
    pub is_match:   bool,
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
