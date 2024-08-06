use super::key_pair::SharesEncryptionKeyPair;
use crate::setup::iris_db::iris::IrisCodeArray;
use base64::{engine::general_purpose, Engine};
use serde::{Deserialize, Serialize};
use sodiumoxide::crypto::{sealedbox, sealedbox::SEALBYTES};

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
    fn decrypt_share(
        code: Vec<u8>,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
        let mut buffer = [0u8; IrisCodeArray::IRIS_CODE_SIZE * 2 + SEALBYTES];
        buffer.copy_from_slice(bytemuck::cast_slice(&code));
        let decrypted = sealedbox::open(&buffer, &decryption_key_pair.pk, &decryption_key_pair.sk);
        match decrypted {
            Ok(bytes) => {
                let mut buffer = [0u16; IrisCodeArray::IRIS_CODE_SIZE];
                buffer.copy_from_slice(bytemuck::cast_slice(&bytes));
                buffer
            }
            Err(_) => panic!("Failed to decrypt iris code"),
        }
    }

    fn decode_bytes(
        bytes: &[u8],
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
        let code = general_purpose::STANDARD.decode(bytes).unwrap();
        if encrypted_shares {
            return Self::decrypt_share(code, decryption_key_pair);
        }
        let mut buffer = [0u16; IrisCodeArray::IRIS_CODE_SIZE];
        buffer.copy_from_slice(bytemuck::cast_slice(&code));
        buffer
    }
    pub fn get_iris_shares(
        &self,
        encrypted_shares: bool,
        decryption_key_pair: SharesEncryptionKeyPair,
    ) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
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
    ) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
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
