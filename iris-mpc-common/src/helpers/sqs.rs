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
    // TODO: make this a message attribute, but the SQS message will anyways be refactored soon.
    pub batch_size: Option<usize>,
    pub request_id: String,
    pub iris_code:  String,
    pub mask_code:  String,
}

impl SMPCRequest {
    fn decode_bytes(bytes: &[u8]) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
        let code = general_purpose::STANDARD.decode(bytes).unwrap();
        let mut buffer = [0u16; IrisCodeArray::IRIS_CODE_SIZE];
        buffer.copy_from_slice(bytemuck::cast_slice(&code));
        buffer
    }
    pub fn get_iris_shares(&self) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
        Self::decode_bytes(self.iris_code.as_bytes())
    }
    pub fn get_mask_shares(&self) -> [u16; IrisCodeArray::IRIS_CODE_SIZE] {
        Self::decode_bytes(self.mask_code.as_bytes())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResultEvent {
    pub node_id:            usize,
    pub serial_id:          u32,
    pub is_match:           bool,
    pub signup_id:          String,
    pub matched_serial_ids: Vec<u32>,
}

impl ResultEvent {
    pub fn new(
        node_id: usize,
        serial_id: u32,
        is_match: bool,
        signup_id: String,
        matched_serial_ids: Vec<u32>,
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
