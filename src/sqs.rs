use base64::{engine::general_purpose, Engine};
use serde::{Deserialize, Serialize};

use crate::setup::iris_db::shamir_iris::ShamirIris;

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
    pub message: SMPCRequest,
    #[serde(rename = "Timestamp")]
    pub timestamp: String,
    #[serde(rename = "UnsubscribeURL")]
    pub unsubscribe_url: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SMPCRequest {
    pub request_type: String,
    pub request_id: String,
    pub iris_code: String,
    pub mask_code: String,
}

impl From<SMPCRequest> for ShamirIris {
    fn from(request: SMPCRequest) -> Self {
        let mut iris = ShamirIris::default();
        let code = general_purpose::STANDARD.decode(request.iris_code.as_bytes()).unwrap();
        let mask = general_purpose::STANDARD.decode(request.mask_code.as_bytes()).unwrap();
        iris.code.copy_from_slice(bytemuck::cast_slice(&code));
        iris.mask.copy_from_slice(bytemuck::cast_slice(&mask));
        iris
    }
}