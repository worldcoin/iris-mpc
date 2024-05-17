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
    pub iris_code: Vec<u16>,
    pub mask_code: Vec<u16>,
}

impl From<SMPCRequest> for ShamirIris {
    fn from(request: SMPCRequest) -> Self {
        let mut iris = ShamirIris::default();
        for i in 0..iris.code.len() {
            iris.code[i] = request.iris_code[i];
            iris.mask[i] = request.mask_code[i];
        }
        iris
    }
}