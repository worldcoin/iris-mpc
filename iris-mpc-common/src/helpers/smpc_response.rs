use aws_sdk_sns::types::MessageAttributeValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const SMPC_MESSAGE_TYPE_ATTRIBUTE: &str = "message_type";
// Error Reasons
pub const ERROR_FAILED_TO_PROCESS_IRIS_SHARES: &str = "failed_to_process_iris_shares";
pub const ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH: &str = "skipped_request_previous_node_batch";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UniquenessResult {
    pub node_id:                   usize,
    pub serial_id:                 Option<u32>,
    pub is_match:                  bool,
    pub signup_id:                 String,
    pub matched_serial_ids:        Option<Vec<u32>>,
    pub matched_serial_ids_left:   Option<Vec<u32>>,
    pub matched_serial_ids_right:  Option<Vec<u32>>,
    pub matched_batch_request_ids: Option<Vec<String>>,
    pub error:                     Option<bool>,
    pub error_reason:              Option<String>,
}

impl UniquenessResult {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_id: usize,
        serial_id: Option<u32>,
        is_match: bool,
        signup_id: String,
        matched_serial_ids: Option<Vec<u32>>,
        matched_serial_ids_left: Option<Vec<u32>>,
        matched_serial_ids_right: Option<Vec<u32>>,
        matched_batch_request_ids: Option<Vec<String>>,
    ) -> Self {
        Self {
            node_id,
            serial_id,
            is_match,
            signup_id,
            matched_serial_ids,
            matched_serial_ids_left,
            matched_serial_ids_right,
            matched_batch_request_ids,
            error: None,
            error_reason: None,
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
