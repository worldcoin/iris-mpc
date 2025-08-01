use aws_sdk_sns::types::MessageAttributeValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const SMPC_MESSAGE_TYPE_ATTRIBUTE: &str = "message_type";
// Error Reasons
pub const ERROR_FAILED_TO_PROCESS_IRIS_SHARES: &str = "failed_to_process_iris_shares";
pub const ERROR_SKIPPED_REQUEST_PREVIOUS_NODE_BATCH: &str = "skipped_request_previous_node_batch";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UniquenessResult {
    pub node_id: usize,
    pub serial_id: Option<u32>,
    pub is_match: bool,
    pub signup_id: String,
    pub matched_serial_ids: Option<Vec<u32>>,
    pub matched_serial_ids_left: Option<Vec<u32>>,
    pub matched_serial_ids_right: Option<Vec<u32>>,
    pub partial_matches_count_right: Option<usize>,
    pub partial_matches_count_left: Option<usize>,
    pub partial_match_rotation_indices_left: Option<Vec<Vec<i8>>>,
    pub partial_match_rotation_indices_right: Option<Vec<Vec<i8>>>,
    pub full_face_mirror_matched_serial_ids: Option<Vec<u32>>,
    pub full_face_mirror_matched_serial_ids_left: Option<Vec<u32>>,
    pub full_face_mirror_matched_serial_ids_right: Option<Vec<u32>>,
    pub full_face_mirror_partial_matches_count_left: Option<usize>,
    pub full_face_mirror_partial_matches_count_right: Option<usize>,
    pub matched_batch_request_ids: Option<Vec<String>>,
    pub error: Option<bool>,
    pub error_reason: Option<String>,
    pub full_face_mirror_attack_detected: bool,
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
        partial_matches_count_right: Option<usize>,
        partial_matches_count_left: Option<usize>,
        partial_match_rotation_indices_left: Option<Vec<Vec<i8>>>,
        partial_match_rotation_indices_right: Option<Vec<Vec<i8>>>,
        full_face_mirror_matched_serial_ids: Option<Vec<u32>>,
        full_face_mirror_matched_serial_ids_left: Option<Vec<u32>>,
        full_face_mirror_matched_serial_ids_right: Option<Vec<u32>>,
        full_face_mirror_partial_matches_count_left: Option<usize>,
        full_face_mirror_partial_matches_count_right: Option<usize>,
        full_face_mirror_attack_detected: bool,
    ) -> Self {
        Self {
            node_id,
            serial_id,
            is_match,
            signup_id,
            matched_serial_ids,
            matched_serial_ids_left,
            matched_serial_ids_right,
            full_face_mirror_matched_serial_ids,
            full_face_mirror_matched_serial_ids_left,
            full_face_mirror_matched_serial_ids_right,
            full_face_mirror_partial_matches_count_left,
            full_face_mirror_partial_matches_count_right,
            matched_batch_request_ids,
            partial_matches_count_right,
            partial_matches_count_left,
            partial_match_rotation_indices_left,
            partial_match_rotation_indices_right,
            error: None,
            error_reason: None,
            full_face_mirror_attack_detected,
        }
    }

    pub fn new_error_result(node_id: usize, signup_id: String, error_reason: &str) -> Self {
        Self {
            node_id,
            serial_id: None,
            is_match: false,
            signup_id,
            matched_serial_ids: None,
            matched_serial_ids_left: None,
            matched_serial_ids_right: None,
            matched_batch_request_ids: None,
            partial_matches_count_right: None,
            partial_matches_count_left: None,
            partial_match_rotation_indices_left: None,
            partial_match_rotation_indices_right: None,
            full_face_mirror_matched_serial_ids: None,
            full_face_mirror_matched_serial_ids_left: None,
            full_face_mirror_matched_serial_ids_right: None,
            full_face_mirror_partial_matches_count_left: None,
            full_face_mirror_partial_matches_count_right: None,
            error: Some(true),
            error_reason: Some(error_reason.to_string()),
            full_face_mirror_attack_detected: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IdentityDeletionResult {
    pub node_id: usize,
    pub serial_id: u32,
    pub success: bool,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReAuthResult {
    pub reauth_id: String,
    pub node_id: usize,
    pub serial_id: u32,
    pub success: bool,
    pub matched_serial_ids: Vec<u32>,
    pub or_rule_used: bool,
    pub error: Option<bool>,
    pub error_reason: Option<String>,
}

impl ReAuthResult {
    pub fn new(
        reauth_id: String,
        node_id: usize,
        serial_id: u32,
        success: bool,
        matched_serial_ids: Vec<u32>,
        or_rule_used: bool,
    ) -> Self {
        Self {
            reauth_id,
            node_id,
            serial_id,
            success,
            matched_serial_ids,
            or_rule_used,
            error: None,
            error_reason: None,
        }
    }

    pub fn new_error_result(
        reauth_id: String,
        node_id: usize,
        serial_id: u32,
        error_reason: &str,
    ) -> Self {
        Self {
            reauth_id,
            node_id,
            serial_id,
            success: false,
            matched_serial_ids: vec![],
            or_rule_used: false,
            error: Some(true),
            error_reason: Some(error_reason.to_string()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResetCheckResult {
    pub reset_id: String,
    pub node_id: usize,
    pub matched_serial_ids: Option<Vec<u32>>,
    pub matched_serial_ids_left: Option<Vec<u32>>,
    pub matched_serial_ids_right: Option<Vec<u32>>,
    pub matched_batch_request_ids: Option<Vec<String>>,
    pub partial_matches_count_right: Option<usize>,
    pub partial_matches_count_left: Option<usize>,
    pub error: Option<bool>,
    pub error_reason: Option<String>,
}

impl ResetCheckResult {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        reset_id: String,
        node_id: usize,
        matched_serial_ids: Option<Vec<u32>>,
        matched_serial_ids_left: Option<Vec<u32>>,
        matched_serial_ids_right: Option<Vec<u32>>,
        matched_batch_request_ids: Option<Vec<String>>,
        partial_matches_count_right: Option<usize>,
        partial_matches_count_left: Option<usize>,
    ) -> Self {
        Self {
            reset_id,
            node_id,
            matched_serial_ids,
            matched_serial_ids_left,
            matched_serial_ids_right,
            matched_batch_request_ids,
            partial_matches_count_right,
            partial_matches_count_left,
            error: None,
            error_reason: None,
        }
    }

    pub fn new_error_result(reset_id: String, node_id: usize, error_reason: &str) -> Self {
        Self {
            reset_id,
            node_id,
            matched_serial_ids: None,
            matched_serial_ids_left: None,
            matched_serial_ids_right: None,
            matched_batch_request_ids: None,
            partial_matches_count_right: None,
            partial_matches_count_left: None,
            error: Some(true),
            error_reason: Some(error_reason.to_string()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResetUpdateAckResult {
    pub reset_id: String,
    pub node_id: usize,
    pub serial_id: u32,
}

impl ResetUpdateAckResult {
    pub fn new(reset_id: String, node_id: usize, serial_id: u32) -> Self {
        Self {
            reset_id,
            node_id,
            serial_id,
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
