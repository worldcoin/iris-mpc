use std::fmt;

use serde::{Deserialize, Serialize};

use super::{RequestBatch, RequestStatus, ResponsePayload};
use crate::constants::N_PARTIES;

/// Encapsulates common data pertinent to a system processing request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestInfo {
    /// Associated request batch ordinal identifier.
    batch_idx: usize,

    /// Associated request batch item ordinal identifier.
    batch_item_idx: usize,

    /// System responses returned by MPC nodes (one per party).
    responses: [Option<ResponsePayload>; N_PARTIES],

    /// User assigned label ... used to associate child/parent requests.
    label: Option<String>,

    /// Set of processing states.
    state_history: Vec<RequestStatus>,

    /// Associated unique identifier.
    uid: uuid::Uuid,
}

impl RequestInfo {
    pub fn new(batch: &RequestBatch, label: Option<String>) -> Self {
        Self::with_indices(batch.batch_idx(), batch.next_item_idx(), label)
    }

    pub fn with_indices(batch_idx: usize, batch_item_idx: usize, label: Option<String>) -> Self {
        let mut state_history = Vec::with_capacity(RequestStatus::VARIANT_COUNT);
        state_history.push(RequestStatus::default());

        Self {
            batch_idx,
            batch_item_idx,
            responses: [const { None }; N_PARTIES],
            label,
            state_history,
            uid: uuid::Uuid::new_v4(),
        }
    }

    pub fn label(&self) -> &Option<String> {
        &self.label
    }

    pub fn uid(&self) -> &uuid::Uuid {
        &self.uid
    }

    pub fn is_complete(&self) -> bool {
        self.responses.iter().all(|c| c.is_some())
    }

    pub fn status(&self) -> &RequestStatus {
        self.state_history.last().unwrap()
    }

    /// Records a response from a node. Returns true if all parties have now responded.
    pub fn record_response(&mut self, response: &ResponsePayload) -> bool {
        self.responses[response.node_id()] = Some(response.clone());
        self.is_complete()
    }

    pub fn set_status(&mut self, new_state: RequestStatus) {
        self.state_history.push(new_state);
    }

    pub fn first_response(&self) -> Option<&ResponsePayload> {
        self.responses.iter().find_map(|c| c.as_ref())
    }

    pub fn responses(&self) -> &[Option<ResponsePayload>; N_PARTIES] {
        &self.responses
    }

    pub fn has_error_response(&self) -> bool {
        self.responses.iter().flatten().any(|r| r.is_error())
    }

    pub fn get_error_msgs(&self) -> String {
        self.responses
            .iter()
            .flatten()
            .filter(|r| r.is_error())
            .map(|r| {
                format!(
                    "node {}: {}",
                    r.node_id(),
                    r.error_reason().unwrap_or("no reason given")
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.label {
            Some(label) => write!(f, "{}", label),
            None => write!(f, "{}.{}", self.batch_idx, self.batch_item_idx),
        }
    }
}
