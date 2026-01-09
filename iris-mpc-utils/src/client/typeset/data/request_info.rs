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

    /// Correlated system responses returned by MPC nodes.
    correlation_set: [Option<ResponsePayload>; N_PARTIES],

    /// Set of processing states.
    state_history: Vec<RequestStatus>,
}

impl RequestInfo {
    pub(super) fn new(batch: &RequestBatch) -> Self {
        let mut state_history = Vec::with_capacity(RequestStatus::VARIANT_COUNT);
        state_history.push(RequestStatus::default());

        Self {
            batch_idx: batch.batch_idx(),
            batch_item_idx: batch.next_item_idx(),
            correlation_set: [const { None }; N_PARTIES],
            state_history,
        }
    }

    pub(super) fn is_fully_correlated(&self) -> bool {
        self.correlation_set.iter().all(|c| c.is_some())
    }

    pub(super) fn set_correlation(&mut self, response: &ResponsePayload) {
        self.correlation_set[response.node_id()] = Some(response.clone());
    }

    pub(super) fn set_status(&mut self, new_state: RequestStatus) {
        self.state_history.push(new_state);
    }

    pub(super) fn status(&self) -> &RequestStatus {
        self.state_history.last().unwrap()
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Request-{}.{}", self.batch_idx, self.batch_item_idx)
    }
}
