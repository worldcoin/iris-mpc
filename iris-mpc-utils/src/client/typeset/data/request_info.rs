use std::fmt;

use super::{request::RequestStatus, request_batch::RequestBatch, response::ResponseBody};
use crate::constants::N_PARTIES;

/// Encapsulates common data pertinent to a system processing request.
#[derive(Clone, Debug)]
pub struct RequestInfo {
    /// Associated request batch ordinal identifier.
    batch_idx: usize,

    /// Associated request batch item ordinal identifier.
    batch_item_idx: usize,

    /// Correlated system responses returned by MPC nodes.
    correlation_set: [Option<ResponseBody>; N_PARTIES],

    /// Set of processing states.
    state_history: Vec<RequestStatus>,
}

impl RequestInfo {
    pub fn new(batch: &RequestBatch) -> Self {
        let mut state_history = Vec::with_capacity(RequestStatus::VARIANT_COUNT);
        state_history.push(RequestStatus::default());

        Self {
            batch_idx: batch.batch_idx(),
            batch_item_idx: batch.requests().len() + 1,
            correlation_set: [const { None }; N_PARTIES],
            state_history,
        }
    }

    pub fn is_correlated(&self) -> bool {
        self.correlation_set.iter().all(|c| c.is_some())
    }

    pub fn set_correlation(&mut self, response: ResponseBody) {
        self.correlation_set[response.node_id()] = Some(response.to_owned());
        tracing::info!("{} :: Correlated -> Node-{}", &self, response.node_id());
        if self.is_correlated() {
            self.set_status(RequestStatus::new_correlated());
        }
    }

    pub fn set_status(&mut self, new_state: RequestStatus) {
        tracing::info!("{} :: State -> {:?}", &self, new_state);
        self.state_history.push(new_state);
    }

    pub fn status(&self) -> &RequestStatus {
        self.state_history.last().unwrap()
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Request {}.{}", self.batch_idx, self.batch_item_idx)
    }
}
