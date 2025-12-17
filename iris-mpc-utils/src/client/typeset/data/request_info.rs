use std::fmt;

use super::{
    request::{Request, RequestStatus},
    request_batch::RequestBatch,
    response::ResponseBody,
};
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

    /// Universally unique request identifer.
    request_id: uuid::Uuid,

    /// Universally unique request identifer of parent request.
    request_id_of_parent: Option<uuid::Uuid>,

    /// Set of processing states.
    state_history: Vec<RequestStatus>,
}

impl RequestInfo {
    pub fn request_id(&self) -> &uuid::Uuid {
        &self.request_id
    }

    pub fn request_id_of_parent(&self) -> &Option<uuid::Uuid> {
        &self.request_id_of_parent
    }

    pub fn new(batch: &RequestBatch, parent: Option<&Request>) -> Self {
        Self {
            batch_idx: batch.batch_idx(),
            batch_item_idx: batch.requests().len() + 1,
            correlation_set: [const { None }; N_PARTIES],
            request_id: uuid::Uuid::new_v4(),
            request_id_of_parent: parent.map(|p| *p.info().request_id()),
            state_history: vec![RequestStatus::default()],
        }
    }

    pub fn is_correlated(&self) -> bool {
        self.correlation_set.iter().all(|c| c.is_some())
    }

    pub fn set_correlation(&mut self, response: &ResponseBody) {
        self.correlation_set[response.node_id()] = Some(response.clone());
    }

    pub fn set_status(&mut self, new_state: RequestStatus) {
        self.state_history.push(new_state);
    }

    pub fn status(&self) -> &RequestStatus {
        self.state_history.last().unwrap()
    }
}

impl fmt::Display for RequestInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Request-{}.{}", self.batch_idx, self.batch_item_idx)
    }
}
