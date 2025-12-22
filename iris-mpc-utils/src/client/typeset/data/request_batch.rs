use std::fmt;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use crate::client::typeset::data::request::RequestStatus;

use super::{request::Request, response::ResponseBody};

/// A data structure representing a batch of requests dispatched for system processing.
#[derive(Clone, Debug)]
pub struct RequestBatch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Requests in batch.
    requests: Vec<Request>,
}

impl RequestBatch {
    pub fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub fn requests(&self) -> &[Request] {
        self.requests.as_slice()
    }

    pub fn requests_mut(&mut self) -> &mut Vec<Request> {
        &mut self.requests
    }

    pub fn new(batch_idx: usize, requests: Option<Vec<Request>>) -> Self {
        Self {
            batch_idx,
            requests: requests.unwrap_or(vec![]),
        }
    }

    pub fn correlate_and_update_child(&mut self, response: ResponseBody) -> Option<()> {
        if let Some(idx_of_correlated) = self.get_idx_of_correlated(&response) {
            self.requests_mut()[idx_of_correlated].set_correlation(&response);
            if let Some(idx_of_child) = self.get_idx_of_child(idx_of_correlated) {
                self.requests_mut()[idx_of_child].set_data_from_parent_response(&response);
            }
            Some(())
        } else {
            None
        }
    }

    fn get_idx_of_correlated(&self, response: &ResponseBody) -> Option<usize> {
        self.requests
            .iter()
            .enumerate()
            .find(|(_, r)| r.is_correlation(response))
            .map(|(idx, _)| idx)
    }

    fn get_idx_of_child(&self, idx_of_parent: usize) -> Option<usize> {
        let parent = &self.requests[idx_of_parent];
        self.requests
            .iter()
            .enumerate()
            .find(|(_, r)| match r.request_id_of_parent() {
                Some(child_request_id) => child_request_id == parent.request_id(),
                None => false,
            })
            .map(|(idx, _)| idx)
    }

    /// Returns true if there are any requests currently enqueued.
    pub fn has_enqueued_items(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueued())
    }

    /// Returns true if there are any requests deemed enqueueable.
    pub fn is_enqueueable(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueueable())
    }

    /// Returns next batch item ordinal identifier.
    pub fn next_item_idx(&self) -> usize {
        &self.requests().len() + 1
    }

    /// Extends requests collection with child/parent requests..
    pub fn push_child_and_maybe_parent(
        &mut self,
        (child, maybe_parent): (Request, Option<Request>),
    ) {
        self.requests_mut().push(child);
        if let Some(parent) = maybe_parent {
            self.requests_mut().push(parent);
        }
    }

    pub fn set_request_status(&mut self, new_state: RequestStatus) {
        for request in self.requests_mut() {
            request.set_status(new_state.clone());
        }
    }
}

impl fmt::Display for RequestBatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Batch:{:03}", self.batch_idx)
    }
}

/// Encapsulates inputs used to derive the kind of request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchKind {
    /// All requests are of same type.
    Simple(&'static str),
}

impl fmt::Display for RequestBatchKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Simple(kind) => write!(f, "{}", kind),
        }
    }
}

impl From<&String> for RequestBatchKind {
    fn from(option: &String) -> Self {
        Self::Simple(match option.as_str() {
            IDENTITY_DELETION_MESSAGE_TYPE => IDENTITY_DELETION_MESSAGE_TYPE,
            REAUTH_MESSAGE_TYPE => REAUTH_MESSAGE_TYPE,
            RESET_CHECK_MESSAGE_TYPE => RESET_CHECK_MESSAGE_TYPE,
            RESET_UPDATE_MESSAGE_TYPE => RESET_UPDATE_MESSAGE_TYPE,
            UNIQUENESS_MESSAGE_TYPE => UNIQUENESS_MESSAGE_TYPE,
            _ => panic!("Unsupported request batch kind"),
        })
    }
}

/// Encapsulates inputs used to compute size of a request batch.
#[derive(Debug, Clone)]
pub enum RequestBatchSize {
    /// Batch size is static.
    Static(usize),
}

impl fmt::Display for RequestBatchSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Static(size) => write!(f, "{}", size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RequestBatch;

    impl RequestBatch {
        pub fn new_1() -> Self {
            Self::new(1, None)
        }

        pub fn new_2() -> Self {
            Self::new(1, Some(vec![]))
        }
    }

    #[tokio::test]
    async fn test_new_1() {
        let batch = RequestBatch::new_1();
        assert!(batch.batch_idx == 1);
        assert!(batch.requests.len() == 0);
    }

    #[tokio::test]
    async fn test_new_2() {
        let batch = RequestBatch::new_2();
        assert!(batch.batch_idx == 1);
        assert!(batch.requests.len() == 0);
    }
}
