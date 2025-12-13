use std::fmt;

use iris_mpc_common::helpers::smpc_request::{
    IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
    RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};

use crate::client::typeset::data::request::RequestStatus;

use super::{request::Request, response::ResponseBody};

/// A data structure representing a batch of requests dispatched for system processing.
#[derive(Debug)]
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

    pub fn new(batch_idx: usize, batch_size: usize) -> Self {
        Self {
            batch_idx,
            requests: Vec::with_capacity(batch_size * 2),
        }
    }

    pub fn correlate_and_update_child(&mut self, response: ResponseBody) -> Option<()> {
        if let Some(idx_of_correlated) = self.get_idx_of_correlated(&response) {
            self.requests_mut()[idx_of_correlated].set_correlation(&response);
            if let Some(idx_of_child) = self.get_idx_of_child(idx_of_correlated) {
                self.requests_mut()[idx_of_child].set_parent_data(&response);
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

    /// Updates a child request with data pulled from a parent's response.
    // pub fn maybe_update_child(&mut self, parent: &Request) {
    //     if let Some(child) = self.get_maybe_child_request(parent) {
    //         match child {
    //             Request::IdentityDeletion {
    //                 uniqueness_serial_id,
    //                 ..
    //             } => *uniqueness_serial_id = parent.info().iris_serial_id(),
    //             Request::Reauthorization {
    //                 uniqueness_serial_id,
    //                 ..
    //             } => *uniqueness_serial_id = parent.info().iris_serial_id(),
    //             Request::ResetUpdate {
    //                 uniqueness_serial_id,
    //                 ..
    //             } => *uniqueness_serial_id = parent.info().iris_serial_id(),
    //             _ => {}
    //         }
    //     }
    // }

    pub fn push_request(&mut self, request: Request) {
        self.requests_mut().push(request);
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
