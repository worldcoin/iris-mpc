use std::fmt;

use serde::{Deserialize, Serialize};

use iris_mpc_common::helpers::smpc_request;

use super::{
    IrisPairDescriptor, Request, RequestInfo, RequestStatus, ResponsePayload,
    UniquenessRequestDescriptor,
};

/// A data structure representing a batch of requests dispatched for system processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestBatch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Requests in batch.
    requests: Vec<Request>,
}

impl RequestBatch {
    pub(crate) fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub(crate) fn requests(&self) -> &[Request] {
        self.requests.as_slice()
    }

    pub(crate) fn requests_mut(&mut self) -> &mut Vec<Request> {
        &mut self.requests
    }

    pub(crate) fn new(batch_idx: usize, requests: Vec<Request>) -> Self {
        Self {
            batch_idx,
            requests,
        }
    }

    /// Maybe returns ordinal identifier of a correlated request.
    pub(crate) fn get_idx_of_correlated(&self, response: &ResponsePayload) -> Option<usize> {
        self.requests
            .iter()
            .enumerate()
            .find(|(_, r)| r.is_correlation(response))
            .map(|(idx, _)| idx)
    }

    /// Maybe returns ordinal identifier a correlated request's child.
    pub(crate) fn get_idx_of_child(&self, idx_of_correlated: usize) -> Option<usize> {
        let correlated = &self.requests[idx_of_correlated];
        self.requests
            .iter()
            .enumerate()
            .find(|(_, r)| r.is_child(correlated))
            .map(|(idx, _)| idx)
    }

    /// Returns true if there are any requests currently enqueued.
    pub(crate) fn has_enqueued_items(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueued())
    }

    /// Returns true if there are any requests deemed enqueueable.
    pub(crate) fn is_enqueueable(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueueable())
    }

    /// Returns next batch item ordinal identifier.
    pub(super) fn next_item_idx(&self) -> usize {
        &self.requests().len() + 1
    }

    /// Extends requests collection with a new IdentityDeletion request.
    pub(crate) fn push_new_identity_deletion(
        &mut self,
        label: Option<String>,
        uniqueness_ref: UniquenessRequestDescriptor,
    ) {
        self.push_request(Request::IdentityDeletion {
            info: RequestInfo::new(self, label),
            uniqueness_ref,
        });
    }

    /// Extends requests collection with a new Reauthorization request.
    pub(crate) fn push_new_reauthorization(
        &mut self,
        label: Option<String>,
        uniqueness_ref: UniquenessRequestDescriptor,
        iris_pair_ref: Option<IrisPairDescriptor>,
    ) {
        self.push_request(Request::Reauthorization {
            info: RequestInfo::new(self, label),
            reauth_id: uuid::Uuid::new_v4(),
            iris_pair_ref,
            uniqueness_ref,
        });
    }

    /// Extends requests collection with a new ResetCheck request.
    pub(crate) fn push_new_reset_check(
        &mut self,
        label: Option<String>,
        iris_pair_ref: Option<IrisPairDescriptor>,
    ) {
        self.push_request(Request::ResetCheck {
            info: RequestInfo::new(self, label),
            iris_pair_ref,
            reset_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new ResetUpdate request.
    pub(crate) fn push_new_reset_update(
        &mut self,
        label: Option<String>,
        uniqueness_ref: UniquenessRequestDescriptor,
        iris_pair_ref: Option<IrisPairDescriptor>,
    ) {
        self.push_request(Request::ResetUpdate {
            info: RequestInfo::new(self, label),
            reset_id: uuid::Uuid::new_v4(),
            iris_pair_ref,
            uniqueness_ref,
        });
    }

    /// Extends requests collection with a new Uniqueness request.
    pub(crate) fn push_new_uniqueness(
        &mut self,
        label: Option<String>,
        iris_pair_ref: Option<IrisPairDescriptor>,
    ) -> uuid::Uuid {
        let signup_id = uuid::Uuid::new_v4();
        self.push_request(Request::Uniqueness {
            info: RequestInfo::new(self, label),
            iris_pair_ref,
            signup_id,
        });
        signup_id
    }

    /// Extends requests collection.
    fn push_request(&mut self, request: Request) {
        self.requests_mut().push(request);
    }

    pub(crate) fn set_request_status(&mut self, new_state: RequestStatus) {
        for request in self.requests_mut() {
            request.set_status(new_state.clone());
        }
    }
}

impl Default for RequestBatch {
    fn default() -> Self {
        Self::new(1, vec![])
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
    fn from(value: &String) -> Self {
        Self::Simple(match value.as_str() {
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
                smpc_request::IDENTITY_DELETION_MESSAGE_TYPE
            }
            smpc_request::REAUTH_MESSAGE_TYPE => smpc_request::REAUTH_MESSAGE_TYPE,
            smpc_request::RESET_CHECK_MESSAGE_TYPE => smpc_request::RESET_CHECK_MESSAGE_TYPE,
            smpc_request::RESET_UPDATE_MESSAGE_TYPE => smpc_request::RESET_UPDATE_MESSAGE_TYPE,
            smpc_request::UNIQUENESS_MESSAGE_TYPE => smpc_request::UNIQUENESS_MESSAGE_TYPE,
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
    use super::{super::UniquenessRequestDescriptor, RequestBatch};

    impl RequestBatch {
        /// New batch of 10 uniqueness requests.
        pub fn new_1() -> Self {
            let mut batch = Self::default();
            for _ in 0..10 {
                batch.push_new_uniqueness(None, None);
            }

            batch
        }

        /// New mixed batch with a parent refrenced by it's request id.
        pub fn new_2() -> Self {
            let mut batch = Self::default();
            for _ in 0..10 {
                let uniqueness_ref =
                    UniquenessRequestDescriptor::SignupId(batch.push_new_uniqueness(None, None));
                batch.push_new_reauthorization(None, uniqueness_ref.clone(), None);
                batch.push_new_reset_check(None, None);
                batch.push_new_reset_update(None, uniqueness_ref.clone(), None);
                batch.push_new_identity_deletion(None, uniqueness_ref.clone());
            }

            batch
        }

        /// New mixed batch with a parent refrenced by it's correlated Iris serial id.
        pub fn new_3() -> Self {
            let mut batch = Self::default();
            for _ in 0..10 {
                let serial_id = 1;
                batch.push_new_reauthorization(
                    None,
                    UniquenessRequestDescriptor::IrisSerialId(serial_id),
                    None,
                );
                batch.push_new_reset_check(None, None);
                batch.push_new_reset_update(
                    None,
                    UniquenessRequestDescriptor::IrisSerialId(serial_id),
                    None,
                );
                batch.push_new_identity_deletion(
                    None,
                    UniquenessRequestDescriptor::IrisSerialId(serial_id),
                );
            }

            batch
        }
    }

    #[tokio::test]
    async fn test_new_default() {
        let batch = RequestBatch::default();
        assert!(batch.batch_idx == 1);
        assert!(batch.requests.is_empty());
    }

    #[tokio::test]
    async fn test_new_1() {
        let batch = RequestBatch::new_1();
        assert!(batch.batch_idx == 1);
        assert!(batch.requests.len() == 10);
        for request in batch.requests {
            assert!(request.is_uniqueness());
        }
    }

    #[tokio::test]
    async fn test_new_2() {
        let batch = RequestBatch::new_2();
        assert!(batch.batch_idx == 1);
        assert!(batch.requests.len() == 50);
    }

    #[tokio::test]
    async fn test_new_3() {
        let batch = RequestBatch::new_3();
        assert!(batch.batch_idx == 1);
        assert!(batch.requests.len() == 40);
    }
}
