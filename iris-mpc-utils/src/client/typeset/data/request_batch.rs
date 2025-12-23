use std::fmt;

use iris_mpc_common::{
    helpers::smpc_request::{
        self, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE,
        RESET_UPDATE_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
    },
    IrisSerialId,
};

use crate::client::typeset::{ParentUniquenessRequest, RequestInfo, RequestStatus};

use super::{Request, ResponseBody};

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

    pub fn new(batch_idx: usize, requests: Vec<Request>) -> Self {
        Self {
            batch_idx,
            requests,
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
            .find(|(_, r)| match r.info().request_id_of_parent() {
                Some(child_request_id) => child_request_id == parent.info().request_id(),
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

    pub fn push_new(&mut self, kind: &str, parent: Option<ParentUniquenessRequest>) {
        assert!(
            ParentUniquenessRequest::is_valid(kind, &parent),
            "Invalid parent request association"
        );

        match kind {
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
                self.push_new_identity_deletion(parent.unwrap());
            }
            smpc_request::REAUTH_MESSAGE_TYPE => {
                self.push_new_reauthorization(parent.unwrap());
            }
            smpc_request::RESET_CHECK_MESSAGE_TYPE => {
                self.push_new_reset_check();
            }
            smpc_request::RESET_UPDATE_MESSAGE_TYPE => {
                self.push_new_reset_update(parent.unwrap());
            }
            smpc_request::UNIQUENESS_MESSAGE_TYPE => {
                self.push_new_uniqueness();
            }
            _ => unreachable!(),
        }
    }

    /// Extends requests collection with a new IdentityDeletion request.
    pub fn push_new_identity_deletion(&mut self, parent: ParentUniquenessRequest) {
        self.push_request(match parent {
            ParentUniquenessRequest::RequestUuid(request_id_of_parent) => {
                Request::IdentityDeletion {
                    info: RequestInfo::new(self, Some(&request_id_of_parent.clone())),
                    uniqueness_serial_id: None,
                }
            }
            ParentUniquenessRequest::IrisSerialId(serial_id) => Request::IdentityDeletion {
                info: RequestInfo::new(self, None),
                uniqueness_serial_id: Some(serial_id),
            },
        });
    }

    /// Extends requests collection with a new Reauthorization request.
    pub fn push_new_reauthorization(&mut self, parent: ParentUniquenessRequest) {
        self.push_request(match parent {
            ParentUniquenessRequest::RequestUuid(request_id_of_parent) => {
                Request::Reauthorization {
                    info: RequestInfo::new(self, Some(&request_id_of_parent.clone())),
                    reauth_id: uuid::Uuid::new_v4(),
                    uniqueness_serial_id: None,
                }
            }
            ParentUniquenessRequest::IrisSerialId(serial_id) => Request::Reauthorization {
                info: RequestInfo::new(self, None),
                reauth_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: Some(serial_id),
            },
        });
    }

    /// Extends requests collection with a new ResetCheck request.
    pub fn push_new_reset_check(&mut self) {
        self.push_request(Request::ResetCheck {
            info: RequestInfo::new(self, None),
            reset_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new ResetUpdate request.
    pub fn push_new_reset_update(&mut self, parent: ParentUniquenessRequest) {
        self.push_request(match parent {
            ParentUniquenessRequest::RequestUuid(request_id_of_parent) => Request::ResetUpdate {
                info: RequestInfo::new(self, Some(&request_id_of_parent.clone())),
                reset_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: None,
            },
            ParentUniquenessRequest::IrisSerialId(serial_id) => Request::ResetUpdate {
                info: RequestInfo::new(self, None),
                reset_id: uuid::Uuid::new_v4(),
                uniqueness_serial_id: Some(serial_id),
            },
        });
    }

    /// Extends requests collection with a new Uniqueness request.
    pub fn push_new_uniqueness(&mut self) -> Request {
        let r = Request::Uniqueness {
            info: RequestInfo::new(self, None),
            signup_id: uuid::Uuid::new_v4(),
        };
        self.push_request(r.clone());

        r
    }

    pub fn push_new_uniqueness_maybe(
        &mut self,
        kind: &str,
        serial_id: Option<IrisSerialId>,
    ) -> Option<ParentUniquenessRequest> {
        match kind {
            smpc_request::RESET_CHECK_MESSAGE_TYPE | smpc_request::UNIQUENESS_MESSAGE_TYPE => None,
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE
            | smpc_request::REAUTH_MESSAGE_TYPE
            | smpc_request::RESET_UPDATE_MESSAGE_TYPE => match serial_id {
                None => Some(ParentUniquenessRequest::RequestUuid(
                    *self.push_new_uniqueness().request_id(),
                )),
                Some(serial_id) => Some(ParentUniquenessRequest::IrisSerialId(serial_id)),
            },
            _ => panic!("Invalid request kind"),
        }
    }

    /// Extends requests collection.
    pub fn push_request(&mut self, request: Request) {
        self.requests_mut().push(request);
    }

    pub fn set_request_status(&mut self, new_state: RequestStatus) {
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
    use super::{super::ParentUniquenessRequest, RequestBatch};

    impl RequestBatch {
        /// New batch of 10 uniqueness requests.
        pub fn new_1() -> Self {
            let mut batch = Self::default();
            for _ in 0..10 {
                batch.push_new_uniqueness();
            }

            batch
        }

        /// New mixed batch with a parent refrenced by it's request id.
        pub fn new_2() -> Self {
            let mut batch = Self::default();
            let parent =
                ParentUniquenessRequest::RequestUuid(*batch.push_new_uniqueness().request_id());
            batch.push_new_reauthorization(parent.clone());
            batch.push_new_reset_check();
            batch.push_new_reset_update(parent.clone());
            batch.push_new_identity_deletion(parent.clone());

            batch
        }

        /// New mixed batch with a parent refrenced by it's correlated Iris serial id.
        pub fn new_3() -> Self {
            let mut batch = Self::default();
            let parent = ParentUniquenessRequest::IrisSerialId(1);
            batch.push_new_reauthorization(parent.clone());
            batch.push_new_reset_check();
            batch.push_new_reset_update(parent.clone());
            batch.push_new_identity_deletion(parent.clone());

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
        assert!(batch.requests.len() == 5);
    }
}
