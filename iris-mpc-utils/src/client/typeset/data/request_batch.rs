use std::fmt;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{IrisPairDescriptor, Request, RequestInfo, RequestStatus};

/// Typed representation of a batch request kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKind {
    IdentityDeletion,
    Reauth,
    ResetCheck,
    ResetUpdate,
    Uniqueness,
}

impl BatchKind {
    /// Parses a batch kind from its SMPC message type string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => Some(Self::IdentityDeletion),
            smpc_request::REAUTH_MESSAGE_TYPE => Some(Self::Reauth),
            smpc_request::RESET_CHECK_MESSAGE_TYPE => Some(Self::ResetCheck),
            smpc_request::RESET_UPDATE_MESSAGE_TYPE => Some(Self::ResetUpdate),
            smpc_request::UNIQUENESS_MESSAGE_TYPE => Some(Self::Uniqueness),
            _ => None,
        }
    }

    /// Returns true if this batch kind requires a parent uniqueness request.
    pub fn requires_parent(&self) -> bool {
        matches!(
            self,
            Self::IdentityDeletion | Self::Reauth | Self::ResetUpdate
        )
    }
}

/// A data structure representing a batch of requests dispatched for system processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestBatch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Active requests in batch.
    requests: Vec<Request>,
}

#[allow(dead_code)]
impl RequestBatch {
    pub(crate) fn batch_idx(&self) -> usize {
        self.batch_idx
    }

    pub(crate) fn requests(&self) -> &[Request] {
        self.requests.as_slice()
    }

    pub(crate) fn new(batch_idx: usize, requests: Vec<Request>) -> Self {
        Self {
            batch_idx,
            requests,
        }
    }

    /// Returns ordered set of unique Iris indexes used across the batch.
    pub(crate) fn iris_pair_indexes(&self) -> Vec<usize> {
        self.requests
            .iter()
            .flat_map(|r| r.iris_pair_indexes())
            .unique()
            .sorted()
            .collect()
    }

    /// Returns true if there are any requests deemed enqueueable.
    pub(crate) fn is_enqueueable(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueueable())
    }

    /// Returns next batch item ordinal identifier.
    pub(super) fn next_item_idx(&self) -> usize {
        self.requests().len() + 1
    }

    /// Extends requests collection with a new IdentityDeletion request.
    pub(crate) fn push_new_identity_deletion(
        &mut self,
        parent: IrisSerialId,
        label: Option<String>,
    ) {
        self.push_request(Request::IdentityDeletion {
            info: RequestInfo::new(self, label),
            parent,
        });
    }

    /// Extends requests collection with a new Reauthorization request.
    pub(crate) fn push_new_reauthorization(
        &mut self,
        parent: IrisSerialId,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
    ) {
        self.push_request(Request::Reauthorization {
            info: RequestInfo::new(self, label),
            iris_pair,
            parent,
            reauth_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new ResetCheck request.
    pub(crate) fn push_new_reset_check(
        &mut self,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
    ) {
        self.push_request(Request::ResetCheck {
            info: RequestInfo::new(self, label),
            iris_pair,
            reset_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new ResetUpdate request.
    pub(crate) fn push_new_reset_update(
        &mut self,
        parent: IrisSerialId,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
    ) {
        self.push_request(Request::ResetUpdate {
            info: RequestInfo::new(self, label),
            iris_pair,
            parent,
            reset_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new Uniqueness request, returning the signup_id.
    pub(crate) fn push_new_uniqueness(
        &mut self,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
    ) -> uuid::Uuid {
        let signup_id = uuid::Uuid::new_v4();
        self.push_request(Request::Uniqueness {
            info: RequestInfo::new(self, label),
            iris_pair,
            signup_id,
        });
        signup_id
    }

    /// Extends requests collection.
    fn push_request(&mut self, request: Request) {
        self.requests.push(request);
    }

    pub(crate) fn set_request_status(&mut self, new_state: RequestStatus) {
        for request in &mut self.requests {
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
