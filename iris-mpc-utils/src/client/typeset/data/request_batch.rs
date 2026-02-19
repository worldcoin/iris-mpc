use std::{collections::HashMap, fmt};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{IrisPairDescriptor, Request, RequestInfo, RequestStatus, ResponsePayload};
use crate::client::options::{Parent, RequestOptions, RequestPayloadOptions};

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
    pub fn from_str(s: &str) -> Option<Self> {
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

/// A child request waiting for its parent's serial ID to be resolved.
#[derive(Debug, Clone)]
pub struct PendingItem {
    /// The key used to match against parent resolution (label string or signup_id string).
    parent_key: String,
    /// Pre-generated operation UUID (used as the S3 key for share upload).
    op_id: uuid::Uuid,
    /// Original request options, used to build the Request on activation.
    opts: RequestOptions,
}

impl PendingItem {
    pub(crate) fn new(parent_key: String, opts: RequestOptions) -> Self {
        Self {
            parent_key,
            op_id: uuid::Uuid::new_v4(),
            opts,
        }
    }

    /// Returns (op_id, optional iris_pair) for share upload.
    pub(crate) fn shares_info(&self) -> (uuid::Uuid, Option<&IrisPairDescriptor>) {
        (self.op_id, self.opts.iris_pair())
    }

    pub(crate) fn op_id(&self) -> uuid::Uuid {
        self.op_id
    }

    pub(crate) fn label(&self) -> Option<String> {
        self.opts.label()
    }

    pub(crate) fn payload(&self) -> &RequestPayloadOptions {
        self.opts.payload()
    }
}

/// A data structure representing a batch of requests dispatched for system processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestBatch {
    /// Ordinal batch identifier to distinguish batches.
    batch_idx: usize,

    /// Active requests in batch.
    requests: Vec<Request>,

    /// Requests waiting for their parent's serial ID to be resolved.
    #[serde(skip)]
    pending: Vec<PendingItem>,
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

    pub(crate) fn pending(&self) -> &[PendingItem] {
        &self.pending
    }

    pub(crate) fn new(batch_idx: usize, requests: Vec<Request>) -> Self {
        Self {
            batch_idx,
            requests,
            pending: vec![],
        }
    }

    /// Adds a pending item waiting for parent resolution.
    pub(crate) fn push_pending(&mut self, item: PendingItem) {
        self.pending.push(item);
    }

    /// Maybe returns ordinal identifier of a request matching the given response.
    pub(crate) fn get_idx_of_correlated(&self, response: &ResponsePayload) -> Option<usize> {
        self.requests
            .iter()
            .enumerate()
            .find(|(_, r)| r.is_correlation(response))
            .map(|(idx, _)| idx)
    }

    /// Returns true if there are any requests currently enqueued.
    pub(crate) fn has_enqueued_items(&self) -> bool {
        self.requests.iter().any(|r| r.is_enqueued())
    }

    /// Returns ordered set of unique Iris indexes used across the batch.
    #[allow(dead_code)]
    pub(crate) fn iris_pair_indexes(&self) -> Vec<usize> {
        self.requests
            .iter()
            .flat_map(|r| r.iris_pair_indexes())
            .unique()
            .sorted()
            .collect()
    }

    /// Removes all pending items whose `parent_key` matches without activating them.
    /// Used when a parent request completes with an error.
    pub(crate) fn drop_pending(&mut self, parent_key: &str) {
        self.pending.retain(|item| item.parent_key != parent_key);
    }

    /// Activates all pending items whose `parent_key` matches, resolving them to full Requests.
    /// The new requests are added with `SharesUploaded` status (shares already uploaded at batch start).
    pub(crate) fn activate_pending(&mut self, parent_key: &str, serial_id: IrisSerialId) {
        let mut to_activate = Vec::new();
        self.pending.retain(|item| {
            if item.parent_key == parent_key {
                to_activate.push(item.clone());
                false
            } else {
                true
            }
        });

        for item in to_activate {
            let batch_idx = self.batch_idx;
            let batch_item_idx = self.next_item_idx();
            let mut request = Request::from_pending(batch_idx, batch_item_idx, &item, serial_id);
            request.set_status(RequestStatus::SharesUploaded);
            self.requests.push(request);
        }
    }

    /// Resolves all pending items whose `parent_key` is in the given resolutions map.
    /// Used at batch start for cross-batch Complex mode parents.
    pub(crate) fn activate_cross_batch_pending(
        &mut self,
        resolutions: &HashMap<String, IrisSerialId>,
    ) {
        let mut to_activate: Vec<(PendingItem, IrisSerialId)> = Vec::new();
        self.pending.retain(|item| {
            if let Some(&serial_id) = resolutions.get(&item.parent_key) {
                to_activate.push((item.clone(), serial_id));
                false
            } else {
                true
            }
        });

        for (item, serial_id) in to_activate {
            let batch_idx = self.batch_idx;
            let batch_item_idx = self.next_item_idx();
            let mut request = Request::from_pending(batch_idx, batch_item_idx, &item, serial_id);
            request.set_status(RequestStatus::SharesUploaded);
            self.requests.push(request);
        }
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
