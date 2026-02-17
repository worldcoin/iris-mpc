use std::{collections::HashMap, fmt};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use iris_mpc_common::{helpers::smpc_request, IrisSerialId};

use super::{
    IrisPairDescriptor, Request, RequestInfo, RequestStatus, ResponsePayload,
    UniquenessRequestDescriptor,
};
use crate::client::options::{RequestOptions, RequestPayloadOptions};

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

    /// Resolves cross-batch parent dependencies using results from previously processed batches.
    pub(crate) fn resolve_cross_batch_parents(
        &mut self,
        resolutions: &HashMap<uuid::Uuid, IrisSerialId>,
    ) {
        for request in self.requests_mut() {
            let parent_desc = match request {
                Request::IdentityDeletion {
                    parent: parent_desc,
                    ..
                }
                | Request::Reauthorization {
                    parent: parent_desc,
                    ..
                }
                | Request::ResetUpdate {
                    parent: parent_desc,
                    ..
                } => parent_desc,
                _ => continue,
            };
            if let UniquenessRequestDescriptor::SignupId(signup_id) = parent_desc {
                if let Some(serial_id) = resolutions.get(signup_id) {
                    *parent_desc = UniquenessRequestDescriptor::IrisSerialId(*serial_id);
                }
            }
        }
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
        parent: UniquenessRequestDescriptor,
        label: Option<String>,
        label_of_parent: Option<String>,
    ) {
        self.push_request(Request::IdentityDeletion {
            info: RequestInfo::new(self, label, label_of_parent),
            parent,
        });
    }

    /// Extends requests collection with a new Reauthorization request.
    pub(crate) fn push_new_reauthorization(
        &mut self,
        parent: UniquenessRequestDescriptor,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
        label_of_parent: Option<String>,
    ) {
        self.push_request(Request::Reauthorization {
            info: RequestInfo::new(self, label, label_of_parent),
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
            info: RequestInfo::new(self, label, None),
            iris_pair,
            reset_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new ResetUpdate request.
    pub(crate) fn push_new_reset_update(
        &mut self,
        parent: UniquenessRequestDescriptor,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
        label_of_parent: Option<String>,
    ) {
        self.push_request(Request::ResetUpdate {
            info: RequestInfo::new(self, label, label_of_parent),
            iris_pair,
            parent,
            reset_id: uuid::Uuid::new_v4(),
        });
    }

    /// Extends requests collection with a new Uniqueness request.
    pub(crate) fn push_new_uniqueness(
        &mut self,
        iris_pair: Option<IrisPairDescriptor>,
        label: Option<String>,
    ) -> uuid::Uuid {
        let signup_id = uuid::Uuid::new_v4();
        self.push_request(Request::Uniqueness {
            info: RequestInfo::new(self, label, None),
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

/// Encapsulates a constructed set of request batches for processing.
pub struct RequestBatchSet {
    // Associated set of request batches.
    batches: Vec<RequestBatch>,
}

impl RequestBatchSet {
    pub(crate) fn new(batches: Vec<RequestBatch>) -> Self {
        Self { batches }
    }

    pub(crate) fn from_options(opts: &[Vec<RequestOptions>]) -> Self {
        let batches: Vec<RequestBatch> = opts
            .iter()
            .enumerate()
            .map(|(batch_idx, opts_batch)| {
                let mut batch = RequestBatch::new(batch_idx, vec![]);
                for opts_request in opts_batch {
                    match opts_request.payload() {
                        RequestPayloadOptions::IdentityDeletion { parent } => {
                            batch.push_new_identity_deletion(
                                UniquenessRequestDescriptor::from_label(parent),
                                opts_request.label(),
                                Some(parent.clone()),
                            );
                        }
                        RequestPayloadOptions::Reauthorisation { iris_pair, parent } => {
                            batch.push_new_reauthorization(
                                UniquenessRequestDescriptor::from_label(parent),
                                Some(*iris_pair),
                                opts_request.label(),
                                Some(parent.clone()),
                            );
                        }
                        RequestPayloadOptions::ResetCheck { iris_pair } => {
                            batch.push_new_reset_check(Some(*iris_pair), opts_request.label());
                        }
                        RequestPayloadOptions::ResetUpdate { iris_pair, parent } => {
                            batch.push_new_reset_update(
                                UniquenessRequestDescriptor::from_label(parent),
                                Some(*iris_pair),
                                opts_request.label(),
                                Some(parent.clone()),
                            );
                        }
                        RequestPayloadOptions::Uniqueness { iris_pair, .. } => {
                            batch.push_new_uniqueness(
                                Some(*iris_pair),
                                opts_request.label().clone(),
                            );
                        }
                    }
                }

                batch
            })
            .collect();

        RequestBatchSet::new(batches)
    }

    pub(crate) fn batches(&self) -> &Vec<RequestBatch> {
        &self.batches
    }

    fn batches_mut(&mut self) -> &mut Vec<RequestBatch> {
        &mut self.batches
    }

    fn requests(&self) -> Vec<&Request> {
        self.batches()
            .iter()
            .flat_map(|batch| batch.requests())
            .collect()
    }

    fn requests_mut(&mut self) -> Vec<&mut Request> {
        self.batches_mut()
            .iter_mut()
            .flat_map(|batch| batch.requests_mut())
            .collect()
    }

    fn requests_with_parent_descriptor_of_label(&self) -> Vec<(Request, uuid::Uuid)> {
        let mut result = vec![];
        for maybe_child in self.requests() {
            if let Some(_parent_descriptor @ UniquenessRequestDescriptor::Label(parent_label)) =
                maybe_child.parent_descriptor()
            {
                for maybe_parent in self.requests() {
                    if let Some(maybe_parent_label) = maybe_parent.label() {
                        if maybe_parent_label == parent_label {
                            if let Request::Uniqueness { signup_id, .. } = maybe_parent {
                                result.push((maybe_child.clone(), *signup_id));
                            }
                        }
                    }
                }
            }
        }

        result
    }

    pub(crate) fn set_child_parent_descriptors_from_labels(&mut self) {
        for (child, parent_signup_id) in self.requests_with_parent_descriptor_of_label() {
            for child_mut in self.requests_mut() {
                if child_mut.info().uid() != child.info().uid() {
                    continue;
                }

                let parent_desc = match child_mut {
                    Request::IdentityDeletion {
                        parent: parent_desc,
                        ..
                    }
                    | Request::Reauthorization {
                        parent: parent_desc,
                        ..
                    }
                    | Request::ResetUpdate {
                        parent: parent_desc,
                        ..
                    } => parent_desc,
                    _ => continue,
                };

                *parent_desc = UniquenessRequestDescriptor::SignupId(parent_signup_id);
            }
        }
    }
}
