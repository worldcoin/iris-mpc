mod actor;
pub mod heartbeat_nccl;
pub mod sync_nccl;

use crate::dot::ROTATIONS;
pub use actor::{get_dummy_shares_for_deletion, ServerActor, ServerActorHandle};
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use std::collections::HashSet;
use tokio::sync::oneshot;

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchQueryEntries {
    pub code: Vec<GaloisRingIrisCodeShare>,
    pub mask: Vec<GaloisRingTrimmedMaskCodeShare>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchMetadata {
    pub node_id:  String,
    pub trace_id: String,
    pub span_id:  String,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchQuery {
    pub request_ids:                Vec<String>,
    pub metadata:                   Vec<BatchMetadata>,
    pub query_left:                 BatchQueryEntries,
    pub db_left:                    BatchQueryEntries,
    pub store_left:                 BatchQueryEntries,
    pub query_right:                BatchQueryEntries,
    pub db_right:                   BatchQueryEntries,
    pub store_right:                BatchQueryEntries,
    pub deletion_requests_indices:  Vec<u32>, // 0-indexed indicies in of entries to be deleted
    pub deletion_requests_metadata: Vec<BatchMetadata>,
    pub valid_entries:              Vec<bool>,
}

macro_rules! filter_by_indices {
    ($data:expr, $indices:expr) => {
        $data = $data
            .iter()
            .enumerate()
            .filter(|(i, _)| $indices.contains(i))
            .map(|(_, v)| v.clone())
            .collect();
    };
}

macro_rules! filter_by_indices_with_rotations {
    ($data:expr, $indices:expr) => {
        $data = $data
            .iter()
            .enumerate()
            .filter(|(i, _)| $indices.contains((&(i / ROTATIONS))))
            .map(|(_, v)| v.clone())
            .collect();
    };
}

impl BatchQuery {
    pub fn retain(&mut self, indices: &[usize]) {
        let indices_set: HashSet<usize> = indices.iter().cloned().collect();
        filter_by_indices!(self.request_ids, indices_set);
        filter_by_indices!(self.metadata, indices_set);
        filter_by_indices!(self.store_left.code, indices_set);
        filter_by_indices!(self.store_left.mask, indices_set);
        filter_by_indices!(self.store_right.code, indices_set);
        filter_by_indices!(self.store_right.mask, indices_set);
        filter_by_indices_with_rotations!(self.query_left.code, indices_set);
        filter_by_indices_with_rotations!(self.query_left.mask, indices_set);
        filter_by_indices_with_rotations!(self.db_left.code, indices_set);
        filter_by_indices_with_rotations!(self.db_left.mask, indices_set);
        filter_by_indices_with_rotations!(self.query_right.code, indices_set);
        filter_by_indices_with_rotations!(self.query_right.mask, indices_set);
        filter_by_indices_with_rotations!(self.db_right.code, indices_set);
        filter_by_indices_with_rotations!(self.db_right.mask, indices_set);
        filter_by_indices!(self.valid_entries, indices_set);
    }
}

#[derive(Debug)]
pub struct ServerJob {
    batch:          BatchQuery,
    return_channel: oneshot::Sender<ServerJobResult>,
}

#[derive(Debug, Clone)]
pub struct ServerJobResult {
    pub merged_results:          Vec<u32>,
    pub request_ids:             Vec<String>,
    pub metadata:                Vec<BatchMetadata>,
    pub matches:                 Vec<bool>,
    pub match_ids:               Vec<Vec<u32>>,
    pub partial_match_ids_left:  Vec<Vec<u32>>,
    pub partial_match_ids_right: Vec<Vec<u32>>,
    pub store_left:              BatchQueryEntries,
    pub store_right:             BatchQueryEntries,
    pub deleted_ids:             Vec<u32>,
}

enum Eye {
    Left,
    Right,
}
