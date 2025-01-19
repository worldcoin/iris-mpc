mod actor;
pub mod sync_nccl;

use crate::dot::{share_db::preprocess_query, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS};
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
pub struct BatchQueryEntriesPreprocessed {
    pub code: Vec<Vec<u8>>,
    pub mask: Vec<Vec<u8>>,
}

impl From<BatchQueryEntries> for BatchQueryEntriesPreprocessed {
    fn from(value: BatchQueryEntries) -> Self {
        let code_coefs = &value.code.iter().flat_map(|e| e.coefs).collect::<Vec<_>>();
        let mask_coefs = &value.mask.iter().flat_map(|e| e.coefs).collect::<Vec<_>>();

        assert_eq!(
            code_coefs.len() / IRIS_CODE_LENGTH,
            mask_coefs.len() / MASK_CODE_LENGTH
        );

        Self {
            code: preprocess_query(code_coefs),
            mask: preprocess_query(mask_coefs),
        }
    }
}

impl BatchQueryEntriesPreprocessed {
    pub fn len(&self) -> usize {
        assert_eq!(self.code.len(), self.mask.len());
        self.code.iter().zip(self.mask.iter()).for_each(|(c, m)| {
            assert_eq!(c.len() / IRIS_CODE_LENGTH, m.len() / MASK_CODE_LENGTH);
        });
        if self.code.is_empty() {
            0
        } else {
            self.code[0].len() / IRIS_CODE_LENGTH
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    pub query_left_preprocessed:    BatchQueryEntriesPreprocessed,
    pub db_left_preprocessed:       BatchQueryEntriesPreprocessed,
    pub query_right:                BatchQueryEntries,
    pub db_right:                   BatchQueryEntries,
    pub store_right:                BatchQueryEntries,
    pub query_right_preprocessed:   BatchQueryEntriesPreprocessed,
    pub db_right_preprocessed:      BatchQueryEntriesPreprocessed,
    pub deletion_requests_indices:  Vec<u32>, // 0-indexed indicies in of entries to be deleted
    pub deletion_requests_metadata: Vec<BatchMetadata>,
    pub or_rule_serial_ids:         Option<Vec<Vec<u32>>>,
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
            .chunks(ROTATIONS)
            .enumerate()
            .filter(|(i, _)| $indices.contains(i))
            .flat_map(|(_, chunk)| chunk.iter().cloned())
            .collect();
    };
}

macro_rules! filter_by_indices_with_rotations_and_code_length {
    ($data:expr, $indices:expr, $code_length:expr) => {
        $data = $data
            .chunks($code_length * ROTATIONS)
            .enumerate()
            .filter(|(i, _)| $indices.contains(i))
            .flat_map(|(_, chunk)| chunk.iter().cloned())
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
        Self::filter_preprocessed_entry(&mut self.query_left_preprocessed, &indices_set);
        Self::filter_preprocessed_entry(&mut self.db_left_preprocessed, &indices_set);
        Self::filter_preprocessed_entry(&mut self.query_right_preprocessed, &indices_set);
        Self::filter_preprocessed_entry(&mut self.db_right_preprocessed, &indices_set);
        filter_by_indices!(self.valid_entries, indices_set);
    }

    fn filter_preprocessed_entry(
        entry: &mut BatchQueryEntriesPreprocessed,
        indices: &HashSet<usize>,
    ) {
        for i in 0..2 {
            filter_by_indices_with_rotations_and_code_length!(
                entry.code[i],
                indices,
                IRIS_CODE_LENGTH
            );
            filter_by_indices_with_rotations_and_code_length!(
                entry.mask[i],
                indices,
                MASK_CODE_LENGTH
            );
        }
    }
}

#[derive(Debug)]
pub struct ServerJob {
    batch:          BatchQuery,
    return_channel: oneshot::Sender<ServerJobResult>,
}

#[derive(Debug, Clone)]
pub struct ServerJobResult {
    pub merged_results:            Vec<u32>,
    pub request_ids:               Vec<String>,
    pub metadata:                  Vec<BatchMetadata>,
    pub matches:                   Vec<bool>,
    pub match_ids:                 Vec<Vec<u32>>,
    pub partial_match_ids_left:    Vec<Vec<u32>>,
    pub partial_match_ids_right:   Vec<Vec<u32>>,
    pub store_left:                BatchQueryEntries,
    pub store_right:               BatchQueryEntries,
    pub deleted_ids:               Vec<u32>,
    pub matched_batch_request_ids: Vec<Vec<String>>,
}

enum Eye {
    Left,
    Right,
}
