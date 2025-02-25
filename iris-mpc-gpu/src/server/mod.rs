mod actor;

use crate::dot::{share_db::preprocess_query, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS};
pub use actor::{generate_luc_records, prepare_or_policy_bitmap, ServerActor, ServerActorHandle};
use iris_mpc_common::{
    helpers::sync::Modification,
    job::{BatchMetadata, BatchQuery, BatchQueryEntries},
};
use std::collections::{HashMap, HashSet};

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

pub struct PreprocessedBatchQuery {
    // Enrollment and reauth specific fields
    pub request_ids:          Vec<String>,
    pub request_types:        Vec<String>,
    pub metadata:             Vec<BatchMetadata>,
    pub query_left:           BatchQueryEntries,
    pub db_left:              BatchQueryEntries,
    pub store_left:           BatchQueryEntries,
    pub query_right:          BatchQueryEntries,
    pub db_right:             BatchQueryEntries,
    pub store_right:          BatchQueryEntries,
    pub or_rule_indices:      Vec<Vec<u32>>,
    pub luc_lookback_records: usize,
    pub valid_entries:        Vec<bool>,
    pub skip_persistence:     Vec<bool>,

    // Only reauth specific fields
    // Map from reauth request id to the index of the target entry to be matched
    pub reauth_target_indices: HashMap<String, u32>,
    pub reauth_use_or_rule:    HashMap<String, bool>,

    // Only deletion specific fields
    pub deletion_requests_indices: Vec<u32>, // 0-indexed indices of entries to be deleted

    // this one is not needed for the GPU actor
    // pub deletion_requests_metadata: Vec<BatchMetadata>,

    // additional fields which are GPU specific
    pub query_left_preprocessed:  BatchQueryEntriesPreprocessed,
    pub db_left_preprocessed:     BatchQueryEntriesPreprocessed,
    pub query_right_preprocessed: BatchQueryEntriesPreprocessed,
    pub db_right_preprocessed:    BatchQueryEntriesPreprocessed,

    // Keeping track of updates & deletions for sync mechanism. Mapping: Serial id -> Modification
    pub modifications: HashMap<u32, Modification>,

    // SNS message ids to assert identical batch processing across parties
    pub sns_message_ids: Vec<String>,
}

impl From<BatchQuery> for PreprocessedBatchQuery {
    fn from(value: BatchQuery) -> Self {
        let mut query_left_preprocessed = None;
        let mut query_right_preprocessed = None;
        let mut db_left_preprocessed = None;
        let mut db_right_preprocessed = None;
        rayon::scope(|s| {
            s.spawn(|_| {
                query_left_preprocessed = Some(BatchQueryEntriesPreprocessed::from(
                    value.query_left.clone(),
                ));
            });
            s.spawn(|_| {
                query_right_preprocessed = Some(BatchQueryEntriesPreprocessed::from(
                    value.query_right.clone(),
                ));
            });
            s.spawn(|_| {
                db_left_preprocessed =
                    Some(BatchQueryEntriesPreprocessed::from(value.db_left.clone()));
            });
            s.spawn(|_| {
                db_right_preprocessed =
                    Some(BatchQueryEntriesPreprocessed::from(value.db_right.clone()));
            });
        });

        Self {
            request_ids:               value.request_ids,
            request_types:             value.request_types,
            metadata:                  value.metadata,
            query_left:                value.query_left,
            db_left:                   value.db_left,
            store_left:                value.store_left,
            query_right:               value.query_right,
            db_right:                  value.db_right,
            store_right:               value.store_right,
            or_rule_indices:           value.or_rule_indices,
            luc_lookback_records:      value.luc_lookback_records,
            valid_entries:             value.valid_entries,
            reauth_target_indices:     value.reauth_target_indices,
            reauth_use_or_rule:        value.reauth_use_or_rule,
            deletion_requests_indices: value.deletion_requests_indices,
            // deletion_requests_metadata: value.deletion_requests_metadata,
            query_left_preprocessed:   query_left_preprocessed.unwrap(),
            db_left_preprocessed:      db_left_preprocessed.unwrap(),
            query_right_preprocessed:  query_right_preprocessed.unwrap(),
            db_right_preprocessed:     db_right_preprocessed.unwrap(),
            modifications:             value.modifications,
            sns_message_ids:           value.sns_message_ids,
            skip_persistence:          value.skip_persistence,
        }
    }
}

impl PreprocessedBatchQuery {
    pub fn retain(&mut self, indices: &[usize]) {
        let indices_set: HashSet<usize> = indices.iter().cloned().collect();
        filter_by_indices!(self.request_ids, indices_set);
        filter_by_indices!(self.request_types, indices_set);
        filter_by_indices!(self.metadata, indices_set);
        filter_by_indices!(self.store_left.code, indices_set);
        filter_by_indices!(self.store_left.mask, indices_set);
        filter_by_indices!(self.store_right.code, indices_set);
        filter_by_indices!(self.store_right.mask, indices_set);
        filter_by_indices!(self.or_rule_indices, indices_set);
        filter_by_indices!(self.skip_persistence, indices_set);
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
