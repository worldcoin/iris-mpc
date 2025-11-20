pub(crate) mod actor;
// public for some integration tests
pub mod anon_stats;

use crate::dot::{share_db::preprocess_query, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS};
pub use actor::{
    generate_luc_records, prepare_or_policy_bitmap, Orientation, ServerActor, ServerActorHandle,
};
use ampc_server_utils::statistics::Eye;
use iris_mpc_common::helpers::sync::ModificationKey;
use iris_mpc_common::job::GaloisSharesBothSides;
use iris_mpc_common::{
    helpers::sync::Modification,
    job::{BatchMetadata, BatchQuery, IrisQueryBatchEntries},
};
use std::collections::{HashMap, HashSet};

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchQueryEntriesPreprocessed {
    pub code: Vec<Vec<u8>>,
    pub mask: Vec<Vec<u8>>,
}

impl From<IrisQueryBatchEntries> for BatchQueryEntriesPreprocessed {
    fn from(value: IrisQueryBatchEntries) -> Self {
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
    pub request_ids: Vec<String>,
    pub request_types: Vec<String>,
    pub metadata: Vec<BatchMetadata>,

    pub left_iris_requests: IrisQueryBatchEntries,
    pub right_iris_requests: IrisQueryBatchEntries,
    pub left_iris_rotated_requests: IrisQueryBatchEntries,
    pub right_iris_rotated_requests: IrisQueryBatchEntries,
    pub left_iris_interpolated_requests: IrisQueryBatchEntries,
    pub right_iris_interpolated_requests: IrisQueryBatchEntries,
    pub left_mirrored_iris_interpolated_requests: IrisQueryBatchEntries,
    pub right_mirrored_iris_interpolated_requests: IrisQueryBatchEntries,

    pub or_rule_indices: Vec<Vec<u32>>,
    pub luc_lookback_records: usize,
    pub valid_entries: Vec<bool>,
    pub skip_persistence: Vec<bool>,

    // Only reauth specific fields
    // Map from reauth request id to the index of the target entry to be matched
    pub reauth_target_indices: HashMap<String, u32>,
    pub reauth_use_or_rule: HashMap<String, bool>,

    // Only deletion specific fields
    pub deletion_requests_indices: Vec<u32>, // 0-indexed indices of entries to be deleted

    // this one is not needed for the GPU actor
    // pub deletion_requests_metadata: Vec<BatchMetadata>,

    // Reset Update specific fields
    pub reset_update_indices: Vec<u32>,
    pub reset_update_request_ids: Vec<String>,
    pub reset_update_shares: Vec<GaloisSharesBothSides>,

    // additional fields which are GPU specific
    pub left_iris_interpolated_requests_preprocessed: BatchQueryEntriesPreprocessed,
    pub right_iris_interpolated_requests_preprocessed: BatchQueryEntriesPreprocessed,
    pub left_iris_rotated_requests_preprocessed: BatchQueryEntriesPreprocessed,
    pub right_iris_rotated_requests_preprocessed: BatchQueryEntriesPreprocessed,
    pub left_mirrored_iris_interpolated_requests_preprocessed: BatchQueryEntriesPreprocessed,
    pub right_mirrored_iris_interpolated_requests_preprocessed: BatchQueryEntriesPreprocessed,

    // Keeping track of updates & deletions for sync mechanism. Mapping: Serial id -> Modification
    pub modifications: HashMap<ModificationKey, Modification>,

    // SNS message ids to assert identical batch processing across parties
    pub sns_message_ids: Vec<String>,

    // If true, anonymized statistics (1D, 2D, mirror) are disabled for this batch.
    pub disable_anonymized_stats: bool,
}

impl PreprocessedBatchQuery {
    pub fn get_iris_requests(&self, eye: Eye) -> &IrisQueryBatchEntries {
        match eye {
            Eye::Left => &self.left_iris_requests,
            Eye::Right => &self.right_iris_requests,
        }
    }

    pub fn get_iris_interpolated_requests_preprocessed(
        &self,
        eye: Eye,
        orientation: Orientation,
    ) -> &BatchQueryEntriesPreprocessed {
        if orientation == Orientation::Mirror {
            // To handle the full-face mirror attack, we want to use:
            // left_mirrored VS right_db
            // right_mirrored VS left_db
            // Hence we need to swap the left and right iris interpolated requests
            match eye {
                Eye::Left => &self.right_mirrored_iris_interpolated_requests_preprocessed,
                Eye::Right => &self.left_mirrored_iris_interpolated_requests_preprocessed,
            }
        } else {
            match eye {
                Eye::Left => &self.left_iris_interpolated_requests_preprocessed,
                Eye::Right => &self.right_iris_interpolated_requests_preprocessed,
            }
        }
    }

    pub fn get_iris_requests_rotated(&self, eye: Eye) -> &IrisQueryBatchEntries {
        match eye {
            Eye::Left => &self.left_iris_rotated_requests,
            Eye::Right => &self.right_iris_rotated_requests,
        }
    }

    pub fn get_iris_requests_rotated_preprocessed(
        &self,
        eye: Eye,
    ) -> &BatchQueryEntriesPreprocessed {
        match eye {
            Eye::Left => &self.left_iris_rotated_requests_preprocessed,
            Eye::Right => &self.right_iris_rotated_requests_preprocessed,
        }
    }
}

impl From<BatchQuery> for PreprocessedBatchQuery {
    fn from(value: BatchQuery) -> Self {
        let mut left_iris_interpolated_requests_preprocessed = None;
        let mut right_iris_interpolated_requests_preprocessed = None;
        let mut left_iris_rotated_requests_preprocessed = None;
        let mut right_iris_rotated_requests_preprocessed = None;
        let mut left_mirrored_iris_interpolated_requests_preprocessed = None;
        let mut right_mirrored_iris_interpolated_requests_preprocessed = None;

        rayon::scope(|s| {
            s.spawn(|_| {
                left_iris_interpolated_requests_preprocessed =
                    Some(BatchQueryEntriesPreprocessed::from(
                        value.left_iris_interpolated_requests.clone(),
                    ));
            });
            s.spawn(|_| {
                right_iris_interpolated_requests_preprocessed =
                    Some(BatchQueryEntriesPreprocessed::from(
                        value.right_iris_interpolated_requests.clone(),
                    ));
            });
            s.spawn(|_| {
                left_iris_rotated_requests_preprocessed = Some(
                    BatchQueryEntriesPreprocessed::from(value.left_iris_rotated_requests.clone()),
                );
            });
            s.spawn(|_| {
                right_iris_rotated_requests_preprocessed = Some(
                    BatchQueryEntriesPreprocessed::from(value.right_iris_rotated_requests.clone()),
                );
            });
            s.spawn(|_| {
                left_mirrored_iris_interpolated_requests_preprocessed =
                    Some(BatchQueryEntriesPreprocessed::from(
                        value.left_mirrored_iris_interpolated_requests.clone(),
                    ));
            });
            s.spawn(|_| {
                right_mirrored_iris_interpolated_requests_preprocessed =
                    Some(BatchQueryEntriesPreprocessed::from(
                        value.right_mirrored_iris_interpolated_requests.clone(),
                    ));
            });
        });

        Self {
            request_ids: value.request_ids,
            request_types: value.request_types,
            metadata: value.metadata,
            left_iris_requests: value.left_iris_requests,
            right_iris_requests: value.right_iris_requests,
            left_iris_rotated_requests: value.left_iris_rotated_requests,
            right_iris_rotated_requests: value.right_iris_rotated_requests,
            left_iris_interpolated_requests: value.left_iris_interpolated_requests,
            right_iris_interpolated_requests: value.right_iris_interpolated_requests,
            left_mirrored_iris_interpolated_requests: value
                .left_mirrored_iris_interpolated_requests,
            right_mirrored_iris_interpolated_requests: value
                .right_mirrored_iris_interpolated_requests,
            or_rule_indices: value.or_rule_indices,
            luc_lookback_records: value.luc_lookback_records,
            valid_entries: value.valid_entries,
            reauth_target_indices: value.reauth_target_indices,
            reauth_use_or_rule: value.reauth_use_or_rule,
            reset_update_indices: value.reset_update_indices,
            reset_update_request_ids: value.reset_update_request_ids,
            reset_update_shares: value.reset_update_shares,
            deletion_requests_indices: value.deletion_requests_indices,
            // deletion_requests_metadata: value.deletion_requests_metadata,
            left_iris_interpolated_requests_preprocessed:
                left_iris_interpolated_requests_preprocessed.unwrap(),
            left_iris_rotated_requests_preprocessed: left_iris_rotated_requests_preprocessed
                .unwrap(),
            right_iris_interpolated_requests_preprocessed:
                right_iris_interpolated_requests_preprocessed.unwrap(),
            right_iris_rotated_requests_preprocessed: right_iris_rotated_requests_preprocessed
                .unwrap(),
            left_mirrored_iris_interpolated_requests_preprocessed:
                left_mirrored_iris_interpolated_requests_preprocessed.unwrap(),
            right_mirrored_iris_interpolated_requests_preprocessed:
                right_mirrored_iris_interpolated_requests_preprocessed.unwrap(),
            modifications: value.modifications,
            sns_message_ids: value.sns_message_ids,
            skip_persistence: value.skip_persistence,
            disable_anonymized_stats: value.disable_anonymized_stats,
        }
    }
}

impl PreprocessedBatchQuery {
    pub fn retain(&mut self, indices: &[usize]) {
        let indices_set: HashSet<usize> = indices.iter().cloned().collect();
        filter_by_indices!(self.request_ids, indices_set);
        filter_by_indices!(self.request_types, indices_set);
        filter_by_indices!(self.metadata, indices_set);
        filter_by_indices!(self.left_iris_requests.code, indices_set);
        filter_by_indices!(self.left_iris_requests.mask, indices_set);
        filter_by_indices!(self.right_iris_requests.code, indices_set);
        filter_by_indices!(self.right_iris_requests.mask, indices_set);
        filter_by_indices!(self.or_rule_indices, indices_set);
        filter_by_indices!(self.skip_persistence, indices_set);
        filter_by_indices_with_rotations!(self.left_iris_interpolated_requests.code, indices_set);
        filter_by_indices_with_rotations!(self.left_iris_interpolated_requests.mask, indices_set);
        filter_by_indices_with_rotations!(self.left_iris_rotated_requests.code, indices_set);
        filter_by_indices_with_rotations!(self.left_iris_rotated_requests.mask, indices_set);
        filter_by_indices_with_rotations!(self.right_iris_interpolated_requests.code, indices_set);
        filter_by_indices_with_rotations!(self.right_iris_interpolated_requests.mask, indices_set);
        filter_by_indices_with_rotations!(self.right_iris_rotated_requests.code, indices_set);
        filter_by_indices_with_rotations!(self.right_iris_rotated_requests.mask, indices_set);
        filter_by_indices_with_rotations!(
            self.left_mirrored_iris_interpolated_requests.code,
            indices_set
        );
        filter_by_indices_with_rotations!(
            self.left_mirrored_iris_interpolated_requests.mask,
            indices_set
        );
        filter_by_indices_with_rotations!(
            self.right_mirrored_iris_interpolated_requests.code,
            indices_set
        );
        filter_by_indices_with_rotations!(
            self.right_mirrored_iris_interpolated_requests.mask,
            indices_set
        );

        Self::filter_preprocessed_entry(
            &mut self.left_iris_interpolated_requests_preprocessed,
            &indices_set,
        );
        Self::filter_preprocessed_entry(
            &mut self.left_iris_rotated_requests_preprocessed,
            &indices_set,
        );
        Self::filter_preprocessed_entry(
            &mut self.right_iris_interpolated_requests_preprocessed,
            &indices_set,
        );
        Self::filter_preprocessed_entry(
            &mut self.right_iris_rotated_requests_preprocessed,
            &indices_set,
        );
        Self::filter_preprocessed_entry(
            &mut self.left_mirrored_iris_interpolated_requests_preprocessed,
            &indices_set,
        );
        Self::filter_preprocessed_entry(
            &mut self.right_mirrored_iris_interpolated_requests_preprocessed,
            &indices_set,
        );
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
