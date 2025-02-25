use crate::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{statistics::BucketStatistics, sync::Modification},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, future::Future};

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

#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct BatchQuery {
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

    // Only reauth specific fields
    // Map from reauth request id to the index of the target entry to be matched
    pub reauth_target_indices: HashMap<String, u32>,
    pub reauth_use_or_rule:    HashMap<String, bool>,

    // Only deletion specific fields
    pub deletion_requests_indices:  Vec<u32>, // 0-indexed indices of entries to be deleted
    pub deletion_requests_metadata: Vec<BatchMetadata>,

    // Keeping track of updates & deletions for sync mechanism. Mapping: Serial id -> Modification
    pub modifications: HashMap<u32, Modification>,

    // SNS message ids to assert identical batch processing across parties
    pub sns_message_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerJobResult<A = ()> {
    pub merged_results: Vec<u32>,
    pub request_ids: Vec<String>,
    pub request_types: Vec<String>,
    pub metadata: Vec<BatchMetadata>,
    pub matches: Vec<bool>,
    pub match_ids: Vec<Vec<u32>>,
    pub partial_match_ids_left: Vec<Vec<u32>>,
    pub partial_match_ids_right: Vec<Vec<u32>>,
    pub partial_match_counters_left: Vec<usize>,
    pub partial_match_counters_right: Vec<usize>,
    pub store_left: BatchQueryEntries,
    pub store_right: BatchQueryEntries,
    pub deleted_ids: Vec<u32>,
    pub matched_batch_request_ids: Vec<Vec<String>>,
    pub anonymized_bucket_statistics_left: BucketStatistics,
    pub anonymized_bucket_statistics_right: BucketStatistics,
    pub successful_reauths: Vec<bool>, // true if request type is reauth and it's successful
    pub reauth_target_indices: HashMap<String, u32>,
    pub reauth_or_rule_used: HashMap<String, bool>,
    pub modifications: HashMap<u32, Modification>,
    /// Actor-specific data (e.g. graph mutations).
    pub actor_data: A,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Eye {
    #[default]
    Left,
    Right,
}

pub trait JobSubmissionHandle {
    type A;

    #[allow(async_fn_in_trait)]
    async fn submit_batch_query(
        &mut self,
        batch: BatchQuery,
    ) -> impl Future<Output = ServerJobResult<Self::A>>;
}
