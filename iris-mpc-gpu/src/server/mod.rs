mod actor;
pub mod sync_nccl;

use crate::helpers::query_processor::CompactQuery;
pub use actor::{preprocess_batch_query, ServerActor, ServerActorHandle};
use iris_mpc_common::galois_engine::degree4::GaloisRingIrisCodeShare;
use tokio::sync::oneshot;

pub const MAX_BATCH_SIZE: usize = 64;

pub type PreprocessedBatchQuery = (CompactQuery, CompactQuery);

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchQueryEntries {
    pub code: Vec<GaloisRingIrisCodeShare>,
    pub mask: Vec<GaloisRingIrisCodeShare>,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchMetadata {
    pub node_id:  String,
    pub trace_id: String,
    pub span_id:  String,
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchQuery {
    pub request_ids: Vec<String>,
    pub metadata:    Vec<BatchMetadata>,
    pub query_left:  BatchQueryEntries,
    pub db_left:     BatchQueryEntries,
    pub store_left:  BatchQueryEntries,
    pub query_right: BatchQueryEntries,
    pub db_right:    BatchQueryEntries,
    pub store_right: BatchQueryEntries,
}

pub struct ServerJob {
    batch:              BatchQuery,
    preprocessed_batch: PreprocessedBatchQuery,
    return_channel:     oneshot::Sender<ServerJobResult>,
}

#[derive(Debug, Clone)]
pub struct ServerJobResult {
    pub merged_results: Vec<u32>,
    pub request_ids:    Vec<String>,
    pub matches:        Vec<bool>,
    pub store_left:     BatchQueryEntries,
    pub store_right:    BatchQueryEntries,
}

enum Eye {
    Left,
    Right,
}
