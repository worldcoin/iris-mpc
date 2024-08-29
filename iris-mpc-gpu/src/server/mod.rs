mod actor;
pub mod heartbeat_nccl;
pub mod sync_nccl;

pub use actor::{ServerActor, ServerActorHandle};
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use tokio::sync::oneshot;

pub const MAX_BATCH_SIZE: usize = 64;

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
    pub request_ids: Vec<String>,
    pub metadata:    Vec<BatchMetadata>,
    pub query_left:  BatchQueryEntries,
    pub db_left:     BatchQueryEntries,
    pub store_left:  BatchQueryEntries,
    pub query_right: BatchQueryEntries,
    pub db_right:    BatchQueryEntries,
    pub store_right: BatchQueryEntries,
}

#[derive(Debug)]
pub struct ServerJob {
    batch:          BatchQuery,
    return_channel: oneshot::Sender<ServerJobResult>,
}

#[derive(Debug, Clone)]
pub struct ServerJobResult {
    pub merged_results: Vec<u32>,
    pub request_ids:    Vec<String>,
    pub matches:        Vec<bool>,
    pub match_ids:      Vec<Vec<u32>>,
    pub store_left:     BatchQueryEntries,
    pub store_right:    BatchQueryEntries,
}

enum Eye {
    Left,
    Right,
}
