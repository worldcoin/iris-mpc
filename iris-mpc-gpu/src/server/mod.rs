mod actor;
pub mod sync_nccl;

pub use actor::{ServerActor, ServerActorHandle};
use iris_mpc_common::galois_engine::degree4::GaloisRingIrisCodeShare;
use tokio::sync::oneshot;

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
    pub query:       BatchQueryEntries,
    pub db:          BatchQueryEntries,
    pub store:       BatchQueryEntries,
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
    pub store:          BatchQueryEntries,
}
