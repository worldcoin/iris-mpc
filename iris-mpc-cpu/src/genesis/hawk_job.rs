use crate::execution::hawk_main::{BothEyes, VecRequests};
use crate::hawkers::aby3::aby3_store::QueryRef;
use eyre::Result;
use iris_mpc_store::DbStoredIris;
use std::sync::Arc;
use tokio::sync::oneshot;

/// An indexation job that materialises an in-mem graph.
#[allow(dead_code)]
pub struct Job {
    // A request encapsulating data for indexation.
    pub(super) request: JobRequest,

    // Tokio channel through which job result will be signalled.
    pub(super) return_channel: oneshot::Sender<Result<JobResult>>,
}

/// An indexation job request.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct JobRequest {
    // Indexation queries over both eyes.
    queries: Arc<BothEyes<VecRequests<QueryRef>>>,
}

/// An indexation job result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {}

/// Convertors.
impl From<&Vec<DbStoredIris>> for JobRequest {
    fn from(_batch: &Vec<DbStoredIris>) -> Self {
        unimplemented!()
    }
}
