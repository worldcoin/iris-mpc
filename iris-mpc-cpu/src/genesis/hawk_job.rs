use crate::execution::hawk_main::{BothEyes, VecRequests};
use crate::hawkers::aby3::aby3_store;
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
    queries: Arc<BothEyes<VecRequests<aby3_store::QueryRef>>>,
}

/// An indexation job result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {}

/// Convertors.
impl From<&Vec<DbStoredIris>> for JobRequest {
    fn from(_value: &Vec<DbStoredIris>) -> Self {
        // From a vec of 64 stored iris codes (left and right)
        // 1. Map each iris to a GaloisRingSharedIris
        // 2. Map each GaloisRingSharedIris to a QueryRef using aby3_store::prepare_query
        // 3. Insert Left.QueryRef + Right.QueryRef into relevant vecs.
        // 4. Return result.
        unimplemented!()
    }
}
