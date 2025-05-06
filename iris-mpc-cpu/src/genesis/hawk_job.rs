use crate::{
    execution::hawk_main::{BothEyes, VecRequests},
    hawkers::aby3::aby3_store::{prepare_query as prepare_aby3_query, QueryRef as Aby3QueryRef},
    protocol::shared_iris::GaloisRingSharedIris,
};
use eyre::Result;
use iris_mpc_store::DbStoredIris;
use std::sync::Arc;
use tokio::sync::oneshot;

// Helper type: Aby3 store batch query.
pub type Aby3BatchQuery = BothEyes<VecRequests<Aby3QueryRef>>;

// Helper type: Aby3 store batch query reference.
pub type Aby3BatchQueryRef = Arc<Aby3BatchQuery>;

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
    queries: Aby3BatchQueryRef,
}

/// An indexation job result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {}

/// Constructor.
impl JobRequest {
    pub fn new(party_id: usize, data: &[DbStoredIris]) -> Self {
        Self {
            queries: Arc::new([
                data.iter()
                    .map(|iris| {
                        prepare_aby3_query(
                            GaloisRingSharedIris::try_from_buffers_inner(
                                party_id,
                                iris.left_code(),
                                iris.left_mask(),
                            )
                            .unwrap(),
                        )
                    })
                    .collect(),
                data.iter()
                    .map(|iris| {
                        prepare_aby3_query(
                            GaloisRingSharedIris::try_from_buffers_inner(
                                party_id,
                                iris.right_code(),
                                iris.right_mask(),
                            )
                            .unwrap(),
                        )
                    })
                    .collect(),
            ]),
        }
    }
}
