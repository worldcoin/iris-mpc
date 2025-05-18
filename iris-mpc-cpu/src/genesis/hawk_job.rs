use crate::{
    execution::hawk_main::{BothEyes, HawkMutation, VecRequests},
    hawkers::aby3::aby3_store::{prepare_query as prepare_aby3_query, QueryRef as Aby3QueryRef},
    protocol::shared_iris::GaloisRingSharedIris,
};
use eyre::Result;
use iris_mpc_common::IrisVectorId;
use iris_mpc_store::DbStoredIris;
use std::sync::Arc;
use tokio::sync::oneshot;

// Helper type: Aby3 store batch query.
pub type Aby3BatchQuery = BothEyes<VecRequests<Aby3QueryRef>>;

// Helper type: Aby3 store batch query reference.
pub type Aby3BatchQueryRef = Arc<Aby3BatchQuery>;

/// An indexation job that materialises an in-mem graph.
pub struct Job {
    // A request encapsulating data for indexation.
    pub(super) request: JobRequest,

    // Tokio channel through which job result will be signalled.
    pub(super) return_channel: oneshot::Sender<Result<JobResult>>,
}

/// An indexation job request.
#[derive(Clone, Debug)]
pub struct JobRequest {
    /// Unique sequential identifier for the job
    pub job_id: usize,

    // Incoming batch of iris identifiers for subsequent correlation.
    pub identifiers: Vec<IrisVectorId>,

    // HNSW indexation queries over both eyes.
    pub queries: Aby3BatchQueryRef,
}

/// Constructor.
impl JobRequest {
    pub fn new(job_id: usize, party_id: usize, data: &[DbStoredIris]) -> Self {
        Self {
            job_id,
            identifiers: data.iter().map(IrisVectorId::from).collect(),
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

/// An indexation result over a set of irises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {
    /// Unique sequential identifier for the job
    pub job_id: usize,

    /// Which identifiers inserted in the job
    pub identifiers: Vec<IrisVectorId>,

    /// Connect plans for updating the HNSW graph in DB
    pub connect_plans: HawkMutation,
}

/// An indexation result over a single iris.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResultOfBatchElement {
    // Identifier of iris.
    iris_vector_id: IrisVectorId,

    // Error flag - TEMP.
    did_error: bool,
}

/// Constructor.
#[allow(dead_code)]
impl JobResultOfBatchElement {
    pub fn new(iris_identifier: IrisVectorId, did_error: bool) -> Self {
        Self {
            iris_vector_id: iris_identifier,
            did_error,
        }
    }
}
