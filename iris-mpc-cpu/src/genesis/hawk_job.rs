use super::{
    utils::{PartyId, COUNT_OF_MPC_PARTIES},
    Batch,
};
use crate::{
    execution::hawk_main::{BothEyes, HawkMutation, VecRequests},
    hawkers::aby3::aby3_store::QueryRef,
};
use eyre::Result;
use iris_mpc_common::{IrisSerialId, IrisVectorId};
use std::{fmt, sync::Arc};
use tokio::sync::oneshot;

// Helper type: Aby3 store batch query.
pub type Aby3BatchQuery = BothEyes<VecRequests<QueryRef>>;

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
    // Incoming batch identifier.
    pub batch_id: usize,

    // Incoming batch of iris identifiers for subsequent correlation.
    pub vector_ids: Vec<IrisVectorId>,

    /// HNSW indexation queries over both eyes.
    pub queries: Aby3BatchQueryRef,
}

/// Constructor.
impl JobRequest {
    pub fn new(
        party_id: PartyId,
        Batch {
            batch_id,
            vector_ids,
            left_queries,
            right_queries,
        }: Batch,
    ) -> Self {
        assert!(party_id < COUNT_OF_MPC_PARTIES, "Invalid party id");
        assert!(!vector_ids.is_empty(), "Invalid batch: is empty");

        Self {
            batch_id,
            vector_ids,
            queries: Arc::new([left_queries, right_queries]),
        }
    }
}

// Methods.
impl JobRequest {
    // Incoming batch size.
    pub fn batch_size(&self) -> usize {
        self.vector_ids.len()
    }
}

/// An indexation result over a set of irises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {
    /// Unique sequential job identifier.
    pub batch_id: usize,

    /// Connect plans for updating HNSW graph in DB.
    pub connect_plans: HawkMutation,

    /// Iris serial id of batch's first element.
    pub first_serial_id: IrisSerialId,

    /// Set of Iris identifiers being indexed.
    pub vector_ids: Vec<IrisVectorId>,

    /// Iris serial id of batch's last element.
    pub last_serial_id: IrisSerialId,
}

/// Constructor.
impl JobResult {
    pub(crate) fn new(request: &JobRequest, connect_plans: HawkMutation) -> Self {
        Self {
            connect_plans,
            batch_id: request.batch_id,
            first_serial_id: request.vector_ids.first().unwrap().serial_id(),
            vector_ids: request.vector_ids.clone(),
            last_serial_id: request.vector_ids.last().unwrap().serial_id(),
        }
    }
}

/// Trait: fmt::Display.
impl fmt::Display for JobResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "batch-id={}, batch-size={}, range=({}..{})",
            self.batch_id,
            self.vector_ids.len(),
            self.first_serial_id,
            self.last_serial_id
        )
    }
}
