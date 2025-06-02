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
use itertools::multiunzip;
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
    pub identifiers: Vec<IrisVectorId>,

    /// HNSW indexation queries over both eyes.
    pub queries: Aby3BatchQueryRef,
}

/// Constructor.
impl JobRequest {
    pub fn new(party_id: PartyId, Batch { data, id: batch_id }: Batch) -> Self {
        assert!(party_id < COUNT_OF_MPC_PARTIES, "Invalid party id");
        assert!(!data.is_empty(), "Invalid batch: is empty");

        let (identifiers, left_queries, right_queries) = multiunzip(data);

        Self {
            batch_id,
            identifiers,
            queries: Arc::new([left_queries, right_queries]),
        }
    }
}

// Methods.
impl JobRequest {
    // Incoming batch size.
    pub fn batch_size(&self) -> usize {
        self.identifiers.len()
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
    pub identifiers: Vec<IrisVectorId>,

    /// Iris serial id of batch's last element.
    pub last_serial_id: IrisSerialId,
}

/// Constructor.
impl JobResult {
    pub(crate) fn new(request: &JobRequest, connect_plans: HawkMutation) -> Self {
        Self {
            connect_plans,
            batch_id: request.batch_id,
            first_serial_id: request.identifiers.first().unwrap().serial_id(),
            identifiers: request.identifiers.clone(),
            last_serial_id: request.identifiers.last().unwrap().serial_id(),
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
            self.identifiers.len(),
            self.first_serial_id,
            self.last_serial_id
        )
    }
}
