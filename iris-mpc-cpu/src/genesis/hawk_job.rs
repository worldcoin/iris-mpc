use super::{
    utils::{PartyId, COUNT_OF_MPC_PARTIES},
    Batch,
};
use crate::{
    execution::hawk_main::{BothEyes, HawkMutation, VecRequests},
    hawkers::aby3::aby3_store::{prepare_query as prepare_aby3_query, QueryRef as Aby3QueryRef},
    protocol::shared_iris::GaloisRingSharedIris,
};
use eyre::Result;
use iris_mpc_common::{IrisSerialId, IrisVectorId};
use std::{fmt, sync::Arc};
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

        Self {
            batch_id,
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
