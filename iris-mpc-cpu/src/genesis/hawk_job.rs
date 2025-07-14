use super::Batch;
use crate::{
    execution::hawk_main::{BothEyes, HawkMutation, VecRequests},
    hawkers::aby3::aby3_store::QueryRef,
};
use eyre::Result;
use iris_mpc_common::{helpers::sync::Modification, IrisSerialId, IrisVectorId};
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
pub enum JobRequest {
    BatchIndexation {
        // Incoming batch identifier.
        batch_id: usize,

        // Incoming batch of iris identifiers for subsequent correlation.
        vector_ids: Vec<IrisVectorId>,

        /// Iris data for persistence.
        vector_ids_to_persist: Vec<IrisVectorId>,

        /// HNSW indexation queries over both eyes.
        queries: Aby3BatchQueryRef,
    },
    Modification {
        // Modification entry for processing
        modification: Modification,
    },
}

/// Constructor.
impl JobRequest {
    pub fn new_batch_indexation(batch: &Batch) -> Self {
        Self::BatchIndexation {
            batch_id: batch.batch_id,
            vector_ids: batch.vector_ids.clone(),
            queries: Arc::new([batch.left_queries.clone(), batch.right_queries.clone()]),
            vector_ids_to_persist: batch.vector_ids_to_persist.clone(),
        }
    }

    pub fn new_modification(modification: Modification) -> Self {
        Self::Modification { modification }
    }
}

/// An indexation result over a set of irises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JobResult {
    BatchIndexation {
        /// Unique sequential job identifier.
        batch_id: usize,

        /// Connect plans for updating HNSW graph in DB.
        connect_plans: HawkMutation,

        /// Set of Iris identifiers being indexed.
        vector_ids: Vec<IrisVectorId>,

        /// Vector ids for persistence
        vector_ids_to_persist: Vec<IrisVectorId>,

        /// Iris serial id of batch's first element.
        first_serial_id: IrisSerialId,
        /// Iris serial id of batch's last element.
        last_serial_id: IrisSerialId,
    },
    Modification {
        /// Modification id of associated modifications table entry.
        modification_id: i64,

        /// Vector id for persistence.
        vector_id_to_persist: IrisVectorId,

        /// Connect plans for updating HNSW graph in DB.
        connect_plans: HawkMutation,
    },
}

/// Constructor.
impl JobResult {
    pub(crate) fn new_batch_result(
        batch_id: usize,
        vector_ids: Vec<IrisVectorId>,
        connect_plans: HawkMutation,
        vector_ids_to_persist: Vec<IrisVectorId>,
    ) -> Self {
        let first_serial_id = vector_ids_to_persist.first().unwrap().serial_id();
        let last_serial_id = vector_ids_to_persist.last().unwrap().serial_id();

        Self::BatchIndexation {
            connect_plans,
            batch_id,
            vector_ids,
            vector_ids_to_persist,
            first_serial_id,
            last_serial_id,
        }
    }

    pub(crate) fn new_modification_result(
        modification_id: i64,
        connect_plans: HawkMutation,
        vector_id_to_persist: IrisVectorId,
    ) -> Self {
        Self::Modification {
            modification_id,
            connect_plans,
            vector_id_to_persist,
        }
    }
}

/// Trait: fmt::Display.
impl fmt::Display for JobResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            JobResult::BatchIndexation {
                batch_id,
                vector_ids,
                first_serial_id,
                last_serial_id,
                ..
            } => {
                write!(
                    f,
                    "batch-id={}, batch-size={}, range=({:?}..{:?})",
                    batch_id,
                    vector_ids.len(),
                    first_serial_id,
                    last_serial_id
                )
            }
            JobResult::Modification {
                modification_id, ..
            } => {
                write!(f, "modification-id={}", modification_id)
            }
        }
    }
}
