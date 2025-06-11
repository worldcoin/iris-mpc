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
    pub fn new_batch_indexation(
        Batch {
            batch_id,
            vector_ids,
            left_queries,
            right_queries,
        }: Batch,
    ) -> Self {
        assert!(!vector_ids.is_empty(), "Invalid batch: is empty");

        Self::BatchIndexation {
            batch_id,
            vector_ids,
            queries: Arc::new([left_queries, right_queries]),
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

        /// Iris serial id of batch's first element.
        first_serial_id: IrisSerialId,

        /// Iris serial id of batch's last element.
        last_serial_id: IrisSerialId,
    },
    Modification {
        /// Modification id of associated modifications table entry
        modification_id: i64,

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
    ) -> Self {
        let first_serial_id = vector_ids.first().unwrap().serial_id();
        let last_serial_id = vector_ids.last().unwrap().serial_id();
        Self::BatchIndexation {
            connect_plans,
            batch_id,
            vector_ids,
            first_serial_id,
            last_serial_id,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn new_modification_result(
        modification_id: i64,
        connect_plans: HawkMutation,
    ) -> Self {
        Self::Modification {
            modification_id,
            connect_plans,
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
                    "batch-id={}, batch-size={}, range=({}..{})",
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
