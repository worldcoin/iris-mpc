use super::Batch;
use crate::{
    execution::hawk_main::{
        iris_worker::IrisPoolHandle, BothEyes, HawkMutation, VecRequests, LEFT, RIGHT,
    },
    hawkers::aby3::aby3_store::Aby3Query,
};
use eyre::Result;
use futures::future::try_join_all;
use iris_mpc_common::{helpers::sync::Modification, IrisSerialId, IrisVectorId};
use std::{fmt, sync::Arc};
use tokio::{
    join,
    sync::{self, oneshot},
};

// Helper type: Aby3 store batch query.
pub type Aby3BatchQuery = BothEyes<VecRequests<Aby3Query>>;

// Helper type: Aby3 store batch query reference.
pub type Aby3BatchQueryRef = Arc<Aby3BatchQuery>;

/// An indexation job that materialises an in-mem graph.
pub struct Job {
    // A request encapsulating data for indexation.
    pub(super) request: JobRequest,

    // Tokio channel through which job result will be signalled.
    pub(super) return_channel: oneshot::Sender<Result<(sync::oneshot::Receiver<()>, JobResult)>>,
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
    // acts as a code barrier
    Sync(bool),
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

    pub async fn numa_realloc(
        queries: Aby3BatchQueryRef,
        workers: BothEyes<IrisPoolHandle>,
    ) -> Aby3BatchQueryRef {
        let (left, right) = join!(
            Self::numa_realloc_side(&queries[LEFT], &workers[LEFT]),
            Self::numa_realloc_side(&queries[RIGHT], &workers[RIGHT]),
        );
        Arc::new([left, right])
    }

    async fn numa_realloc_side(
        requests: &VecRequests<Aby3Query>,
        worker: &IrisPoolHandle,
    ) -> VecRequests<Aby3Query> {
        // Iterate over all the irises.
        let all_irises_iter = requests
            .iter()
            .flat_map(|query| [&query.iris, &query.iris_proc]);

        // Go realloc the irises in parallel.
        let tasks = all_irises_iter.map(|iris| worker.numa_realloc(iris.clone()).unwrap());

        // Iterate over the results in the same order.
        let mut new_irises_iter = try_join_all(tasks).await.unwrap().into_iter();

        // Rebuild the same structure with the new irises.
        let new_requests = requests
            .iter()
            .map(|_old_query| {
                let iris = new_irises_iter.next().unwrap();
                let iris_proc = new_irises_iter.next().unwrap();
                Aby3Query { iris, iris_proc }
            })
            .collect::<Vec<_>>();

        assert!(new_irises_iter.next().is_none());
        new_requests
    }
}

/// An indexation result over a set of irises.
#[derive(Debug)]
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
        done_tx: sync::oneshot::Sender<()>,
    },
    Modification {
        /// Modification id of associated modifications table entry.
        modification_id: i64,

        /// Vector id for persistence.
        vector_id_to_persist: IrisVectorId,

        /// Connect plans for updating HNSW graph in DB.
        connect_plans: HawkMutation,
        done_tx: sync::oneshot::Sender<()>,
    },
    Sync(bool),
}

/// Constructor.
impl JobResult {
    pub(crate) fn new_batch_result(
        batch_id: usize,
        vector_ids: Vec<IrisVectorId>,
        connect_plans: HawkMutation,
        vector_ids_to_persist: Vec<IrisVectorId>,
        done_tx: sync::oneshot::Sender<()>,
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
            done_tx,
        }
    }

    pub(crate) fn new_modification_result(
        modification_id: i64,
        connect_plans: HawkMutation,
        vector_id_to_persist: IrisVectorId,
        done_tx: sync::oneshot::Sender<()>,
    ) -> Self {
        Self::Modification {
            modification_id,
            connect_plans,
            vector_id_to_persist,
            done_tx,
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
            JobResult::Sync(x) => write!(f, "sync={x}"),
        }
    }
}
