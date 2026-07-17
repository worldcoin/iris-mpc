use super::Batch;
use crate::{
    execution::hawk_main::{iris_worker::QueryId, BothEyes, VecRequests},
    graph_checkpoint::GraphCheckpointState,
    hawkers::aby3::aby3_store::Aby3Query,
    protocol::shared_iris::ArcIris,
};
use eyre::Result;
use iris_mpc_common::{SerialId, VectorId};
use std::{
    fmt,
    sync::{atomic::AtomicU8, Arc},
};

use tokio::sync::{self, oneshot};

// constants for managing the persistence synchronization logic
pub const SYNC_RUNNING: u8 = 0;
pub const SYNC_DONE: u8 = 1;
pub const SYNC_ERROR: u8 = 2;

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
        vector_ids: Vec<VectorId>,

        /// Iris data for persistence.
        vector_ids_to_persist: Vec<VectorId>,

        /// HNSW indexation queries over both eyes.
        queries: Aby3BatchQueryRef,

        /// Irises to cache in worker pools before search, per eye [LEFT, RIGHT].
        irises_to_cache: BothEyes<Vec<(QueryId, ArcIris)>>,
    },
    /// Graph surgery for one serial (version-join delta): remove every
    /// existing graph key, then optionally search + re-insert the current
    /// source version.
    VersionReplay {
        serial_id: SerialId,
        /// Existing graph keys per eye, removed before the insertion search.
        removals: BothEyes<Vec<VectorId>>,
        /// False for deletion tombstones: remove only.
        reinsert: bool,
    },
    /// Acts as a code barrier for inter-node synchronization.
    SyncState {
        /// Whether this node has been signaled to shut down.
        shutdown: bool,
        sync_status: Arc<AtomicU8>,
    },
    SyncPeers,
}

/// Constructor.
impl JobRequest {
    pub fn new_batch_indexation(batch: &Batch) -> Self {
        Self::BatchIndexation {
            batch_id: batch.batch_id,
            vector_ids: batch.vector_ids.clone(),
            queries: Arc::new([batch.left_queries.clone(), batch.right_queries.clone()]),
            vector_ids_to_persist: batch.vector_ids_to_persist.clone(),
            irises_to_cache: batch.irises_to_cache.clone(),
        }
    }

    pub fn new_version_replay(
        serial_id: SerialId,
        removals: BothEyes<Vec<VectorId>>,
        reinsert: bool,
    ) -> Self {
        Self::VersionReplay {
            serial_id,
            removals,
            reinsert,
        }
    }
}

/// An indexation result over a set of irises.
#[derive(Debug)]
pub enum JobResult {
    BatchIndexation {
        /// Unique sequential job identifier.
        batch_id: usize,

        /// Set of Iris identifiers being indexed.
        vector_ids: Vec<VectorId>,

        /// Vector ids for persistence
        vector_ids_to_persist: Vec<VectorId>,

        /// Iris serial id of batch's first element.
        first_serial_id: SerialId,
        /// Iris serial id of batch's last element.
        last_serial_id: SerialId,
        done_tx: sync::oneshot::Sender<()>,
    },
    VersionReplay {
        /// Serial the surgery targeted (present for remove-only completions
        /// too, so every completion is attributable in logs).
        serial_id: SerialId,

        /// Vector id for persistence; `None` for remove-only surgery.
        vector_id_to_persist: Option<VectorId>,

        done_tx: sync::oneshot::Sender<()>,
    },
    S3Checkpoint {
        checkpoint_state: GraphCheckpointState,
        done_tx: sync::oneshot::Sender<()>,
    },
    SyncState {
        /// Whether the shutdown states of different nodes' Sync jobs
        /// were mismatched.
        mismatched: bool,
    },
    SyncPeers,
}

/// Constructor.
impl JobResult {
    pub(crate) fn new_batch_result(
        batch_id: usize,
        vector_ids: Vec<VectorId>,
        vector_ids_to_persist: Vec<VectorId>,
        done_tx: sync::oneshot::Sender<()>,
    ) -> Self {
        let first_serial_id = vector_ids_to_persist.first().unwrap().serial_id();
        let last_serial_id = vector_ids_to_persist.last().unwrap().serial_id();

        Self::BatchIndexation {
            batch_id,
            vector_ids,
            vector_ids_to_persist,
            first_serial_id,
            last_serial_id,
            done_tx,
        }
    }

    pub(crate) fn new_version_replay_result(
        serial_id: SerialId,
        vector_id_to_persist: Option<VectorId>,
        done_tx: sync::oneshot::Sender<()>,
    ) -> Self {
        Self::VersionReplay {
            serial_id,
            vector_id_to_persist,
            done_tx,
        }
    }

    pub fn new_s3_checkpoint(
        checkpoint_state: GraphCheckpointState,
        done_tx: sync::oneshot::Sender<()>,
    ) -> Self {
        Self::S3Checkpoint {
            checkpoint_state,
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
                    "JobResult::BatchIndexation: batch-id={}, batch-size={}, range=({:?}..{:?})",
                    batch_id,
                    vector_ids.len(),
                    first_serial_id,
                    last_serial_id
                )
            }
            JobResult::VersionReplay {
                serial_id,
                vector_id_to_persist,
                ..
            } => match vector_id_to_persist {
                Some(vid) => write!(
                    f,
                    "JobResult::VersionReplay: serial-id={}, version-id={}",
                    serial_id,
                    vid.version_id()
                ),
                None => write!(
                    f,
                    "JobResult::VersionReplay: serial-id={}, remove-only",
                    serial_id
                ),
            },
            JobResult::SyncState { mismatched } => {
                write!(f, "JobResult::SyncState: mismatched={}", mismatched)
            }
            JobResult::SyncPeers => {
                write!(f, "JobResult::SyncPeers")
            }
            JobResult::S3Checkpoint {
                checkpoint_state, ..
            } => {
                write!(
                    f,
                    "JobResult::S3Checkpoint: iris-id={}, modification-id={}",
                    checkpoint_state.last_indexed_iris_id,
                    checkpoint_state.last_indexed_modification_id
                )
            }
        }
    }
}
