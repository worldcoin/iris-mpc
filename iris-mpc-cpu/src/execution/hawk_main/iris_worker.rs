use crate::{
    execution::hawk_main::{BothEyes, HAWK_MIN_DIST_ROTATIONS},
    hawkers::aby3::aby3_store::DistanceMode,
    hawkers::shared_irises::SharedIrisesRef,
    protocol::{
        ops::{
            galois_ring_pairwise_distance, non_existent_distance, pairwise_distance,
            rotation_aware_pairwise_distance, rotation_aware_pairwise_distance_rowmajor,
        },
        shared_iris::{ArcIris, GaloisRingSharedIris},
    },
    shares::RingElement,
};
use ampc_actor_utils::fast_metrics::FastHistogram;
use core_affinity::CoreId;
use crossbeam::channel::{Receiver, Sender};
use eyre::Result;
use futures::future::{try_join_all, BoxFuture};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    SerialId, VectorId, ROTATIONS,
};
use iris_mpc_store::Store;
use itertools::{izip, Itertools};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt::Debug,
    iter,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::Instant,
};
use tokio::sync::{mpsc, oneshot, RwLock as AsyncRwLock};
use tracing::info;

/// Production task size for full-rotation linear-scan dot products.
/// Full-scan dot work per pinned worker task. 256 amortizes channel and task
/// overhead without the load imbalance observed at 512 on r8g.24xlarge.
pub const DEFAULT_FULL_ROTATION_TASK_SIZE: usize = 256;

fn default_full_rotation_task_size() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_FULL_ROTATION_TASK_SIZE)
        .expect("the default full-rotation task size must be nonzero")
}

/// Defines the types of tasks that can be offloaded to an `IrisWorker`.
///
/// This enum represents the commands that can be sent to the worker pool for processing.
/// Each variant includes the necessary data for the operation and usually a `oneshot::Sender` (`rsp`)
/// to return the result to the caller.
#[derive(Debug)]
enum IrisTask {
    /// A synchronization barrier to ensure all preceding tasks in the channel are completed.
    Sync { rsp: oneshot::Sender<()> },
    /// Reallocates an `ArcIris` to NUMA-local memory.
    ///
    /// This task takes a shared iris pointer and creates a new `Arc` with the data
    /// allocated on the memory node local to the worker's CPU core. This is a key
    /// optimization for NUMA architectures, reducing memory access latency.
    Realloc {
        iris: ArcIris,
        rsp: oneshot::Sender<ArcIris>,
    },
    /// Inserts a new iris in the vector store.
    Insert { vector_id: VectorId, iris: ArcIris },
    /// Pre-allocates memory in the iris store to accommodate a number of new irises.
    Reserve { additional: usize },
    /// Computes the dot product for a list of iris pairs.
    DotProductPairs {
        pairs: Vec<(ArcIris, VectorId)>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Computes the dot product between a single query iris and a batch of database irises.
    DotProductBatch {
        query: ArcIris,
        vector_ids: Vec<VectorId>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Dot products against transient targets loaded from cold storage.
    DotProductIrisesBatch {
        query: ArcIris,
        targets: Arc<[ArcIris]>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Computes the rotation-aware dot product between a query and a batch of database irises.
    RotationAwareDotProductBatch {
        query: ArcIris,
        vector_ids: Arc<[VectorId]>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Rotation-aware dot products against transient cold-storage targets.
    RotationAwareDotProductIrisesBatch {
        query: ArcIris,
        targets: Arc<[ArcIris]>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Computes all 31 rotations in one pass over a target range. This is
    /// reserved for exact linear scans; HNSW intentionally uses three
    /// independent 11-rotation windows for candidate recall.
    FullRotationDotProductBatch {
        query: ArcIris,
        vector_ids: Arc<[VectorId]>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Full 31-rotation dot products against transient cold-storage targets.
    FullRotationDotProductIrisesBatch {
        query: ArcIris,
        targets: Arc<[ArcIris]>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Computes the pairwise distance for pairs of irises in the Galois Ring.
    RingPairwiseDistance {
        input: Vec<Option<(ArcIris, ArcIris)>>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Computes the rotation-aware pairwise distance for a single pair of irises.
    RotationAwarePairwiseDistance {
        pair: (ArcIris, ArcIris),
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
}

/// A handle to a pool of `IrisWorker` threads.
///
/// This struct provides an interface to a pool of background workers that are responsible
/// for CPU-intensive computations and NUMA-aware data management.
///
/// # NUMA Awareness
/// When NUMA is enabled, each worker is pinned to a specific CPU core, and tasks like
/// `numa_realloc` ensure that data is moved to memory local to that core before processing.
/// This minimizes memory latency and is important for performance on multi-socket servers.
///
/// # Task Distribution
/// Tasks are distributed among the workers to parallelize work. For read-only tasks
/// (like dot products), a round-robin strategy is used. For tasks that mutate the
/// underlying iris store (like `insert`), a consistent worker is chosen based on the
/// `VectorId` to ensure data consistency without requiring locks.
#[derive(Clone, Debug)]
pub struct IrisPoolHandle {
    /// Senders for each worker thread's task channel.
    workers: Arc<[Sender<IrisTask>]>,
    /// A counter used for round-robin task distribution.
    next_counter: Arc<AtomicU64>,
    /// Latency metric for dot_product_batch (used with Simple distance).
    metric_dot_product_batch_latency: FastHistogram,
    /// Latency metric for rotation_aware_dot_product_batch (used with MinRotation distance).
    metric_rotation_aware_dot_product_latency: FastHistogram,
}

impl IrisPoolHandle {
    pub fn numa_realloc(&self, iris: ArcIris) -> Result<oneshot::Receiver<ArcIris>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::Realloc { iris, rsp: tx };
        self.get_next_worker().send(task)?;
        Ok(rx)
    }

    pub async fn wait_completion(&self) -> Result<()> {
        try_join_all(self.workers.iter().map(|w| {
            let (rsp, rx) = oneshot::channel();
            w.send(IrisTask::Sync { rsp }).unwrap();
            rx
        }))
        .await?;
        Ok(())
    }

    pub fn insert(&self, vector_id: VectorId, iris: ArcIris) -> Result<()> {
        let task = IrisTask::Insert { vector_id, iris };
        self.get_mut_worker().send(task)?;
        Ok(())
    }

    pub fn reserve(&self, additional: usize) -> Result<()> {
        let task = IrisTask::Reserve { additional };
        self.get_mut_worker().send(task)?;
        Ok(())
    }

    pub async fn dot_product_pairs(
        &self,
        pairs: Vec<(ArcIris, VectorId)>,
    ) -> Result<Vec<RingElement<u16>>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::DotProductPairs { pairs, rsp: tx };
        self.submit(task, rx).await
    }

    pub async fn dot_product_batch(
        &mut self,
        query: ArcIris,
        vector_ids: Vec<VectorId>,
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::DotProductBatch {
            query,
            vector_ids,
            rsp: tx,
        };
        let result = self.submit(task, rx).await;
        self.metric_dot_product_batch_latency
            .record(start.elapsed().as_secs_f64());
        result
    }

    async fn dot_product_irises_batch(
        &self,
        query: ArcIris,
        targets: Vec<ArcIris>,
    ) -> Result<Vec<RingElement<u16>>> {
        let targets: Arc<[ArcIris]> = Arc::from(targets);
        let mut responses = Vec::with_capacity(Self::n_batch_chunks(targets.len()));
        for (i, _) in targets.chunks(Self::ROT_AWARE_BATCH_CHUNK_SIZE).enumerate() {
            let start = i * Self::ROT_AWARE_BATCH_CHUNK_SIZE;
            let end = (start + Self::ROT_AWARE_BATCH_CHUNK_SIZE).min(targets.len());
            let (tx, rx) = oneshot::channel();
            self.get_next_worker()
                .send(IrisTask::DotProductIrisesBatch {
                    query: query.clone(),
                    targets: targets.clone(),
                    range: start..end,
                    rsp: tx,
                })?;
            responses.push(rx);
        }
        Ok(try_join_all(responses)
            .await?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Maximum size of batches for rotation aware dot product batch tasks.
    const ROT_AWARE_BATCH_CHUNK_SIZE: usize = 128;

    /// Number of chunks a batch is split into for rotation aware dot product
    /// batch evaluation.
    #[inline(always)]
    fn n_batch_chunks(batch_len: usize) -> usize {
        batch_len.div_ceil(Self::ROT_AWARE_BATCH_CHUNK_SIZE)
    }

    /// Dispatch a batch of rotation aware dot product evaluations, splitting
    /// into tasks over chunks of maximum size `ROT_AWARE_BATCH_CHUNK_SIZE`.
    ///
    /// Response channels are appended to `responses` for caller to await.
    #[inline(always)]
    fn dispatch_rotation_dot_product_batch(
        &self,
        query: ArcIris,
        vector_ids: &[VectorId],
        responses: &mut Vec<oneshot::Receiver<Vec<RingElement<u16>>>>,
    ) -> Result<()> {
        let shared_ids: Arc<[VectorId]> = Arc::from(vector_ids);
        for (i, _) in shared_ids
            .chunks(Self::ROT_AWARE_BATCH_CHUNK_SIZE)
            .enumerate()
        {
            let start = i * Self::ROT_AWARE_BATCH_CHUNK_SIZE;
            let end = (start + Self::ROT_AWARE_BATCH_CHUNK_SIZE).min(shared_ids.len());
            let (tx, rx) = oneshot::channel();
            let task = IrisTask::RotationAwareDotProductBatch {
                query: query.clone(),
                vector_ids: shared_ids.clone(),
                range: start..end,
                rsp: tx,
            };
            self.get_next_worker().send(task)?;
            responses.push(rx);
        }

        Ok(())
    }

    pub async fn rotation_aware_dot_product_pairs(
        &self,
        pairs: Vec<(ArcIris, VectorId)>,
    ) -> Result<Vec<RingElement<u16>>> {
        let mut responses = Vec::with_capacity(pairs.len());
        for (query, id) in pairs {
            self.dispatch_rotation_dot_product_batch(query, &[id], &mut responses)?;
        }

        let results = futures::future::try_join_all(responses).await?;
        let results = results.into_iter().flatten().collect();

        Ok(results)
    }

    pub async fn rotation_aware_dot_product_batch(
        &mut self,
        query: ArcIris,
        vector_ids: &[VectorId],
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();

        let mut responses = Vec::with_capacity(Self::n_batch_chunks(vector_ids.len()));
        self.dispatch_rotation_dot_product_batch(query, vector_ids, &mut responses)?;

        let results = futures::future::try_join_all(responses).await?;
        let results = results.into_iter().flatten().collect();

        self.metric_rotation_aware_dot_product_latency
            .record(start.elapsed().as_secs_f64());
        Ok(results)
    }

    async fn rotation_aware_dot_product_irises_batch(
        &self,
        query: ArcIris,
        targets: Vec<ArcIris>,
    ) -> Result<Vec<RingElement<u16>>> {
        let targets: Arc<[ArcIris]> = Arc::from(targets);
        let mut responses = Vec::with_capacity(Self::n_batch_chunks(targets.len()));
        for (i, _) in targets.chunks(Self::ROT_AWARE_BATCH_CHUNK_SIZE).enumerate() {
            let start = i * Self::ROT_AWARE_BATCH_CHUNK_SIZE;
            let end = (start + Self::ROT_AWARE_BATCH_CHUNK_SIZE).min(targets.len());
            let (tx, rx) = oneshot::channel();
            self.get_next_worker()
                .send(IrisTask::RotationAwareDotProductIrisesBatch {
                    query: query.clone(),
                    targets: targets.clone(),
                    range: start..end,
                    rsp: tx,
                })?;
            responses.push(rx);
        }
        Ok(try_join_all(responses)
            .await?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Compute every iris rotation while traversing each target exactly once.
    /// Production passes 128 records per task; benchmarks can explicitly vary
    /// the task size without changing the dot-product computation.
    pub async fn full_rotation_dot_product_batch(
        &mut self,
        query: ArcIris,
        vector_ids: &[VectorId],
        task_size: NonZeroUsize,
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();
        let shared_ids: Arc<[VectorId]> = Arc::from(vector_ids);
        let task_size = task_size.get();
        let mut responses = Vec::with_capacity(shared_ids.len().div_ceil(task_size));

        for (i, _) in shared_ids.chunks(task_size).enumerate() {
            let range_start = i * task_size;
            let range_end = (range_start + task_size).min(shared_ids.len());
            let (tx, rx) = oneshot::channel();
            self.get_next_worker()
                .send(IrisTask::FullRotationDotProductBatch {
                    query: query.clone(),
                    vector_ids: shared_ids.clone(),
                    range: range_start..range_end,
                    rsp: tx,
                })?;
            responses.push(rx);
        }

        let results = futures::future::try_join_all(responses)
            .await?
            .into_iter()
            .flatten()
            .collect();
        self.metric_rotation_aware_dot_product_latency
            .record(start.elapsed().as_secs_f64());
        Ok(results)
    }

    async fn full_rotation_dot_product_irises_batch(
        &self,
        query: ArcIris,
        targets: Vec<ArcIris>,
        task_size: NonZeroUsize,
    ) -> Result<Vec<RingElement<u16>>> {
        let targets: Arc<[ArcIris]> = Arc::from(targets);
        let task_size = task_size.get();
        let mut responses = Vec::with_capacity(targets.len().div_ceil(task_size));
        for (i, _) in targets.chunks(task_size).enumerate() {
            let range_start = i * task_size;
            let range_end = (range_start + task_size).min(targets.len());
            let (tx, rx) = oneshot::channel();
            self.get_next_worker()
                .send(IrisTask::FullRotationDotProductIrisesBatch {
                    query: query.clone(),
                    targets: targets.clone(),
                    range: range_start..range_end,
                    rsp: tx,
                })?;
            responses.push(rx);
        }
        Ok(try_join_all(responses)
            .await?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Computes rotation-aware dot products for multiple (query, vectors) batches.
    ///
    /// Each query's prerotation is reused across all its target vectors, making this
    /// more efficient than `rotation_aware_dot_product_pairs` when the same query
    /// is compared against multiple vectors.
    ///
    /// Returns results grouped by input batch.
    pub async fn rotation_aware_dot_product_multibatch(
        &self,
        batches: Vec<(ArcIris, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<RingElement<u16>>>> {
        // Track batch index for each chunk to enable reassembly
        let chunk_batch_indices = batches
            .iter()
            .enumerate()
            .flat_map(|(batch_idx, (_, vids))| vec![batch_idx; Self::n_batch_chunks(vids.len())])
            .collect_vec();
        let n_chunks = chunk_batch_indices.len();

        // Preallocate vectors for results
        let mut results = batches
            .iter()
            .map(|(_, vids)| Vec::with_capacity(2 * HAWK_MIN_DIST_ROTATIONS * vids.len()))
            .collect_vec();

        // Dispatch dot product batches
        let mut responses = Vec::with_capacity(n_chunks);
        for (query, ref vector_ids) in batches {
            self.dispatch_rotation_dot_product_batch(query, vector_ids, &mut responses)?;
        }

        // Reassemble results by batch
        let chunk_results = futures::future::try_join_all(responses).await?;
        for (batch_idx, chunk_result) in izip!(chunk_batch_indices, chunk_results) {
            results[batch_idx].extend(chunk_result);
        }

        Ok(results)
    }

    pub async fn bench_batch_dot(
        &self,
        per_worker: usize,
        query: ArcIris,
        vector_ids: &[VectorId],
    ) -> Result<Vec<RingElement<u16>>> {
        let shared_ids: Arc<[VectorId]> = Arc::from(vector_ids);
        let mut responses = Vec::with_capacity(shared_ids.len().div_ceil(per_worker));
        // Does not call `dispatch_rotation_dot_product_batch` because chunking
        // is controlled dynamically.
        for (i, _) in shared_ids.chunks(per_worker).enumerate() {
            let start = i * per_worker;
            let end = (start + per_worker).min(shared_ids.len());
            let (tx, rx) = oneshot::channel();
            let task = IrisTask::RotationAwareDotProductBatch {
                query: query.clone(),
                vector_ids: shared_ids.clone(),
                range: start..end,
                rsp: tx,
            };
            self.get_next_worker().send(task)?;
            responses.push(rx);
        }

        let r = futures::future::try_join_all(responses).await?;
        let flattened = r.into_iter().flatten().collect();

        Ok(flattened)
    }

    pub async fn galois_ring_pairwise_distances(
        &self,
        input: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::RingPairwiseDistance { input, rsp: tx };
        self.submit(task, rx).await
    }

    pub async fn rotation_aware_pairwise_distances(
        &self,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let mut responses = Vec::with_capacity(pairs.len());
        for pair in pairs {
            let (tx, rx) = oneshot::channel();
            responses.push(rx);

            match pair {
                None => {
                    let _ = tx.send(non_existent_distance());
                }
                Some(pair) => {
                    let task = IrisTask::RotationAwarePairwiseDistance { pair, rsp: tx };
                    self.get_next_worker().send(task)?;
                }
            }
        }
        let results = futures::future::try_join_all(responses).await?;
        let results = results.into_iter().flatten().collect();
        Ok(results)
    }

    async fn submit(
        &self,
        task: IrisTask,
        rx: oneshot::Receiver<Vec<RingElement<u16>>>,
    ) -> Result<Vec<RingElement<u16>>> {
        self.get_next_worker().send(task)?;
        Ok(rx.await?)
    }

    fn get_next_worker(&self) -> &Sender<IrisTask> {
        // fetch_add() wraps around on overflow
        let idx = self.next_counter.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = idx % self.workers.len();
        &self.workers[idx]
    }

    /// Get the worker responsible for store mutations.
    fn get_mut_worker(&self) -> &Sender<IrisTask> {
        &self.workers[0]
    }
}

pub fn init_workers(
    shard_index: usize,
    iris_store: SharedIrisesRef<ArcIris>,
    numa: bool,
) -> IrisPoolHandle {
    let core_ids = select_core_ids(shard_index);
    info!(
        "Dot product shard {} running on {} cores ({:?})",
        shard_index,
        core_ids.len(),
        core_ids
    );

    let mut channels = vec![];
    for core_id in core_ids {
        let (tx, rx) = crossbeam::channel::unbounded::<IrisTask>();
        channels.push(tx);
        let iris_store = iris_store.clone();
        std::thread::spawn(move || {
            let _ = core_affinity::set_for_current(core_id);
            worker_thread(rx, iris_store, numa);
        });
    }

    IrisPoolHandle {
        workers: channels.into(),
        next_counter: Arc::new(AtomicU64::new(0)),
        metric_dot_product_batch_latency: FastHistogram::new(
            "iris_worker.dot_product_batch_latency",
        ),
        metric_rotation_aware_dot_product_latency: FastHistogram::new(
            "iris_worker.rotation_aware_dot_product_latency",
        ),
    }
}

fn worker_thread(ch: Receiver<IrisTask>, iris_store: SharedIrisesRef<ArcIris>, numa: bool) {
    while let Ok(task) = ch.recv() {
        match task {
            IrisTask::Realloc { iris, rsp } => {
                // Re-allocate from this thread.
                // This attempts to use the NUMA-aware first-touch policy of the OS.
                let new_iris = if numa {
                    Arc::new((*iris).clone())
                } else {
                    iris
                };
                let _ = rsp.send(new_iris);
            }

            IrisTask::Sync { rsp } => {
                let _ = rsp.send(());
            }

            IrisTask::Insert { vector_id, iris } => {
                let iris = if numa {
                    Arc::new((*iris).clone())
                } else {
                    iris
                };

                let mut store = iris_store.data.blocking_write();
                store.insert(vector_id, iris);
            }

            IrisTask::Reserve { additional } => {
                let mut store = iris_store.data.blocking_write();
                store.reserve(additional);
            }

            IrisTask::DotProductPairs { pairs, rsp } => {
                let store = iris_store.data.blocking_read();

                let iris_pairs = pairs
                    .iter()
                    .map(|(q, vid)| store.get_vector(vid).map(|iris| (q, iris)));

                let r = pairwise_distance(iris_pairs);
                let _ = rsp.send(r);
            }

            IrisTask::DotProductBatch {
                query,
                vector_ids,
                rsp,
            } => {
                let store = iris_store.data.blocking_read();

                let iris_pairs = vector_ids
                    .iter()
                    .map(|v| store.get_vector(v).map(|iris| (&query, iris)));

                let r = pairwise_distance(iris_pairs);
                let _ = rsp.send(r);
            }

            IrisTask::DotProductIrisesBatch {
                query,
                targets,
                range,
                rsp,
            } => {
                let iris_pairs = targets[range].iter().map(|target| Some((&query, target)));
                let _ = rsp.send(pairwise_distance(iris_pairs));
            }

            IrisTask::RotationAwareDotProductBatch {
                query,
                vector_ids,
                range,
                rsp,
            } => {
                let store = iris_store.data.blocking_read();
                let targets = vector_ids[range].iter().map(|v| store.get_vector(v));
                let result = rotation_aware_pairwise_distance_rowmajor::<HAWK_MIN_DIST_ROTATIONS, _>(
                    &query, targets,
                );
                let _ = rsp.send(result);
            }

            IrisTask::RotationAwareDotProductIrisesBatch {
                query,
                targets,
                range,
                rsp,
            } => {
                let targets = targets[range].iter().map(Some);
                let result = rotation_aware_pairwise_distance_rowmajor::<HAWK_MIN_DIST_ROTATIONS, _>(
                    &query, targets,
                );
                let _ = rsp.send(result);
            }

            IrisTask::FullRotationDotProductBatch {
                query,
                vector_ids,
                range,
                rsp,
            } => {
                let store = iris_store.data.blocking_read();
                let targets = vector_ids[range].iter().map(|v| store.get_vector(v));
                let result =
                    rotation_aware_pairwise_distance_rowmajor::<ROTATIONS, _>(&query, targets);
                let _ = rsp.send(result);
            }

            IrisTask::FullRotationDotProductIrisesBatch {
                query,
                targets,
                range,
                rsp,
            } => {
                let targets = targets[range].iter().map(Some);
                let result =
                    rotation_aware_pairwise_distance_rowmajor::<ROTATIONS, _>(&query, targets);
                let _ = rsp.send(result);
            }

            IrisTask::RingPairwiseDistance { input, rsp } => {
                let r = galois_ring_pairwise_distance(input);
                let _ = rsp.send(r);
            }

            IrisTask::RotationAwarePairwiseDistance { pair, rsp } => {
                let r = rotation_aware_pairwise_distance::<HAWK_MIN_DIST_ROTATIONS, _>(
                    &pair.0,
                    iter::once(Some(&pair.1)),
                );
                let _ = rsp.send(r);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IrisWorkerPool trait — abstracts over local/remote worker implementations
// ---------------------------------------------------------------------------

/// Unique identifier for a cached query in the worker pool.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct QueryId(pub u64);

static QUERY_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl QueryId {
    pub fn new() -> Self {
        Self(QUERY_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Identifies a specific preprocessed variant of a cached query.
///
/// Each cached iris produces 31 rotations × 2 orientations (normal + mirrored).
/// `QuerySpec` selects which variant to use for a given distance computation.
///
/// Also used as the `QueryRef` type in `VectorStore` (via the `Aby3Query` alias
/// in `aby3_store`).
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub struct QuerySpec {
    pub query_id: QueryId,
    /// Rotation index (0–30). Index 15 is the identity (center).
    pub rotation: usize,
    /// If true, use the mirrored-then-preprocessed variant.
    pub mirrored: bool,
}

impl QuerySpec {
    /// Create a query handle for the identity rotation, non-mirrored.
    pub fn new(query_id: QueryId) -> Self {
        Self {
            query_id,
            rotation: CENTER_ROTATION,
            mirrored: false,
        }
    }

    /// Create a query handle with explicit rotation and mirror flag.
    pub fn with_rotation(query_id: QueryId, rotation: usize, mirrored: bool) -> Self {
        Self {
            query_id,
            rotation,
            mirrored,
        }
    }
}

/// Rotation index for the identity (no rotation). This is index 15 in the
/// 31-element output of `all_rotations()`.
pub const CENTER_ROTATION: usize = 15;

/// Trait abstracting over iris worker pool implementations.
///
/// Mirrors the design from the "Iris Db Sharding and Memory Optimization"
/// document. The worker owns all iris data — callers interact through
/// opaque `QueryId` handles.
///
/// - **cache_queries**: push raw irises; worker mirrors, preprocesses, rotates
/// - **compute_dot_products**: distance computation using cached preprocessed rotations
/// - **fetch_irises**: pull iris data from the worker's store
/// - **insert_irises**: persist cached iris into the worker's store
/// - **evict_queries**: free cached query data
pub trait IrisWorkerPool: Debug + Send + Sync {
    /// Cache query irises for subsequent computation.
    ///
    /// The worker performs the full preprocessing pipeline on each iris:
    /// mirror, preprocess (Lagrange interpolation), and generate all 31
    /// rotations for both normal and mirrored variants.
    ///
    /// Caching an already-cached `QueryId` is a no-op.
    ///
    // TODO: Accept a rotation/mirror mask so callers can request only the
    // variants they need. Currently every call generates all 31 rotations ×
    // 2 orientations (62 variants), but:
    //   - Hawk main only uses HAWK_BASE_ROTATIONS_MASK (3 rotations) × 2
    //     orientations → 6 out of 62 used
    //   - Genesis and compaction only use CENTER_ROTATION, no mirror → 1 out
    //     of 62 used
    // A signature like `cache_queries(queries, rotation_mask: u32, mirror: bool)`
    // would let LocalIrisWorkerPool skip generating + NUMA-reallocating unused
    // variants.
    fn cache_queries<'a>(&'a self, queries: Vec<(QueryId, ArcIris)>) -> BoxFuture<'a, Result<()>>;

    /// Compute dot products for batches of (query_spec, targets).
    ///
    /// Each `QuerySpec` selects a specific preprocessed rotation from the
    /// cache. Returns one `Vec<RingElement<u16>>` per batch.
    fn compute_dot_products<'a>(
        &'a self,
        batches: Vec<(QuerySpec, Vec<VectorId>)>,
    ) -> BoxFuture<'a, Result<Vec<Vec<RingElement<u16>>>>>;

    /// Compute the 31 rotation dot products for one query in a single target
    /// traversal. Exact linear scan uses this instead of HNSW's 3 x 11
    /// rotation-window task decomposition.
    fn compute_dot_products_full_rotations<'a>(
        &'a self,
        query: QuerySpec,
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>>;

    /// Fetch iris data from the worker's store by vector ID.
    ///
    /// Returns one `ArcIris` per input ID in the same order. Database-backed
    /// workers return an error when an exact `(serial_id, version_id)` row is
    /// missing; substituting an empty sharing would reconstruct to zero
    /// distance and fail open under the threshold convention.
    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>>;

    /// Hint that these records will soon be fetched. Database-backed workers
    /// enqueue the read without blocking the caller; resident workers do
    /// nothing. Any asynchronous fetch failure is reported by
    /// [`IrisWorkerPool::wait_for_prefetch`].
    fn prefetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>>;

    /// Wait until all prefetch hints accepted before this call have completed,
    /// returning an error if any of those reads failed.
    fn wait_for_prefetch<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    /// Insert a cached iris into the worker's persistent store.
    ///
    /// The worker looks up the original (un-rotated) iris from the cache
    /// by `QueryId` and inserts it at the given `VectorId`.
    ///
    /// Returns the store checksum after all inserts are applied.
    fn insert_irises<'a>(&'a self, inserts: Vec<(QueryId, VectorId)>)
        -> BoxFuture<'a, Result<u64>>;

    /// Compute pairwise distances between pairs of cached queries.
    ///
    /// Used for intra-batch matching where both irises are cached queries
    /// (not stored vectors). Each `None` pair produces a max-distance sentinel.
    ///
    /// Convention: the first `QuerySpec` selects a **preprocessed** rotation,
    /// the second `QueryId` selects the **raw (original)** iris (only the
    /// query identity matters — rotation/mirrored are not applicable for the
    /// raw operand). This matches `trick_dot` (one preprocessed, one raw).
    fn compute_pairwise_distances<'a>(
        &'a self,
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>>;

    /// Evict cached queries, freeing memory.
    fn evict_queries<'a>(&'a self, query_ids: Vec<QueryId>) -> BoxFuture<'a, Result<()>>;

    /// Delete irises by replacing them with party-specific dummy sentinels
    /// that produce max-distance in dot products. The party_id is a config
    /// field on the implementer.
    fn delete_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>>;

    /// Acknowledge database commits for mutations previously applied to the
    /// worker. Resident pools need no acknowledgement; a database-backed cold
    /// pool uses it to discard its write-behind overlay safely.
    fn acknowledge_persisted<'a>(&'a self, serial_ids: Vec<SerialId>) -> BoxFuture<'a, Result<()>>;
}

/// Blanket impl so any `Arc<T: IrisWorkerPool>` (including
/// `Arc<dyn IrisWorkerPool>`) can be passed wherever an `impl IrisWorkerPool`
/// or `&dyn IrisWorkerPool` is expected.
impl<T: ?Sized + IrisWorkerPool> IrisWorkerPool for Arc<T> {
    fn cache_queries<'a>(&'a self, queries: Vec<(QueryId, ArcIris)>) -> BoxFuture<'a, Result<()>> {
        (**self).cache_queries(queries)
    }
    fn compute_dot_products<'a>(
        &'a self,
        batches: Vec<(QuerySpec, Vec<VectorId>)>,
    ) -> BoxFuture<'a, Result<Vec<Vec<RingElement<u16>>>>> {
        (**self).compute_dot_products(batches)
    }
    fn compute_dot_products_full_rotations<'a>(
        &'a self,
        query: QuerySpec,
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>> {
        (**self).compute_dot_products_full_rotations(query, vector_ids)
    }
    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>> {
        (**self).fetch_irises(ids)
    }
    fn prefetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>> {
        (**self).prefetch_irises(ids)
    }
    fn wait_for_prefetch<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        (**self).wait_for_prefetch()
    }
    fn insert_irises<'a>(
        &'a self,
        inserts: Vec<(QueryId, VectorId)>,
    ) -> BoxFuture<'a, Result<u64>> {
        (**self).insert_irises(inserts)
    }
    fn compute_pairwise_distances<'a>(
        &'a self,
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>> {
        (**self).compute_pairwise_distances(pairs)
    }
    fn evict_queries<'a>(&'a self, query_ids: Vec<QueryId>) -> BoxFuture<'a, Result<()>> {
        (**self).evict_queries(query_ids)
    }
    fn delete_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>> {
        (**self).delete_irises(ids)
    }
    fn acknowledge_persisted<'a>(&'a self, serial_ids: Vec<SerialId>) -> BoxFuture<'a, Result<()>> {
        (**self).acknowledge_persisted(serial_ids)
    }
}

/// Cloneable bridge from the asynchronous result-persistence task back to the
/// worker pools. Each committed mutation releases one queued cold-eye value.
#[derive(Clone)]
pub struct IrisPersistenceAckHandle {
    pools: BothEyes<Arc<dyn IrisWorkerPool>>,
}

impl IrisPersistenceAckHandle {
    pub(crate) fn new(pools: BothEyes<Arc<dyn IrisWorkerPool>>) -> Self {
        Self { pools }
    }

    pub async fn acknowledge_persisted(&self, serial_ids: Vec<SerialId>) -> Result<()> {
        futures::try_join!(
            self.pools[0].acknowledge_persisted(serial_ids.clone()),
            self.pools[1].acknowledge_persisted(serial_ids),
        )?;
        Ok(())
    }
}

/// Cache a single iris and return a query handle (center rotation, non-mirrored).
/// Helper used by tests, benches, and example bins — production code paths
/// manage the cache lifecycle explicitly via `cache_queries` / `evict_queries`.
pub async fn cache_iris(pool: &dyn IrisWorkerPool, iris: ArcIris) -> Result<QuerySpec> {
    let qid = QueryId::new();
    pool.cache_queries(vec![(qid, iris)]).await?;
    Ok(QuerySpec::new(qid))
}

/// Cache multiple irises and return query handles in input order.
pub async fn cache_irises(
    pool: &dyn IrisWorkerPool,
    irises: Vec<ArcIris>,
) -> Result<Vec<QuerySpec>> {
    let pairs: Vec<_> = irises
        .into_iter()
        .map(|iris| (QueryId::new(), iris))
        .collect();
    let specs: Vec<_> = pairs.iter().map(|(qid, _)| QuerySpec::new(*qid)).collect();
    pool.cache_queries(pairs).await?;
    Ok(specs)
}

// ---------------------------------------------------------------------------
// LocalIrisWorkerPool — wraps IrisPoolHandle + query cache
// ---------------------------------------------------------------------------

/// Cached preprocessing results for a single base iris.
struct CachedQuery {
    /// The original (un-rotated, un-preprocessed) iris, for `insert_irises`.
    original: ArcIris,
    /// `all_rotations(preprocess(original))` — 31 entries.
    preprocessed_rotations: Vec<ArcIris>,
    /// `all_rotations(preprocess(mirror(original)))` — 31 entries.
    mirrored_preprocessed_rotations: Vec<ArcIris>,
}

/// Local implementation of `IrisWorkerPool` that wraps `IrisPoolHandle` with
/// a query cache. The cache holds the full preprocessing output (rotations of
/// both normal and mirrored preprocessed variants).
#[derive(Clone)]
pub struct LocalIrisWorkerPool {
    inner: IrisPoolHandle,
    query_cache: Arc<RwLock<HashMap<QueryId, CachedQuery>>>,
    iris_store: SharedIrisesRef<ArcIris>,
    mode: DistanceMode,
    party_id: usize,
    /// Number of records assigned to one full-rotation dot-product task.
    /// Normal constructors retain the production default; benchmarks can opt
    /// into another nonzero value through `with_full_rotation_task_size`.
    full_rotation_task_size: NonZeroUsize,
    /// When set, the complete iris column stays in Postgres. RAM holds only
    /// the rolling LUC window and mutations awaiting a database commit; older
    /// explicit candidates are fetched sparsely.
    cold_storage: Option<ColdStorage>,
}

#[derive(Clone)]
struct ColdStorage {
    store: Store,
    side: usize,
    state: Arc<AsyncRwLock<ColdStorageState>>,
    prefetch_tx: mpsc::Sender<ColdPrefetchCommand>,
}

#[derive(Default)]
struct ColdStorageState {
    luc_cache: RollingLucCache,
    pending_mutations: PendingMutations,
    prefetched: HashMap<VectorId, PrefetchEntry>,
}

/// One full scan chunk is the largest amount of cold data retained solely for
/// lookahead. At 38,400 bytes per eye this caps the optimization near 150 MiB,
/// while ordinary sparse matches occupy far less.
const COLD_PREFETCH_MAX_RECORDS: usize = 1 << 12;
const COLD_PREFETCH_QUEUE_DEPTH: usize = 64;

enum ColdPrefetchCommand {
    Fetch(Vec<VectorId>),
    Barrier(oneshot::Sender<Result<()>>),
}

enum PrefetchEntry {
    Loading {
        remaining_uses: usize,
    },
    Ready {
        iris: ArcIris,
        remaining_uses: usize,
    },
}

impl PrefetchEntry {
    fn add_use(&mut self) {
        let remaining_uses = match self {
            Self::Loading { remaining_uses } | Self::Ready { remaining_uses, .. } => remaining_uses,
        };
        *remaining_uses = remaining_uses.saturating_add(1);
    }
}

impl ColdStorageState {
    fn cached_iris(&self, id: &VectorId) -> Option<ArcIris> {
        self.pending_mutations
            .get(id)
            .or_else(|| self.luc_cache.get(id))
    }

    /// Take one promised use of a prefetched value. Loading entries are also
    /// decremented so an on-demand fallback cannot leak a reservation.
    fn take_prefetched(&mut self, id: &VectorId) -> Option<ArcIris> {
        let (iris, remove) = match self.prefetched.get_mut(id) {
            Some(PrefetchEntry::Ready {
                iris,
                remaining_uses,
            }) => {
                let iris = iris.clone();
                *remaining_uses = remaining_uses.saturating_sub(1);
                (Some(iris), *remaining_uses == 0)
            }
            Some(PrefetchEntry::Loading { remaining_uses }) => {
                *remaining_uses = remaining_uses.saturating_sub(1);
                (None, *remaining_uses == 0)
            }
            None => (None, false),
        };
        if remove {
            self.prefetched.remove(id);
        }
        iris
    }
}

/// The current LUC lookback window for the non-resident eye. Entries are
/// indexed by serial ID so an update replaces the previous vector version.
#[derive(Default)]
struct RollingLucCache {
    capacity: usize,
    latest_serial_id: SerialId,
    entries: BTreeMap<SerialId, (VectorId, ArcIris)>,
}

impl RollingLucCache {
    fn new(
        capacity: usize,
        latest_serial_id: SerialId,
        entries: impl IntoIterator<Item = (VectorId, ArcIris)>,
    ) -> Self {
        let mut cache = Self {
            capacity,
            latest_serial_id,
            entries: BTreeMap::new(),
        };
        for (id, iris) in entries {
            if cache.contains_serial(id.serial_id()) {
                cache.entries.insert(id.serial_id(), (id, iris));
            }
        }
        cache
    }

    fn first_serial_id(&self) -> SerialId {
        let width = u32::try_from(self.capacity)
            .unwrap_or(u32::MAX)
            .saturating_sub(1);
        self.latest_serial_id.saturating_sub(width).max(1)
    }

    fn contains_serial(&self, serial_id: SerialId) -> bool {
        self.capacity > 0
            && serial_id >= self.first_serial_id()
            && serial_id <= self.latest_serial_id
    }

    fn get(&self, id: &VectorId) -> Option<ArcIris> {
        self.entries
            .get(&id.serial_id())
            .filter(|(cached_id, _)| cached_id == id)
            .map(|(_, iris)| iris.clone())
    }

    fn apply_mutation(&mut self, id: VectorId, iris: ArcIris) {
        if self.capacity == 0 {
            return;
        }
        if id.serial_id() > self.latest_serial_id {
            self.latest_serial_id = id.serial_id();
            let first = self.first_serial_id();
            self.entries.retain(|serial_id, _| *serial_id >= first);
        }
        if self.contains_serial(id.serial_id()) {
            self.entries.insert(id.serial_id(), (id, iris));
        }
    }
}

/// FIFO per serial ID: persistence runs in batch order, while a later batch
/// may already have updated the same identity before the older commit is
/// acknowledged. Popping only the oldest value preserves that newer write.
#[derive(Default)]
struct PendingMutations {
    entries: HashMap<SerialId, VecDeque<(VectorId, ArcIris)>>,
}

/// Startup parameters for a database-backed non-resident eye.
pub struct ColdStorageInit {
    pub store: Store,
    pub side: usize,
    /// Exact current vector IDs in the configured LUC lookback window.
    pub luc_window_ids: Vec<VectorId>,
    /// Highest allocated serial ID, including an empty database (`0`).
    pub latest_serial_id: SerialId,
    /// Number of serial positions retained. This intentionally includes the
    /// extra position used by `HawkRequest::luc_ids`.
    pub luc_window_capacity: usize,
}

impl PendingMutations {
    fn get(&self, id: &VectorId) -> Option<ArcIris> {
        self.entries
            .get(&id.serial_id())
            .and_then(|values| values.back())
            .filter(|(pending_id, _)| pending_id == id)
            .map(|(_, iris)| iris.clone())
    }

    fn push(&mut self, id: VectorId, iris: ArcIris) {
        self.entries
            .entry(id.serial_id())
            .or_default()
            .push_back((id, iris));
    }

    fn acknowledge(&mut self, serial_id: SerialId) {
        let remove = if let Some(values) = self.entries.get_mut(&serial_id) {
            values.pop_front();
            values.is_empty()
        } else {
            false
        };
        if remove {
            self.entries.remove(&serial_id);
        }
    }
}

async fn run_cold_prefetch_worker(
    store: Store,
    side: usize,
    party_id: usize,
    state: Arc<AsyncRwLock<ColdStorageState>>,
    mut rx: mpsc::Receiver<ColdPrefetchCommand>,
) {
    let mut pending_error = None;
    while let Some(command) = rx.recv().await {
        match command {
            ColdPrefetchCommand::Fetch(ids) => {
                if let Err(error) = prefetch_cold_irises(&store, side, party_id, &state, ids).await
                {
                    tracing::error!(
                        ?error,
                        "Cold-eye database prefetch failed; failing the current scan"
                    );
                    metrics::counter!("linear_scan_cold_prefetch_errors_total").increment(1);
                    if pending_error.is_none() {
                        pending_error = Some(error);
                    }
                }
            }
            ColdPrefetchCommand::Barrier(done) => {
                let _ = done.send(pending_error.take().map_or(Ok(()), Err));
            }
        }
    }
}

async fn prefetch_cold_irises(
    store: &Store,
    side: usize,
    party_id: usize,
    state: &Arc<AsyncRwLock<ColdStorageState>>,
    mut ids: Vec<VectorId>,
) -> Result<()> {
    ids.sort_unstable();
    ids.dedup();

    // Reserve bounded cache slots before doing I/O. The single prefetch worker
    // serializes reservation, while foreground fetches may consume entries.
    let to_fetch = {
        let mut state = state.write().await;
        let mut to_fetch = Vec::new();
        for id in ids {
            if state.cached_iris(&id).is_some() {
                continue;
            }
            if let Some(entry) = state.prefetched.get_mut(&id) {
                entry.add_use();
            } else if state.prefetched.len() < COLD_PREFETCH_MAX_RECORDS {
                state
                    .prefetched
                    .insert(id, PrefetchEntry::Loading { remaining_uses: 1 });
                to_fetch.push(id);
            } else {
                metrics::counter!("linear_scan_cold_prefetch_capacity_skips_total").increment(1);
            }
        }
        to_fetch
    };
    if to_fetch.is_empty() {
        return Ok(());
    }

    let db_ids = to_fetch
        .iter()
        .map(|id| id.serial_id() as i64)
        .collect_vec();
    let db_start = Instant::now();
    let fetched = async {
        let rows = store.get_iris_data_by_ids_for_side(&db_ids, side).await?;
        let mut fetched = HashMap::with_capacity(rows.len());
        for row in rows {
            fetched.insert(
                row.vector_id(),
                GaloisRingSharedIris::try_from_buffers(party_id, row.code(), row.mask())?,
            );
        }
        Ok::<_, eyre::Report>(fetched)
    }
    .await;
    metrics::histogram!("linear_scan_cold_prefetch_db_duration")
        .record(db_start.elapsed().as_secs_f64());
    metrics::histogram!("linear_scan_cold_prefetch_batch_size").record(db_ids.len() as f64);

    let mut state = state.write().await;
    match fetched {
        Ok(mut fetched) => {
            let missing = to_fetch
                .iter()
                .filter(|id| !fetched.contains_key(id))
                .copied()
                .collect_vec();
            if !missing.is_empty() {
                for id in &to_fetch {
                    if matches!(
                        state.prefetched.get(id),
                        Some(PrefetchEntry::Loading { .. })
                    ) {
                        state.prefetched.remove(id);
                    }
                }
                return Err(cold_db_miss_error(party_id, side, &missing));
            }
            for id in to_fetch {
                let Some(entry) = state.prefetched.remove(&id) else {
                    // A foreground read already consumed this reservation.
                    continue;
                };
                let PrefetchEntry::Loading { remaining_uses } = entry else {
                    unreachable!("new cold prefetch reservation must still be loading");
                };
                state.prefetched.insert(
                    id,
                    PrefetchEntry::Ready {
                        iris: fetched
                            .remove(&id)
                            .expect("cold-eye missing rows were checked above"),
                        remaining_uses,
                    },
                );
            }
            metrics::counter!("linear_scan_cold_prefetched_records_total")
                .increment(db_ids.len() as u64);
            Ok(())
        }
        Err(error) => {
            for id in to_fetch {
                if matches!(
                    state.prefetched.get(&id),
                    Some(PrefetchEntry::Loading { .. })
                ) {
                    state.prefetched.remove(&id);
                }
            }
            Err(error)
        }
    }
}

fn cold_db_miss_error(party_id: usize, side: usize, missing: &[VectorId]) -> eyre::Report {
    let examples = missing.iter().take(8).copied().collect_vec();
    metrics::counter!("linear_scan_cold_db_misses_total").increment(missing.len() as u64);
    tracing::error!(
        party_id,
        side,
        missing_count = missing.len(),
        ?examples,
        "Cold-eye database did not return the exact requested vector versions"
    );
    eyre::eyre!(
        "cold-eye database missing {} exact vector ID(s) for party {party_id}, side {side}; \
         examples: {examples:?}",
        missing.len()
    )
}

impl Debug for LocalIrisWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalIrisWorkerPool")
            .field("inner", &self.inner)
            .finish()
    }
}

impl LocalIrisWorkerPool {
    pub fn new(
        inner: IrisPoolHandle,
        iris_store: SharedIrisesRef<ArcIris>,
        mode: DistanceMode,
        party_id: usize,
    ) -> Self {
        Self {
            inner,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            iris_store,
            mode,
            party_id,
            full_rotation_task_size: default_full_rotation_task_size(),
            cold_storage: None,
        }
    }

    /// Construct the non-resident eye of a linear-scan actor. Its complete
    /// iris column remains in Postgres. The current LUC window is retained in
    /// memory; older explicit candidates are fetched sparsely. A bounded FIFO
    /// overlay keeps mutations visible until asynchronous persistence commits.
    pub async fn new_cold(
        inner: IrisPoolHandle,
        iris_store: SharedIrisesRef<ArcIris>,
        mode: DistanceMode,
        party_id: usize,
        init: ColdStorageInit,
    ) -> Result<Self> {
        let ColdStorageInit {
            store,
            side,
            luc_window_ids,
            latest_serial_id,
            luc_window_capacity,
        } = init;
        let db_ids = luc_window_ids
            .iter()
            .map(|id| id.serial_id() as i64)
            .collect_vec();
        let rows = store.get_iris_data_by_ids_for_side(&db_ids, side).await?;
        let mut cached = Vec::with_capacity(rows.len());
        for row in rows {
            let vector_id = row.vector_id();
            let iris = GaloisRingSharedIris::try_from_buffers(party_id, row.code(), row.mask())?;
            cached.push((vector_id, iris));
        }
        let missing = luc_window_ids
            .iter()
            .filter(|id| !cached.iter().any(|(cached_id, _)| cached_id == *id))
            .copied()
            .collect_vec();
        if !missing.is_empty() {
            return Err(cold_db_miss_error(party_id, side, &missing));
        }
        let state = Arc::new(AsyncRwLock::new(ColdStorageState {
            luc_cache: RollingLucCache::new(luc_window_capacity, latest_serial_id, cached),
            pending_mutations: PendingMutations::default(),
            prefetched: HashMap::new(),
        }));
        let (prefetch_tx, prefetch_rx) = mpsc::channel(COLD_PREFETCH_QUEUE_DEPTH);
        tokio::spawn(run_cold_prefetch_worker(
            store.clone(),
            side,
            party_id,
            state.clone(),
            prefetch_rx,
        ));

        Ok(Self {
            inner,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            iris_store,
            mode,
            party_id,
            full_rotation_task_size: default_full_rotation_task_size(),
            cold_storage: Some(ColdStorage {
                store,
                side,
                state,
                prefetch_tx,
            }),
        })
    }

    /// Create a local worker pool for shard 0 with NUMA pinning.
    /// Standard construction for tests, benchmarks, and single-node tools.
    pub fn new_local(
        iris_store: SharedIrisesRef<ArcIris>,
        mode: DistanceMode,
        party_id: usize,
    ) -> Self {
        let pool = init_workers(0, iris_store.clone(), true);
        Self::new(pool, iris_store, mode, party_id)
    }

    /// Override full-rotation task granularity for an explicitly configured
    /// benchmark or tool. Production constructors always default to 128.
    pub fn with_full_rotation_task_size(mut self, task_size: NonZeroUsize) -> Self {
        self.full_rotation_task_size = task_size;
        self
    }

    async fn fetch_irises_resident_or_cold(&self, ids: &[VectorId]) -> Result<Vec<ArcIris>> {
        let Some(cold) = &self.cold_storage else {
            let store = self.iris_store.data.read().await;
            return Ok(ids
                .iter()
                .map(|id| store.get_vector_or_empty(id).clone())
                .collect());
        };

        // Prefer uncommitted writes, then the resident LUC window. This makes
        // a just-inserted or just-deleted iris visible before result
        // persistence finishes, without retaining committed cold history.
        let cached = {
            let mut state = cold.state.write().await;
            ids.iter()
                .map(|id| state.cached_iris(id).or_else(|| state.take_prefetched(id)))
                .collect::<Vec<_>>()
        };
        let missing_db_ids = ids
            .iter()
            .zip(&cached)
            .filter_map(|(id, value)| value.is_none().then_some(id.serial_id() as i64))
            .unique()
            .collect_vec();
        let rows = cold
            .store
            .get_iris_data_by_ids_for_side(&missing_db_ids, cold.side)
            .await?;
        let mut fetched = HashMap::with_capacity(rows.len());
        for row in rows {
            let vector_id = row.vector_id();
            let iris =
                GaloisRingSharedIris::try_from_buffers(self.party_id, row.code(), row.mask())?;
            fetched.insert(vector_id, iris);
        }

        let mut resolved = Vec::with_capacity(ids.len());
        let mut missing = Vec::new();
        for (id, value) in ids.iter().zip(cached) {
            if let Some(iris) = value.or_else(|| fetched.get(id).cloned()) {
                resolved.push(iris);
            } else {
                missing.push(*id);
            }
        }
        missing.sort_unstable();
        missing.dedup();
        if !missing.is_empty() {
            return Err(cold_db_miss_error(self.party_id, cold.side, &missing));
        }
        Ok(resolved)
    }
}

/// Build 31 `ArcIris` rotations from code and mask rotation vecs.
fn zip_rotations(
    code_rots: Vec<GaloisRingIrisCodeShare>,
    mask_rots: Vec<GaloisRingTrimmedMaskCodeShare>,
) -> Vec<ArcIris> {
    code_rots
        .into_iter()
        .zip(mask_rots)
        .map(|(code, mask)| Arc::new(GaloisRingSharedIris { code, mask }))
        .collect()
}

impl IrisWorkerPool for LocalIrisWorkerPool {
    fn cache_queries<'a>(&'a self, queries: Vec<(QueryId, ArcIris)>) -> BoxFuture<'a, Result<()>> {
        let query_cache = self.query_cache.clone();
        let inner = self.inner.clone();
        Box::pin(async move {
            let start = Instant::now();
            // Filter out already-cached queries.
            let new_queries: Vec<_> = {
                let cache = query_cache.read().unwrap();
                queries
                    .into_iter()
                    .filter(|(qid, _)| !cache.contains_key(qid))
                    .collect()
            };
            if new_queries.is_empty() {
                return Ok(());
            }

            // Preprocess + rotate, collecting all resulting ArcIris values.
            let mut entries: Vec<(QueryId, CachedQuery)> = Vec::with_capacity(new_queries.len());
            for (query_id, iris) in new_queries {
                // --- Normal: preprocess then rotate ---
                let mut code_proc = iris.code.clone();
                let mut mask_proc = iris.mask.clone();
                code_proc.preprocess_iris_code_query_share();
                mask_proc.preprocess_mask_code_query_share();
                let preprocessed_rotations =
                    zip_rotations(code_proc.all_rotations(), mask_proc.all_rotations());

                // --- Mirrored: mirror, preprocess, then rotate ---
                let mut code_mirror = iris.code.mirrored_code();
                let mut mask_mirror = iris.mask.mirrored();
                code_mirror.preprocess_iris_code_query_share();
                mask_mirror.preprocess_mask_code_query_share();
                let mirrored_preprocessed_rotations =
                    zip_rotations(code_mirror.all_rotations(), mask_mirror.all_rotations());

                entries.push((
                    query_id,
                    CachedQuery {
                        original: iris,
                        preprocessed_rotations,
                        mirrored_preprocessed_rotations,
                    },
                ));
            }

            // NUMA-realloc all irises onto the worker pool's NUMA node.
            // The query iris is the "left" operand in every trick_dot and is
            // read once per stored vector — for ef=128 that's 128 × ~38KB per
            // search step, so NUMA locality matters.
            let mut realloc_futures = Vec::new();
            for (_, entry) in &entries {
                realloc_futures.push(inner.numa_realloc(entry.original.clone()));
                for rot in &entry.preprocessed_rotations {
                    realloc_futures.push(inner.numa_realloc(rot.clone()));
                }
                for rot in &entry.mirrored_preprocessed_rotations {
                    realloc_futures.push(inner.numa_realloc(rot.clone()));
                }
            }
            let receivers: Vec<_> = realloc_futures.into_iter().collect::<Result<Vec<_>>>()?;
            let reallocated = try_join_all(receivers).await?;

            // Write NUMA-local copies back into the entries.
            let mut idx = 0;
            for (_, entry) in &mut entries {
                entry.original = reallocated[idx].clone();
                idx += 1;
                for rot in &mut entry.preprocessed_rotations {
                    *rot = reallocated[idx].clone();
                    idx += 1;
                }
                for rot in &mut entry.mirrored_preprocessed_rotations {
                    *rot = reallocated[idx].clone();
                    idx += 1;
                }
            }

            // Store in cache.
            let mut cache = query_cache.write().unwrap();
            for (query_id, entry) in entries {
                cache.entry(query_id).or_insert(entry);
            }
            metrics::histogram!("cache_queries_duration").record(start.elapsed().as_secs_f64());
            Ok(())
        })
    }

    fn compute_dot_products<'a>(
        &'a self,
        batches: Vec<(QuerySpec, Vec<VectorId>)>,
    ) -> BoxFuture<'a, Result<Vec<Vec<RingElement<u16>>>>> {
        let query_cache = self.query_cache.clone();
        let mut inner = self.inner.clone();
        let mode = self.mode;
        let pool = self.clone();
        let is_cold = self.cold_storage.is_some();
        Box::pin(async move {
            // Look up the correct preprocessed rotation for each batch
            let iris_batches: Vec<(ArcIris, Vec<VectorId>)> = {
                let cache = query_cache.read().unwrap();
                batches
                    .into_iter()
                    .map(|(spec, tids)| {
                        let cached = cache
                            .get(&spec.query_id)
                            .ok_or_else(|| eyre::eyre!("Query {:?} not cached", spec.query_id))?;
                        let rotations = if spec.mirrored {
                            &cached.mirrored_preprocessed_rotations
                        } else {
                            &cached.preprocessed_rotations
                        };
                        Ok((rotations[spec.rotation].clone(), tids))
                    })
                    .collect::<Result<Vec<_>>>()?
            };

            if is_cold {
                let mut results = Vec::with_capacity(iris_batches.len());
                for (iris_proc, target_ids) in iris_batches {
                    let targets = pool.fetch_irises_resident_or_cold(&target_ids).await?;
                    let result = match mode {
                        DistanceMode::Simple => {
                            inner.dot_product_irises_batch(iris_proc, targets).await?
                        }
                        DistanceMode::MinRotation => {
                            inner
                                .rotation_aware_dot_product_irises_batch(iris_proc, targets)
                                .await?
                        }
                    };
                    results.push(result);
                }
                return Ok(results);
            }

            match mode {
                DistanceMode::Simple => {
                    let mut results = Vec::with_capacity(iris_batches.len());
                    for (iris_proc, targets) in iris_batches {
                        let r = inner.dot_product_batch(iris_proc, targets).await?;
                        results.push(r);
                    }
                    Ok(results)
                }
                DistanceMode::MinRotation => {
                    inner
                        .rotation_aware_dot_product_multibatch(iris_batches)
                        .await
                }
            }
        })
    }

    fn compute_dot_products_full_rotations<'a>(
        &'a self,
        query: QuerySpec,
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>> {
        let query_cache = self.query_cache.clone();
        let mut inner = self.inner.clone();
        let pool = self.clone();
        let is_cold = self.cold_storage.is_some();
        let task_size = self.full_rotation_task_size;
        Box::pin(async move {
            let iris = {
                let cache = query_cache.read().unwrap();
                let cached = cache
                    .get(&query.query_id)
                    .ok_or_else(|| eyre::eyre!("Query {:?} not cached", query.query_id))?;
                let rotations = if query.mirrored {
                    &cached.mirrored_preprocessed_rotations
                } else {
                    &cached.preprocessed_rotations
                };
                rotations[query.rotation].clone()
            };
            if is_cold {
                let targets = pool.fetch_irises_resident_or_cold(&vector_ids).await?;
                inner
                    .full_rotation_dot_product_irises_batch(iris, targets, task_size)
                    .await
            } else {
                inner
                    .full_rotation_dot_product_batch(iris, &vector_ids, task_size)
                    .await
            }
        })
    }

    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>> {
        Box::pin(async move { self.fetch_irises_resident_or_cold(&ids).await })
    }

    fn prefetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>> {
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            if ids.is_empty() {
                return Ok(());
            }
            let Some(cold) = cold_storage else {
                return Ok(());
            };
            match cold.prefetch_tx.try_send(ColdPrefetchCommand::Fetch(ids)) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    metrics::counter!("linear_scan_cold_prefetch_queue_skips_total").increment(1);
                    Ok(())
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    Err(eyre::eyre!("cold-eye prefetch worker stopped"))
                }
            }
        })
    }

    fn wait_for_prefetch<'a>(&'a self) -> BoxFuture<'a, Result<()>> {
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            let Some(cold) = cold_storage else {
                return Ok(());
            };
            let (done_tx, done_rx) = oneshot::channel();
            cold.prefetch_tx
                .send(ColdPrefetchCommand::Barrier(done_tx))
                .await
                .map_err(|_| eyre::eyre!("cold-eye prefetch worker stopped"))?;
            done_rx
                .await
                .map_err(|_| eyre::eyre!("cold-eye prefetch barrier was dropped"))?
        })
    }

    fn insert_irises<'a>(
        &'a self,
        inserts: Vec<(QueryId, VectorId)>,
    ) -> BoxFuture<'a, Result<u64>> {
        let query_cache = self.query_cache.clone();
        let iris_store = self.iris_store.clone();
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            // Resolve query IDs to irises (release cache lock before await).
            let resolved: Vec<_> = {
                let cache = query_cache.read().unwrap();
                inserts
                    .into_iter()
                    .map(|(qid, vid)| {
                        let iris = cache
                            .get(&qid)
                            .ok_or_else(|| eyre::eyre!("Query {:?} not cached for insert", qid))?
                            .original
                            .clone();
                        Ok((vid, iris))
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            if let Some(cold) = cold_storage {
                let mut state = cold.state.write().await;
                for (vector_id, iris) in resolved {
                    state.pending_mutations.push(vector_id, iris.clone());
                    state.luc_cache.apply_mutation(vector_id, iris);
                }
                return Ok(0);
            }

            // Write directly to the shared store (not via IrisPoolHandle::insert
            // which is fire-and-forget). HNSW insertion needs the iris to be
            // visible in the store immediately after this returns.
            let mut store = iris_store.data.write().await;
            for (vector_id, iris) in resolved {
                store.insert(vector_id, iris);
            }
            Ok(store.set_hash.checksum())
        })
    }

    fn compute_pairwise_distances<'a>(
        &'a self,
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>> {
        let query_cache = self.query_cache.clone();
        let inner = self.inner.clone();
        let mode = self.mode;
        Box::pin(async move {
            // Resolve pairs to ArcIris pairs.
            // First = preprocessed rotation, second = raw (original) iris.
            let iris_pairs: Vec<Option<(ArcIris, ArcIris)>> = {
                let cache = query_cache.read().unwrap();
                pairs
                    .into_iter()
                    .map(|pair| -> Result<_> {
                        match pair {
                            None => Ok(None),
                            Some((a, b_id)) => {
                                let ca = cache.get(&a.query_id).ok_or_else(|| {
                                    eyre::eyre!(
                                        "Query {:?} not cached for pairwise (a)",
                                        a.query_id
                                    )
                                })?;
                                let cb = cache.get(&b_id).ok_or_else(|| {
                                    eyre::eyre!("Query {:?} not cached for pairwise (b)", b_id)
                                })?;
                                let iris_a = if a.mirrored {
                                    &ca.mirrored_preprocessed_rotations
                                } else {
                                    &ca.preprocessed_rotations
                                }[a.rotation]
                                    .clone();
                                let iris_b = cb.original.clone();
                                Ok(Some((iris_a, iris_b)))
                            }
                        }
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            match mode {
                DistanceMode::Simple => inner.galois_ring_pairwise_distances(iris_pairs).await,
                DistanceMode::MinRotation => {
                    inner.rotation_aware_pairwise_distances(iris_pairs).await
                }
            }
        })
    }

    fn evict_queries<'a>(&'a self, query_ids: Vec<QueryId>) -> BoxFuture<'a, Result<()>> {
        let query_cache = self.query_cache.clone();
        Box::pin(async move {
            let mut cache = query_cache.write().unwrap();
            for qid in query_ids {
                cache.remove(&qid);
            }
            Ok(())
        })
    }

    fn delete_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>> {
        let iris_store = self.iris_store.clone();
        let party_id = self.party_id;
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            let dummy = Arc::new(GaloisRingSharedIris::dummy_for_party(party_id));
            if let Some(cold) = cold_storage {
                let mut state = cold.state.write().await;
                for id in ids {
                    let next_id = id.next_version();
                    state.pending_mutations.push(next_id, dummy.clone());
                    state.luc_cache.apply_mutation(next_id, dummy.clone());
                }
                return Ok(());
            }
            let mut store = iris_store.data.write().await;
            for id in ids {
                store.update(id, dummy.clone());
            }
            Ok(())
        })
    }

    fn acknowledge_persisted<'a>(&'a self, serial_ids: Vec<SerialId>) -> BoxFuture<'a, Result<()>> {
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            if let Some(cold) = cold_storage {
                let mut state = cold.state.write().await;
                for serial_id in serial_ids {
                    state.pending_mutations.acknowledge(serial_id);
                }
            }
            Ok(())
        })
    }
}

pub fn select_core_ids(shard_index: usize) -> Vec<CoreId> {
    use iris_mpc_common::helpers::numactl;

    let numa_nodes = numactl::get_numa_nodes();
    let node = numa_nodes[shard_index % numa_nodes.len()];

    let cpu_ids = numactl::get_cores_for_node(node);

    assert!(
        !cpu_ids.is_empty(),
        "No CPUs available for NUMA node {}",
        node
    );

    cpu_ids.into_iter().map(|id| CoreId { id }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkers::shared_irises::SharedIrises;

    const TEST_TARGETS: usize = 513;

    #[test]
    fn rolling_luc_cache_tracks_versions_and_evicts_old_serials() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let mut cache = RollingLucCache::new(
            3,
            5,
            (3..=5).map(|serial_id| (VectorId::from_serial_id(serial_id), iris.clone())),
        );

        let updated_four = VectorId::from_serial_id(4).next_version();
        cache.apply_mutation(updated_four, iris.clone());
        assert!(cache.get(&updated_four).is_some());
        assert!(cache.get(&VectorId::from_serial_id(4)).is_none());

        cache.apply_mutation(VectorId::from_serial_id(6), iris.clone());
        assert_eq!(
            cache.entries.keys().copied().collect::<Vec<_>>(),
            vec![4, 5, 6]
        );

        cache.apply_mutation(VectorId::from_serial_id(2), iris);
        assert!(!cache.entries.contains_key(&2));
    }

    #[test]
    fn pending_mutation_ack_is_fifo_for_repeated_identity_updates() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let first = VectorId::from_serial_id(7).next_version();
        let second = first.next_version();
        let mut pending = PendingMutations::default();
        pending.push(first, iris.clone());
        pending.push(second, iris);

        assert!(pending.get(&second).is_some());
        pending.acknowledge(7);
        assert!(pending.get(&second).is_some());
        assert!(pending.get(&first).is_none());
        pending.acknowledge(7);
        assert!(!pending.entries.contains_key(&7));
    }

    #[test]
    fn prefetched_iris_is_retained_for_each_promised_use_then_evicted() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let id = VectorId::from_serial_id(11);
        let mut state = ColdStorageState::default();
        state.prefetched.insert(
            id,
            PrefetchEntry::Ready {
                iris: iris.clone(),
                remaining_uses: 2,
            },
        );

        assert_eq!(state.take_prefetched(&id), Some(iris.clone()));
        assert!(state.prefetched.contains_key(&id));
        assert_eq!(state.take_prefetched(&id), Some(iris));
        assert!(!state.prefetched.contains_key(&id));
    }

    #[test]
    fn full_rotation_task_size_only_changes_decomposition() -> Result<()> {
        std::thread::Builder::new()
            .name("full_rotation_task_size".to_owned())
            .stack_size(32 * 1024 * 1024)
            .spawn(run_full_rotation_task_size_test)?
            .join()
            .expect("full-rotation task-size test thread panicked")
    }

    fn run_full_rotation_task_size_test() -> Result<()> {
        let runtime = tokio::runtime::Runtime::new()?;
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let vector_ids = (0..TEST_TARGETS)
            .map(|index| VectorId::from_0_index(index as u32))
            .collect::<Vec<_>>();
        let points = vector_ids
            .iter()
            .copied()
            .map(|id| (id, iris.clone()))
            .collect::<HashMap<_, _>>();
        let storage = SharedIrises::new(points, iris.clone()).to_arc();
        let workers = init_workers(0, storage, false);

        let mut results = Vec::new();
        for task_size in [64, 128, 256, 512] {
            let mut workers = workers.clone();
            results.push(runtime.block_on(workers.full_rotation_dot_product_batch(
                iris.clone(),
                &vector_ids,
                NonZeroUsize::new(task_size).unwrap(),
            ))?);
        }

        assert!(results.iter().all(|result| result == &results[0]));
        assert_eq!(results[0].len(), TEST_TARGETS * ROTATIONS * 2);
        Ok(())
    }
}
