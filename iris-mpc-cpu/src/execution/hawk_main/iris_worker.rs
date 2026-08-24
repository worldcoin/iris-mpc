use crate::{
    execution::hawk_main::HAWK_MIN_DIST_ROTATIONS,
    hawkers::aby3::aby3_store::DistanceMode,
    hawkers::shared_irises::SharedIrisesRef,
    protocol::{
        ops::{
            galois_ring_pairwise_distance, non_existent_distance, pairwise_distance,
            rotation_aware_pairwise_distance, rotation_aware_pairwise_distance_rowmajor,
        },
        shared_iris::{ArcIris, GaloisRingSharedIris, ResidentIris, ResidentLayout},
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
use moka::sync::Cache;
use std::{
    collections::{HashMap, HashSet, VecDeque},
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
    // Overridable for scheduling experiments; the default is the tuned
    // production value.
    static TASK_SIZE: std::sync::OnceLock<NonZeroUsize> = std::sync::OnceLock::new();
    *TASK_SIZE.get_or_init(|| {
        std::env::var("IRIS_MPC_FULL_ROTATION_TASK_SIZE")
            .ok()
            .and_then(|value| value.parse().ok())
            .and_then(NonZeroUsize::new)
            .unwrap_or_else(|| {
                NonZeroUsize::new(DEFAULT_FULL_ROTATION_TASK_SIZE)
                    .expect("the default full-rotation task size must be nonzero")
            })
    })
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
        targets: Arc<Vec<ArcIris>>,
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
        targets: Arc<Vec<ArcIris>>,
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
        targets: Arc<Vec<ArcIris>>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    /// Both orientations' 31-rotation dot products in one pass over a resident
    /// target range. Targets are looked up and streamed once; each loaded
    /// target row feeds both queries' rotation tiles.
    FullRotationDotProductPairBatch {
        queries: [ArcIris; 2],
        vector_ids: Arc<[VectorId]>,
        range: std::ops::Range<usize>,
        rsp: oneshot::Sender<[Vec<RingElement<u16>>; 2]>,
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
        let targets = Arc::new(targets);
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
        let targets = Arc::new(targets);
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

    /// Paired-query variant of [`Self::full_rotation_dot_product_batch`]: one
    /// target traversal per task computes both queries' full rotation sets.
    pub async fn full_rotation_dot_product_pair_batch(
        &mut self,
        queries: [ArcIris; 2],
        vector_ids: &[VectorId],
        task_size: NonZeroUsize,
    ) -> Result<[Vec<RingElement<u16>>; 2]> {
        let start = Instant::now();
        let shared_ids: Arc<[VectorId]> = Arc::from(vector_ids);
        let task_size = task_size.get();
        let mut responses = Vec::with_capacity(shared_ids.len().div_ceil(task_size));

        for (i, _) in shared_ids.chunks(task_size).enumerate() {
            let range_start = i * task_size;
            let range_end = (range_start + task_size).min(shared_ids.len());
            let (tx, rx) = oneshot::channel();
            self.get_next_worker()
                .send(IrisTask::FullRotationDotProductPairBatch {
                    queries: queries.clone(),
                    vector_ids: shared_ids.clone(),
                    range: range_start..range_end,
                    rsp: tx,
                })?;
            responses.push(rx);
        }

        let mut results = [
            Vec::with_capacity(2 * ROTATIONS * shared_ids.len()),
            Vec::with_capacity(2 * ROTATIONS * shared_ids.len()),
        ];
        for task_results in futures::future::try_join_all(responses).await? {
            let [first, second] = task_results;
            results[0].extend(first);
            results[1].extend(second);
        }
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
        let targets = Arc::new(targets);
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
    iris_store: SharedIrisesRef<ResidentIris>,
    numa: bool,
    layout: ResidentLayout,
) -> IrisPoolHandle {
    let core_ids = select_core_ids(shard_index);
    info!(
        "Dot product shard {} running on {} cores ({:?}), resident layout {:?}",
        shard_index,
        core_ids.len(),
        core_ids,
        layout,
    );

    let mut channels = vec![];
    for core_id in core_ids {
        let (tx, rx) = crossbeam::channel::unbounded::<IrisTask>();
        channels.push(tx);
        let iris_store = iris_store.clone();
        std::thread::spawn(move || {
            let _ = core_affinity::set_for_current(core_id);
            worker_thread(rx, iris_store, numa, layout);
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

fn worker_thread(
    ch: Receiver<IrisTask>,
    iris_store: SharedIrisesRef<ResidentIris>,
    numa: bool,
    layout: ResidentLayout,
) {
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
                // `from_arc` writes the resident representation from this
                // thread, so first-touch places it NUMA-locally. The extra
                // u16 clone is only needed when the resident layout keeps
                // the incoming allocation.
                let resident = match layout {
                    ResidentLayout::U16 if numa => ResidentIris::U16(Arc::new((*iris).clone())),
                    _ => ResidentIris::from_arc(iris, layout),
                };

                let mut store = iris_store.data.blocking_write();
                store.insert(vector_id, resident);
            }

            IrisTask::Reserve { additional } => {
                let mut store = iris_store.data.blocking_write();
                store.reserve(additional);
            }

            IrisTask::DotProductPairs { pairs, rsp } => {
                let store = iris_store.data.blocking_read();

                let targets: Vec<Option<ArcIris>> = pairs
                    .iter()
                    .map(|(_, vid)| store.get_vector(vid).map(ResidentIris::to_arc))
                    .collect();
                let iris_pairs = pairs
                    .iter()
                    .zip(&targets)
                    .map(|((q, _), target)| target.as_ref().map(|iris| (q, iris)));

                let r = pairwise_distance(iris_pairs);
                let _ = rsp.send(r);
            }

            IrisTask::DotProductBatch {
                query,
                vector_ids,
                rsp,
            } => {
                let store = iris_store.data.blocking_read();

                let targets: Vec<Option<ArcIris>> = vector_ids
                    .iter()
                    .map(|v| store.get_vector(v).map(ResidentIris::to_arc))
                    .collect();
                let iris_pairs = targets
                    .iter()
                    .map(|target| target.as_ref().map(|iris| (&query, iris)));

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
                let targets: Vec<Option<ArcIris>> = vector_ids[range]
                    .iter()
                    .map(|v| store.get_vector(v).map(ResidentIris::to_arc))
                    .collect();
                let result = rotation_aware_pairwise_distance_rowmajor::<HAWK_MIN_DIST_ROTATIONS, _>(
                    &query,
                    targets.iter().map(Option::as_ref),
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
                let result = full_rotation_distance_resident(
                    &query,
                    vector_ids[range].iter().map(|v| store.get_vector(v)),
                );
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

            IrisTask::FullRotationDotProductPairBatch {
                queries,
                vector_ids,
                range,
                rsp,
            } => {
                let store = iris_store.data.blocking_read();
                let result = full_rotation_distance_resident_pair(
                    &queries,
                    vector_ids[range].iter().map(|v| store.get_vector(v)),
                );
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

/// Full 31-rotation distances against resident targets, dispatching on the
/// resident representation: mixed-plane targets use the UMMLA kernel,
/// u16 targets the MLA kernel. Results are bit-identical between the two.
fn full_rotation_distance_resident<'a, I>(query: &ArcIris, targets: I) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<&'a ResidentIris>> + ExactSizeIterator,
{
    let residents: Vec<Option<&ResidentIris>> = targets.collect();

    #[cfg(target_arch = "aarch64")]
    {
        use crate::protocol::ops::rotation_aware_pairwise_distance_mixed;
        use crate::protocol::shared_iris::MixedPlaneIris;

        let all_mixed = residents
            .iter()
            .all(|target| target.is_none_or(|resident| resident.as_mixed().is_some()));
        if all_mixed && residents.iter().any(Option::is_some) {
            let mixed: Vec<Option<&MixedPlaneIris>> = residents
                .iter()
                .map(|target| target.and_then(ResidentIris::as_mixed))
                .collect();
            return rotation_aware_pairwise_distance_mixed::<ROTATIONS>(query, &mixed);
        }
    }

    let owned: Vec<Option<ArcIris>> = residents
        .iter()
        .map(|target| target.map(ResidentIris::to_arc))
        .collect();
    rotation_aware_pairwise_distance_rowmajor::<ROTATIONS, _>(
        query,
        owned.iter().map(Option::as_ref),
    )
}

/// Paired-query variant of [`full_rotation_distance_resident`]: mixed-plane
/// residents use the fused UMMLA pair kernel (targets streamed once for both
/// queries); the u16 fallback evaluates the queries sequentially, which is
/// bit-identical and still benefits from the two-slot prerotation cache.
fn full_rotation_distance_resident_pair<'a, I>(
    queries: &[ArcIris; 2],
    targets: I,
) -> [Vec<RingElement<u16>>; 2]
where
    I: Iterator<Item = Option<&'a ResidentIris>> + ExactSizeIterator,
{
    let residents: Vec<Option<&ResidentIris>> = targets.collect();

    #[cfg(target_arch = "aarch64")]
    {
        use crate::protocol::ops::rotation_aware_pairwise_distance_mixed_pair;
        use crate::protocol::shared_iris::MixedPlaneIris;

        let all_mixed = residents
            .iter()
            .all(|target| target.is_none_or(|resident| resident.as_mixed().is_some()));
        if all_mixed && residents.iter().any(Option::is_some) {
            let mixed: Vec<Option<&MixedPlaneIris>> = residents
                .iter()
                .map(|target| target.and_then(ResidentIris::as_mixed))
                .collect();
            return rotation_aware_pairwise_distance_mixed_pair::<ROTATIONS>(
                [&queries[0], &queries[1]],
                &mixed,
            );
        }
    }

    let owned: Vec<Option<ArcIris>> = residents
        .iter()
        .map(|target| target.map(ResidentIris::to_arc))
        .collect();
    [&queries[0], &queries[1]].map(|query| {
        rotation_aware_pairwise_distance_rowmajor::<ROTATIONS, _>(
            query,
            owned.iter().map(Option::as_ref),
        )
    })
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

    /// Compute two queries' full 31-rotation dot products over one shared
    /// target traversal. The exact scan uses this to evaluate the normal and
    /// mirror orientations while streaming the resident database once.
    ///
    /// Each returned side is identical to a separate
    /// [`IrisWorkerPool::compute_dot_products_full_rotations`] call. The
    /// default implementation evaluates the queries sequentially; pools with
    /// fused kernels override it.
    fn compute_dot_products_full_rotations_pair<'a>(
        &'a self,
        queries: [QuerySpec; 2],
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, Result<[Vec<RingElement<u16>>; 2]>> {
        Box::pin(async move {
            let first = self
                .compute_dot_products_full_rotations(queries[0], vector_ids.clone())
                .await?;
            let second = self
                .compute_dot_products_full_rotations(queries[1], vector_ids)
                .await?;
            Ok([first, second])
        })
    }

    /// Fetch iris data from the worker's store by vector ID.
    ///
    /// Returns one `ArcIris` per input ID in the same order. Database-backed
    /// workers return an error when an exact `(serial_id, version_id)` row is
    /// missing; substituting an empty sharing would reconstruct to zero
    /// distance and fail open under the threshold convention.
    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>>;

    /// Hint that these records will soon be fetched. Database-backed workers
    /// enqueue the read without blocking the caller; resident workers do
    /// nothing. Each record is reserved once and handed to the first
    /// [`IrisWorkerPool::fetch_irises`] that asks for it; repeated hints for a
    /// record already reserved or cached are no-ops. A failed asynchronous
    /// read is logged and counted only: the foreground fetch repeats the read
    /// and is the path that fails closed on a missing exact version.
    fn prefetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>>;

    /// Wait until all prefetch hints accepted before this call have completed.
    fn wait_for_prefetch<'a>(&'a self) -> BoxFuture<'a, Result<()>>;

    /// Drop every outstanding prefetch reservation, moving records that did
    /// arrive into the reusable cache. Called once a batch has finished all
    /// of its scans so that reservations never outlive the batch that made
    /// them. Resident workers do nothing. Returns the number of reservations
    /// dropped.
    fn release_prefetched<'a>(&'a self) -> BoxFuture<'a, usize>;

    /// Insert a cached iris into the worker's persistent store.
    ///
    /// The worker looks up the original (un-rotated) iris from the cache
    /// by `QueryId` and inserts it at the given `VectorId`.
    ///
    /// Returns the store checksum after all inserts are applied.
    fn insert_irises<'a>(&'a self, inserts: Vec<(QueryId, VectorId)>)
        -> BoxFuture<'a, Result<u64>>;

    /// Drop exact vector versions from a cold mutation overlay after their
    /// database transaction commits. Resident workers have nothing to drop.
    ///
    /// Exact-version removal is intentional: a later mutation of the same
    /// serial ID may already be queued, and acknowledging the older version
    /// must not make that newer mutation invisible before it is persisted.
    fn acknowledge_persisted_irises<'a>(
        &'a self,
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, usize>;

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
    fn compute_dot_products_full_rotations_pair<'a>(
        &'a self,
        queries: [QuerySpec; 2],
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, Result<[Vec<RingElement<u16>>; 2]>> {
        (**self).compute_dot_products_full_rotations_pair(queries, vector_ids)
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
    fn release_prefetched<'a>(&'a self) -> BoxFuture<'a, usize> {
        (**self).release_prefetched()
    }
    fn insert_irises<'a>(
        &'a self,
        inserts: Vec<(QueryId, VectorId)>,
    ) -> BoxFuture<'a, Result<u64>> {
        (**self).insert_irises(inserts)
    }
    fn acknowledge_persisted_irises<'a>(
        &'a self,
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, usize> {
        (**self).acknowledge_persisted_irises(vector_ids)
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
    iris_store: SharedIrisesRef<ResidentIris>,
    layout: ResidentLayout,
    mode: DistanceMode,
    party_id: usize,
    /// When set, the complete iris column stays in Postgres. RAM holds the
    /// rolling LUC window, a bounded frequency cache, and mutations awaiting a
    /// database commit; older explicit candidates are fetched sparsely.
    cold_storage: Option<ColdStorage>,
    /// The HNSW-style windowed dot products read resident records through
    /// `ResidentIris::to_arc`, which rebuilds a u16 iris per target when the
    /// resident layout is the mixed-plane scan layout. The exact scan never
    /// takes that path (it streams planes directly), so production pools
    /// refuse it; the cross-kernel parity tests opt in explicitly.
    windowed_ops_on_mixed_residents: bool,
}

#[derive(Clone)]
struct ColdStorage {
    store: Store,
    side: usize,
    state: Arc<AsyncRwLock<ColdStorageState>>,
    prefetch_tx: mpsc::Sender<ColdPrefetchCommand>,
}

struct ColdStorageState {
    luc_cache: RollingLucCache,
    /// Frequency-aware cache for older exact-version records. Moka's default
    /// TinyLFU admission protects hot supermatchers from one-off scan traffic.
    lfu_cache: Cache<VectorId, ArcIris>,
    lfu_cache_enabled: bool,
    lfu_cache_eye: &'static str,
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
    Barrier(oneshot::Sender<()>),
}

/// A single-use reservation: the first foreground fetch of the record
/// consumes it (promoting the value into the LFU cache), and any reservation
/// still present when the batch ends is released by
/// [`IrisWorkerPool::release_prefetched`]. Reservations are deliberately not
/// use-counted: the same record is hinted by several independent paths (known
/// candidates, every first-eye chunk that matches it, both orientations), and
/// reconciling those counts against the deduplicated second-stage fetch is
/// exactly what leaked entries before.
enum PrefetchEntry {
    Loading,
    Ready(ArcIris),
}

impl ColdStorageState {
    fn new(luc_cache: RollingLucCache, lfu_cache_capacity: usize, side: usize) -> Self {
        Self {
            luc_cache,
            lfu_cache: Cache::builder()
                .max_capacity(lfu_cache_capacity as u64)
                .build(),
            lfu_cache_enabled: lfu_cache_capacity > 0,
            lfu_cache_eye: match side {
                0 => "left",
                1 => "right",
                _ => "unknown",
            },
            pending_mutations: PendingMutations::default(),
            prefetched: HashMap::new(),
        }
    }

    fn mutation_or_luc_iris(&self, id: &VectorId) -> Option<ArcIris> {
        self.pending_mutations
            .get(id)
            .or_else(|| self.luc_cache.get(id))
    }

    fn contains_cached_iris(&self, id: &VectorId) -> bool {
        self.mutation_or_luc_iris(id).is_some()
            || (self.lfu_cache_enabled && self.lfu_cache.contains_key(id))
    }

    fn get_lfu_iris(&self, id: &VectorId) -> Option<ArcIris> {
        if !self.lfu_cache_enabled {
            return None;
        }
        if let Some(iris) = self.lfu_cache.get(id) {
            metrics::counter!(
                "linear_scan_cold_lfu_cache_hits_total",
                "eye" => self.lfu_cache_eye,
            )
            .increment(1);
            Some(iris)
        } else {
            metrics::counter!(
                "linear_scan_cold_lfu_cache_misses_total",
                "eye" => self.lfu_cache_eye,
            )
            .increment(1);
            None
        }
    }

    fn insert_lfu_iris(&self, id: VectorId, iris: ArcIris) {
        if self.lfu_cache_enabled {
            self.lfu_cache.insert(id, iris);
            metrics::counter!(
                "linear_scan_cold_lfu_cache_inserts_total",
                "eye" => self.lfu_cache_eye,
            )
            .increment(1);
        }
    }

    fn invalidate_previous_lfu_version(&self, id: VectorId) {
        if self.lfu_cache_enabled && id.version_id() > 0 {
            self.lfu_cache
                .invalidate(&VectorId::new(id.serial_id(), id.version_id() - 1));
        }
    }

    /// Prefer the authoritative mutation/LUC overlays, then consume any
    /// promised prefetch before consulting the LFU. Consuming in this order
    /// preserves prefetch use counts while promoting successful DB reads into
    /// the reusable cache.
    fn take_cached_or_prefetched(&mut self, id: &VectorId) -> Option<ArcIris> {
        if let Some(iris) = self.mutation_or_luc_iris(id) {
            return Some(iris);
        }
        if let Some(iris) = self.take_prefetched(id) {
            metrics::counter!(
                "linear_scan_cold_prefetch_hits_total",
                "eye" => self.lfu_cache_eye,
            )
            .increment(1);
            self.insert_lfu_iris(*id, iris.clone());
            return Some(iris);
        }
        self.get_lfu_iris(id)
    }

    /// Consume a prefetch reservation. A still-loading reservation is dropped
    /// as well: the caller reads the record itself, and the worker skips
    /// reservations that disappeared while its database read was in flight.
    fn take_prefetched(&mut self, id: &VectorId) -> Option<ArcIris> {
        match self.prefetched.remove(id)? {
            PrefetchEntry::Ready(iris) => Some(iris),
            PrefetchEntry::Loading => None,
        }
    }

    /// Drop all reservations, keeping arrived records in the LFU cache.
    fn release_prefetched(&mut self) -> usize {
        let released = self.prefetched.len();
        // Local bookkeeping only, but keep the crate-wide rule of never
        // iterating a hash map in an unspecified order.
        let mut drained = self.prefetched.drain().collect::<Vec<_>>();
        drained.sort_unstable_by_key(|(id, _)| *id);
        for (id, entry) in drained {
            if let PrefetchEntry::Ready(iris) = entry {
                // `insert_lfu_iris` borrows `self` immutably; the cache is
                // internally synchronized, so go through it directly.
                if self.lfu_cache_enabled {
                    self.lfu_cache.insert(id, iris);
                }
            }
        }
        if released > 0 {
            metrics::counter!("linear_scan_cold_prefetch_released_total")
                .increment(released as u64);
        }
        released
    }
}

/// The current LUC lookback window for the non-resident eye. Serial IDs are
/// dense, so the front is the oldest retained ID and the back is the latest.
#[derive(Default)]
struct RollingLucCache {
    capacity: usize,
    entries: VecDeque<(VectorId, ArcIris)>,
}

impl RollingLucCache {
    /// Build the window from serial-ordered entries. The window is a pure
    /// cache in front of the exact-version overlay, LFU, and database, so a
    /// gap in the loaded serial IDs (a registry hole, a partial load) only
    /// shrinks the window to its dense tail instead of refusing to start.
    fn new(capacity: usize, mut entries: Vec<(VectorId, ArcIris)>) -> Self {
        let dense_tail_start = entries
            .windows(2)
            .rposition(|pair| pair[1].0.serial_id() != pair[0].0.serial_id() + 1)
            .map_or(0, |gap| gap + 1);
        if dense_tail_start > 0 {
            tracing::warn!(
                dropped = dense_tail_start,
                retained = entries.len() - dense_tail_start,
                "rolling LUC window is not dense; keeping only its newest contiguous run"
            );
            entries.drain(..dense_tail_start);
        }
        if entries.len() > capacity {
            let excess = entries.len() - capacity;
            entries.drain(..excess);
        }
        Self {
            capacity,
            entries: entries.into(),
        }
    }

    fn get(&self, id: &VectorId) -> Option<ArcIris> {
        let first_serial_id = self.entries.front()?.0.serial_id();
        let offset = usize::try_from(id.serial_id().checked_sub(first_serial_id)?).ok()?;
        self.entries
            .get(offset)
            .filter(|(cached_id, _)| cached_id == id)
            .map(|(_, iris)| iris.clone())
    }

    fn apply_mutation(&mut self, id: VectorId, iris: ArcIris) {
        if self.capacity == 0 {
            return;
        }

        let Some((latest_id, _)) = self.entries.back() else {
            // An empty window (empty database, or reset below) starts at
            // whatever serial ID arrives first.
            self.entries.push_back((id, iris));
            return;
        };

        let latest_serial_id = latest_id.serial_id();
        if id.serial_id() <= latest_serial_id {
            let first_serial_id = self.entries.front().unwrap().0.serial_id();
            let Some(offset) = id.serial_id().checked_sub(first_serial_id) else {
                return;
            };
            let Some(entry) = self.entries.get_mut(offset as usize) else {
                return;
            };
            *entry = (id, iris);
        } else {
            if id.serial_id() != latest_serial_id.wrapping_add(1) {
                // Serial IDs normally advance by one. Rather than trusting an
                // offset-addressed window across a hole, restart it at the new
                // record; older records are still served exactly by the
                // overlay, LFU, or database.
                tracing::warn!(
                    latest_serial_id,
                    serial_id = id.serial_id(),
                    "rolling LUC insertion skipped serial IDs; restarting the window"
                );
                metrics::counter!("linear_scan_cold_luc_window_resets_total").increment(1);
                self.entries.clear();
            } else if self.entries.len() == self.capacity {
                self.entries.pop_front();
            }
            self.entries.push_back((id, iris));
        }
    }
}

/// Pending writes grouped by serial ID. Acknowledgements remove only the exact
/// committed vector version, preserving any later update of the same identity.
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
    /// Number of serial positions retained. This intentionally includes the
    /// extra position used by `HawkRequest::luc_ids`.
    pub luc_window_capacity: usize,
    /// Maximum number of exact-version records outside the LUC window kept by
    /// the frequency-aware cache. Zero disables this cache.
    pub lfu_cache_capacity: usize,
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
        self.record_size();
    }

    fn acknowledge(&mut self, vector_id: VectorId) -> bool {
        let serial_id = vector_id.serial_id();
        let mut removed = false;
        let remove_serial = if let Some(values) = self.entries.get_mut(&serial_id) {
            if let Some(index) = values.iter().position(|(id, _)| *id == vector_id) {
                removed = values.remove(index).is_some();
            }
            values.is_empty()
        } else {
            false
        };
        if remove_serial {
            self.entries.remove(&serial_id);
        }
        self.record_size();
        if removed {
            metrics::counter!("linear_scan_cold_mutation_overlay_evictions_total").increment(1);
        }
        removed
    }

    fn record_size(&self) {
        let len = self.entries.values().map(VecDeque::len).sum::<usize>();
        metrics::gauge!("linear_scan_cold_mutation_overlay_entries").set(len as f64);
    }
}

async fn run_cold_prefetch_worker(
    store: Store,
    side: usize,
    party_id: usize,
    state: Arc<AsyncRwLock<ColdStorageState>>,
    mut rx: mpsc::Receiver<ColdPrefetchCommand>,
) {
    while let Some(command) = rx.recv().await {
        match command {
            ColdPrefetchCommand::Fetch(ids) => {
                // A prefetch is only a hint. Its reservations are released on
                // failure, so the foreground fetch repeats the read and is the
                // path that fails closed; surfacing the error here as well
                // would fail a batch for a transient read that the foreground
                // would have served, and could attribute a failure from an
                // earlier, already-abandoned scan to the next barrier.
                if let Err(error) = prefetch_cold_irises(&store, side, party_id, &state, ids).await
                {
                    tracing::warn!(
                        ?error,
                        "Cold-eye database prefetch failed; the scan will read these records directly"
                    );
                    metrics::counter!("linear_scan_cold_prefetch_errors_total").increment(1);
                }
            }
            ColdPrefetchCommand::Barrier(done) => {
                let _ = done.send(());
            }
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
            if state.contains_cached_iris(&id) || state.prefetched.contains_key(&id) {
                continue;
            }
            if state.prefetched.len() < COLD_PREFETCH_MAX_RECORDS {
                state.prefetched.insert(id, PrefetchEntry::Loading);
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

    let fetched = fetched.and_then(|fetched| {
        let missing = to_fetch
            .iter()
            .filter(|id| !fetched.contains_key(id))
            .copied()
            .collect_vec();
        if missing.is_empty() {
            Ok(fetched)
        } else {
            Err(cold_db_miss_error(party_id, side, &missing))
        }
    });

    let mut state = state.write().await;
    match fetched {
        Ok(mut fetched) => {
            for id in to_fetch {
                // A reservation that disappeared while the read was in flight
                // was consumed by a foreground read or released at batch end.
                if let Some(entry) = state.prefetched.get_mut(&id) {
                    *entry = PrefetchEntry::Ready(
                        fetched
                            .remove(&id)
                            .expect("cold-eye missing rows were checked above"),
                    );
                }
            }
            metrics::counter!("linear_scan_cold_prefetched_records_total")
                .increment(db_ids.len() as u64);
            Ok(())
        }
        Err(error) => {
            // Release the still-loading reservations so the foreground fetch
            // reads (and, for a genuine miss, fails closed on) these records.
            for id in to_fetch {
                if matches!(state.prefetched.get(&id), Some(PrefetchEntry::Loading)) {
                    state.prefetched.remove(&id);
                }
            }
            Err(error)
        }
    }
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
        iris_store: SharedIrisesRef<ResidentIris>,
        layout: ResidentLayout,
        mode: DistanceMode,
        party_id: usize,
    ) -> Self {
        Self {
            inner,
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            iris_store,
            layout,
            mode,
            party_id,
            cold_storage: None,
            windowed_ops_on_mixed_residents: false,
        }
    }

    /// Construct the non-resident eye of a linear-scan actor. Its complete
    /// iris column remains in Postgres. The current LUC window is retained in
    /// memory; frequently used older candidates enter a bounded TinyLFU cache.
    /// A bounded FIFO overlay keeps mutations visible until asynchronous
    /// persistence commits.
    pub async fn new_cold(
        inner: IrisPoolHandle,
        iris_store: SharedIrisesRef<ResidentIris>,
        layout: ResidentLayout,
        mode: DistanceMode,
        party_id: usize,
        init: ColdStorageInit,
    ) -> Result<Self> {
        let ColdStorageInit {
            store,
            side,
            luc_window_ids,
            luc_window_capacity,
            lfu_cache_capacity,
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
        let cached_ids = cached
            .iter()
            .map(|(cached_id, _)| *cached_id)
            .collect::<HashSet<_>>();
        let missing = luc_window_ids
            .iter()
            .filter(|id| !cached_ids.contains(id))
            .copied()
            .collect_vec();
        if !missing.is_empty() {
            return Err(cold_db_miss_error(party_id, side, &missing));
        }
        let state = Arc::new(AsyncRwLock::new(ColdStorageState::new(
            RollingLucCache::new(luc_window_capacity, cached),
            lfu_cache_capacity,
            side,
        )));
        metrics::gauge!(
            "linear_scan_cold_lfu_cache_capacity_records",
            "eye" => match side {
                0 => "left",
                1 => "right",
                _ => "unknown",
            },
        )
        .set(lfu_cache_capacity as f64);
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
            layout,
            mode,
            party_id,
            cold_storage: Some(ColdStorage {
                store,
                side,
                state,
                prefetch_tx,
            }),
            windowed_ops_on_mixed_residents: false,
        })
    }

    /// Allow windowed (HNSW-style) dot products against mixed-plane
    /// residents. Each target is rebuilt as a u16 iris, so this is only for
    /// tests and tools that compare the two kernels on the same data.
    pub fn with_windowed_ops_on_mixed_residents(mut self) -> Self {
        self.windowed_ops_on_mixed_residents = true;
        self
    }

    /// Create a local worker pool for shard 0 with NUMA pinning.
    /// Standard construction for tests, benchmarks, and single-node tools.
    pub fn new_local(
        iris_store: SharedIrisesRef<ResidentIris>,
        layout: ResidentLayout,
        mode: DistanceMode,
        party_id: usize,
    ) -> Self {
        let pool = init_workers(0, iris_store.clone(), true, layout);
        Self::new(pool, iris_store, layout, mode, party_id)
    }

    async fn fetch_irises_resident_or_cold(&self, ids: &[VectorId]) -> Result<Vec<ArcIris>> {
        let Some(cold) = &self.cold_storage else {
            let store = self.iris_store.data.read().await;
            return Ok(ids
                .iter()
                .map(|id| store.get_vector_or_empty(id).to_arc())
                .collect());
        };

        // Prefer uncommitted writes, then the resident LUC window. This makes
        // a just-inserted or just-deleted iris visible before result
        // persistence finishes; the exact-version LFU is subordinate to both.
        let cached = {
            let mut state = cold.state.write().await;
            ids.iter()
                .map(|id| state.take_cached_or_prefetched(id))
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
        if !fetched.is_empty() {
            let state = cold.state.read().await;
            for id in ids.iter().copied().unique() {
                if let Some(iris) = fetched.remove(&id) {
                    state.insert_lfu_iris(id, iris);
                }
            }
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
        let layout = self.layout;
        let windowed_ops_on_mixed_residents = self.windowed_ops_on_mixed_residents;
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

            eyre::ensure!(
                layout == ResidentLayout::U16 || windowed_ops_on_mixed_residents,
                "windowed dot products are not served from mixed-plane residents: this path \
                 rebuilds a u16 iris per target, and the exact scan does not use it"
            );
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
        let task_size = default_full_rotation_task_size();
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

    fn compute_dot_products_full_rotations_pair<'a>(
        &'a self,
        queries: [QuerySpec; 2],
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, Result<[Vec<RingElement<u16>>; 2]>> {
        let query_cache = self.query_cache.clone();
        let mut inner = self.inner.clone();
        let pool = self.clone();
        let is_cold = self.cold_storage.is_some();
        let task_size = default_full_rotation_task_size();
        Box::pin(async move {
            let irises = {
                let cache = query_cache.read().unwrap();
                let mut resolved = Vec::with_capacity(2);
                for query in queries {
                    let cached = cache
                        .get(&query.query_id)
                        .ok_or_else(|| eyre::eyre!("Query {:?} not cached", query.query_id))?;
                    let rotations = if query.mirrored {
                        &cached.mirrored_preprocessed_rotations
                    } else {
                        &cached.preprocessed_rotations
                    };
                    resolved.push(rotations[query.rotation].clone());
                }
                let second = resolved.pop().expect("two resolved queries");
                let first = resolved.pop().expect("two resolved queries");
                [first, second]
            };
            if is_cold {
                // Cold targets are fetched once and reused by both queries;
                // the dot passes themselves stay sequential on this rare path.
                let targets = pool.fetch_irises_resident_or_cold(&vector_ids).await?;
                let [query_a, query_b] = irises;
                let first = inner
                    .full_rotation_dot_product_irises_batch(query_a, targets.clone(), task_size)
                    .await?;
                let second = inner
                    .full_rotation_dot_product_irises_batch(query_b, targets, task_size)
                    .await?;
                Ok([first, second])
            } else {
                inner
                    .full_rotation_dot_product_pair_batch(irises, &vector_ids, task_size)
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
                .map_err(|_| eyre::eyre!("cold-eye prefetch barrier was dropped"))
        })
    }

    fn release_prefetched<'a>(&'a self) -> BoxFuture<'a, usize> {
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            let Some(cold) = cold_storage else {
                return 0;
            };
            let released = cold.state.write().await.release_prefetched();
            released
        })
    }

    fn insert_irises<'a>(
        &'a self,
        inserts: Vec<(QueryId, VectorId)>,
    ) -> BoxFuture<'a, Result<u64>> {
        let query_cache = self.query_cache.clone();
        let iris_store = self.iris_store.clone();
        let cold_storage = self.cold_storage.clone();
        let layout = self.layout;
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
                    state.invalidate_previous_lfu_version(vector_id);
                    state.pending_mutations.push(vector_id, iris.clone());
                    state.luc_cache.apply_mutation(vector_id, iris);
                }
                return Ok(0);
            }

            // Build the resident representation before taking the lock; the
            // mixed-plane interleave has no reason to run under it.
            let resident = resolved
                .into_iter()
                .map(|(vector_id, iris)| (vector_id, ResidentIris::from_arc(iris, layout)))
                .collect::<Vec<_>>();

            // Write directly to the shared store (not via IrisPoolHandle::insert
            // which is fire-and-forget). HNSW insertion needs the iris to be
            // visible in the store immediately after this returns.
            let mut store = iris_store.data.write().await;
            for (vector_id, iris) in resident {
                store.insert(vector_id, iris);
            }
            Ok(store.set_hash.checksum())
        })
    }

    fn acknowledge_persisted_irises<'a>(
        &'a self,
        vector_ids: Vec<VectorId>,
    ) -> BoxFuture<'a, usize> {
        let cold_storage = self.cold_storage.clone();
        Box::pin(async move {
            let Some(cold) = cold_storage else {
                return 0;
            };
            let mut state = cold.state.write().await;
            vector_ids
                .into_iter()
                .filter(|vector_id| state.pending_mutations.acknowledge(*vector_id))
                .count()
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
        let layout = self.layout;
        Box::pin(async move {
            let dummy = Arc::new(GaloisRingSharedIris::dummy_for_party(party_id));
            if let Some(cold) = cold_storage {
                let mut state = cold.state.write().await;
                for id in ids {
                    let next_id = id.next_version();
                    state.lfu_cache.invalidate(&id);
                    state.pending_mutations.push(next_id, dummy.clone());
                    state.luc_cache.apply_mutation(next_id, dummy.clone());
                }
                return Ok(());
            }
            let resident_dummy = ResidentIris::from_arc(dummy, layout);
            let mut store = iris_store.data.write().await;
            for id in ids {
                store.update(id, resident_dummy.clone());
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
            (3..=5)
                .map(|serial_id| (VectorId::from_serial_id(serial_id), iris.clone()))
                .collect(),
        );

        let updated_four = VectorId::from_serial_id(4).next_version();
        cache.apply_mutation(updated_four, iris.clone());
        assert!(cache.get(&updated_four).is_some());
        assert!(cache.get(&VectorId::from_serial_id(4)).is_none());

        cache.apply_mutation(VectorId::from_serial_id(6), iris.clone());
        assert_eq!(
            cache
                .entries
                .iter()
                .map(|(id, _)| id.serial_id())
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );

        cache.apply_mutation(VectorId::from_serial_id(2), iris);
        assert!(cache.entries.iter().all(|(id, _)| id.serial_id() != 2));
    }

    #[test]
    fn rolling_luc_cache_fills_from_an_empty_database() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let mut cache = RollingLucCache::new(2, Vec::new());

        for serial_id in 1..=3 {
            cache.apply_mutation(VectorId::from_serial_id(serial_id), iris.clone());
        }

        assert_eq!(
            cache
                .entries
                .iter()
                .map(|(id, _)| id.serial_id())
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn pending_mutation_acknowledges_only_exact_persisted_versions() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let first = VectorId::from_serial_id(7).next_version();
        let second = first.next_version();
        let mut pending = PendingMutations::default();
        pending.push(first, iris.clone());
        pending.push(second, iris);

        assert!(pending.get(&second).is_some());
        assert!(pending.acknowledge(first));
        assert!(pending.get(&second).is_some());
        assert!(pending.get(&first).is_none());
        assert!(!pending.acknowledge(first));
        assert!(pending.acknowledge(second));
        assert!(!pending.entries.contains_key(&7));
    }

    #[test]
    fn prefetched_iris_is_consumed_once_and_promoted_to_the_lfu() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let id = VectorId::from_serial_id(11);
        let loading = VectorId::from_serial_id(12);
        let mut state = ColdStorageState::new(RollingLucCache::default(), 8, 0);
        state
            .prefetched
            .insert(id, PrefetchEntry::Ready(iris.clone()));
        state.prefetched.insert(loading, PrefetchEntry::Loading);

        // The first consumer takes the reservation and the value moves to the
        // LFU, where a second consumer (the other orientation) finds it.
        assert_eq!(state.take_cached_or_prefetched(&id), Some(iris.clone()));
        assert!(!state.prefetched.contains_key(&id));
        assert_eq!(state.take_cached_or_prefetched(&id), Some(iris));

        // A foreground read of a still-loading record drops the reservation
        // so the worker's late completion is ignored.
        assert!(state.take_cached_or_prefetched(&loading).is_none());
        assert!(!state.prefetched.contains_key(&loading));
    }

    #[test]
    fn releasing_prefetched_drops_reservations_and_keeps_arrived_values() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let ready = VectorId::from_serial_id(21);
        let loading = VectorId::from_serial_id(22);
        let mut state = ColdStorageState::new(RollingLucCache::default(), 8, 0);
        state
            .prefetched
            .insert(ready, PrefetchEntry::Ready(iris.clone()));
        state.prefetched.insert(loading, PrefetchEntry::Loading);

        assert_eq!(state.release_prefetched(), 2);
        assert!(state.prefetched.is_empty());
        assert_eq!(state.lfu_cache.get(&ready), Some(iris));
        assert!(state.lfu_cache.get(&loading).is_none());
        assert_eq!(state.release_prefetched(), 0);
    }

    #[test]
    fn rolling_luc_window_tolerates_gaps() {
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let entry = |serial_id: u32| (VectorId::from_serial_id(serial_id), iris.clone());

        // A hole in the loaded window keeps only the newest contiguous run.
        let cache = RollingLucCache::new(8, vec![entry(3), entry(4), entry(6), entry(7)]);
        assert!(cache.get(&VectorId::from_serial_id(4)).is_none());
        assert!(cache.get(&VectorId::from_serial_id(6)).is_some());
        assert!(cache.get(&VectorId::from_serial_id(7)).is_some());

        // An empty window starts at whatever serial ID arrives first.
        let mut cache = RollingLucCache::new(3, Vec::new());
        cache.apply_mutation(VectorId::from_serial_id(40), iris.clone());
        cache.apply_mutation(VectorId::from_serial_id(41), iris.clone());
        assert!(cache.get(&VectorId::from_serial_id(40)).is_some());

        // A skipped serial ID restarts the window instead of panicking.
        cache.apply_mutation(VectorId::from_serial_id(50), iris.clone());
        assert!(cache.get(&VectorId::from_serial_id(40)).is_none());
        assert!(cache.get(&VectorId::from_serial_id(41)).is_none());
        assert!(cache.get(&VectorId::from_serial_id(50)).is_some());
        cache.apply_mutation(VectorId::from_serial_id(51), iris.clone());
        cache.apply_mutation(VectorId::from_serial_id(52), iris.clone());
        cache.apply_mutation(VectorId::from_serial_id(53), iris);
        assert!(cache.get(&VectorId::from_serial_id(50)).is_none());
        assert!(cache.get(&VectorId::from_serial_id(53)).is_some());
    }

    #[test]
    fn cold_lfu_cache_retains_a_hot_record_during_unique_churn() {
        let mut state = ColdStorageState::new(RollingLucCache::default(), 2, 0);
        let iris = Arc::new(GaloisRingSharedIris::default_for_party(0));
        let hot = VectorId::from_serial_id(1);
        let initial_cold = VectorId::from_serial_id(2);
        state.lfu_cache.insert(hot, iris.clone());
        state.lfu_cache.insert(initial_cold, iris.clone());
        state.lfu_cache.run_pending_tasks();

        for _ in 0..128 {
            assert!(state.take_cached_or_prefetched(&hot).is_some());
        }
        state.lfu_cache.run_pending_tasks();

        for serial_id in 3..64 {
            state
                .lfu_cache
                .insert(VectorId::from_serial_id(serial_id), iris.clone());
            state.lfu_cache.run_pending_tasks();
        }

        assert!(state.lfu_cache.get(&hot).is_some());
        assert!(state.lfu_cache.entry_count() <= 2);
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

        // Cover every supported resident layout; results must be identical.
        let mut per_layout = Vec::new();
        for layout in [
            ResidentLayout::U16,
            crate::protocol::shared_iris::preferred_scan_layout(),
        ] {
            let points = vector_ids
                .iter()
                .copied()
                .map(|id| (id, ResidentIris::from_arc(iris.clone(), layout)))
                .collect::<HashMap<_, _>>();
            let storage =
                SharedIrises::new(points, ResidentIris::from_arc(iris.clone(), layout)).to_arc();
            let workers = init_workers(0, storage, false, layout);

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
            per_layout.push(results.remove(0));
        }
        assert!(
            per_layout.iter().all(|result| result == &per_layout[0]),
            "resident layouts must produce identical distances"
        );
        Ok(())
    }
}
