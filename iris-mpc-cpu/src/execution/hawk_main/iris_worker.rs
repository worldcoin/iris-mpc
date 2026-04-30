use crate::{
    execution::hawk_main::HAWK_MIN_DIST_ROTATIONS,
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
use ampc_actor_utils::{fast_metrics::FastHistogram, network::workpool::leader::LeaderHandle};
use bytes::Bytes;
use core_affinity::CoreId;
use crossbeam::channel::{Receiver, Sender};
use eyre::{eyre, Result};
use futures::future::{try_join_all, BoxFuture};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    vector_id::VectorId,
};
use iris_mpc_worker_protocol as wire;
use itertools::{izip, Itertools};
use std::{
    collections::HashMap,
    fmt::Debug,
    iter,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::Instant,
};
use tokio::sync::oneshot;
use tracing::info;

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
    /// Computes the rotation-aware dot product between a query and a batch of database irises.
    RotationAwareDotProductBatch {
        query: ArcIris,
        vector_ids: Arc<[VectorId]>,
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
    // variants. This is the main remaining performance gap vs the old design
    // (which only preprocessed the selected rotations).
    fn cache_queries<'a>(&'a self, queries: Vec<(QueryId, ArcIris)>) -> BoxFuture<'a, Result<()>>;

    /// Compute dot products for batches of (query_spec, targets).
    ///
    /// Each `QuerySpec` selects a specific preprocessed rotation from the
    /// cache. Returns one `Vec<RingElement<u16>>` per batch.
    fn compute_dot_products<'a>(
        &'a self,
        batches: Vec<(QuerySpec, Vec<VectorId>)>,
    ) -> BoxFuture<'a, Result<Vec<Vec<RingElement<u16>>>>>;

    /// Fetch iris data from the worker's store by vector ID.
    ///
    /// Returns one `ArcIris` per input ID in the same order.  Missing
    /// entries produce the store's default empty iris (which yields
    /// max-distance in dot products).
    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>>;

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
    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>> {
        (**self).fetch_irises(ids)
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
        }
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

    /// Access the underlying `IrisPoolHandle` for operations not on the trait
    /// (e.g., `numa_realloc`, `reserve`, `wait_completion`).
    pub fn inner(&self) -> &IrisPoolHandle {
        &self.inner
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

    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>> {
        let iris_store = self.iris_store.clone();
        Box::pin(async move {
            let store = iris_store.data.read().await;
            Ok(ids
                .iter()
                .map(|id| store.get_vector_or_empty(id).clone())
                .collect())
        })
    }

    fn insert_irises<'a>(
        &'a self,
        inserts: Vec<(QueryId, VectorId)>,
    ) -> BoxFuture<'a, Result<u64>> {
        let query_cache = self.query_cache.clone();
        let iris_store = self.iris_store.clone();
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
        Box::pin(async move {
            let dummy = Arc::new(GaloisRingSharedIris::dummy_for_party(party_id));
            let mut store = iris_store.data.write().await;
            for id in ids {
                store.update(id, dummy.clone());
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

// ---------------------------------------------------------------------------
// RemoteIrisWorkerPool — drives a fleet of remote workers via the
// ampc-actor-utils workpool `LeaderHandle`. Wire types come from
// `iris-mpc-worker-protocol`. The strawman worker (and, eventually, the
// production worker binary) sit on the other end of these calls.
//
// **v1 scope: single-shard only.** `num_shards == 1` is enforced at
// construction. Once the sharding-key hash is settled with the worker
// author, scatter-gather routing of `VectorId`s and per-shard checksum
// combination land in a follow-up. Until then, broadcast and
// scatter-gather collapse to "send to the one worker we have."
// ---------------------------------------------------------------------------

/// Distance computations farmed out to remote workers over TCP.
pub struct RemoteIrisWorkerPool {
    leader: Arc<LeaderHandle>,
    /// Number of worker shards. v1 enforces == 1.
    num_shards: usize,
    /// Drives `delete_irises`'s dummy iris choice and is forwarded to
    /// workers via construction config (not on the wire).
    party_id: usize,
}

impl Debug for RemoteIrisWorkerPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteIrisWorkerPool")
            .field("num_shards", &self.num_shards)
            .field("party_id", &self.party_id)
            .finish()
    }
}

impl RemoteIrisWorkerPool {
    /// Wrap a connected `LeaderHandle` as an `IrisWorkerPool`.
    ///
    /// The caller is responsible for calling
    /// `LeaderHandle::wait_for_all_connections` *before* wrapping in `Arc`,
    /// since that method requires `&mut`. After this returns, the pool is
    /// ready to dispatch.
    pub fn new(leader: Arc<LeaderHandle>, num_shards: usize, party_id: usize) -> Self {
        assert_eq!(
            num_shards, 1,
            "RemoteIrisWorkerPool currently requires num_shards == 1; \
             multi-shard routing lands in a follow-up PR"
        );
        Self {
            leader,
            num_shards,
            party_id,
        }
    }

    pub fn party_id(&self) -> usize {
        self.party_id
    }

    pub fn num_shards(&self) -> usize {
        self.num_shards
    }
}

/// Broadcast `req` to every worker, decode each response with `extract`,
/// and fail loudly on any worker error / protocol error / wrong variant.
async fn broadcast_each<F, T>(
    leader: &LeaderHandle,
    req: wire::WorkerRequest,
    method_name: &'static str,
    mut extract: F,
) -> Result<Vec<T>>
where
    F: FnMut(wire::WorkerResponse) -> Result<T>,
{
    let bytes = wire::encode_request(&req).map_err(|e| eyre!("encode {method_name}: {e}"))?;
    let job = leader
        .broadcast(Bytes::from(bytes))
        .await
        .map_err(|e| eyre!("broadcast {method_name}: {e:?}"))?;
    let responses = job
        .await
        .map_err(|e| eyre!("workpool {method_name}: {e:?}"))?;
    let mut out = Vec::with_capacity(responses.len());
    for rsp in responses {
        let payload = rsp
            .payload
            .map_err(|e| eyre!("worker {method_name}: {e:?}"))?;
        let response = wire::decode_response(&payload.to_bytes())
            .map_err(|e| eyre!("decode {method_name}: {e}"))?;
        if let wire::WorkerResponse::ProtocolError(msg) = &response {
            return Err(eyre!("{method_name} protocol error: {msg}"));
        }
        out.push(extract(response)?);
    }
    Ok(out)
}

fn share_to_arc_iris(s: wire::IrisShare) -> ArcIris {
    Arc::new(GaloisRingSharedIris {
        code: s.code,
        mask: s.mask,
    })
}

fn arc_iris_to_share(arc: &ArcIris) -> wire::IrisShare {
    wire::IrisShare {
        code: arc.code.clone(),
        mask: arc.mask.clone(),
    }
}

fn qid_to_wire(q: QueryId) -> wire::QueryId {
    wire::QueryId(q.0)
}

fn spec_to_wire(s: QuerySpec) -> wire::QuerySpec {
    let rotation =
        u8::try_from(s.rotation).expect("rotation index fits in u8 (max 30 in the codebase)");
    wire::QuerySpec {
        query_id: qid_to_wire(s.query_id),
        rotation,
        mirrored: s.mirrored,
    }
}

fn wrap_u16_vec(v: Vec<u16>) -> Vec<RingElement<u16>> {
    v.into_iter().map(RingElement).collect()
}

impl IrisWorkerPool for RemoteIrisWorkerPool {
    fn cache_queries<'a>(&'a self, queries: Vec<(QueryId, ArcIris)>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let wire_queries = queries
                .into_iter()
                .map(|(qid, iris)| (qid_to_wire(qid), arc_iris_to_share(&iris)))
                .collect();
            let req = wire::WorkerRequest::CacheQueries {
                queries: wire_queries,
            };
            broadcast_each(
                &self.leader,
                req,
                "cache_queries",
                |response| match response {
                    wire::WorkerResponse::CacheQueries(Ok(())) => Ok(()),
                    wire::WorkerResponse::CacheQueries(Err(msg)) => {
                        Err(eyre!("cache_queries: {msg}"))
                    }
                    other => Err(eyre!("cache_queries: unexpected variant {other:?}")),
                },
            )
            .await?;
            Ok(())
        })
    }

    fn compute_dot_products<'a>(
        &'a self,
        batches: Vec<(QuerySpec, Vec<VectorId>)>,
    ) -> BoxFuture<'a, Result<Vec<Vec<RingElement<u16>>>>> {
        Box::pin(async move {
            // Single-shard v1: all batches go to the one worker as-is.
            // Multi-shard v2 will partition each batch's `VectorId`s by
            // shard and reassemble per-target outputs back into input
            // order; the hash function is part of the worker-author
            // contract.
            let wire_batches = batches
                .into_iter()
                .map(|(spec, vids)| (spec_to_wire(spec), vids))
                .collect();
            let req = wire::WorkerRequest::ComputeDotProducts {
                batches: wire_batches,
            };
            let mut all =
                broadcast_each(
                    &self.leader,
                    req,
                    "compute_dot_products",
                    |response| match response {
                        wire::WorkerResponse::ComputeDotProducts(Ok(batches)) => Ok(batches),
                        wire::WorkerResponse::ComputeDotProducts(Err(msg)) => {
                            Err(eyre!("compute_dot_products: {msg}"))
                        }
                        other => Err(eyre!("compute_dot_products: unexpected variant {other:?}")),
                    },
                )
                .await?;
            // v1: one worker → one set of batches. Take it directly.
            let batches = all
                .pop()
                .ok_or_else(|| eyre!("compute_dot_products: no responses"))?;
            Ok(batches.into_iter().map(wrap_u16_vec).collect())
        })
    }

    fn fetch_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<Vec<ArcIris>>> {
        Box::pin(async move {
            let req = wire::WorkerRequest::FetchIrises { ids };
            let mut all =
                broadcast_each(
                    &self.leader,
                    req,
                    "fetch_irises",
                    |response| match response {
                        wire::WorkerResponse::FetchIrises(Ok(irises)) => Ok(irises),
                        wire::WorkerResponse::FetchIrises(Err(msg)) => {
                            Err(eyre!("fetch_irises: {msg}"))
                        }
                        other => Err(eyre!("fetch_irises: unexpected variant {other:?}")),
                    },
                )
                .await?;
            let irises = all
                .pop()
                .ok_or_else(|| eyre!("fetch_irises: no responses"))?;
            Ok(irises.into_iter().map(share_to_arc_iris).collect())
        })
    }

    fn insert_irises<'a>(
        &'a self,
        inserts: Vec<(QueryId, VectorId)>,
    ) -> BoxFuture<'a, Result<u64>> {
        Box::pin(async move {
            let wire_inserts = inserts
                .into_iter()
                .map(|(qid, vid)| (qid_to_wire(qid), vid))
                .collect();
            let req = wire::WorkerRequest::InsertIrises {
                inserts: wire_inserts,
            };
            let mut all =
                broadcast_each(
                    &self.leader,
                    req,
                    "insert_irises",
                    |response| match response {
                        wire::WorkerResponse::InsertIrises(Ok(checksum)) => Ok(checksum),
                        wire::WorkerResponse::InsertIrises(Err(msg)) => {
                            Err(eyre!("insert_irises: {msg}"))
                        }
                        other => Err(eyre!("insert_irises: unexpected variant {other:?}")),
                    },
                )
                .await?;
            // v1: single shard → single checksum.
            // v2 with multi-shard will need to combine per-shard
            // `set_hash` checksums; combining policy is TBD and depends
            // on the SetHash algebra (XOR if it's order-independent).
            all.pop()
                .ok_or_else(|| eyre!("insert_irises: no responses"))
        })
    }

    fn compute_pairwise_distances<'a>(
        &'a self,
        pairs: Vec<Option<(QuerySpec, QueryId)>>,
    ) -> BoxFuture<'a, Result<Vec<RingElement<u16>>>> {
        Box::pin(async move {
            let wire_pairs = pairs
                .into_iter()
                .map(|p| p.map(|(spec, qid)| (spec_to_wire(spec), qid_to_wire(qid))))
                .collect();
            let req = wire::WorkerRequest::ComputePairwiseDistances { pairs: wire_pairs };
            let mut all = broadcast_each(
                &self.leader,
                req,
                "compute_pairwise_distances",
                |response| match response {
                    wire::WorkerResponse::ComputePairwiseDistances(Ok(values)) => Ok(values),
                    wire::WorkerResponse::ComputePairwiseDistances(Err(msg)) => {
                        Err(eyre!("compute_pairwise_distances: {msg}"))
                    }
                    other => Err(eyre!(
                        "compute_pairwise_distances: unexpected variant {other:?}"
                    )),
                },
            )
            .await?;
            let values = all
                .pop()
                .ok_or_else(|| eyre!("compute_pairwise_distances: no responses"))?;
            Ok(wrap_u16_vec(values))
        })
    }

    fn evict_queries<'a>(&'a self, query_ids: Vec<QueryId>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let wire_ids = query_ids.into_iter().map(qid_to_wire).collect();
            let req = wire::WorkerRequest::EvictQueries {
                query_ids: wire_ids,
            };
            broadcast_each(
                &self.leader,
                req,
                "evict_queries",
                |response| match response {
                    wire::WorkerResponse::EvictQueries(Ok(())) => Ok(()),
                    wire::WorkerResponse::EvictQueries(Err(msg)) => {
                        Err(eyre!("evict_queries: {msg}"))
                    }
                    other => Err(eyre!("evict_queries: unexpected variant {other:?}")),
                },
            )
            .await?;
            Ok(())
        })
    }

    fn delete_irises<'a>(&'a self, ids: Vec<VectorId>) -> BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            let req = wire::WorkerRequest::DeleteIrises { ids };
            broadcast_each(
                &self.leader,
                req,
                "delete_irises",
                |response| match response {
                    wire::WorkerResponse::DeleteIrises(Ok(())) => Ok(()),
                    wire::WorkerResponse::DeleteIrises(Err(msg)) => {
                        Err(eyre!("delete_irises: {msg}"))
                    }
                    other => Err(eyre!("delete_irises: unexpected variant {other:?}")),
                },
            )
            .await?;
            Ok(())
        })
    }
}
