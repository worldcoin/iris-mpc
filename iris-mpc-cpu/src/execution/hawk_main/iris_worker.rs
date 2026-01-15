use crate::{
    execution::hawk_main::HAWK_MINFHD_ROTATIONS,
    hawkers::shared_irises::SharedIrisesRef,
    protocol::{
        ops::{
            galois_ring_pairwise_distance, non_existent_distance, pairwise_distance,
            rotation_aware_pairwise_distance, rotation_aware_pairwise_distance_rowmajor,
        },
        shared_iris::ArcIris,
    },
    shares::RingElement,
};
use ampc_actor_utils::fast_metrics::FastHistogram;
use core_affinity::CoreId;
use crossbeam::channel::{Receiver, Sender};
use eyre::Result;
use futures::future::try_join_all;
use iris_mpc_common::vector_id::VectorId;
use std::{
    cmp, iter,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
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
        vector_ids: Vec<VectorId>,
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
    /// A metric to record the latency of task execution.
    metric_latency: FastHistogram,
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
        &mut self,
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
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::DotProductBatch {
            query,
            vector_ids,
            rsp: tx,
        };
        self.submit(task, rx).await
    }

    pub async fn rotation_aware_dot_product_pairs(
        &mut self,
        pairs: Vec<(ArcIris, VectorId)>,
    ) -> Result<Vec<RingElement<u16>>> {
        let mut responses = Vec::new();

        for (query, id) in pairs {
            let (tx, rx) = oneshot::channel();
            let task = IrisTask::RotationAwareDotProductBatch {
                query,
                vector_ids: vec![id],
                rsp: tx,
            };
            self.get_next_worker().send(task)?;
            responses.push(rx);
        }
        let results = futures::future::try_join_all(responses).await?;
        let results = results.into_iter().flatten().collect();
        Ok(results)
    }

    pub async fn rotation_aware_dot_product_batch(
        &mut self,
        query: ArcIris,
        vector_ids: Vec<VectorId>,
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();

        const CHUNK_SIZE: usize = 128;
        let mut responses = Vec::with_capacity(vector_ids.len().div_ceil(CHUNK_SIZE));

        for chunk in vector_ids.chunks(CHUNK_SIZE) {
            let (tx, rx) = oneshot::channel();
            let task = IrisTask::RotationAwareDotProductBatch {
                query: query.clone(),
                vector_ids: chunk.to_vec(),
                rsp: tx,
            };
            self.get_next_worker().send(task)?;
            responses.push(rx);
        }

        let results = futures::future::try_join_all(responses).await?;
        let results = results.into_iter().flatten().collect();

        self.metric_latency.record(start.elapsed().as_secs_f64());
        Ok(results)
    }

    /// Computes rotation-aware dot products for multiple (query, vectors) batches.
    ///
    /// Each query's prerotation is reused across all its target vectors, making this
    /// more efficient than `rotation_aware_dot_product_pairs` when the same query
    /// is compared against multiple vectors.
    ///
    /// Returns results grouped by input batch.
    pub async fn rotation_aware_dot_product_batches(
        &mut self,
        batches: Vec<(ArcIris, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<RingElement<u16>>>> {
        let start = Instant::now();
        const CHUNK_SIZE: usize = 128;

        // Track batch_idx for each chunk to enable reassembly
        let mut chunk_batch_indices: Vec<usize> = Vec::new();
        let mut responses = Vec::new();

        for (batch_idx, (query, vector_ids)) in batches.iter().enumerate() {
            for chunk in vector_ids.chunks(CHUNK_SIZE) {
                let (tx, rx) = oneshot::channel();
                let task = IrisTask::RotationAwareDotProductBatch {
                    query: query.clone(),
                    vector_ids: chunk.to_vec(),
                    rsp: tx,
                };
                self.get_next_worker().send(task)?;
                chunk_batch_indices.push(batch_idx);
                responses.push(rx);
            }
        }

        let chunk_results = futures::future::try_join_all(responses).await?;

        // Reassemble results by batch
        let mut results: Vec<Vec<RingElement<u16>>> = batches
            .iter()
            .map(|(_, ids)| Vec::with_capacity(2 * HAWK_MINFHD_ROTATIONS * ids.len()))
            .collect();

        for (batch_idx, chunk_result) in chunk_batch_indices.into_iter().zip(chunk_results) {
            results[batch_idx].extend(chunk_result);
        }

        self.metric_latency.record(start.elapsed().as_secs_f64());
        Ok(results)
    }

    pub async fn bench_batch_dot(
        &mut self,
        per_worker: usize,
        query: ArcIris,
        vector_ids: Vec<VectorId>,
    ) -> Result<Vec<RingElement<u16>>> {
        let mut responses = Vec::with_capacity(vector_ids.len() / per_worker);
        for vector_id_chunk in vector_ids.chunks(per_worker) {
            let (tx, rx) = oneshot::channel();
            let task = IrisTask::RotationAwareDotProductBatch {
                query: query.clone(),
                vector_ids: vector_id_chunk.to_vec(),
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
        &mut self,
        input: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::RingPairwiseDistance { input, rsp: tx };
        self.submit(task, rx).await
    }

    pub async fn rotation_aware_pairwise_distances(
        &mut self,
        pairs: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let mut responses = Vec::new();
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
        &mut self,
        task: IrisTask,
        rx: oneshot::Receiver<Vec<RingElement<u16>>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();

        self.get_next_worker().send(task)?;
        let res = rx.await?;

        self.metric_latency.record(start.elapsed().as_secs_f64());
        Ok(res)
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
        metric_latency: FastHistogram::new("iris_worker.latency"),
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
                rsp,
            } => {
                let store = iris_store.data.blocking_read();
                let targets = vector_ids.iter().map(|v| store.get_vector(v));
                let result = rotation_aware_pairwise_distance_rowmajor::<HAWK_MINFHD_ROTATIONS, _>(
                    &query, targets,
                );
                let _ = rsp.send(result);
            }

            IrisTask::RingPairwiseDistance { input, rsp } => {
                let r = galois_ring_pairwise_distance(input);
                let _ = rsp.send(r);
            }

            IrisTask::RotationAwarePairwiseDistance { pair, rsp } => {
                let r = rotation_aware_pairwise_distance::<HAWK_MINFHD_ROTATIONS, _>(
                    &pair.0,
                    iter::once(Some(&pair.1)),
                );
                let _ = rsp.send(r);
            }
        }
    }
}

const SHARD_COUNT: usize = 2;

pub fn select_core_ids(shard_index: usize) -> Vec<CoreId> {
    let mut core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.sort();
    assert!(!core_ids.is_empty());

    let shard_count = cmp::min(SHARD_COUNT, core_ids.len());
    let shard_index = shard_index % shard_count;

    let shard_size = core_ids.len() / shard_count;
    let start = shard_index * shard_size;
    let end = cmp::min(start + shard_size, core_ids.len());

    core_ids[start..end].to_vec()
}
