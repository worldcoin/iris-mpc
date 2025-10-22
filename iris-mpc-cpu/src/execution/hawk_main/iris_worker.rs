use crate::{
    hawkers::shared_irises::SharedIrisesRef,
    protocol::{
        ops::{galois_ring_pairwise_distance, pairwise_distance, rotation_aware_pairwise_distance},
        shared_iris::ArcIris,
    },
    shares::RingElement,
};
use core_affinity::CoreId;
use crossbeam::channel::{Receiver, Sender};
use eyre::Result;
use futures::future::try_join_all;
use iris_mpc_common::{
    fast_metrics::FastHistogram, vector_id::VectorId, MIN_DOT_BATCH_SIZE, ROTATIONS,
};
use std::{
    cmp,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
use tokio::sync::oneshot;
use tracing::info;

#[derive(Debug)]
enum IrisTask {
    Sync {
        rsp: oneshot::Sender<()>,
    },
    /// Move an iris code to memory closer to the pool (NUMA-awareness).
    Realloc {
        iris: ArcIris,
        rsp: oneshot::Sender<ArcIris>,
    },
    Insert {
        vector_id: VectorId,
        iris: ArcIris,
    },
    Reserve {
        additional: usize,
    },
    DotProductPairs {
        pairs: Vec<(ArcIris, VectorId)>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    DotProductBatch {
        query: ArcIris,
        vector_ids: Vec<VectorId>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    RotationAwareDotProductBatch {
        query: ArcIris,
        vector_ids: Arc<Vec<VectorId>>,
        // the worker threads write to non-overlapping parts of the result
        result: Arc<Vec<RingElement<u16>>>,
        input_offset: usize,
        input_len: usize,
        rsp: oneshot::Sender<()>,
    },
    // potential candidate to replace RotationAwareDotProductBatch
    BenchBatchDot {
        query: ArcIris,
        vector_ids: Vec<VectorId>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
    RingPairwiseDistance {
        input: Vec<Option<(ArcIris, ArcIris)>>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
}

#[derive(Clone, Debug)]
pub struct IrisPoolHandle {
    workers: Arc<[Sender<IrisTask>]>,
    next_counter: Arc<AtomicU64>,
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

    pub async fn rotation_aware_dot_product_batch(
        &mut self,
        query: ArcIris,
        vector_ids: Vec<VectorId>,
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();

        // result will be unsafely handled by the worker threads.
        // would use an UnsafeCell but that is not Send.
        // getting a slice from result seems better than just sending a raw pointer.
        let result = Arc::new(vec![RingElement(0_u16); vector_ids.len() * ROTATIONS * 2]);
        let vector_ids = Arc::new(vector_ids);
        let input_len = vector_ids.len();
        let num_slices = input_len.div_ceil(MIN_DOT_BATCH_SIZE);

        // Compute base size and remainder
        let base_size = input_len / num_slices;
        let remainder = input_len % num_slices;

        let mut input_idx = 0;
        let mut responses = Vec::with_capacity(num_slices);
        for i in 0..num_slices {
            let (tx, rx) = oneshot::channel();

            // Distribute the remainder to the first slices
            let elements_to_process = base_size + if i < remainder { 1 } else { 0 };

            let task = IrisTask::RotationAwareDotProductBatch {
                query: query.clone(),
                vector_ids: vector_ids.clone(),
                result: result.clone(),
                input_offset: input_idx,
                input_len: elements_to_process,
                rsp: tx,
            };
            self.get_next_worker().send(task)?;
            responses.push(rx);

            input_idx += elements_to_process;
        }

        let _results = futures::future::join_all(responses)
            .await
            .into_iter()
            .collect::<Result<Vec<()>, _>>()?;

        let result =
            Arc::try_unwrap(result).expect("Failed to unwrap Arc: other references still exist");

        self.metric_latency.record(start.elapsed().as_secs_f64());
        Ok(result)
    }

    pub async fn bench_batch_dot(
        &mut self,
        per_worker: usize,
        inputs: Vec<(ArcIris, Vec<VectorId>)>,
    ) -> Result<Vec<Vec<RingElement<u16>>>> {
        let mut responses = Vec::with_capacity(inputs.len());
        for (query, vector_ids) in inputs.into_iter() {
            let num_tasks = vector_ids.len().div_ceil(per_worker);
            for vector_id_chunk in vector_ids.chunks(per_worker) {
                let (tx, rx) = oneshot::channel();
                let task = IrisTask::BenchBatchDot {
                    query: query.clone(),
                    vector_ids: vector_id_chunk.to_vec(),
                    rsp: tx,
                };
                self.get_next_worker().send(task)?;
                responses.push(rx);
            }
        }

        let results = futures::future::try_join_all(responses).await?;
        Ok(results)
    }

    pub async fn galois_ring_pairwise_distances(
        &mut self,
        input: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::RingPairwiseDistance { input, rsp: tx };
        self.submit(task, rx).await
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
                result,
                input_offset,
                input_len,
                rsp,
            } => {
                let input_slice = &vector_ids[input_offset..input_offset + input_len];

                let p_result = result.as_ptr() as *mut RingElement<u16>;
                let multiplier = ROTATIONS * 2;

                // safety: the worker threads must write to non-overlapping regions of result
                // and result must not be used until all the worker threads are finished.
                let r_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        p_result.add(input_offset * multiplier),
                        (input_offset + input_len) * multiplier,
                    )
                };

                let store = iris_store.data.blocking_read();
                let targets = input_slice.iter().map(|v| store.get_vector(v));
                rotation_aware_pairwise_distance(&query, targets, r_slice);
                drop(vector_ids);
                drop(result);
                let _ = rsp.send(());
            }

            IrisTask::BenchBatchDot {
                query,
                vector_ids,
                rsp,
            } => {
                let store = iris_store.data.blocking_read();
                let targets = vector_ids.iter().map(|v| store.get_vector(v));
                let mut result = vec![RingElement(0); vector_ids.len() * ROTATIONS * 2];
                rotation_aware_pairwise_distance(&query, targets, &mut result);
                let _ = rsp.send(result);
            }

            IrisTask::RingPairwiseDistance { input, rsp } => {
                let r = galois_ring_pairwise_distance(input);
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
