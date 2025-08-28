use crate::{
    hawkers::shared_irises::SharedIrisesRef,
    protocol::{ops::galois_ring_pairwise_distance, shared_iris::ArcIris},
    shares::RingElement,
};
use core_affinity::CoreId;
use crossbeam::channel::{Receiver, Sender};
use eyre::Result;
use iris_mpc_common::vector_id::VectorId;
use itertools::Itertools;
use metrics::histogram;
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
    /// Move an iris code to memory closer to the pool (NUMA-awareness).
    Realloc {
        iris: ArcIris,
        rsp: oneshot::Sender<ArcIris>,
    },
    Insert {
        vector_id: VectorId,
        iris: ArcIris,
        rsp: oneshot::Sender<VectorId>,
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
    RingPairwiseDistance {
        input: Vec<Option<(ArcIris, ArcIris)>>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
}

#[derive(Clone, Debug)]
pub struct IrisPoolHandle {
    workers: Arc<Vec<Sender<IrisTask>>>,
    next_counter: Arc<AtomicU64>,
}

impl IrisPoolHandle {
    pub fn realloc(&self, iris: ArcIris) -> Result<oneshot::Receiver<ArcIris>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::Realloc { iris, rsp: tx };
        self.get_next_worker().send(task)?;
        Ok(rx)
    }

    pub async fn insert(&self, vector_id: VectorId, iris: ArcIris) -> Result<VectorId> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::Insert {
            vector_id,
            iris,
            rsp: tx,
        };
        self.get_next_worker().send(task)?;
        Ok(rx.await?)
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
        &self,
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

    pub async fn galois_ring_pairwise_distances(
        &self,
        input: Vec<Option<(ArcIris, ArcIris)>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let (tx, rx) = oneshot::channel();
        let task = IrisTask::RingPairwiseDistance { input, rsp: tx };
        self.submit(task, rx).await
    }

    async fn submit(
        &self,
        task: IrisTask,
        rx: oneshot::Receiver<Vec<RingElement<u16>>>,
    ) -> Result<Vec<RingElement<u16>>> {
        let start = Instant::now();

        self.get_next_worker().send(task)?;
        let res = rx.await?;

        histogram!("iris_worker.latency", "histogram" => "histogram")
            .record(start.elapsed().as_secs_f64());
        Ok(res)
    }

    fn get_next_worker(&self) -> &Sender<IrisTask> {
        // fetch_add() wraps around on overflow
        let idx = self.next_counter.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = idx % self.workers.len();
        &self.workers[idx]
    }
}

pub fn init_workers(shard_index: usize, iris_store: SharedIrisesRef<ArcIris>) -> IrisPoolHandle {
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
            worker_thread(rx, iris_store);
        });
    }

    IrisPoolHandle {
        workers: Arc::new(channels),
        next_counter: Arc::new(AtomicU64::new(0)),
    }
}

fn worker_thread(ch: Receiver<IrisTask>, iris_store: SharedIrisesRef<ArcIris>) {
    while let Ok(task) = ch.recv() {
        match task {
            IrisTask::Realloc { iris, rsp } => {
                // Re-allocate from this thread.
                // This attempts to use the NUMA-aware first-touch policy of the OS.
                let new_iris = Arc::new((*iris).clone());
                let _ = rsp.send(new_iris);
            }

            IrisTask::Insert {
                vector_id,
                iris,
                rsp,
            } => {
                let mut store = iris_store.data.blocking_write();
                let vector_id = store.insert(vector_id, iris);
                let _ = rsp.send(vector_id);
            }

            IrisTask::DotProductPairs { pairs, rsp } => {
                let pairs = {
                    let store = iris_store.data.blocking_read();

                    pairs
                        .into_iter()
                        .map(|(q, vid)| {
                            let vector = store.get_vector(&vid);
                            vector.map(|v| (q, v))
                        })
                        .collect_vec()
                };

                let r = galois_ring_pairwise_distance(pairs);
                let _ = rsp.send(r);
            }

            IrisTask::DotProductBatch {
                query,
                vector_ids,
                rsp,
            } => {
                let pairs = {
                    let store = iris_store.data.blocking_read();

                    vector_ids
                        .iter()
                        .map(|v| store.get_vector(v).map(|v| (query.clone(), v)))
                        .collect_vec()
                };

                let r = galois_ring_pairwise_distance(pairs);
                let _ = rsp.send(r);
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
    let core_ids = core_affinity::get_core_ids().unwrap();
    assert!(!core_ids.is_empty());

    let shard_count = cmp::min(SHARD_COUNT, core_ids.len());
    let shard_index = shard_index % shard_count;

    let shard_size = core_ids.len() / shard_count;
    let start = shard_index * shard_size;
    let end = cmp::min(start + shard_size, core_ids.len());

    core_ids[start..end].to_vec()
}
