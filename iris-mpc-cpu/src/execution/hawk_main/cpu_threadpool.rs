use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use crossbeam::channel::{Receiver, Sender};
use tokio::sync::oneshot;

use crate::{
    hawkers::aby3::aby3_store::QueryInput, protocol::ops::galois_ring_pairwise_distance,
    shares::RingElement,
};

#[derive(Debug)]
enum CpuTask {
    RingPairwiseDistance {
        input: Vec<Option<(QueryInput, QueryInput)>>,
        rsp: oneshot::Sender<Vec<RingElement<u16>>>,
    },
}

#[derive(Clone, Debug)]
pub struct CpuWorkerHandle {
    workers: Arc<Vec<Sender<CpuTask>>>,
    next_counter: Arc<AtomicU64>,
}

impl CpuWorkerHandle {
    pub async fn galois_ring_pairwise_distances(
        &self,
        input: Vec<Option<(QueryInput, QueryInput)>>,
    ) -> Vec<RingElement<u16>> {
        let (tx, rx) = oneshot::channel();
        let task = CpuTask::RingPairwiseDistance { input, rsp: tx };
        let _ = self.get_next_worker().send(task);
        rx.await.unwrap()
    }

    fn get_next_worker(&self) -> &Sender<CpuTask> {
        // fetch_add() wraps around on overflow
        let idx = self.next_counter.fetch_add(1, Ordering::Relaxed) as usize;
        let idx = idx % self.workers.len();
        &self.workers[idx]
    }
}

pub fn init_workers(num_workers: usize) -> CpuWorkerHandle {
    // context switching is bad for CPU bound work because it invalidates the cache.
    // ensure the CPU workers don't context switch.
    let mut core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.reverse();
    assert!(core_ids.len() >= 1);

    // need to use at least one core
    let cores_to_use = std::cmp::max(
        1,
        // minus one to leave core 0 alone.
        // other stuff probably gets scheduled on core 0
        std::cmp::min(num_workers, core_ids.len().saturating_sub(1)),
    );

    let mut channels = vec![];
    for &core_id in core_ids.iter().take(cores_to_use) {
        let (tx, rx) = crossbeam::channel::unbounded::<CpuTask>();
        channels.push(tx);
        std::thread::spawn(move || {
            let _ = core_affinity::set_for_current(core_id);
            worker_thread(rx);
        });
    }

    CpuWorkerHandle {
        workers: Arc::new(channels),
        next_counter: Arc::new(AtomicU64::new(0)),
    }
}

fn worker_thread(ch: Receiver<CpuTask>) {
    while let Ok(task) = ch.recv() {
        match task {
            CpuTask::RingPairwiseDistance { input, rsp } => {
                let r = galois_ring_pairwise_distance(input);
                let _ = rsp.send(r);
            }
        }
    }
}
