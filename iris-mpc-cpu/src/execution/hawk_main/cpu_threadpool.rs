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
    ch: Sender<CpuTask>,
}

impl CpuWorkerHandle {
    pub async fn galois_ring_pairwise_distances(
        &self,
        input: Vec<Option<(QueryInput, QueryInput)>>,
    ) -> Vec<RingElement<u16>> {
        let (tx, rx) = oneshot::channel();
        let task = CpuTask::RingPairwiseDistance { input, rsp: tx };
        let _ = self.ch.send(task);
        rx.await.unwrap()
    }
}

pub fn init_workers(num_workers: usize) -> CpuWorkerHandle {
    // context switching is bad for CPU bound work because it invalidates the cache.
    // ensure the CPU workers don't context switch.
    let mut core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.reverse();

    // minus one to leave core 0 alone.
    // other stuff probably gets scheduled on core 0
    let mut it = core_ids
        .iter()
        .take(std::cmp::min(num_workers, core_ids.len() - 1));

    let scheduler_core_id = *it.next().unwrap();

    let mut channels = vec![];
    for &core_id in it {
        let (tx, rx) = crossbeam::channel::unbounded::<CpuTask>();
        channels.push(tx);
        std::thread::spawn(move || {
            let _ = core_affinity::set_for_current(core_id);
            worker_thread(rx);
        });
    }

    let (tx, rx) = crossbeam::channel::unbounded::<CpuTask>();
    std::thread::spawn(move || {
        let _ = core_affinity::set_for_current(scheduler_core_id);
        scheduler_thread(rx, channels);
    });

    CpuWorkerHandle { ch: tx }
}

fn scheduler_thread(cmd_ch: Receiver<CpuTask>, workers: Vec<Sender<CpuTask>>) {
    let mut worker_idx = 0;
    while let Ok(task) = cmd_ch.recv() {
        let _ = workers[worker_idx].send(task);
        worker_idx = (worker_idx + 1) % workers.len();
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
