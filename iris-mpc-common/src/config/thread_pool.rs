use core_affinity::CoreId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

// this is currently just being used to pin tokio threads to cores

static TOKIO_CORES: LazyLock<Vec<CoreId>> = LazyLock::new(get_tokio_core_ids);
static TOKIO_THREADS: AtomicUsize = AtomicUsize::new(0);

// these functions are used to build a tokio runtime
pub fn num_tokio_cores() -> usize {
    TOKIO_CORES.len()
}

pub fn pin_next_tokio_core_id() {
    let i = TOKIO_THREADS.fetch_add(1, Ordering::Relaxed);
    let core = i % num_tokio_cores();
    let _ = core_affinity::set_for_current(TOKIO_CORES[core]);
}

// hack: really only need 1 thread pool
pub fn get_iris_worker_core_ids(half_idx: usize) -> Vec<CoreId> {
    let mut core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.sort();
    assert!(!core_ids.is_empty());

    let num_worker_cores = get_num_iris_worker_cores();
    let half = num_worker_cores / 2;

    core_ids
        .into_iter()
        //        .rev()
        .skip(half * half_idx)
        .take(half)
        .collect()
}

fn get_num_tokio_cores() -> usize {
    let half = num_cpus::get() / 2;
    // leave some cores for the OS i guess
    std::cmp::max(4, half.saturating_sub(4))
}

fn get_num_iris_worker_cores() -> usize {
    num_cpus::get() / 2
}

fn get_tokio_core_ids() -> Vec<CoreId> {
    let mut core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.sort();
    assert!(!core_ids.is_empty());

    core_ids
        .into_iter()
        .rev()
        .take(get_num_tokio_cores())
        .collect()
}
