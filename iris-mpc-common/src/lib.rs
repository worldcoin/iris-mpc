#![allow(clippy::needless_range_loop)]
pub mod config;
pub mod error;
pub mod galois_engine;
pub mod helpers;
pub mod iris_db;
pub mod job;
pub mod postgres;
pub mod shamir;
#[cfg(feature = "helpers")]
pub mod test;
pub mod tracing;
pub mod vector_id;

pub const IRIS_CODE_LENGTH: usize = 12_800;
pub const MASK_CODE_LENGTH: usize = 6_400;
pub const ROTATIONS: usize = 31;

pub const PRE_PROC_ROW_PADDING: usize = 120;
pub const IRIS_CODE_ROWS: usize = 16;
// 16 = 12800 / 800 = (IRIS_CODE_LENGTH) / (CODE_COLS * 4)
pub const PRE_PROC_IRIS_CODE_LENGTH: usize =
    IRIS_CODE_LENGTH + (IRIS_CODE_ROWS * PRE_PROC_ROW_PADDING);
pub const PRE_PROC_MASK_CODE_LENGTH: usize = MASK_CODE_LENGTH + (8 * PRE_PROC_ROW_PADDING);

/// Iris code database type; .0 = iris code, .1 = mask
pub type IrisCodeDb = (Vec<u16>, Vec<u16>);
/// Borrowed version of iris database; .0 = iris code, .1 = mask
pub type IrisCodeDbSlice<'a> = (&'a [u16], &'a [u16]);

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

pub use ampc_secret_sharing::galois;
pub use ampc_secret_sharing::id;
pub use vector_id::SerialId as IrisSerialId;
pub use vector_id::VectorId as IrisVectorId;
pub use vector_id::VersionId as IrisVersionId;

/// Static counter that increments each time `next_worker_index` is called,
/// cycling through 0..num_workers. Used for round-robin worker selection.
static WORKER_CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Returns the next worker index, cycling from 0 to num_workers-1.
/// Each call increments the counter and returns the previous value mod num_workers.
pub fn next_worker_index(num_workers: usize) -> usize {
    if num_workers == 0 {
        return 0;
    }
    WORKER_CALL_COUNTER.fetch_add(1, Ordering::Relaxed) % num_workers
}

pub fn get_num_tokio_threads() -> usize {
    let core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.len() / 2
}
