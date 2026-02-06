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

pub use ampc_secret_sharing::galois;
pub use ampc_secret_sharing::id;
pub use vector_id::SerialId as IrisSerialId;
pub use vector_id::VectorId as IrisVectorId;
pub use vector_id::VersionId as IrisVersionId;

use std::fs;
use std::sync::LazyLock;

/// Caches the number of NUMA nodes detected on the system.
/// Defaults to 1 if not on Linux or if detection fails.
pub static SHARD_COUNT: LazyLock<usize> = LazyLock::new(|| {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            let count = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("node"))
                .count();

            // Handle edge case where dir exists but is empty
            return if count > 0 { count } else { 1 };
        }
    }
    1
});

// uses libc to avoid the need to install numa specific tools on the target
#[cfg(target_os = "linux")]
pub fn restrict_to_node_zero() {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, MPOL_BIND};
    use nix::libc::{self, c_int, c_ulong};
    use std::mem;

    // Define raw syscall wrapper if not provided by libc
    // mode: e.g., MPOL_BIND, MPOL_INTERLEAVE
    // nodemask: pointer to a bitmask of nodes
    // maxnode: number of nodes in the mask
    extern "C" {
        fn set_mempolicy(mode: c_int, nodemask: *const c_ulong, maxnode: c_ulong) -> c_int;
    }

    unsafe {
        let mut cpuset: cpu_set_t = mem::zeroed();
        for core in 0..get_node_zero_cores() {
            CPU_SET(core, &mut cpuset);
        }

        if sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpuset) != 0 {
            eprintln!("Warning: Failed to set CPU affinity");
        }

        // This prevents RAM allocations from "bleeding" into Node 1
        let nodemask: libc::c_ulong = 1 << 0; // Bit 0 = Node 0
        let maxnode: libc::c_ulong = *SHARD_COUNT as _; // Total nodes in the mask

        let res = set_mempolicy(MPOL_BIND, &nodemask, maxnode);

        if res != 0 {
            eprintln!("Warning: set_mempolicy syscall failed. Check permissions/capabilities.");
        }
    }
}

// On Mac or other OSs, this function becomes a "No-Op" (does nothing)
#[cfg(not(target_os = "linux"))]
pub fn restrict_to_node_zero() {
    // macOS uses Unified Memory; NUMA pinning isn't applicable/available via libc
}

/// assumes the NIC and the first half of the CPU cores are on NUMA node 0, and restricts tokio to that node. for a single node system, there should be no effect.
pub fn get_node_zero_cores() -> usize {
    let core_ids = core_affinity::get_core_ids().unwrap();
    core_ids.len() / *SHARD_COUNT
}
