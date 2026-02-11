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
            // return if count > 0 { count } else { 1 };
        }
    }
    1
});

/// Caches the CPU IDs for NUMA node 0.
pub static NODE_ZERO_CPUS: LazyLock<Vec<usize>> = LazyLock::new(|| get_cpus_for_node(0));

/// Parses a Linux cpulist format string (e.g., "0-15,32-47") into a vector of CPU IDs.
fn parse_cpulist(cpulist: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in cpulist.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(s), Ok(e)) = (start.parse::<usize>(), end.parse::<usize>()) {
                cpus.extend(s..=e);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Returns the CPU IDs belonging to the specified NUMA node.
/// On non-Linux or if detection fails, returns all available CPU IDs for node 0,
/// or an empty vec for other nodes.
pub fn get_cpus_for_node(node: usize) -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        let path = format!("/sys/devices/system/node/node{}/cpulist", node);
        if let Ok(contents) = fs::read_to_string(&path) {
            return parse_cpulist(&contents);
        }
    }

    // Fallback for non-Linux or if sysfs read fails:
    // Node 0 gets all CPUs, other nodes get none
    if node == 0 {
        core_affinity::get_core_ids()
            .unwrap_or_default()
            .into_iter()
            .map(|c| c.id)
            .collect()
    } else {
        Vec::new()
    }
}

// uses libc to avoid the need to install numa specific tools on the target
#[cfg(target_os = "linux")]
pub fn restrict_to_node_zero() {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET};
    use nix::libc::{self};
    use std::io::Error;
    use std::mem;

    unsafe {
        let mut cpuset: cpu_set_t = mem::zeroed();
        for &cpu in NODE_ZERO_CPUS.iter() {
            CPU_SET(cpu, &mut cpuset);
        }

        if sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpuset) != 0 {
            let err = Error::last_os_error();
            eprintln!("Warning: Failed to set CPU affinity: {}", err);
        }

        // set_mempolicy_for_node(0);
    }
}

// On Mac or other OSs, this function becomes a no-op
#[cfg(not(target_os = "linux"))]
pub fn restrict_to_node_zero() {
    // macOS uses Unified Memory; NUMA pinning isn't applicable/available via libc
}

/// Returns the number of CPU cores on NUMA node 0.
/// For a single-node system, this returns the total number of cores.
pub fn get_node_zero_cores() -> usize {
    NODE_ZERO_CPUS.len()
}

/// Sets the memory policy for the current thread to bind allocations to the specified NUMA node.
/// On non-Linux systems, this is a no-op.
#[allow(dead_code)]
#[cfg(target_os = "linux")]
pub fn set_mempolicy_for_node(node: usize) {
    use libc::MPOL_BIND;
    use nix::libc;
    use std::io::Error;

    unsafe {
        // Set bit for the target node
        let nodemask: libc::c_ulong = 1 << node;
        // maxnode must be > highest node number
        let maxnode: libc::c_ulong = 2;

        let res = libc::syscall(
            libc::SYS_set_mempolicy,
            MPOL_BIND,
            &nodemask as *const libc::c_ulong,
            maxnode,
        );

        if res != 0 {
            let err = Error::last_os_error();
            eprintln!("Warning: set_mempolicy for node {} failed: {}", node, err);
        }
    }
}

#[allow(dead_code)]
#[cfg(not(target_os = "linux"))]
pub fn set_mempolicy_for_node(_node: usize) {
    // No-op on non-Linux systems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cpulist_simple_range() {
        let cpus = parse_cpulist("0-95");
        assert_eq!(cpus.len(), 96);
        assert_eq!(cpus, (0..=95).collect::<Vec<_>>());
    }

    #[test]
    fn test_parse_cpulist_multiple_ranges() {
        let cpus = parse_cpulist("0-15,32-47");
        let expected: Vec<usize> = (0..=15).chain(32..=47).collect();
        assert_eq!(cpus, expected);
    }

    #[test]
    fn test_parse_cpulist_single_values() {
        let cpus = parse_cpulist("0,5,10");
        assert_eq!(cpus, vec![0, 5, 10]);
    }

    #[test]
    fn test_parse_cpulist_mixed() {
        let cpus = parse_cpulist("0-3,8,12-14");
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 12, 13, 14]);
    }

    #[test]
    fn test_parse_cpulist_with_whitespace() {
        let cpus = parse_cpulist("  0-3 \n");
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_parse_cpulist_empty() {
        let cpus = parse_cpulist("");
        assert!(cpus.is_empty());
    }
}
