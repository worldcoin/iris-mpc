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

/// Returns a list of all available NUMA node IDs on the system.
/// On non-Linux or if detection fails, returns vec![0].
pub fn get_numa_nodes() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
            let mut nodes: Vec<usize> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let name = e.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("node") {
                        name_str.strip_prefix("node")?.parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .collect();

            nodes.sort_unstable();
            if !nodes.is_empty() {
                return nodes;
            }
        }
    }

    vec![0]
}

/// Returns the number of CPU cores on the specified NUMA node.
pub fn get_cores_for_node(node: usize) -> usize {
    get_cpus_for_node(node).len()
}

/// Restricts the current process to run only on CPUs belonging to NUMA node 0.
/// On non-Linux systems, this is a no-op.
#[cfg(target_os = "linux")]
pub fn restrict_to_node_zero() {
    restrict_to_node(0);
}

// On Mac or other OSs, this function becomes a no-op
#[cfg(not(target_os = "linux"))]
pub fn restrict_to_node_zero() {
    // macOS uses Unified Memory; NUMA pinning isn't applicable/available via libc
}

/// Restricts the current process to run only on CPUs belonging to the specified NUMA node.
/// On non-Linux systems, this is a no-op.
#[cfg(target_os = "linux")]
pub fn restrict_to_node(node: usize) {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;

    let cpus = get_cpus_for_node(node);
    if cpus.is_empty() {
        eprintln!("Warning: No CPUs found for NUMA node {}", node);
        return;
    }

    let mut cpuset = CpuSet::new();
    for &cpu in cpus.iter() {
        if let Err(e) = cpuset.set(cpu) {
            eprintln!("Warning: Failed to set CPU {} in cpuset: {}", cpu, e);
            continue;
        }
    }

    if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpuset) {
        eprintln!(
            "Warning: Failed to set CPU affinity for node {}: {}",
            node, e
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn restrict_to_node(_node: usize) {
    // No-op on non-Linux systems
}

/// Pins the current thread to a specific CPU core.
/// On non-Linux systems, this is a no-op.
#[cfg(target_os = "linux")]
pub fn pin_thread_to_cpu(cpu: usize) {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;

    let mut cpuset = CpuSet::new();
    if let Err(e) = cpuset.set(cpu) {
        eprintln!("Warning: Failed to set CPU {} in cpuset: {}", cpu, e);
        return;
    }

    if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpuset) {
        eprintln!("Warning: Failed to pin to CPU {}: {}", cpu, e);
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_thread_to_cpu(_cpu: usize) {
    // No-op on non-Linux systems
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
    use nix::libc::{self, MPOL_BIND};
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

    #[test]
    fn test_get_numa_nodes_returns_at_least_one() {
        let nodes = get_numa_nodes();
        assert!(!nodes.is_empty());
        assert!(nodes.contains(&0));
    }

    #[test]
    fn test_get_numa_nodes_sorted() {
        let nodes = get_numa_nodes();
        let mut sorted = nodes.clone();
        sorted.sort_unstable();
        assert_eq!(nodes, sorted);
    }

    #[test]
    fn test_get_cores_for_node_zero() {
        let cores = get_cores_for_node(0);
        assert!(cores > 0, "Node 0 should have at least one core");
    }

    #[test]
    fn test_get_node_zero_cores_matches_get_cores_for_node() {
        assert_eq!(get_node_zero_cores(), get_cores_for_node(0));
    }

    #[test]
    fn test_shard_count_positive() {
        assert!(*SHARD_COUNT > 0);
    }
}
