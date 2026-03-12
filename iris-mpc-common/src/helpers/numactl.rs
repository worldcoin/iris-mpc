use std::collections::HashSet;
#[cfg(target_os = "linux")]
use std::fs;
use std::sync::{LazyLock, RwLock};

// =============================================================================
// Static Variables
// =============================================================================

/// The number of cores reserved for the tokio runtime per NUMA node.
/// Set via `init()` before using other functions in this module.
/// If None, no cores are reserved and overlapping is allowed.
static TOKIO_THREAD_COUNT: LazyLock<RwLock<Option<usize>>> = LazyLock::new(|| RwLock::new(None));

/// Caches the set of CPU IDs that have NIC (ENA) IRQs pinned to them.
/// Detected once at startup by scanning /proc/irq/.
/// On non-Linux, this is empty.
static NIC_IRQ_CPUS: LazyLock<HashSet<usize>> = LazyLock::new(|| {
    #[cfg(target_os = "linux")]
    {
        return detect_nic_irq_cpus();
    }
    #[cfg(not(target_os = "linux"))]
    HashSet::new()
});

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

// =============================================================================
// Public API
// =============================================================================

/// Initialize the sysfs module with the number of tokio runtime threads per NUMA node.
/// If Some(n), reserves the first n cores on EACH node for tokio and skips them in worker allocation.
/// Total tokio threads will be n * number_of_numa_nodes.
/// If None, allows core overlap (no reservation) and restrict_tokio_runtime() becomes a no-op.
pub fn init(tokio_threads: Option<usize>) {
    if let Ok(mut count) = TOKIO_THREAD_COUNT.write() {
        *count = tokio_threads;
    }
}

/// Returns the CPU IDs belonging to the specified NUMA node that are available
/// for worker threads (i.e., excluding cores reserved for tokio).
///
/// Tokio core selection strategy per node:
/// 1. Prefer cores that have NIC IRQs pinned to them (colocates network
///    interrupt processing with the tokio I/O runtime).
/// 2. If fewer NIC IRQ cores exist than requested, fill the remainder from
///    the lowest-numbered non-NIC cores on the node.
///
/// Worker threads get the remaining cores (non-tokio, non-NIC-IRQ).
///
/// If init was called with None, no cores are reserved (overlapping allowed).
pub fn get_cores_for_node(node: usize) -> Vec<usize> {
    let tokio_cores = get_tokio_cores_for_node(node);
    let tokio_set: HashSet<usize> = tokio_cores.into_iter().collect();
    let all_cpus = _get_cores_for_node(node);
    all_cpus
        .into_iter()
        .filter(|cpu| !tokio_set.contains(cpu))
        .collect()
}

/// Returns the CPU IDs that should be used for tokio threads on the given
/// NUMA node. Prefers NIC IRQ cores, then fills from lowest-numbered cores.
pub fn get_tokio_cores_for_node(node: usize) -> Vec<usize> {
    let tokio_count = TOKIO_THREAD_COUNT
        .read()
        .ok()
        .and_then(|guard| *guard)
        .unwrap_or(0);
    if tokio_count == 0 {
        return Vec::new();
    }

    let all_cpus = _get_cores_for_node(node);
    let nic_cpus = &*NIC_IRQ_CPUS;

    // Partition into NIC-IRQ cores and non-NIC cores (preserving order)
    let mut nic_on_node: Vec<usize> = all_cpus.iter().copied().filter(|c| nic_cpus.contains(c)).collect();
    let non_nic_on_node: Vec<usize> = all_cpus.iter().copied().filter(|c| !nic_cpus.contains(c)).collect();

    // Take up to tokio_count from NIC cores first, then fill from non-NIC
    nic_on_node.truncate(tokio_count);
    let remaining = tokio_count.saturating_sub(nic_on_node.len());
    let mut tokio_cores = nic_on_node;
    tokio_cores.extend(non_nic_on_node.into_iter().take(remaining));
    tokio_cores.sort_unstable();
    tokio_cores
}

/// Returns the total number of tokio worker threads across all NUMA nodes.
/// If set via init(Some(n)), returns n * number_of_numa_nodes.
/// If not set via init(), defaults to the number of available CPU cores.
pub fn get_tokio_worker_threads() -> usize {
    let r = TOKIO_THREAD_COUNT.read().ok().and_then(|g| *g);
    r.map(|count| count * *SHARD_COUNT).unwrap_or_else(|| {
        core_affinity::get_core_ids()
            .map(|ids| ids.len())
            .unwrap_or(1)
    })
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

/// Restricts the current thread to run only on the tokio-designated CPUs
/// (NIC IRQ cores preferred, then lowest-numbered cores per NUMA node).
/// If init was called with None, this function does nothing (no restriction).
/// On non-Linux systems, this is a no-op.
#[cfg(target_os = "linux")]
pub fn restrict_tokio_runtime() {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;

    if matches!(TOKIO_THREAD_COUNT.read().ok().and_then(|guard| *guard), None | Some(0)) {
        return;
    }

    let mut cpuset = CpuSet::new();
    let numa_nodes = get_numa_nodes();

    for node in &numa_nodes {
        let cpus = get_tokio_cores_for_node(*node);

        if cpus.is_empty() {
            eprintln!("Warning: No CPUs found for tokio runtime on node {}", node);
            continue;
        }

        eprintln!("Tokio runtime node {}: CPUs {:?}", node, cpus);
        for &cpu in cpus.iter() {
            if let Err(e) = cpuset.set(cpu) {
                eprintln!("Warning: Failed to set CPU {} in cpuset: {}", cpu, e);
                continue;
            }
        }
    }

    if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpuset) {
        eprintln!(
            "Warning: Failed to set CPU affinity for tokio runtime: {}",
            e
        );
    }
}

#[cfg(not(target_os = "linux"))]
pub fn restrict_tokio_runtime() {
    // macOS uses Unified Memory; NUMA pinning isn't applicable/available via libc
}

// =============================================================================
// Private Helper Functions
// =============================================================================

/// Scans /proc/irq/ to find CPUs that have NIC (ENA) interrupt queues pinned to them.
/// Looks for IRQ handlers whose name contains "ens" or "ena" (the AWS ENA driver pattern).
#[cfg(target_os = "linux")]
fn detect_nic_irq_cpus() -> HashSet<usize> {
    let mut nic_cpus = HashSet::new();
    let irq_dir = match fs::read_dir("/proc/irq") {
        Ok(d) => d,
        Err(_) => return nic_cpus,
    };

    for entry in irq_dir.filter_map(|e| e.ok()) {
        let irq_path = entry.path();
        if !irq_path.is_dir() {
            continue;
        }

        // Check if any handler subdirectory matches ENA NIC pattern
        let has_nic_handler = fs::read_dir(&irq_path)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .any(|e| {
                let name = e.file_name();
                let name_str = name.to_string_lossy();
                name_str.contains("ens") || name_str.starts_with("ena")
            });

        if !has_nic_handler {
            continue;
        }

        // Read which CPU(s) this IRQ is pinned to
        let affinity_path = irq_path.join("smp_affinity_list");
        if let Ok(contents) = fs::read_to_string(&affinity_path) {
            for cpu in parse_cpulist(&contents) {
                nic_cpus.insert(cpu);
            }
        }
    }

    if !nic_cpus.is_empty() {
        let mut sorted: Vec<_> = nic_cpus.iter().copied().collect();
        sorted.sort_unstable();
        eprintln!(
            "Detected {} NIC IRQ CPUs: {:?}",
            nic_cpus.len(),
            sorted
        );
    }

    nic_cpus
}

/// Parses a Linux cpulist format string (e.g., "0-15,32-47") into a vector of CPU IDs.
#[cfg(target_os = "linux")]
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

/// Internal helper that returns all CPU IDs for a node without skipping any.
fn _get_cores_for_node(node: usize) -> Vec<usize> {
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_simple_range() {
        let cpus = parse_cpulist("0-95");
        assert_eq!(cpus.len(), 96);
        assert_eq!(cpus, (0..=95).collect::<Vec<_>>());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_multiple_ranges() {
        let cpus = parse_cpulist("0-15,32-47");
        let expected: Vec<usize> = (0..=15).chain(32..=47).collect();
        assert_eq!(cpus, expected);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_single_values() {
        let cpus = parse_cpulist("0,5,10");
        assert_eq!(cpus, vec![0, 5, 10]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_mixed() {
        let cpus = parse_cpulist("0-3,8,12-14");
        assert_eq!(cpus, vec![0, 1, 2, 3, 8, 12, 13, 14]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_cpulist_with_whitespace() {
        let cpus = parse_cpulist("  0-3 \n");
        assert_eq!(cpus, vec![0, 1, 2, 3]);
    }

    #[cfg(target_os = "linux")]
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
    fn test_shard_count_positive() {
        assert!(*SHARD_COUNT > 0);
    }
}
