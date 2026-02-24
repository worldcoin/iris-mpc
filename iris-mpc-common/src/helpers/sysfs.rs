use std::fs;
use std::sync::{LazyLock, OnceLock};

// =============================================================================
// Static Variables
// =============================================================================

/// The number of cores reserved for the tokio runtime.
/// Set via `init()` before using other functions in this module.
static TOKIO_THREAD_COUNT: OnceLock<usize> = OnceLock::new();

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

/// Initialize the sysfs module with the number of tokio runtime threads.
/// This must be called before using `restrict_tokio_runtime()` or the CPU
/// allocation for worker threads will properly skip tokio-reserved cores.
pub fn init(tokio_threads: usize) {
    TOKIO_THREAD_COUNT.set(tokio_threads).ok();
}

/// Returns the CPU IDs belonging to the specified NUMA node, skipping
/// the first X cores reserved for the tokio runtime (where X is set via `init()`).
/// This ensures balanced core allocation across NUMA nodes for worker threads.
/// On non-Linux or if detection fails, returns available CPU IDs for node 0
/// (minus reserved cores), or an empty vec for other nodes.
pub fn get_cores_for_node(node: usize) -> Vec<usize> {
    let cpus = _get_cores_for_node(node);
    let skip_count = *TOKIO_THREAD_COUNT.get().unwrap_or(&0);
    cpus.into_iter().skip(skip_count).collect()
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

/// Restricts the current process to run only on the first X CPUs of NUMA node 0,
/// where X is the tokio thread count set via `init()`.
/// On non-Linux systems, this is a no-op.
#[cfg(target_os = "linux")]
pub fn restrict_tokio_runtime() {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;

    let tokio_count = *TOKIO_THREAD_COUNT.get().unwrap_or(&0);
    if tokio_count == 0 {
        return; // No restriction if not initialized
    }

    let all_cpus = _get_cores_for_node(0);
    let cpus: Vec<_> = all_cpus.into_iter().take(tokio_count).collect();

    if cpus.is_empty() {
        eprintln!("Warning: No CPUs found for tokio runtime on node 0");
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
    fn test_shard_count_positive() {
        assert!(*SHARD_COUNT > 0);
    }
}
