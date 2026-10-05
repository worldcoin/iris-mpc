use std::sync::OnceLock;

// =============================================================================
// Static Variables
// =============================================================================

/// How CPUs are split between the tokio runtime and pinned worker threads.
/// Set via `init()` or `init_with_smt_siblings()` before using other functions
/// in this module.
#[derive(Clone, Copy, Debug, Default)]
struct TokioReservation {
    /// Tokio CPUs per NUMA node. `None` without SMT placement means no
    /// reservation: tokio and workers may overlap.
    per_node: Option<usize>,
    /// Place tokio on the second hardware thread of each physical core.
    smt_siblings: bool,
}

static TOKIO_RESERVATION: OnceLock<TokioReservation> = OnceLock::new();

// =============================================================================
// Public API
// =============================================================================

/// Initialize the numactl module with the number of tokio runtime threads per NUMA node.
/// If Some(n), reserves the first n cores on EACH node for tokio and skips them in worker allocation.
/// Total tokio threads will be n * number_of_numa_nodes.
/// If None, allows core overlap (no reservation) and restrict_tokio_runtime() becomes a no-op.
///
/// Panics if the reservation exceeds available cores on any NUMA node.
pub fn init(tokio_threads: Option<usize>) {
    init_with_smt_siblings(tokio_threads, false);
}

/// Like [`init`], but with `smt_siblings` the tokio runtime runs on the
/// second hardware thread of every physical core (at most `tokio_threads` per
/// NUMA node when set) and workers keep the first hardware threads. Falls
/// back to [`init`]'s placement on hosts without SMT.
pub fn init_with_smt_siblings(tokio_threads: Option<usize>, smt_siblings: bool) {
    let smt_siblings = smt_siblings && {
        let available = get_numa_nodes()
            .into_iter()
            .all(|node| !smt_secondaries(&all_cores_for_node(node)).is_empty());
        if !available {
            eprintln!("Warning: tokio_on_smt_siblings is set, but the host has no SMT siblings");
        }
        available
    };
    if let Some(count) = tokio_threads {
        for node in get_numa_nodes() {
            let all = all_cores_for_node(node);
            let available = if smt_siblings {
                smt_secondaries(&all).len()
            } else {
                all.len()
            };
            assert!(
                count <= available,
                "separate_tokio_cores_per_node ({count}) exceeds available cores ({available}) on NUMA node {node}"
            );
        }
    }
    TOKIO_RESERVATION
        .set(TokioReservation {
            per_node: tokio_threads,
            smt_siblings,
        })
        .expect("numactl::init() called more than once");
}

fn reservation() -> TokioReservation {
    TOKIO_RESERVATION.get().copied().unwrap_or_default()
}

/// The CPUs of `node` reserved for the tokio runtime, or `None` if the
/// runtime is not restricted.
fn tokio_cores_for_node(node: usize) -> Option<Vec<usize>> {
    let reservation = reservation();
    let all = all_cores_for_node(node);
    if reservation.smt_siblings {
        let siblings = smt_secondaries(&all);
        Some(match reservation.per_node {
            Some(count) => siblings.into_iter().take(count).collect(),
            None => siblings,
        })
    } else {
        reservation
            .per_node
            .map(|count| all.into_iter().take(count).collect())
    }
}

/// Returns the CPU IDs belonging to the specified NUMA node, skipping
/// the CPUs reserved for the tokio runtime on this node (see `init()`).
/// If init was called with None, no cores are skipped (overlapping allowed).
/// Each NUMA node independently reserves its tokio CPUs.
/// This ensures balanced core allocation across NUMA nodes for worker threads.
/// On non-Linux or if detection fails, returns available CPU IDs for node 0
/// (minus reserved cores), or an empty vec for other nodes.
pub fn get_cores_for_node(node: usize) -> Vec<usize> {
    let cpus = all_cores_for_node(node);
    match tokio_cores_for_node(node) {
        Some(reserved) => cpus
            .into_iter()
            .filter(|cpu| !reserved.contains(cpu))
            .collect(),
        None => cpus,
    }
}

/// One worker CPU per physical core of `node`: the first hardware thread of
/// every core whose first hardware thread is not reserved for tokio. Kernels
/// that saturate a per-core unit (such as AMX tiles) lose throughput when two
/// hardware threads of one core share it. With `tokio_on_smt_siblings`, tokio
/// shares every core through its second hardware thread; otherwise cores
/// reserved for tokio get no worker.
pub fn get_physical_cores_for_node(node: usize) -> Vec<usize> {
    let mut all = all_cores_for_node(node);
    all.sort_unstable();
    let reserved = tokio_cores_for_node(node).unwrap_or_default();
    let mut seen = Vec::new();
    all.into_iter()
        .filter(|&cpu| {
            let core = core_key(cpu);
            if core.is_some_and(|core| seen.contains(&core)) {
                return false;
            }
            seen.extend(core);
            !reserved.contains(&cpu)
        })
        .collect()
}

/// Returns the total number of tokio worker threads across all NUMA nodes.
/// With a reservation this is the number of reserved CPUs; otherwise it
/// defaults to the number of available CPU cores.
pub fn get_tokio_worker_threads() -> usize {
    let reserved = get_numa_nodes()
        .into_iter()
        .map(tokio_cores_for_node)
        .try_fold(0, |total, cpus| cpus.map(|cpus| total + cpus.len()));
    match reserved {
        Some(count) if count > 0 => count,
        _ => core_affinity::get_core_ids()
            .map(|ids| ids.len())
            .unwrap_or(1),
    }
}

/// Returns a list of all available NUMA node IDs on the system.
/// On non-Linux or if detection fails, returns vec![0].
pub fn get_numa_nodes() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
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

/// Restricts the current process (or thread) to the CPUs reserved for the
/// tokio runtime by `init()`. Without a reservation this does nothing.
/// On non-Linux systems, this is a no-op.
#[cfg(target_os = "linux")]
pub fn restrict_tokio_runtime() {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;

    let mut cpuset = CpuSet::new();
    let mut any = false;
    for node in get_numa_nodes() {
        let Some(cpus) = tokio_cores_for_node(node) else {
            return;
        };
        if cpus.is_empty() {
            eprintln!("Warning: No CPUs found for tokio runtime on node {}", node);
            continue;
        }
        for cpu in cpus {
            if let Err(e) = cpuset.set(cpu) {
                eprintln!("Warning: Failed to set CPU {} in cpuset: {}", cpu, e);
                continue;
            }
            any = true;
        }
    }
    if !any {
        return;
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

/// `(package, core)` of a CPU, or `None` if the topology is unknown.
fn core_key(cpu: usize) -> Option<(u32, u32)> {
    #[cfg(target_os = "linux")]
    {
        let read = |name: &str| {
            std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/{name}"))
                .ok()?
                .trim()
                .parse::<u32>()
                .ok()
        };
        Some((read("physical_package_id")?, read("core_id")?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpu;
        None
    }
}

/// The CPUs of `cpus` that share their physical core with a lower-numbered
/// CPU of the list, i.e. every hardware thread but the first of each core.
fn smt_secondaries(cpus: &[usize]) -> Vec<usize> {
    let mut sorted = cpus.to_vec();
    sorted.sort_unstable();
    let mut seen = Vec::new();
    sorted
        .into_iter()
        .filter(|&cpu| match core_key(cpu) {
            Some(core) if seen.contains(&core) => true,
            Some(core) => {
                seen.push(core);
                false
            }
            None => false,
        })
        .collect()
}

/// Parses a Linux cpulist format string (e.g., "0-15,32-47") into a vector of CPU IDs.
#[cfg(any(target_os = "linux", test))]
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

/// Returns all CPU IDs for a node without skipping any.
fn all_cores_for_node(node: usize) -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        let path = format!("/sys/devices/system/node/node{}/cpulist", node);
        if let Ok(contents) = std::fs::read_to_string(&path) {
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
}
