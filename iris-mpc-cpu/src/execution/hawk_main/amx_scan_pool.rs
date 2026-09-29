//! Pinned worker threads for the AMX full-rotation scan.
//!
//! One thread runs on each physical core outside the tokio reservation (see
//! `numactl::get_physical_cores_for_node`). Each NUMA node has its own task
//! queue, and a scan task only touches groups of arena segments placed on the
//! node of the threads that serve it.
// Without AMX the pool never starts, so its tasks are never run.
#![cfg_attr(
    not(all(target_arch = "x86_64", target_os = "linux")),
    allow(dead_code)
)]

use crate::protocol::{
    amx_scan::{AmxQuery, GroupArena, GROUP},
    ops::SHARE_OF_MAX_DISTANCE_TRIMMED,
};
use eyre::{eyre, Result};
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
use iris_mpc_common::helpers::numactl;
use iris_mpc_common::ROTATIONS;
use std::{
    mem::MaybeUninit,
    sync::{Arc, OnceLock},
};

/// Values per record and query: `(code, trimmed mask)` for every rotation.
const RECORD_VALUES: usize = 2 * ROTATIONS;
/// Group-query evaluations per task (about 1 ms of AMX work): enough to
/// amortize the dispatch, small enough to balance the threads of a node.
const TASK_GROUP_QUERIES: usize = 64;

/// One query of a scan and where its records land.
struct ScanQuery {
    query: Arc<AmxQuery>,
    /// Output buffer of this query.
    buffer: usize,
    /// Position of the query's first record in its buffer, in records.
    record_offset: usize,
}

/// Arena group of a scan and the output record of each of its lanes.
struct GroupJob {
    group: usize,
    /// Output record per lane; `u32::MAX` marks a lane outside the scan.
    records: [u32; GROUP],
}

/// Output buffers of one scan, written concurrently by its tasks.
///
/// Every element is written exactly once before the buffers are read: each
/// record of each query is either a missing target, which the scan fills
/// with the sentinel before dispatch, or a lane of exactly one job, which
/// one task writes completely. The buffers therefore start uninitialized
/// (zeroing them would cost as much memory traffic as the result itself), are
/// only written through raw pointers while tasks run, and become readable
/// once every task has reported completion.
struct ScanOutput {
    buffers: Vec<Vec<MaybeUninit<u16>>>,
    pointers: Vec<*mut u16>,
}

// SAFETY: see the type documentation; the raw pointers stay valid because
// `buffers` is never resized while tasks hold the output.
unsafe impl Send for ScanOutput {}
// SAFETY: as above.
unsafe impl Sync for ScanOutput {}

impl ScanOutput {
    fn new(lens: impl Iterator<Item = usize>) -> Self {
        let mut buffers = lens
            .map(|len| {
                let mut buffer = Vec::with_capacity(len);
                // SAFETY: `MaybeUninit` elements need no initialization.
                unsafe { buffer.set_len(len) };
                buffer
            })
            .collect::<Vec<Vec<MaybeUninit<u16>>>>();
        let pointers = buffers
            .iter_mut()
            .map(|buffer| buffer.as_mut_ptr().cast::<u16>())
            .collect();
        Self { buffers, pointers }
    }

    /// Write `values` to element `index` onward of buffer `buffer`.
    ///
    /// # Safety
    /// The elements must be in bounds and written by no other task.
    unsafe fn write(&self, buffer: usize, index: usize, values: &[u16]) {
        debug_assert!(index + values.len() <= self.buffers[buffer].len());
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            self.pointers[buffer].add(index),
            values.len(),
        );
    }

    /// The written buffers.
    ///
    /// # Safety
    /// Every element must have been written (see the type documentation).
    unsafe fn into_buffers(self) -> Vec<Vec<u16>> {
        self.buffers
            .into_iter()
            .map(|buffer| {
                let mut buffer = std::mem::ManuallyDrop::new(buffer);
                // SAFETY: `MaybeUninit<u16>` has the layout of `u16`, and the
                // caller guarantees that every element is initialized.
                Vec::from_raw_parts(
                    buffer.as_mut_ptr().cast::<u16>(),
                    buffer.len(),
                    buffer.capacity(),
                )
            })
            .collect()
    }
}

struct Task {
    arena: Arc<GroupArena>,
    queries: Arc<[ScanQuery]>,
    jobs: Arc<[GroupJob]>,
    range: std::ops::Range<usize>,
    output: Arc<ScanOutput>,
    done: tokio::sync::oneshot::Sender<()>,
}

pub struct AmxScanPool {
    /// Task queue per NUMA node, indexed like [`GroupArena::node_of_group`].
    queues: Vec<crossbeam::channel::Sender<Task>>,
}

impl AmxScanPool {
    /// The process-wide pool, started on first use. `None` without AMX.
    pub fn global() -> Option<&'static Self> {
        static POOL: OnceLock<Option<AmxScanPool>> = OnceLock::new();
        POOL.get_or_init(Self::start).as_ref()
    }

    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    fn start() -> Option<Self> {
        use crate::protocol::amx_scan::{amx_available, AmxThread};

        if !amx_available() {
            return None;
        }
        let nodes = numactl::get_numa_nodes();
        let mut node_cpus = nodes
            .iter()
            .map(|&node| numactl::get_physical_cores_for_node(node))
            .collect::<Vec<_>>();
        if node_cpus.iter().all(Vec::is_empty) {
            // Tokio holds the first hardware thread of every core. Sharing
            // cores still beats scanning the grouped layout without AMX.
            tracing::warn!(
                "No physical cores outside the tokio reservation; \
                 AMX scan workers run on the remaining hardware threads"
            );
            node_cpus = nodes
                .iter()
                .map(|&node| numactl::get_cores_for_node(node))
                .collect();
        }
        let Some(fallback) = node_cpus.iter().position(|cpus| !cpus.is_empty()) else {
            tracing::error!("No CPUs for the AMX scan workers; the AMX kernel is disabled");
            return None;
        };
        let channels = node_cpus
            .iter()
            .map(|_| crossbeam::channel::unbounded::<Task>())
            .collect::<Vec<_>>();
        let mut threads = 0;
        for (node, cpus) in node_cpus.iter().enumerate() {
            // A node without worker CPUs is served by the first node with some.
            let receiver = if cpus.is_empty() {
                &channels[fallback].1
            } else {
                &channels[node].1
            };
            for &cpu in cpus {
                let receiver = receiver.clone();
                threads += 1;
                std::thread::Builder::new()
                    .name(format!("amx-scan-{cpu}"))
                    .spawn(move || {
                        let _ = core_affinity::set_for_current(core_affinity::CoreId { id: cpu });
                        let mut thread = AmxThread::new().expect("AMX was detected");
                        while let Ok(task) = receiver.recv() {
                            run_task(&mut thread, task);
                        }
                    })
                    .expect("failed to spawn an AMX scan thread");
            }
        }
        tracing::info!(
            numa_nodes = nodes.len(),
            threads,
            cpus = ?node_cpus,
            "Started AMX exact-scan workers"
        );
        // Senders of nodes without CPUs forward to the fallback queue.
        let queues = node_cpus
            .iter()
            .enumerate()
            .map(|(node, cpus)| {
                if cpus.is_empty() {
                    channels[fallback].0.clone()
                } else {
                    channels[node].0.clone()
                }
            })
            .collect();
        Some(Self { queues })
    }

    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    fn start() -> Option<Self> {
        None
    }

    /// Full-rotation `(code, trimmed mask)` contributions of every query
    /// against the arena records at `targets`, streaming each group once for
    /// all queries. `targets[i]` is the arena index of record `i`, or `None`
    /// for a record that is not resident, which gets the max-distance
    /// sentinel. Output buffer `b` holds, for each query of `queries[b]` in
    /// order, all records as `2 * ROTATIONS` interleaved values.
    pub async fn scan(
        &self,
        arena: Arc<GroupArena>,
        queries: Vec<Vec<Arc<AmxQuery>>>,
        targets: &[Option<usize>],
    ) -> Result<Vec<Vec<u16>>> {
        let records = targets.len();
        let output = ScanOutput::new(
            queries
                .iter()
                .map(|queries| queries.len() * records * RECORD_VALUES),
        );
        let queries = queries
            .into_iter()
            .enumerate()
            .flat_map(|(buffer, queries)| {
                queries
                    .into_iter()
                    .enumerate()
                    .map(move |(position, query)| ScanQuery {
                        query,
                        buffer,
                        record_offset: position * records,
                    })
            })
            .collect::<Vec<_>>();

        let sentinel = sentinel_record();
        let mut jobs: Vec<GroupJob> = Vec::new();
        for (record, target) in targets.iter().enumerate() {
            let Some(index) = *target else {
                for query in &queries {
                    let start = (query.record_offset + record) * RECORD_VALUES;
                    // SAFETY: in bounds, and no task writes a missing record.
                    unsafe { output.write(query.buffer, start, &sentinel) };
                }
                continue;
            };
            let (group, lane) = (index / GROUP, index % GROUP);
            match jobs.last_mut() {
                Some(job) if job.group == group && job.records[lane] == u32::MAX => {
                    job.records[lane] = record as u32;
                }
                _ => {
                    let mut records = [u32::MAX; GROUP];
                    records[lane] = record as u32;
                    jobs.push(GroupJob { group, records });
                }
            }
        }
        if jobs.is_empty() || queries.is_empty() {
            // SAFETY: every record was a missing target and holds the sentinel.
            return Ok(unsafe { output.into_buffers() });
        }

        // Tasks: runs of jobs on one NUMA node, bounded in group-queries.
        let groups_per_task = (TASK_GROUP_QUERIES / queries.len()).max(1);
        let mut tasks = Vec::new();
        let mut start = 0;
        for index in 1..=jobs.len() {
            let node = arena.node_of_group(jobs[start].group);
            let split = index == jobs.len()
                || index - start == groups_per_task
                || arena.node_of_group(jobs[index].group) != node;
            if split {
                tasks.push((node, start..index));
                start = index;
            }
        }

        let output = Arc::new(output);
        let queries: Arc<[ScanQuery]> = queries.into();
        let jobs: Arc<[GroupJob]> = jobs.into();
        let mut pending = Vec::with_capacity(tasks.len());
        for (node, range) in tasks {
            let (done, receiver) = tokio::sync::oneshot::channel();
            self.queues[node % self.queues.len()]
                .send(Task {
                    arena: arena.clone(),
                    queries: queries.clone(),
                    jobs: jobs.clone(),
                    range,
                    output: output.clone(),
                    done,
                })
                .map_err(|_| eyre!("AMX scan workers stopped"))?;
            pending.push(receiver);
        }
        for receiver in pending {
            receiver
                .await
                .map_err(|_| eyre!("an AMX scan task did not complete"))?;
        }
        let output = Arc::try_unwrap(output)
            .map_err(|_| eyre!("AMX scan output is still shared after all tasks completed"))?;
        // SAFETY: every task completed, and together with the sentinels they
        // wrote every element (see `ScanOutput`).
        Ok(unsafe { output.into_buffers() })
    }
}

/// `(code, trimmed mask)` of a missing target for every rotation.
fn sentinel_record() -> [u16; RECORD_VALUES] {
    let (code, mask) = SHARE_OF_MAX_DISTANCE_TRIMMED;
    std::array::from_fn(|index| if index % 2 == 0 { code } else { mask })
}

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn run_task(thread: &mut crate::protocol::amx_scan::AmxThread, task: Task) {
    let Task {
        arena,
        queries,
        jobs,
        range,
        output,
        done,
    } = task;
    let sentinel = sentinel_record();
    let mut record = [0_u16; RECORD_VALUES * GROUP];
    {
        let groups = arena.groups();
        for job in &jobs[range] {
            let Some(bytes) = groups.group(job.group) else {
                // A resident record's segment is always allocated; treat a
                // missing one like a missing record rather than reading it.
                for query in queries.iter() {
                    for &lane_record in job.records.iter().filter(|&&r| r != u32::MAX) {
                        let index = (query.record_offset + lane_record as usize) * RECORD_VALUES;
                        // SAFETY: this lane's record is written by this task only.
                        unsafe { output.write(query.buffer, index, &sentinel) };
                    }
                }
                continue;
            };
            let lanes: [u32; GROUP] = std::array::from_fn(|lane| {
                if job.records[lane] == u32::MAX {
                    u32::MAX
                } else {
                    lane as u32
                }
            });
            for query in queries.iter() {
                thread.scan_group(&query.query, &bytes, |component, acc| {
                    acc.scatter(component, &lanes, &mut record)
                });
                for (lane, &lane_record) in job.records.iter().enumerate() {
                    if lane_record == u32::MAX {
                        continue;
                    }
                    let index = (query.record_offset + lane_record as usize) * RECORD_VALUES;
                    // SAFETY: this lane's record is written by this task only,
                    // and the scan checked that it lies inside the buffer.
                    unsafe {
                        output.write(
                            query.buffer,
                            index,
                            &record[lane * RECORD_VALUES..(lane + 1) * RECORD_VALUES],
                        )
                    };
                }
            }
        }
    }
    drop(output);
    let _ = done.send(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{
        amx_scan::{scan_lane_reference, SEGMENT_RECORDS},
        shared_iris::GaloisRingSharedIris,
    };
    use aes_prng::AesRng;
    use rand::{Rng, SeedableRng};

    fn random_share(rng: &mut AesRng) -> GaloisRingSharedIris {
        let mut iris = GaloisRingSharedIris::default_for_party(0);
        rng.fill(&mut iris.code.coefs[..]);
        rng.fill(&mut iris.mask.coefs[..]);
        iris
    }

    /// Two output buffers with several queries each, targets spread over
    /// groups and segments, a missing record and one in an unallocated segment.
    #[tokio::test]
    async fn scan_matches_reference_layout() -> Result<()> {
        let Some(pool) = AmxScanPool::global() else {
            eprintln!("AMX not available; skipping");
            return Ok(());
        };
        let mut rng = AesRng::seed_from_u64(9);
        let arena = Arc::new(GroupArena::new(0, vec![0, 1]));
        for index in [0, 5, 15, 16, 40, SEGMENT_RECORDS + 3] {
            arena.write(index, &random_share(&mut rng));
        }
        let targets = [
            Some(5),
            Some(0),
            None,
            Some(40),
            Some(16),
            Some(15),
            Some(SEGMENT_RECORDS + 3),
            Some(7 * SEGMENT_RECORDS),
        ];
        let queries = (0..3)
            .map(|_| Arc::new(AmxQuery::new(&random_share(&mut rng))))
            .collect::<Vec<_>>();
        let buffers = pool
            .scan(
                arena.clone(),
                vec![
                    vec![queries[0].clone(), queries[1].clone()],
                    vec![queries[2].clone()],
                ],
                &targets,
            )
            .await?;
        assert_eq!(buffers[0].len(), 2 * targets.len() * RECORD_VALUES);
        assert_eq!(buffers[1].len(), targets.len() * RECORD_VALUES);

        let sentinel = sentinel_record();
        let groups = arena.groups();
        let layout = [(0, 0, 0), (1, 0, targets.len()), (2, 1, 0)];
        for (query, buffer, record_offset) in layout {
            for (record, target) in targets.iter().enumerate() {
                let start = (record_offset + record) * RECORD_VALUES;
                let got = &buffers[buffer][start..start + RECORD_VALUES];
                let expected = match target.and_then(|index| {
                    groups
                        .group(index / GROUP)
                        .map(|group| scan_lane_reference(&queries[query], &group, index % GROUP))
                }) {
                    Some(pairs) => pairs.iter().flatten().copied().collect::<Vec<_>>(),
                    None => sentinel.to_vec(),
                };
                assert_eq!(
                    got,
                    expected.as_slice(),
                    "record {record} target {target:?}"
                );
            }
        }
        Ok(())
    }
}
