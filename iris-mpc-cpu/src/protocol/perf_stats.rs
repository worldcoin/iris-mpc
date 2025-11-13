//-- AI generated --//
// perf_stats.rs
// Thread-safe, low-overhead performance stats with batch-size buckets.

use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

type Bucket = usize; // actual batch size or the upper-bound bucket
type Label = &'static str; // keep labels 'static to avoid allocs

/// Aggregates per (label, bucket). All atomics for low-overhead lock-free updates.
/// We intentionally do NOT store individual samples to keep overhead tiny.
#[derive(Debug)]
struct Agg {
    count: AtomicU64,
    sum_ns: AtomicU64, // total duration in nanoseconds
    min_ns: AtomicU64, // initialized to u64::MAX
    max_ns: AtomicU64, // initialized to 0
}

impl Agg {
    fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            sum_ns: AtomicU64::new(0),
            min_ns: AtomicU64::new(u64::MAX),
            max_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    fn record_ns(&self, ns: u64) {
        // Count & sum: relaxed is sufficient for stats aggregation.
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(ns, Ordering::Relaxed);

        // Min: CAS loop.
        let mut cur = self.min_ns.load(Ordering::Relaxed);
        while ns < cur {
            match self
                .min_ns
                .compare_exchange_weak(cur, ns, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }

        // Max: CAS loop.
        let mut cur_max = self.max_ns.load(Ordering::Relaxed);
        while ns > cur_max {
            match self.max_ns.compare_exchange_weak(
                cur_max,
                ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur_max = actual,
            }
        }
    }
}

/// Global enable switch to make timing a near no-op in hot paths when disabled.
static ENABLED: AtomicBool = AtomicBool::new(true);

/// Map: (label, bucket) -> Agg
static STATS: Lazy<DashMap<(Label, Bucket), Agg>> = Lazy::new(|| DashMap::new());

#[inline]
fn bucket_for(batch_size: usize, upper_bound: usize) -> Bucket {
    if batch_size > upper_bound {
        upper_bound
    } else {
        batch_size
    }
}

/// Record a single duration sample for a labeled code block and batch size.
/// - `label`: a compile-time string identifying the measured block
/// - `batch_size`: current x.len()
/// - `upper_bound`: bucket cap; sizes > upper_bound are folded into `upper_bound` bucket
/// - `dur`: measured time
#[inline]
pub fn record(label: Label, batch_size: usize, upper_bound: usize, dur: Duration) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let ns = dur.as_nanos() as u64; // if your durations can exceed u64::MAX ns, reconsider
    let b = bucket_for(batch_size, upper_bound);
    let entry = STATS.entry((label, b)).or_insert_with(Agg::new);
    entry.record_ns(ns);
}

/// Disable all recording (fast path early-return).
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

/// Enable recording.
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// Clear all accumulated stats.
pub fn clear() {
    STATS.clear();
}

/// Snapshot row returned by `snapshot()`.
#[derive(Debug, Clone)]
pub struct SnapshotRow {
    pub label: Label,
    pub bucket: Bucket,
    pub count: u64,
    pub sum_ns: u64,
    pub mean_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
}

impl SnapshotRow {
    pub fn mean_ms(&self) -> f64 {
        self.mean_ns / 1_000_000.0
    }
    pub fn min_ms(&self) -> f64 {
        self.min_ns as f64 / 1_000_000.0
    }
    pub fn max_ms(&self) -> f64 {
        self.max_ns as f64 / 1_000_000.0
    }
}

/// Take a consistent snapshot for reporting/logging (no locking held after copy).
pub fn snapshot() -> Vec<SnapshotRow> {
    let mut rows = Vec::with_capacity(STATS.len());
    for item in STATS.iter() {
        let ((label, bucket), agg) = (item.key(), item.value());

        let count = agg.count.load(Ordering::Relaxed);
        if count == 0 {
            continue;
        }

        let sum_ns = agg.sum_ns.load(Ordering::Relaxed);
        let min_ns = agg.min_ns.load(Ordering::Relaxed);
        let max_ns = agg.max_ns.load(Ordering::Relaxed);
        let mean_ns = (sum_ns as f64) / (count as f64);

        rows.push(SnapshotRow {
            label: *label,
            bucket: *bucket,
            count,
            sum_ns,
            mean_ns,
            min_ns,
            max_ns,
        });
    }

    // Sort by (label, bucket) for stable output
    rows.sort_by(|a, b| match a.label.cmp(&b.label) {
        std::cmp::Ordering::Equal => a.bucket.cmp(&b.bucket),
        other => other,
    });
    rows
}

/// Print the top-N performance stats rows by sample count.
/// Useful for quickly identifying the most frequently measured
/// code paths or batch sizes in the current snapshot.
pub fn print_top_by_count(top_n: usize) {
    let mut rows = snapshot();

    // Sort descending by count
    rows.sort_by(|a, b| b.count.cmp(&a.count));

    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Label", "Bucket", "Count", "Min(ms)", "Mean(ms)", "Max(ms)"
    );
    for row in rows.iter().take(top_n) {
        println!(
            "{:<20} {:>10} {:>10} {:>10.3} {:>10.3} {:>10.3}",
            row.label,
            row.bucket,
            row.count,
            row.min_ms(),
            row.mean_ms(),
            row.max_ms()
        );
    }
}

// /// Total time recorded for `label` across all perf buckets (via `snapshot()`).
// /// Non-blocking; returns `0ns` if the label has no rows.
// pub fn total_duration(label: Label) -> Duration {
//     let rows = snapshot();
//     let total_ns: u64 = rows
//         .into_iter()
//         .filter(|r| r.label == label)
//         .map(|r| r.sum_ns)
//         .sum();
//     Duration::from_nanos(total_ns)
// }

// This is to record just the block of FSS vs RSS in cross_compare

// ---------------------------------------------------------
// Scoped timer: records elapsed time to `record(...)` on drop
// ---------------------------------------------------------
pub struct ScopedTimer {
    label: &'static str,
    n: usize,
    upper_bound: usize,
    start: Instant,
}

impl ScopedTimer {
    #[inline]
    pub fn new(label: &'static str, n: usize, upper_bound: usize) -> Self {
        Self {
            label,
            n,
            upper_bound,
            start: Instant::now(),
        }
    }
}

impl Drop for ScopedTimer {
    #[inline]
    fn drop(&mut self) {
        // Uses the repo's existing atomic aggregator:
        // fn record(label: Label, n: usize, upper_bound: usize, dur: Duration)
        record(self.label, self.n, self.upper_bound, self.start.elapsed());
    }
}

// ---------------------------------------------------------
// Per-party scoped timer macro (explicit bucket size param)
// ---------------------------------------------------------
// Usage in ops.rs:
//   let _tt = perf_scoped_for_party!("cross_compare.extract_open",
//                                    session.party_id(), diff.len(), 150);
// The base label must be a literal so we can `concat!` at compile time.
#[macro_export]
macro_rules! perf_scoped_for_party {
    ($base_label:literal, $party:expr, $n:expr, $bucket_cap:expr) => {{
        let label: &'static str = match ($party as usize) {
            0 => concat!($base_label, ".p0"),
            1 => concat!($base_label, ".p1"),
            2 => concat!($base_label, ".p2"),
            _ => concat!($base_label, ".p2"), // fallback
        };
        $crate::protocol::perf_stats::ScopedTimer::new(label, $n, $bucket_cap)
    }};
}

// ---------------------------------------------------------
// Readback helpers (for printing totals elsewhere)
// ---------------------------------------------------------

/// Total time recorded for `label` across all buckets.
/// Returns `0ns` if the label has no rows.
pub fn total_duration(label: &str) -> Duration {
    let rows = snapshot();
    let total_ns: u64 = rows
        .into_iter()
        .filter(|r| r.label == label)
        .map(|r| r.sum_ns)
        .sum();
    Duration::from_nanos(total_ns)
}

/// Totals for `{base}.p0`, `{base}.p1`, `{base}.p2` (in that order).
pub fn total_duration_per_party(base_label: &str) -> [Duration; 3] {
    let want = [
        format!("{base_label}.p0"),
        format!("{base_label}.p1"),
        format!("{base_label}.p2"),
    ];
    let mut acc = [0u64; 3];

    for row in snapshot() {
        if row.label == want[0].as_str() {
            acc[0] += row.sum_ns;
        } else if row.label == want[1].as_str() {
            acc[1] += row.sum_ns;
        } else if row.label == want[2].as_str() {
            acc[2] += row.sum_ns;
        }
    }

    [
        Duration::from_nanos(acc[0]),
        Duration::from_nanos(acc[1]),
        Duration::from_nanos(acc[2]),
    ]
}

/// Time a *block* for a specific party; returns the block's value.
/// Use for multi-statement code or unit-returning code (e.g., `open_bin_fss(...).await?;`).
/// Run-and-time a block for a specific party, with explicit bucket cap.
/// Returns whatever the block returns, and measures across `await`s and early `?`.
#[macro_export]
macro_rules! perf_time_block_for_party {
    ($base_label:literal, $party:expr, $n:expr, $bucket_cap:expr, $body:block) => {{
        let __label: &'static str = match ($party as usize) {
            0 => concat!($base_label, ".p0"),
            1 => concat!($base_label, ".p1"),
            2 => concat!($base_label, ".p2"),
            _ => concat!($base_label, ".p2"),
        };
        let __perf_result = {
            let __perf_guard =
                $crate::protocol::perf_stats::ScopedTimer::new(__label, $n, $bucket_cap);
            $body
            // guard drops here → time recorded
        };
        __perf_result
    }};
}

/// Time a single *expression*; returns the expression's value.
/// Nice for inline use like: `let x = perf_time_expr_for_party!(..., expr)?;`
#[macro_export]
macro_rules! perf_time_expr_for_party {
    ($base_label:literal, $party:expr, $n:expr, $bucket_cap:expr, $expr:expr) => {{
        let __label: &'static str = match ($party as usize) {
            0 => concat!($base_label, ".p0"),
            1 => concat!($base_label, ".p1"),
            2 => concat!($base_label, ".p2"),
            _ => concat!($base_label, ".p2"),
        };
        {
            let __perf_guard =
                $crate::protocol::perf_stats::ScopedTimer::new(__label, $n, $bucket_cap);
            $expr
        }
        // guard drops at the end of the inner block
    }};
}

// Replace your existing perf_time_let_for_party! with this version
#[macro_export]
macro_rules! perf_time_let_for_party {
    ($base_label:literal, $party:expr, $n:expr, $bucket_cap:expr, let $pat:pat = $expr:expr) => {
        let $pat = {
            let __label: &'static str = match ($party as usize) {
                0 => concat!($base_label, ".p0"),
                1 => concat!($base_label, ".p1"),
                2 => concat!($base_label, ".p2"),
                _ => concat!($base_label, ".p2"),
            };
            let __perf_guard =
                $crate::protocol::perf_stats::ScopedTimer::new(__label, $n, $bucket_cap);
            $expr
            // guard drops here
        };
    };
}
