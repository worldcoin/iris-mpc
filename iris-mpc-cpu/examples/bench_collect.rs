//! Benchmark: collection strategies for awaiting parallel worker results.
//!
//! Emulates the pattern in `IrisPoolHandle::rotation_aware_dot_product_batch`:
//! N concurrent tokio tasks each dispatch M work items to a pool of OS worker
//! threads (via crossbeam channels), then collect results back via oneshot
//! channels. Results must be returned in the same order as dispatched (matching
//! the `try_join_all` + `flatten` pattern in production).
//!
//! Work durations are jittered (uniform ±30% by default) to simulate realistic
//! out-of-order completion, since different chunks hit different memory regions.
//!
//! Usage:
//!   cargo run --release --example bench_collect -p iris-mpc-cpu
//!   cargo run --release --example bench_collect -p iris-mpc-cpu -- --concurrent 16 --items 32
//!   cargo run --release --example bench_collect -p iris-mpc-cpu -- --sweep
//!   cargo run --release --example bench_collect -p iris-mpc-cpu -- --jitter 50

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use crossbeam::channel::{self, Sender};
use futures::future::{join_all, try_join_all};
use futures::stream::{FuturesUnordered, StreamExt};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tokio::sync::{mpsc, oneshot, Notify};
use tokio_metrics::TaskMonitor;

// ── CLI ─────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "bench_collect",
    about = "Benchmark result-collection strategies for parallel worker dispatch"
)]
struct Args {
    /// Number of OS worker threads (simulates iris worker pool size).
    /// Workers sleep (no CPU burn), so this can be large.
    #[arg(short = 'w', long, default_value_t = 256)]
    workers: usize,

    /// Number of tokio runtime worker threads
    #[arg(short = 't', long, default_value_t = 4)]
    tokio_threads: usize,

    /// Number of concurrent async tasks (simulates parallel search sessions)
    #[arg(short = 'c', long, default_value_t = 8)]
    concurrent: usize,

    /// Number of work items per task (simulates chunks per dot-product call)
    #[arg(short = 'm', long, default_value_t = 8)]
    items: usize,

    /// Base work duration per item in microseconds
    #[arg(long, default_value_t = 5000)]
    work_us: u64,

    /// Jitter as a percentage of base work duration (e.g. 30 means ±30%)
    #[arg(long, default_value_t = 30)]
    jitter: u64,

    /// Number of measurement iterations
    #[arg(short = 'n', long, default_value_t = 50)]
    iterations: usize,

    /// Warmup iterations
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Run a sweep over multiple configurations
    #[arg(long)]
    sweep: bool,

    /// Number of background tokio tasks simulating MPC networking load.
    /// Each task cycles: CPU work (--bg-cpu-us) then async sleep (--bg-sleep-us),
    /// competing for tokio worker threads with the collection tasks.
    #[arg(long, default_value_t = 0)]
    bg_tasks: usize,

    /// Microseconds of CPU work per background task cycle (simulates message
    /// serialization/deserialization on tokio thread)
    #[arg(long, default_value_t = 50)]
    bg_cpu_us: u64,

    /// Microseconds of async sleep per background task cycle (simulates
    /// network round-trip wait)
    #[arg(long, default_value_t = 200)]
    bg_sleep_us: u64,
}

// ── Worker Pool ─────────────────────────────────────────────────────────────

type WorkFn = Box<dyn FnOnce() + Send>;

struct WorkerPool {
    senders: Vec<Sender<WorkFn>>,
    next: AtomicUsize,
}

impl WorkerPool {
    fn new(n_workers: usize) -> Self {
        let senders = (0..n_workers)
            .map(|_| {
                let (tx, rx) = channel::unbounded::<WorkFn>();
                std::thread::spawn(move || {
                    while let Ok(f) = rx.recv() {
                        f();
                    }
                });
                tx
            })
            .collect();
        WorkerPool {
            senders,
            next: AtomicUsize::new(0),
        }
    }

    fn send_round_robin(&self, f: WorkFn) {
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        self.senders[idx].send(f).unwrap();
    }

    fn send_to(&self, worker_idx: usize, f: WorkFn) {
        self.senders[worker_idx % self.senders.len()]
            .send(f)
            .unwrap();
    }

    fn n_workers(&self) -> usize {
        self.senders.len()
    }
}

// ── Simulated Work & Jitter ─────────────────────────────────────────────────

/// Simulate work by sleeping. We're measuring tokio collection overhead, not
/// CPU work — so workers just block for the given duration without burning cores.
#[inline(never)]
fn simulate_work(duration: Duration) {
    std::thread::sleep(duration);
}

/// Generate jittered work durations: uniform in [base*(1-pct/100), base*(1+pct/100)].
fn jittered_durations(
    n_items: usize,
    base_us: u64,
    jitter_pct: u64,
    rng: &mut impl Rng,
) -> Vec<Duration> {
    let lo = base_us.saturating_sub(base_us * jitter_pct / 100);
    let hi = base_us + base_us * jitter_pct / 100;
    (0..n_items)
        .map(|_| Duration::from_micros(rng.gen_range(lo..=hi)))
        .collect()
}

// ── Collection Strategies ───────────────────────────────────────────────────
//
// All strategies:
//   - Dispatch items to OS workers with per-item jittered durations
//   - Return Vec<usize> of sequence numbers in original dispatch order [0,1,..,N-1]
//   - Workers return their sequence number; the collection step reassembles order

/// Current approach: dispatch round-robin, `try_join_all(oneshots).await`.
/// Naturally preserves order (results come back in input-future order).
async fn strategy_try_join_all(pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
    let receivers: Vec<_> = durations
        .iter()
        .enumerate()
        .map(|(seq, &dur)| {
            let (tx, rx) = oneshot::channel::<usize>();
            pool.send_round_robin(Box::new(move || {
                simulate_work(dur);
                let _ = tx.send(seq);
            }));
            rx
        })
        .collect();
    try_join_all(receivers).await.unwrap()
}

/// FuturesUnordered: ready-queue based polling, avoids the O(N) scan that
/// JoinAll does on each wakeup. Results arrive out-of-order, so we reassemble
/// into a pre-allocated indexed buffer.
async fn strategy_futures_unordered(pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
    let n = durations.len();
    let mut futs: FuturesUnordered<_> = durations
        .iter()
        .enumerate()
        .map(|(seq, &dur)| {
            let (tx, rx) = oneshot::channel::<usize>();
            pool.send_round_robin(Box::new(move || {
                simulate_work(dur);
                let _ = tx.send(seq);
            }));
            rx
        })
        .collect();

    let mut buffer = vec![0usize; n];
    while let Some(r) = futs.next().await {
        let seq = r.unwrap();
        buffer[seq] = seq;
    }
    buffer
}

/// Workers send (seq, result) to a shared mpsc channel. Batched wakeups may
/// reduce tokio overhead. Results arrive out-of-order, reassembled via indexed
/// buffer.
async fn strategy_mpsc(pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
    let n = durations.len();
    let (tx, mut rx) = mpsc::unbounded_channel::<usize>();
    for (seq, &dur) in durations.iter().enumerate() {
        let tx = tx.clone();
        pool.send_round_robin(Box::new(move || {
            simulate_work(dur);
            let _ = tx.send(seq);
        }));
    }
    drop(tx);

    let mut buffer = vec![0usize; n];
    for _ in 0..n {
        let seq = rx.recv().await.unwrap();
        buffer[seq] = seq;
    }
    buffer
}

/// Partition work across workers: each worker gets a contiguous slice and
/// processes items sequentially. Reduces oneshot count from N_items to
/// min(N_items, N_workers). Order is preserved: worker k returns items for
/// its slice, and we try_join_all workers in order.
///
/// Trade-off: serializes items within each worker, so the slowest worker
/// determines latency. But generates far fewer tokio wakeups.
async fn strategy_per_worker_batch(pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
    let n = durations.len();
    let n_workers = pool.n_workers();
    let items_per_worker = n.div_ceil(n_workers);
    let mut receivers = Vec::with_capacity(n_workers.min(n));

    for (w, chunk) in durations.chunks(items_per_worker).enumerate() {
        let chunk_durations: Vec<Duration> = chunk.to_vec();
        let base_seq = w * items_per_worker;
        let (tx, rx) = oneshot::channel::<Vec<usize>>();
        pool.send_to(
            w,
            Box::new(move || {
                let results: Vec<usize> = chunk_durations
                    .iter()
                    .enumerate()
                    .map(|(i, dur)| {
                        simulate_work(*dur);
                        base_seq + i
                    })
                    .collect();
                let _ = tx.send(results);
            }),
        );
        receivers.push(rx);
    }
    let chunks = try_join_all(receivers).await.unwrap();
    chunks.into_iter().flatten().collect()
}

/// Dispatch all items round-robin (parallel execution), then await oneshots
/// sequentially. Naturally preserves order. Baseline for comparison.
async fn strategy_sequential(pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
    let receivers: Vec<_> = durations
        .iter()
        .enumerate()
        .map(|(seq, &dur)| {
            let (tx, rx) = oneshot::channel::<usize>();
            pool.send_round_robin(Box::new(move || {
                simulate_work(dur);
                let _ = tx.send(seq);
            }));
            rx
        })
        .collect();
    let mut results = Vec::with_capacity(receivers.len());
    for rx in receivers {
        results.push(rx.await.unwrap());
    }
    results
}

/// Single-wakeup strategy: workers write results to a shared indexed buffer
/// and decrement an atomic counter. The last worker to finish notifies the
/// async task via `tokio::sync::Notify`. Generates exactly 1 tokio wakeup.
/// The atomic counter + Notify acts as a completion barrier; after it fires,
/// all indexed writes are visible (AcqRel on the counter).
async fn strategy_atomic_notify(pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
    let n = durations.len();
    let buffer: Arc<Vec<AtomicUsize>> =
        Arc::new((0..n).map(|_| AtomicUsize::new(usize::MAX)).collect());
    let remaining = Arc::new(AtomicUsize::new(n));
    let notify = Arc::new(Notify::new());

    for (seq, &dur) in durations.iter().enumerate() {
        let buffer = buffer.clone();
        let remaining = remaining.clone();
        let notify = notify.clone();
        pool.send_round_robin(Box::new(move || {
            simulate_work(dur);
            buffer[seq].store(seq, Ordering::Release);
            if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                notify.notify_one();
            }
        }));
    }

    // Wait for all workers. The loop handles the edge case where the permit
    // arrives between Notified creation and first poll.
    loop {
        if remaining.load(Ordering::Acquire) == 0 {
            break;
        }
        let notified = notify.notified();
        if remaining.load(Ordering::Acquire) == 0 {
            break;
        }
        notified.await;
    }

    buffer.iter().map(|a| a.load(Ordering::Acquire)).collect()
}

// ── Strategy Enum ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Strategy {
    TryJoinAll,
    FuturesUnordered,
    Mpsc,
    PerWorkerBatch,
    Sequential,
    AtomicNotify,
}

impl Strategy {
    const ALL: &[Strategy] = &[
        Strategy::TryJoinAll,
        Strategy::FuturesUnordered,
        Strategy::Mpsc,
        Strategy::PerWorkerBatch,
        Strategy::Sequential,
        Strategy::AtomicNotify,
    ];

    fn name(self) -> &'static str {
        match self {
            Strategy::TryJoinAll => "try_join_all",
            Strategy::FuturesUnordered => "futures_unordered",
            Strategy::Mpsc => "mpsc",
            Strategy::PerWorkerBatch => "per_worker_batch",
            Strategy::Sequential => "sequential",
            Strategy::AtomicNotify => "atomic_notify",
        }
    }

    async fn run(self, pool: &WorkerPool, durations: &[Duration]) -> Vec<usize> {
        match self {
            Strategy::TryJoinAll => strategy_try_join_all(pool, durations).await,
            Strategy::FuturesUnordered => strategy_futures_unordered(pool, durations).await,
            Strategy::Mpsc => strategy_mpsc(pool, durations).await,
            Strategy::PerWorkerBatch => strategy_per_worker_batch(pool, durations).await,
            Strategy::Sequential => strategy_sequential(pool, durations).await,
            Strategy::AtomicNotify => strategy_atomic_notify(pool, durations).await,
        }
    }
}

// ── Measurement ─────────────────────────────────────────────────────────────

struct BenchResult {
    strategy: &'static str,
    wall_clock_us: Vec<f64>,
    per_task_us: Vec<f64>,
    mean_scheduled_us: f64,
    mean_poll_us: f64,
    total_poll_count: u64,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Expected result: [0, 1, 2, ..., n-1]
fn expected_order(n: usize) -> Vec<usize> {
    (0..n).collect()
}

async fn run_strategy(
    pool: Arc<WorkerPool>,
    strategy: Strategy,
    n_concurrent: usize,
    n_items: usize,
    work_us: u64,
    jitter_pct: u64,
    warmup: usize,
    iterations: usize,
) -> BenchResult {
    let mut rng = SmallRng::from_entropy();
    let expected = expected_order(n_items);

    // Warmup
    for _ in 0..warmup {
        let task_durations: Vec<Vec<Duration>> = (0..n_concurrent)
            .map(|_| jittered_durations(n_items, work_us, jitter_pct, &mut rng))
            .collect();
        let handles: Vec<_> = task_durations
            .into_iter()
            .map(|durs| {
                let pool = pool.clone();
                tokio::spawn(async move { strategy.run(&pool, &durs).await })
            })
            .collect();
        join_all(handles).await;
    }

    let monitor = TaskMonitor::new();
    let mut wall_clock_us = Vec::with_capacity(iterations);
    let mut per_task_us = Vec::with_capacity(iterations * n_concurrent);

    for _ in 0..iterations {
        let task_durations: Vec<Vec<Duration>> = (0..n_concurrent)
            .map(|_| jittered_durations(n_items, work_us, jitter_pct, &mut rng))
            .collect();

        let start = Instant::now();
        let handles: Vec<_> = task_durations
            .into_iter()
            .map(|durs| {
                let pool = pool.clone();
                let expected = expected.clone();
                tokio::spawn(monitor.instrument(async move {
                    let t = Instant::now();
                    let results = strategy.run(&pool, &durs).await;
                    assert_eq!(results, expected, "{}: ordering violated", strategy.name());
                    t.elapsed()
                }))
            })
            .collect();
        let results = join_all(handles).await;
        wall_clock_us.push(start.elapsed().as_secs_f64() * 1e6);
        for r in results {
            per_task_us.push(r.unwrap().as_secs_f64() * 1e6);
        }
    }

    wall_clock_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    per_task_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let metrics = monitor.cumulative();
    BenchResult {
        strategy: strategy.name(),
        wall_clock_us,
        per_task_us,
        mean_scheduled_us: metrics.mean_scheduled_duration().as_secs_f64() * 1e6,
        mean_poll_us: metrics.mean_poll_duration().as_secs_f64() * 1e6,
        total_poll_count: metrics.total_poll_count,
    }
}

fn print_header() {
    println!(
        "{:<22} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} {:>8}",
        "Strategy", "Wall p50", "Wall p95", "Wall p99", "Task p50", "Task p95", "Sched (us)",
        "Poll (us)", "Polls"
    );
    println!("{}", "-".repeat(114));
}

fn print_result(r: &BenchResult) {
    println!(
        "{:<22} {:>9.0}u {:>9.0}u {:>9.0}u {:>9.0}u {:>9.0}u {:>12.1} {:>10.2} {:>8}",
        r.strategy,
        percentile(&r.wall_clock_us, 50.0),
        percentile(&r.wall_clock_us, 95.0),
        percentile(&r.wall_clock_us, 99.0),
        percentile(&r.per_task_us, 50.0),
        percentile(&r.per_task_us, 95.0),
        r.mean_scheduled_us,
        r.mean_poll_us,
        r.total_poll_count,
    );
}

fn run_config(
    rt: &tokio::runtime::Runtime,
    pool: &Arc<WorkerPool>,
    concurrent: usize,
    items: usize,
    work_us: u64,
    jitter_pct: u64,
    warmup: usize,
    iterations: usize,
) {
    print_header();
    for &strategy in Strategy::ALL {
        let r = rt.block_on(run_strategy(
            pool.clone(),
            strategy,
            concurrent,
            items,
            work_us,
            jitter_pct,
            warmup,
            iterations,
        ));
        print_result(&r);
    }
}

fn main() {
    let args = Args::parse();

    let pool = Arc::new(WorkerPool::new(args.workers));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.tokio_threads)
        .enable_all()
        .build()
        .unwrap();

    // Spawn background tasks that compete for tokio worker threads.
    let bg_handles: Vec<tokio::task::JoinHandle<()>> = (0..args.bg_tasks)
        .map(|_| {
            let cpu_us = args.bg_cpu_us;
            let sleep_us = args.bg_sleep_us;
            rt.spawn(async move {
                let cpu_dur = Duration::from_micros(cpu_us);
                let sleep_dur = Duration::from_micros(sleep_us);
                loop {
                    // Simulate MPC message processing (blocks tokio thread)
                    let start = Instant::now();
                    while start.elapsed() < cpu_dur {}
                    // Simulate network RTT (yields tokio thread, re-enters run queue after sleep)
                    tokio::time::sleep(sleep_dur).await;
                }
            })
        })
        .collect();

    println!(
        "Workers: {}, Tokio threads: {}, Jitter: +/-{}%, BG tasks: {} ({}us cpu + {}us sleep)\n",
        args.workers, args.tokio_threads, args.jitter, args.bg_tasks, args.bg_cpu_us, args.bg_sleep_us
    );

    if args.sweep {
        // (concurrent_tasks, items_per_task, work_us)
        let configs = [
            (8, 8, 5000),   // baseline: 8 sessions, 8 chunks, 5ms work
            (16, 8, 5000),  // more sessions
            (8, 32, 5000),  // more chunks per session
            (64, 64, 5000), // high contention
            (4, 4, 10000),  // few tasks, longer work
        ];

        for (concurrent, items, work_us) in configs {
            println!("=== concurrent={concurrent}, items={items}, work_us={work_us} ===\n");
            run_config(
                &rt,
                &pool,
                concurrent,
                items,
                work_us,
                args.jitter,
                args.warmup,
                args.iterations,
            );
            println!();
        }
    } else {
        println!(
            "concurrent={}, items={}, work_us={}\n",
            args.concurrent, args.items, args.work_us
        );
        run_config(
            &rt,
            &pool,
            args.concurrent,
            args.items,
            args.work_us,
            args.jitter,
            args.warmup,
            args.iterations,
        );
    }

    for h in bg_handles {
        h.abort();
    }
}
