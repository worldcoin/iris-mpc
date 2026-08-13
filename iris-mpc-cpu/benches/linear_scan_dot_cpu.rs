use eyre::{ensure, Result};
use iris_mpc_common::{VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS};
use iris_mpc_cpu::{
    execution::hawk_main::iris_worker::{
        init_workers, IrisWorkerPool, LocalIrisWorkerPool, QueryId, QuerySpec,
    },
    hawkers::{aby3::aby3_store::DistanceMode, shared_irises::SharedIrises},
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    env,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

const DEFAULT_DB_SIZE: usize = 1_572_864;
const DEFAULT_WARMUP_RUNS: usize = 1;
const DEFAULT_MEASURED_RUNS: usize = 3;
const DEFAULT_TOKIO_CORES: usize = 8;

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err.into()),
    }
}

fn build_pool(db_size: usize, numa_shard: usize) -> (LocalIrisWorkerPool, Vec<VectorId>, ArcIris) {
    let query = Arc::new(GaloisRingSharedIris::default_for_party(0));
    println!(
        "BENCH_LOADING db_size={db_size} payload_gib={:.3}",
        db_size as f64 * 2.0 * (IRIS_CODE_LENGTH + MASK_CODE_LENGTH) as f64
            / (1024.0 * 1024.0 * 1024.0),
    );
    let load_started = Instant::now();

    // Deep-clone every record so the scan reads a production-sized working
    // set instead of repeatedly hitting one shared cache-resident allocation.
    let irises = (0..db_size)
        .into_par_iter()
        .map(|_| Arc::new((*query).clone()))
        .collect::<Vec<_>>();
    let mut store = SharedIrises::new(
        HashMap::new(),
        Arc::new(GaloisRingSharedIris::default_for_party(0)),
    );
    store.reserve(db_size);
    let mut vector_ids = Vec::with_capacity(db_size);
    for iris in irises {
        vector_ids.push(store.append(iris));
    }
    println!(
        "BENCH_LOADED seconds={:.3}",
        load_started.elapsed().as_secs_f64()
    );

    let store = store.to_arc();
    let workers = init_workers(numa_shard, store.clone(), true);
    (
        LocalIrisWorkerPool::new(workers, store, DistanceMode::MinRotation, 0),
        vector_ids,
        query,
    )
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

fn report(backend: &str, db_size: usize, orientations: usize, samples: &[Duration]) {
    let elapsed = median(samples).as_secs_f64();
    let comparisons = db_size * orientations;
    println!(
        "BENCH_RESULT backend={backend} db_size={db_size} orientations={orientations} \
         median_seconds={elapsed:.6} logical_comparisons_per_second={:.3} \
         full_scan_equivalent_seconds={:.6}",
        comparisons as f64 / elapsed,
        elapsed,
    );
}

fn main() -> Result<()> {
    let db_size = env_usize("IRIS_MPC_DOT_BENCH_DB_SIZE", DEFAULT_DB_SIZE)?;
    let numa_shard = env_usize("IRIS_MPC_DOT_BENCH_NUMA_SHARD", 0)?;
    let warmup_runs = env_usize("IRIS_MPC_DOT_BENCH_WARMUP", DEFAULT_WARMUP_RUNS)?;
    let measured_runs = env_usize("IRIS_MPC_DOT_BENCH_RUNS", DEFAULT_MEASURED_RUNS)?;
    let tokio_cores = env_usize("IRIS_MPC_DOT_BENCH_TOKIO_CORES", DEFAULT_TOKIO_CORES)?;
    let tokio_cores = (tokio_cores > 0).then_some(tokio_cores);
    ensure!(db_size > 0, "DB size must be positive");
    ensure!(measured_runs > 0, "measured run count must be positive");
    iris_mpc_common::helpers::numactl::init(tokio_cores);
    println!(
        "BENCH_CONFIG batch_size=1 db_size={db_size} numa_shard={numa_shard} \
         warmup_runs={warmup_runs} measured_runs={measured_runs} rotations={ROTATIONS} \
         tokio_cores={} dot_cores={}",
        tokio_cores.unwrap_or_else(iris_mpc_common::helpers::numactl::get_tokio_worker_threads),
        iris_mpc_cpu::execution::hawk_main::iris_worker::select_core_ids(numa_shard).len(),
    );

    let (pool, vector_ids, query) = build_pool(db_size, numa_shard);
    iris_mpc_common::helpers::numactl::restrict_tokio_runtime();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(iris_mpc_common::helpers::numactl::get_tokio_worker_threads())
        .on_thread_start(iris_mpc_common::helpers::numactl::restrict_tokio_runtime)
        .enable_all()
        .build()?;
    let normal_id = QueryId::new();
    let mirror_id = QueryId::new();
    runtime.block_on(pool.cache_queries(vec![(normal_id, query.clone()), (mirror_id, query)]))?;
    let normal = QuerySpec::new(normal_id);
    let mirror = QuerySpec::new(mirror_id);
    let expected_len = 2 * ROTATIONS * db_size;

    let run_single = || -> Result<Duration> {
        let started = Instant::now();
        let output = runtime
            .block_on(pool.compute_dot_products_full_rotations(normal, vector_ids.clone()))?;
        let elapsed = started.elapsed();
        ensure!(output.len() == expected_len, "unexpected result length");
        black_box(output);
        Ok(elapsed)
    };
    for _ in 0..warmup_runs {
        black_box(run_single()?);
    }
    let mut single_samples = Vec::with_capacity(measured_runs);
    for run in 0..measured_runs {
        let elapsed = run_single()?;
        println!(
            "BENCH_SAMPLE backend=cpu_single_orientation run={run} seconds={:.6}",
            elapsed.as_secs_f64()
        );
        single_samples.push(elapsed);
    }

    let run_mirror = || -> Result<Duration> {
        let started = Instant::now();
        let (normal_output, mirror_output) = runtime.block_on(async {
            tokio::join!(
                pool.compute_dot_products_full_rotations(normal, vector_ids.clone()),
                pool.compute_dot_products_full_rotations(mirror, vector_ids.clone()),
            )
        });
        let normal_output = normal_output?;
        let mirror_output = mirror_output?;
        let elapsed = started.elapsed();
        ensure!(
            normal_output.len() == expected_len && mirror_output.len() == expected_len,
            "unexpected mirror result length"
        );
        black_box((normal_output, mirror_output));
        Ok(elapsed)
    };
    for _ in 0..warmup_runs {
        black_box(run_mirror()?);
    }
    let mut mirror_samples = Vec::with_capacity(measured_runs);
    for run in 0..measured_runs {
        let elapsed = run_mirror()?;
        println!(
            "BENCH_SAMPLE backend=cpu_normal_and_mirror run={run} seconds={:.6}",
            elapsed.as_secs_f64()
        );
        mirror_samples.push(elapsed);
    }

    report("cpu_single_orientation", db_size, 1, &single_samples);
    report("cpu_normal_and_mirror", db_size, 2, &mirror_samples);
    println!(
        "BENCH_MIRROR_RATIO wall_time_ratio={:.3}",
        median(&mirror_samples).as_secs_f64() / median(&single_samples).as_secs_f64()
    );
    Ok(())
}
