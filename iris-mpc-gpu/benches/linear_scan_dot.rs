use eyre::{ensure, Result};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS,
};
use iris_mpc_cpu::{
    execution::hawk_main::iris_worker::{IrisWorkerPool, LocalIrisWorkerPool, QueryId, QuerySpec},
    hawkers::{aby3::aby3_store::DistanceMode, shared_irises::SharedIrises},
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
};
use iris_mpc_gpu::{
    dot::share_db::{preprocess_query, ProcessedDatabase, ShareDB, SlicedProcessedDatabase},
    helpers::{
        device_manager::DeviceManager,
        query_processor::{
            CudaVec2DSlicerRawPointer, CudaVec2DSlicerU32, CudaVec2DSlicerU8, StreamAwareCudaSlice,
        },
    },
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    env,
    hint::black_box,
    sync::Arc,
    time::{Duration, Instant},
};

/// This is the chunk size used by `ServerActor` in production. Keeping it here
/// (instead of exporting an actor implementation detail) makes the benchmark
/// fail visibly if its production-parity assumption is changed.
const PRODUCTION_DB_CHUNK_SIZE: usize = 1 << 15;
const DEFAULT_DB_SIZE: usize = 3 * 4 * PRODUCTION_DB_CHUNK_SIZE;
const DEFAULT_WARMUP_RUNS: usize = 1;
const DEFAULT_MEASURED_RUNS: usize = 5;

struct GpuScan<'a> {
    device_manager: Arc<DeviceManager>,
    code_engine: ShareDB,
    mask_engine: ShareDB,
    code_db: SlicedProcessedDatabase,
    mask_db: SlicedProcessedDatabase,
    db_sizes: Vec<usize>,
    code_queries: CudaVec2DSlicerU8,
    mask_queries: CudaVec2DSlicerU8,
    code_query_sums: CudaVec2DSlicerU32,
    mask_query_sums: CudaVec2DSlicerU32,
    streams: [Vec<cudarc::driver::CudaStream>; 2],
    blass: [Vec<cudarc::cublas::CudaBlas>; 2],
    code_buffers: [iris_mpc_gpu::dot::share_db::DBChunkBuffers; 2],
    mask_buffers: [iris_mpc_gpu::dot::share_db::DBChunkBuffers; 2],
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl GpuScan<'_> {
    fn chunk_sizes(&self, chunk_idx: usize) -> Vec<usize> {
        self.db_sizes
            .iter()
            .map(|&size| {
                size.saturating_sub(PRODUCTION_DB_CHUNK_SIZE * chunk_idx)
                    .min(PRODUCTION_DB_CHUNK_SIZE)
            })
            .collect()
    }

    fn n_chunks(&self) -> usize {
        self.db_sizes
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .div_ceil(PRODUCTION_DB_CHUNK_SIZE)
    }

    fn prefetch(&self, chunk_idx: usize, buffer: usize, stream: usize) {
        let sizes = self.chunk_sizes(chunk_idx);
        let offsets = self
            .db_sizes
            .iter()
            .map(|_| chunk_idx * PRODUCTION_DB_CHUNK_SIZE)
            .collect::<Vec<_>>();
        self.code_db.prefetch_chunk(
            &self.code_engine,
            &self.code_buffers[buffer],
            &sizes,
            &offsets,
            &self.db_sizes,
            &self.streams[stream],
        );
        self.mask_db.prefetch_chunk(
            &self.mask_engine,
            &self.mask_buffers[buffer],
            &sizes,
            &offsets,
            &self.db_sizes,
            &self.streams[stream],
        );
    }

    fn dot_chunk(&mut self, logical_chunk_idx: usize, buffer: usize, stream: usize) {
        let sizes = self.chunk_sizes(logical_chunk_idx);
        let offset = logical_chunk_idx * PRODUCTION_DB_CHUNK_SIZE;
        self.code_engine.dot(
            &self.code_queries,
            &CudaVec2DSlicerRawPointer::from(&self.code_buffers[buffer]),
            &sizes,
            0,
            &self.streams[stream],
            &self.blass[stream],
        );
        self.mask_engine.dot(
            &self.mask_queries,
            &CudaVec2DSlicerRawPointer::from(&self.mask_buffers[buffer]),
            &sizes,
            0,
            &self.streams[stream],
            &self.blass[stream],
        );
        self.code_engine.dot_reduce(
            &self.code_query_sums,
            &self.code_db.code_sums_gr,
            &sizes,
            offset,
            &self.streams[stream],
        );
        self.mask_engine.dot_reduce_and_multiply(
            &self.mask_query_sums,
            &self.mask_db.code_sums_gr,
            &sizes,
            offset,
            &self.streams[stream],
            2,
        );
    }

    /// Production-shaped copy-only scan: page-locked host DB, async HtoD, and
    /// alternating buffers/streams, without any dot kernels.
    fn run_copy_only_once(&mut self) -> Duration {
        let started = Instant::now();
        self.prefetch(0, 0, 0);
        for chunk_idx in 1..self.n_chunks() {
            let slot = chunk_idx % 2;
            self.prefetch(chunk_idx, slot, slot);
        }
        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);
        started.elapsed()
    }

    /// Put two full chunks in the production buffers before the timed
    /// resident-dot measurement. Each buffer is much larger than GPU L2, so
    /// alternating them cannot turn the scan into an L2-cache benchmark.
    fn prepare_resident_dot(&self) {
        self.prefetch(0, 0, 0);
        self.prefetch(1.min(self.n_chunks() - 1), 1, 1);
        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);
    }

    fn run_resident_dot_once(&mut self) -> Duration {
        let started = Instant::now();
        for chunk_idx in 0..self.n_chunks() {
            let slot = chunk_idx % 2;
            self.dot_chunk(chunk_idx, slot, slot);
            // ShareDB reuses one result/intermediate allocation per device.
            // Production events serialize that reuse while the other stream
            // prefetches; a stream sync provides the same dependency here.
            self.device_manager.await_streams(&self.streams[slot]);
        }
        started.elapsed()
    }

    fn run_combined_once(&mut self) -> Duration {
        let n_chunks = self.n_chunks();

        let started = Instant::now();

        // Production starts chunk zero on stream set zero. Every later chunk
        // is loaded into the other buffer/stream while the current chunk's
        // code and mask dots and reductions execute.
        self.prefetch(0, 0, 0);

        for chunk_idx in 0..n_chunks {
            let current = chunk_idx % 2;
            let next = (chunk_idx + 1) % 2;
            if chunk_idx + 1 < n_chunks {
                self.prefetch(chunk_idx + 1, next, next);
            }
            self.dot_chunk(chunk_idx, current, current);
            self.device_manager.await_streams(&self.streams[current]);
        }

        self.device_manager.await_streams(&self.streams[0]);
        self.device_manager.await_streams(&self.streams[1]);
        started.elapsed()
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(err) => Err(err.into()),
    }
}

fn deterministic_iris(party_id: usize) -> GaloisRingSharedIris {
    let mut code = [0u16; IRIS_CODE_LENGTH];
    let mut mask = [0u16; MASK_CODE_LENGTH];
    for (idx, value) in code.iter_mut().enumerate() {
        *value = (idx as u16).wrapping_mul(17).wrapping_add(23);
    }
    for (idx, value) in mask.iter_mut().enumerate() {
        *value = (idx as u16).wrapping_mul(29).wrapping_add(11);
    }
    GaloisRingSharedIris {
        code: GaloisRingIrisCodeShare::new(code, party_id),
        mask: GaloisRingTrimmedMaskCodeShare::new(mask, party_id),
    }
}

/// Upload the signed-limb row sums consumed by the production reduction
/// kernel. `ShareDB::query_sums` obtains the same values with GEMM, but that
/// helper requires its row count to be divisible by four. A literal B=1 query
/// has 31 rows, so doing this once on the host preserves the exact 31-row dot
/// workload instead of padding the measured GPU scan with phantom queries.
fn upload_query_sums(
    device_manager: &DeviceManager,
    preprocessed_query: &[Vec<u8>],
    query_rows: usize,
    row_width: usize,
) -> Result<CudaVec2DSlicerU32> {
    ensure!(preprocessed_query.len() == 2, "a ring share has two limbs");
    ensure!(
        preprocessed_query
            .iter()
            .all(|limb| limb.len() == query_rows * row_width),
        "unexpected preprocessed query shape"
    );
    let sums = preprocessed_query
        .iter()
        .map(|limb| {
            limb.chunks_exact(row_width)
                .map(|row| row.iter().map(|&value| value as i8 as i32).sum::<i32>() as u32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let limb_0 = device_manager
        .devices()
        .iter()
        .map(|device| {
            device
                .htod_sync_copy(&sums[0])
                .map(StreamAwareCudaSlice::from)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let limb_1 = device_manager
        .devices()
        .iter()
        .map(|device| {
            device
                .htod_sync_copy(&sums[1])
                .map(StreamAwareCudaSlice::from)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(CudaVec2DSlicerU32 { limb_0, limb_1 })
}

fn build_gpu_scan(db_size: usize, query: &GaloisRingSharedIris) -> Result<GpuScan<'static>> {
    let device_manager = Arc::new(DeviceManager::init());
    let n_devices = device_manager.device_count();
    ensure!(n_devices > 0, "benchmark requires at least one GPU");
    ensure!(
        db_size.is_multiple_of(n_devices * PRODUCTION_DB_CHUNK_SIZE),
        "DB size must be a multiple of {} so every GPU chunk is full",
        n_devices * PRODUCTION_DB_CHUNK_SIZE
    );

    let code_engine = ShareDB::init(
        0,
        device_manager.clone(),
        PRODUCTION_DB_CHUNK_SIZE,
        ROTATIONS,
        IRIS_CODE_LENGTH,
        ([0u32; 8], [0u32; 8]),
        vec![],
    );
    let mask_engine = ShareDB::init(
        0,
        device_manager.clone(),
        PRODUCTION_DB_CHUNK_SIZE,
        ROTATIONS,
        MASK_CODE_LENGTH,
        ([0u32; 8], [0u32; 8]),
        vec![],
    );

    let streams = [device_manager.fork_streams(), device_manager.fork_streams()];
    let blass = [
        device_manager.create_cublas(&streams[0]),
        device_manager.create_cublas(&streams[1]),
    ];
    let code_buffers = [
        code_engine.alloc_db_chunk_buffer(PRODUCTION_DB_CHUNK_SIZE),
        code_engine.alloc_db_chunk_buffer(PRODUCTION_DB_CHUNK_SIZE),
    ];
    let mask_buffers = [
        mask_engine.alloc_db_chunk_buffer(PRODUCTION_DB_CHUNK_SIZE),
        mask_engine.alloc_db_chunk_buffer(PRODUCTION_DB_CHUNK_SIZE),
    ];

    let mut code_db = code_engine.alloc_db(db_size);
    let mut mask_db = mask_engine.alloc_db(db_size);
    let code_limb_0 = query
        .code
        .coefs
        .iter()
        .map(|x| *x as u8)
        .collect::<Vec<_>>();
    let code_limb_1 = query
        .code
        .coefs
        .iter()
        .map(|x| (*x >> 8) as u8)
        .collect::<Vec<_>>();
    let mask_limb_0 = query
        .mask
        .coefs
        .iter()
        .map(|x| *x as u8)
        .collect::<Vec<_>>();
    let mask_limb_1 = query
        .mask
        .coefs
        .iter()
        .map(|x| (*x >> 8) as u8)
        .collect::<Vec<_>>();

    println!("loading GPU host database ({db_size} records)");
    (0..db_size).into_par_iter().for_each(|idx| {
        ShareDB::load_single_record_from_s3(
            idx,
            &code_db.code_gr,
            &code_limb_0,
            &code_limb_1,
            n_devices,
            IRIS_CODE_LENGTH,
        );
        ShareDB::load_single_record_from_s3(
            idx,
            &mask_db.code_gr,
            &mask_limb_0,
            &mask_limb_1,
            n_devices,
            MASK_CODE_LENGTH,
        );
    });
    let db_sizes = vec![db_size / n_devices; n_devices];
    code_db.preprocess(&code_engine, &db_sizes);
    mask_db.preprocess(&mask_engine, &db_sizes);
    device_manager.register_host_memory(&code_db, db_size, IRIS_CODE_LENGTH);
    device_manager.register_host_memory(&mask_db, db_size, MASK_CODE_LENGTH);

    let code_query = (0..ROTATIONS)
        .flat_map(|rotation| {
            query
                .code
                .coefs
                .iter()
                .map(move |value| value.wrapping_add(rotation as u16))
        })
        .collect::<Vec<_>>();
    let mask_query = (0..ROTATIONS)
        .flat_map(|rotation| {
            query
                .mask
                .coefs
                .iter()
                .map(move |value| value.wrapping_add(rotation as u16))
        })
        .collect::<Vec<_>>();
    let code_query = preprocess_query(&code_query);
    let mask_query = preprocess_query(&mask_query);
    let code_queries =
        device_manager.htod_transfer_query(&code_query, &streams[0], 1, IRIS_CODE_LENGTH)?;
    let mask_queries =
        device_manager.htod_transfer_query(&mask_query, &streams[0], 1, MASK_CODE_LENGTH)?;
    let code_query_sums =
        upload_query_sums(&device_manager, &code_query, ROTATIONS, IRIS_CODE_LENGTH)?;
    let mask_query_sums =
        upload_query_sums(&device_manager, &mask_query, ROTATIONS, MASK_CODE_LENGTH)?;
    device_manager.await_streams(&streams[0]);

    Ok(GpuScan {
        device_manager,
        code_engine,
        mask_engine,
        code_db,
        mask_db,
        db_sizes,
        code_queries,
        mask_queries,
        code_query_sums,
        mask_query_sums,
        streams,
        blass,
        code_buffers,
        mask_buffers,
        _lifetime: std::marker::PhantomData,
    })
}

fn build_cpu_pool(db_size: usize, query: &ArcIris) -> (LocalIrisWorkerPool, Vec<VectorId>) {
    println!("loading CPU database ({db_size} records)");
    let mut store = SharedIrises::new(
        HashMap::new(),
        Arc::new(GaloisRingSharedIris::default_for_party(0)),
    );
    store.reserve(db_size);
    let mut vector_ids = Vec::with_capacity(db_size);
    for _ in 0..db_size {
        // A deep clone gives every logical record its own backing memory, so
        // the scan cannot turn into an unrealistically cache-resident test.
        vector_ids.push(store.append(Arc::new((**query).clone())));
    }
    let store = store.to_arc();
    (
        LocalIrisWorkerPool::new_local(store, DistanceMode::MinRotation, 0),
        vector_ids,
    )
}

fn median(samples: &[Duration]) -> Duration {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn print_result(backend: &str, db_size: usize, internal_rotations: usize, samples: &[Duration]) {
    let median = median(samples).as_secs_f64();
    let mean = samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64;
    let comparisons_per_second = db_size as f64 / median;
    let rotation_pairs_per_second = db_size as f64 * internal_rotations as f64 / median;
    println!(
        "BENCH_RESULT backend={backend} batch_size=1 db_size={db_size} samples={} \
         median_seconds={median:.6} mean_seconds={mean:.6} comparisons_per_second={comparisons_per_second:.3} \
         internal_rotations={internal_rotations} rotation_pairs_per_second={rotation_pairs_per_second:.3}",
        samples.len(),
    );
}

fn measure(
    backend: &str,
    warmup_runs: usize,
    measured_runs: usize,
    mut run: impl FnMut() -> Duration,
) -> Vec<Duration> {
    for _ in 0..warmup_runs {
        black_box(run());
    }
    (0..measured_runs)
        .map(|sample| {
            let elapsed = run();
            println!(
                "BENCH_SAMPLE backend={backend} run={sample} seconds={:.6}",
                elapsed.as_secs_f64()
            );
            elapsed
        })
        .collect()
}

fn main() -> Result<()> {
    let db_size = env_usize("IRIS_MPC_DOT_BENCH_DB_SIZE", DEFAULT_DB_SIZE)?;
    let warmup_runs = env_usize("IRIS_MPC_DOT_BENCH_WARMUP", DEFAULT_WARMUP_RUNS)?;
    let measured_runs = env_usize("IRIS_MPC_DOT_BENCH_RUNS", DEFAULT_MEASURED_RUNS)?;
    ensure!(db_size > 0, "DB size must be positive");
    ensure!(measured_runs > 0, "measured run count must be positive");

    println!(
        "BENCH_CONFIG batch_size=1 db_size={db_size} warmup_runs={warmup_runs} \
         measured_runs={measured_runs} chunk_size={PRODUCTION_DB_CHUNK_SIZE} \
         gpu_loading=page_locked_async_double_buffered scope=code_mask_dot_and_reduce"
    );

    let query = Arc::new(deterministic_iris(0));

    let mut gpu = build_gpu_scan(db_size, &query)?;
    println!("BENCH_GPU gpu_count={}", gpu.device_manager.device_count());
    let gpu_copy_samples = measure("gpu_copy_only", warmup_runs, measured_runs, || {
        gpu.run_copy_only_once()
    });
    gpu.prepare_resident_dot();
    let gpu_dot_samples = measure("gpu_resident_dot", warmup_runs, measured_runs, || {
        gpu.run_resident_dot_once()
    });
    let gpu_samples = measure("gpu_combined", warmup_runs, measured_runs, || {
        gpu.run_combined_once()
    });

    let runtime = tokio::runtime::Runtime::new()?;
    let (cpu, vector_ids) = build_cpu_pool(db_size, &query);
    let query_id = QueryId::new();
    runtime.block_on(cpu.cache_queries(vec![(query_id, query.clone())]))?;
    let query = QuerySpec::new(query_id);

    let run_cpu = |run: usize, report: bool| -> Result<Duration> {
        let ids = vector_ids.clone();
        let started = Instant::now();
        let output = runtime.block_on(cpu.compute_dot_products_full_rotations(query, ids))?;
        let elapsed = started.elapsed();
        ensure!(
            output.len() == 2 * ROTATIONS * db_size,
            "CPU worker returned an unexpected dot-product result count"
        );
        black_box(output);
        if report {
            println!(
                "BENCH_SAMPLE backend=cpu run={run} seconds={:.6}",
                elapsed.as_secs_f64()
            );
        }
        Ok(elapsed)
    };
    for run in 0..warmup_runs {
        black_box(run_cpu(run, false)?);
    }
    let cpu_samples = (0..measured_runs)
        .map(|run| run_cpu(run, true))
        .collect::<Result<Vec<_>>>()?;

    print_result("cpu", db_size, ROTATIONS, &cpu_samples);
    print_result("gpu_copy_only", db_size, 0, &gpu_copy_samples);
    print_result("gpu_resident_dot", db_size, ROTATIONS, &gpu_dot_samples);
    print_result("gpu_combined", db_size, ROTATIONS, &gpu_samples);
    let copy_seconds = median(&gpu_copy_samples).as_secs_f64();
    let dot_seconds = median(&gpu_dot_samples).as_secs_f64();
    let combined_seconds = median(&gpu_samples).as_secs_f64();
    let transferred_bytes = db_size * 2 * (IRIS_CODE_LENGTH + MASK_CODE_LENGTH);
    let copy_gbps = transferred_bytes as f64 / copy_seconds / 1e9;
    let limiting_component = if copy_seconds >= dot_seconds {
        "pci_copy"
    } else {
        "dot_compute"
    };
    println!(
        "GPU_BOTTLENECK limiting_component={limiting_component} copy_gbps={copy_gbps:.3} \
         copy_seconds={copy_seconds:.6} resident_dot_seconds={dot_seconds:.6} \
         combined_seconds={combined_seconds:.6} combined_over_max_component={:.3}",
        combined_seconds / copy_seconds.max(dot_seconds),
    );
    let speedup = median(&cpu_samples).as_secs_f64() / median(&gpu_samples).as_secs_f64();
    println!("BENCH_COMPARISON metric=logical_comparisons_per_second gpu_over_cpu={speedup:.3}");

    Ok(())
}
