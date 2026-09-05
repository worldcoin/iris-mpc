//! Local production NTT kernel only; excludes MPC and database preprocessing.
use eyre::{ensure, Result};
use iris_mpc_cpu::protocol::ntt::{score_chunk, FieldIris, SpectralIris, SpectralQuery, MODULUS};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::{env, hint::black_box, time::Instant};

fn setting(name: &str, default: usize) -> Result<usize> {
    Ok(env::var(name).map_or(Ok(default), |x| x.parse())?)
}

fn main() -> Result<()> {
    let records = setting("IRIS_MPC_DOT_BENCH_DB_SIZE", 1_048_576)?;
    let workers = setting("IRIS_MPC_NTT_BENCH_WORKERS", 85)?;
    let first_core = setting("IRIS_MPC_NTT_BENCH_FIRST_CORE", 11)?;
    let runs = setting("IRIS_MPC_DOT_BENCH_RUNS", 9)?;
    ensure!(
        records > 0 && workers > 0 && runs > 0,
        "positive benchmark sizes required"
    );
    let cores =
        core_affinity::get_core_ids().ok_or_else(|| eyre::eyre!("CPU affinity unavailable"))?;
    ensure!(
        first_core + workers <= cores.len(),
        "requested CPU range unavailable"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .start_handler(move |i| {
            assert!(core_affinity::set_for_current(cores[first_core + i]));
        })
        .build()?;
    let mut rng = ChaCha20Rng::seed_from_u64(522012386);
    let field = FieldIris {
        code: (0..12800).map(|_| rng.gen_range(0..MODULUS)).collect(),
        mask: (0..6400).map(|_| rng.gen_range(0..MODULUS)).collect(),
    };
    let query = SpectralQuery::prepare(&[&field, &field.mirrored()], 0);
    let templates: Vec<_> = (0..256)
        .map(|_| {
            let record = FieldIris {
                code: (0..12800).map(|_| rng.gen_range(0..MODULUS)).collect(),
                mask: (0..6400).map(|_| rng.gen_range(0..MODULUS)).collect(),
            };
            SpectralIris::prepare(&record)
        })
        .collect();
    // Deep copies give every database record its own physical payload.
    let db: Vec<_> = pool.install(|| {
        (0..records)
            .into_par_iter()
            .map(|i| templates[i % templates.len()].clone())
            .collect()
    });
    println!("NTT_CONFIG p={MODULUS} records={records} workers={workers} first_core={first_core} orientations=2 rotations=31 payload_bytes={}", records * 38400);
    let mut samples = Vec::new();
    for run in 0..=runs {
        let start = Instant::now();
        pool.install(|| {
            db.par_chunks(256).for_each(|chunk| {
                let refs: Vec<_> = chunk.iter().collect();
                black_box(score_chunk(&query, &refs));
            })
        });
        let seconds = start.elapsed().as_secs_f64();
        println!("NTT_SAMPLE run={run} seconds={seconds:.6}");
        if run != 0 {
            samples.push(seconds);
        }
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];
    println!(
        "NTT_RESULT median_seconds={median:.6} comp_per_second={:.0}",
        2.0 * records as f64 / median
    );
    Ok(())
}
