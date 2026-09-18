use iris_mpc_common::VectorId;
use std::{
    collections::HashSet,
    env,
    hint::black_box,
    mem::size_of,
    time::{Duration, Instant},
};

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let records = env_usize("IRIS_MPC_CANDIDATE_BENCH_RECORDS", 2_000_000);
    let candidate_count = env_usize("IRIS_MPC_CANDIDATE_BENCH_CANDIDATES", 128);
    let repetitions = env_usize("IRIS_MPC_CANDIDATE_BENCH_REPETITIONS", 5);
    assert!(records > 0 && records <= u32::MAX as usize);
    assert!(repetitions > 0);

    let live_ids = (0..records)
        .map(|index| VectorId::from_0_index(index as u32))
        .collect::<Vec<_>>();
    let candidates = (0..candidate_count)
        .map(|index| {
            let record = index * records / candidate_count.max(1);
            VectorId::from_0_index(record.min(records - 1) as u32)
        })
        .collect::<Vec<_>>();

    let mut hash_build = Duration::ZERO;
    let mut hash_lookup = Duration::ZERO;
    let mut hash_capacity = 0usize;
    for _ in 0..repetitions {
        let started = Instant::now();
        let live_ids_set = live_ids.iter().copied().collect::<HashSet<_>>();
        hash_build += started.elapsed();
        hash_capacity = live_ids_set.capacity();

        let started = Instant::now();
        let found = candidates
            .iter()
            .filter(|id| live_ids_set.contains(id))
            .count();
        hash_lookup += started.elapsed();
        black_box(found);
    }

    let mut binary_lookup = Duration::ZERO;
    for _ in 0..repetitions {
        let started = Instant::now();
        let found = candidates
            .iter()
            .filter(|id| live_ids.binary_search(id).is_ok())
            .count();
        binary_lookup += started.elapsed();
        black_box(found);
    }

    let divisor = repetitions as f64;
    println!(
        "LINEAR_SCAN_CANDIDATE_RESULT records={records} candidates={} repetitions={repetitions} \
         vector_id_bytes={} hash_capacity={hash_capacity} hash_table_lower_bound_bytes={} \
         hash_build_seconds={:.9} hash_lookup_seconds={:.9} binary_lookup_seconds={:.9}",
        candidates.len(),
        size_of::<VectorId>(),
        hash_capacity * size_of::<VectorId>(),
        hash_build.as_secs_f64() / divisor,
        hash_lookup.as_secs_f64() / divisor,
        binary_lookup.as_secs_f64() / divisor,
    );
}
