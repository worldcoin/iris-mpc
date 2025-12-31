use std::sync::Arc;

use aes_prng::AesRng;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use iris_mpc_cpu::{
    execution::{local::LocalRuntime, session::Session},
    protocol::ops::{batch_signed_lift_vec, cross_compare},
    shares::{share::DistanceShare, Share},
};
use rand::SeedableRng;
use tokio::{sync::Mutex, task::JoinSet};

#[path = "bench_utils.rs"]
mod bench_utils;
use bench_utils::create_random_sharing;

criterion_group!(networking, bench_is_match_batch_tcp,);
criterion_main!(networking);

async fn run_jobs(
    num_iterations: usize,
    sessions: &[Arc<Mutex<Session>>],
    d1: Vec<Share<u16>>,
    d2: Vec<Share<u16>>,
    t1: Vec<Share<u16>>,
    t2: Vec<Share<u16>>,
) {
    let mut jobs = JoinSet::new();
    for (index, player_session) in sessions.iter().enumerate() {
        // each vec of shares is of length 3 - per 3PC. the sessions were created
        // so that if split by 3, each chunk has the same session id, and each idx
        // corresponds to a party.
        let d1i = d1[index % 3];
        let d2i = d2[index % 3];
        let t1i = t1[index % 3];
        let t2i = t2[index % 3];
        let player_session = player_session.clone();
        jobs.spawn(async move {
            let mut player_session = player_session.lock().await;
            for _ in 0..num_iterations {
                let ds_and_ts =
                    batch_signed_lift_vec(&mut player_session, vec![d1i, d2i, t1i, t2i])
                        .await
                        .unwrap();
                cross_compare(
                    &mut player_session,
                    &[(
                        DistanceShare::new(ds_and_ts[0], ds_and_ts[1]),
                        DistanceShare::new(ds_and_ts[2], ds_and_ts[3]),
                    )],
                )
                .await
                .unwrap();
            }
        });
    }
    let _outputs = black_box(jobs.join_all().await);
}

fn bench_is_match_batch_tcp(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_match_batch_tcp");
    group.sample_size(10);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    #[allow(clippy::single_element_loop)]
    for (nj, rp) in [(1024, 32)] {
        {
            let cp = 1;
            let mut rng = AesRng::seed_from_u64(0_u64);
            let d1 = create_random_sharing(&mut rng, 10_u16);
            let d2 = create_random_sharing(&mut rng, 10_u16);
            let t1 = create_random_sharing(&mut rng, 10_u16);
            let t2 = create_random_sharing(&mut rng, 10_u16);

            let sessions = rt
                .block_on(async move { LocalRuntime::mock_sessions_with_tcp(cp, rp).await })
                .unwrap();

            let num_parties = 3;
            assert_eq!(sessions.len(), rp * num_parties);

            group.bench_function(
                BenchmarkId::new("local", format!("cp: {}, rp: {}, nj: {}", cp, rp, nj)),
                |b| {
                    b.iter(|| {
                        let (d1, d2, t1, t2) = (d1.clone(), d2.clone(), t1.clone(), t2.clone());
                        let sessions = &sessions;
                        rt.block_on(async move {
                            for _ in 0..nj / rp {
                                run_jobs(
                                    1,
                                    sessions,
                                    d1.clone(),
                                    d2.clone(),
                                    t1.clone(),
                                    t2.clone(),
                                )
                                .await;
                            }
                        });
                    })
                },
            );
        }
    }
}
