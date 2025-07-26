use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::protocol::{
    ops::galois_ring_pairwise_distance, shared_iris::GaloisRingSharedIris,
};
use rand::thread_rng;

pub fn bench_galois_ring_pairwise_distance(c: &mut Criterion) {
    let rng = &mut thread_rng();

    let iris = IrisCode::random_rng(rng);
    let shares = GaloisRingSharedIris::generate_shares_locally(rng, iris);

    let pairs = vec![Some((&shares[0], &shares[1])); 1];

    c.bench_function("galois_ring_pairwise_distance", |b| {
        b.iter(|| galois_ring_pairwise_distance(black_box(&pairs)))
    });
}

criterion_group!(benches, bench_galois_ring_pairwise_distance);
criterion_main!(benches);
