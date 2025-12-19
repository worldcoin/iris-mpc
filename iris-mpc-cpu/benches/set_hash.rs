use criterion::{black_box, criterion_group, criterion_main, Criterion};
use iris_mpc_common::vector_id::VectorId;
use iris_mpc_cpu::execution::hawk_main::state_check::SetHash;

/// Benchmark that repeatedly calls insert on SharedIrises.
fn bench_set_hash(c: &mut Criterion) {
    c.bench_function("set_hash_vector_id", |b| {
        let mut set_hash = SetHash::default();
        let v = black_box(VectorId::from_serial_id(1111));

        b.iter(|| {
            set_hash.add_unordered(v);
            set_hash.checksum()
        });
    });

    c.bench_function("set_hash_250_links", |b| {
        let mut set_hash = SetHash::default();

        let lc = 1_u8;
        let v = VectorId::from_serial_id(1);
        let links = &vec![VectorId::from_serial_id(2); 250];
        let item = black_box((lc, v, links));

        b.iter(|| {
            set_hash.add_unordered(item);
            set_hash.checksum()
        });
    });
}

criterion_group! {benches, bench_set_hash}
criterion_main!(benches);
