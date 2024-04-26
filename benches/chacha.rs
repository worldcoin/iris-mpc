use criterion::{criterion_group, criterion_main, Criterion};
use gpu_iris_mpc::rng::chacha::ChaChaCudaRng;

pub fn criterion_benchmark_chacha12_runner(c: &mut Criterion, buf_size: usize) {
    let mut group = c.benchmark_group(format!(
        "ChaCha12 (buf_size = {}MB)",
        buf_size * 4 / (1024 * 1024)
    ));

    group.throughput(criterion::Throughput::Bytes(
        (buf_size * std::mem::size_of::<u32>()) as u64,
    ));
    let mut chacha = ChaChaCudaRng::init(buf_size);
    group.bench_function("with copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng();
        })
    });
    let mut chacha = ChaChaCudaRng::init(buf_size);
    group.bench_function("without copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng_no_host_copy();
        })
    });
    group.finish();
}

pub fn criterion_benchmark_chacha12(c: &mut Criterion) {
    for log_buf_size in 20..30 {
        let buf_size = (1usize << log_buf_size) / 4;
        criterion_benchmark_chacha12_runner(c, buf_size);
    }
}
criterion_group!(
    name = rng_benches;
    config = Criterion::default();
    targets = criterion_benchmark_chacha12
);
criterion_main!(rng_benches);
