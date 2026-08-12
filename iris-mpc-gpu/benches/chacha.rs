use criterion::{criterion_group, criterion_main, Criterion};
use cudarc::driver::CudaDevice;
use iris_mpc_gpu::rng::chacha::ChaChaCudaRng;

pub fn criterion_benchmark_chacha12_runner(c: &mut Criterion, buf_size_bytes: usize) {
    let mut group = c.benchmark_group(format!(
        "ChaCha12 (buf_size = {}MB)",
        buf_size_bytes / (1024 * 1024)
    ));

    group.throughput(criterion::Throughput::Bytes(buf_size_bytes as u64));
    let mut chacha = ChaChaCudaRng::init(buf_size_bytes, CudaDevice::new(0).unwrap(), [0u32; 8]);
    group.bench_function("with copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng();
        })
    });
    let mut chacha = ChaChaCudaRng::init(buf_size_bytes, CudaDevice::new(0).unwrap(), [0u32; 8]);
    let dev = CudaDevice::new(0).unwrap();
    let stream = dev.fork_default_stream().unwrap();
    group.bench_function("without copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng_no_host_copy(buf_size_bytes, &stream);
        })
    });
    group.finish();
}

pub fn criterion_benchmark_chacha12(c: &mut Criterion) {
    for log_buf_size in 20..=30 {
        criterion_benchmark_chacha12_runner(c, 1usize << log_buf_size);
    }
}

criterion_group!(
    name = rng_benches;
    config = Criterion::default();
    targets = criterion_benchmark_chacha12
);
criterion_main!(rng_benches);
