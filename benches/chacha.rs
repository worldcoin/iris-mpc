use criterion::{criterion_group, criterion_main, Criterion};
use cudarc::driver::CudaDevice;
use gpu_iris_mpc::rng::{aes::AesCudaRng, chacha::ChaChaCudaRng, chacha_field};

pub fn criterion_benchmark_chacha12_field_runner(c: &mut Criterion, buf_size: usize) {
    let mut chacha =
        chacha_field::ChaChaCudaFeRng::init(buf_size, CudaDevice::new(0).unwrap(), [0u32; 8]);
    let mut group = c.benchmark_group(format!(
        "ChaCha12 Field (buf_size = {} kEl)",
        chacha.num_valid() / 1000
    ));
    group.throughput(criterion::Throughput::Elements((chacha.num_valid()) as u64));
    group.bench_function("with copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng();
        })
    });
    let mut chacha =
        chacha_field::ChaChaCudaFeRng::init(buf_size, CudaDevice::new(0).unwrap(), [0u32; 8]);
    group.bench_function("without copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng_no_host_copy();
        })
    });
    group.finish();
}

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

pub fn criterion_benchmark_aes_runner(c: &mut Criterion, buf_size: usize) {
    let mut group = c.benchmark_group(format!(
        "AES (buf_size = {}MB)",
        buf_size * 4 / (1024 * 1024)
    ));

    group.throughput(criterion::Throughput::Bytes(
        (buf_size * std::mem::size_of::<u32>()) as u64,
    ));
    let mut chacha = AesCudaRng::init(buf_size);
    group.bench_function("with copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng();
        })
    });
    let mut chacha = AesCudaRng::init(buf_size);
    group.bench_function("without copy to host", move |b| {
        b.iter(|| {
            chacha.fill_rng_no_host_copy();
        })
    });
    group.finish();
}

pub fn criterion_benchmark_chacha12(c: &mut Criterion) {
    for log_buf_size in 20..=30 {
        let buf_size = (1usize << log_buf_size) / 4;
        criterion_benchmark_chacha12_runner(c, buf_size);
    }
}

pub fn criterion_benchmark_chacha12_field(c: &mut Criterion) {
    for log_buf_size in 10..=20 {
        let buf_size = (1usize << log_buf_size) * 1000;
        criterion_benchmark_chacha12_field_runner(c, buf_size);
    }
}

pub fn criterion_benchmark_aes(c: &mut Criterion) {
    for log_buf_size in 20..=30 {
        let buf_size = (1usize << log_buf_size) / 4;
        criterion_benchmark_aes_runner(c, buf_size);
    }
}
criterion_group!(
    name = rng_benches;
    config = Criterion::default();
    targets = criterion_benchmark_chacha12, criterion_benchmark_chacha12_field, criterion_benchmark_aes
);
criterion_main!(rng_benches);
