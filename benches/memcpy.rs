use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cudarc::driver::{
    result::{free_sync, malloc_async, memcpy_dtod_async, memcpy_htod_async, stream::null},
    CudaDevice,
};

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");
    let dev1 = CudaDevice::new(0).unwrap();
    let stream1 = dev.fork_default_stream().unwrap();

    const QUERY_LEN: usize = 930 * 12800;

    let query = vec![0u8; QUERY_LEN];

    // group.throughput(Throughput::Bytes((query_size * 31 * WIDTH * 2) as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(100));

    group.bench_function("malloc_async", |b| {
        let mut dev_ptr = 0;
        b.iter(|| {
            dev_ptr = unsafe {
                malloc_async(stream.stream, std::mem::size_of::<u8>() * QUERY_LEN).unwrap()
            };
            unsafe {
                memcpy_htod_async(dev_ptr, &query, stream.stream).unwrap();
            }

            memcpy_dtod_async(dst, src, num_bytes, stream)
        });
    });
}

criterion_group!(benches, bench_memcpy);
criterion_main!(benches);
