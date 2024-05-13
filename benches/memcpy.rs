use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cudarc::driver::{
    result::{free_sync, malloc_async, memcpy_dtod_async, memcpy_htod_async, stream::null},
    CudaDevice,
};

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");
    let dev1 = CudaDevice::new(0).unwrap();
    let stream1 = dev1.fork_default_stream().unwrap();

    let dev2 = CudaDevice::new(1).unwrap();
    let stream2 = dev2.fork_default_stream().unwrap();

    const QUERY_LEN: usize = 930 * 12800;

    let query1 = vec![0u8; QUERY_LEN];
    let query2 = vec![1u8; QUERY_LEN];

    // group.throughput(Throughput::Bytes((query_size * 31 * WIDTH * 2) as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(100));

    group.bench_function("malloc_async", |b| {
        b.iter(|| {
            dev1.bind_to_thread();
            let dev_ptr1 = unsafe {
                malloc_async(stream1.stream, std::mem::size_of::<u8>() * QUERY_LEN).unwrap()
            };

            unsafe {
                memcpy_htod_async(dev_ptr1, &query1, stream1.stream).unwrap();
            }

            // let dev_ptr2 = unsafe {
            //     malloc_async(stream1.stream, std::mem::size_of::<u8>() * QUERY_LEN).unwrap()
            // };

            // unsafe {
            //     memcpy_htod_async(dev_ptr2, &query2, stream1.stream).unwrap();
            // }

            // dev2.bind_to_thread();

            // let dev_ptr2 = unsafe {
            //     malloc_async(stream2.stream, std::mem::size_of::<u8>() * QUERY_LEN).unwrap()
            // };

            // unsafe {
            //     memcpy_htod_async(dev_ptr2, &query, stream2.stream).unwrap();
            // }
            // dev1.bind_to_thread();
            // unsafe {
            //     memcpy_dtod_async(dev_ptr2, dev_ptr1, std::mem::size_of::<u8>() * QUERY_LEN, stream1.stream);
            // }
        });
    });
}

criterion_group!(benches, bench_memcpy);
criterion_main!(benches);
