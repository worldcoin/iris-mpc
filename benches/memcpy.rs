use std::{ffi::c_void, time::Duration};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use cudarc::driver::{
    result::{free_sync, malloc_async, memcpy_dtod_async, memcpy_htod_async, stream::null},
    sys::{lib, CUstream_st},
    CudaDevice,
};

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");

    let mut devs = vec![];
    let mut streams = vec![];

    for i in 0..CudaDevice::count().unwrap() {
        let dev = CudaDevice::new(0).unwrap();
        streams.push(dev.fork_default_stream().unwrap());
        devs.push(dev);
    }

    const QUERY_LEN: usize = 930 * 12800;

    let query1 = vec![0u8; QUERY_LEN];
    let query0 = vec![1u8; QUERY_LEN];


    // group.throughput(Throughput::Bytes((query_size * 31 * WIDTH * 2) as u64));
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(10));
    group.warm_up_time(Duration::from_millis(1));

    group.bench_function("malloc_async", |b| {
        b.iter(|| {
            // dev1.bind_to_thread();
            // let dev_ptr1 = unsafe {
            //     malloc_async(stream1.stream, std::mem::size_of::<u8>() * QUERY_LEN).unwrap()
            // };

            // unsafe {
            //     memcpy_htod_async(dev_ptr1, &query1, stream1.stream).unwrap();
            // }

            // let mut thread_handles = vec![];

            let mut query_pinned1: *mut c_void = std::ptr::null_mut();
            unsafe {
                let _ = lib().cuMemAllocHost_v2(&mut query_pinned1, QUERY_LEN);
            }

            let mut query_pinned0: *mut c_void = std::ptr::null_mut();
            unsafe {
                let _ = lib().cuMemAllocHost_v2(&mut query_pinned0, QUERY_LEN);
            }

            for idx in 0..CudaDevice::count().unwrap() {
                let q1 = query1.clone();
                let q0 = query0.clone();
                let s = streams[idx as usize].stream as u64;

                let query1 = unsafe {
                    malloc_async(s as *mut CUstream_st, std::mem::size_of::<u8>() * q1.len())
                        .unwrap()
                };

                let query0 = unsafe {
                    malloc_async(s as *mut CUstream_st, std::mem::size_of::<u8>() * q0.len())
                        .unwrap()
                };

                // let handle = std::thread::spawn(move || unsafe {
                unsafe {
                    // memcpy_htod_async(query1, &q1, s as *mut CUstream_st).unwrap();
                    // memcpy_htod_async(query0, &q0, s as *mut CUstream_st).unwrap();

                    lib().cuMemcpyHtoDAsync_v2(
                        query1,
                        query_pinned1 as *const _,
                        QUERY_LEN,
                        s as *mut CUstream_st,
                    );

                    lib().cuMemcpyHtoDAsync_v2(
                        query0,
                        query_pinned0 as *const _,
                        QUERY_LEN,
                        s as *mut CUstream_st,
                    );
                }
                // });
                // thread_handles.push(handle);
            }

            // for handle in thread_handles {
            //     handle.join();
            // }

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
