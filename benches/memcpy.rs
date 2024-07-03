use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use cudarc::driver::{result, sys::lib};
use gpu_iris_mpc::dot::device_manager::DeviceManager;
use std::{ffi::c_void, sync::Arc, time::Duration};

const DB_SIZE: usize = 8 * 10_000;
const WIDTH: usize = 12_800;

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");

    let db = vec![0u16; DB_SIZE * WIDTH];
    let dbs = db.chunks(db.len() / 8).collect::<Vec<_>>();
    let device_manager = Arc::new(DeviceManager::init());

    // group.throughput(Throughput::Bytes((DB_SIZE * WIDTH * 2) as u64));
    group.throughput(Throughput::Elements((DB_SIZE / 2) as u64)); // code + mask
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);

    let mut host_mems = vec![];
    for i in 0..8 {
        let mut pinned: *mut c_void = std::ptr::null_mut();
        unsafe {
            lib().cuMemAllocHost_v2(&mut pinned, db.len() / 8 * 2);
            std::ptr::copy(dbs[i].as_ptr() as *const _, pinned as *mut _, db.len() / 8);
            host_mems.push(pinned);
        }
    }

    let streams = device_manager.fork_streams();

    let mut dsts = vec![];
    for i in 0..8 {
        unsafe {
            dsts.push(result::malloc_async(streams[i].stream, db.len() / 8 * 2).unwrap());
        }
    }

    device_manager.await_streams(&streams);

    group.bench_function(format!("dtoh memcpy {}", DB_SIZE), |b| {
        b.iter(|| {
            for i in 0..8 {
                device_manager.device(i).bind_to_thread().unwrap();
                unsafe {
                    lib().cuMemcpyHtoDAsync_v2(
                        dsts[i],
                        host_mems[i],
                        db.len() / 8 * 2,
                        streams[i].stream,
                    );
                }
            }
            device_manager.await_streams(&streams);
        });
    });
}

criterion_group!(benches, bench_memcpy);
criterion_main!(benches);
