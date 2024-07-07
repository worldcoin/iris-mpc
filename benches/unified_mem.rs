use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use cudarc::driver::{
    result::{malloc_managed, mem_prefetch_async},
    sys::{CUmemAttach_flags, CUmemLocationType, CUmemLocation_st},
};
use gpu_iris_mpc::dot::device_manager::DeviceManager;
use std::{sync::Arc, time::Duration};

const DB_SIZE: usize = 8 * 10_000;
const WIDTH: usize = 12_800;

fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_memcpy");

    let db = vec![0u16; DB_SIZE * WIDTH];
    let device_manager = Arc::new(DeviceManager::init());

    group.throughput(Throughput::Bytes((DB_SIZE * WIDTH * 2) as u64));
    // group.throughput(Throughput::Elements((DB_SIZE / 2) as u64)); // code + mask
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);

    let mem =
        unsafe { malloc_managed(db.len() * 2, CUmemAttach_flags::CU_MEM_ATTACH_GLOBAL).unwrap() };

    unsafe {
        std::ptr::copy(db.as_ptr() as *const _, mem as *mut _, db.len());
    }

    let streams = device_manager.fork_streams();
    let buffer_size = db.len() / 8 * 2;

    group.bench_function(format!("dtoh memcpy {}", DB_SIZE), |b| {
        b.iter(|| {
            for i in 0..8 {
                device_manager.device(i).bind_to_thread().unwrap();
                unsafe {
                    mem_prefetch_async(
                        mem + (buffer_size * i) as u64,
                        buffer_size,
                        CUmemLocation_st {
                            type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                            id:    i as i32,
                        },
                        streams[i].stream,
                    )
                    .unwrap();
                }
            }
            device_manager.await_streams(&streams);
        });
    });
}

criterion_group!(benches, bench_memcpy);
criterion_main!(benches);
