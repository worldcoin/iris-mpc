use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};
use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::{compile_ptx, Ptx},
};

fn lauch_config(num_threads: u32, num_total: u32) -> LaunchConfig {
    let num_blocks = (num_total + num_threads - 1) / num_threads;
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (num_threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

struct Kernels {
    pub(crate) transpose_64x64: CudaFunction,
    pub(crate) transpose_64x64_shared: CudaFunction,
}

impl Kernels {
    const MOD_NAME: &'static str = "TComp";

    pub(crate) fn new(dev: Arc<CudaDevice>, ptx: Ptx) -> Kernels {
        dev.load_ptx(
            ptx.clone(),
            Self::MOD_NAME,
            &[
                "shared_xor",
                "shared_xor_assign",
                "shared_and_pre",
                "shared_not_inplace",
                "shared_not",
                "shared_lift_mul_sub_split",
                "shared_u64_transpose_pack_u64",
                "shared_u64_transpose_pack_u64_global_mem",
                "shared_u32_transpose_pack_u64",
                "shared_split1",
                "shared_split2",
            ],
        )
        .unwrap();
        let transpose_64x64 = dev
            .get_func(Self::MOD_NAME, "shared_u64_transpose_pack_u64_global_mem")
            .unwrap();
        let transpose_64x64_shared = dev
            .get_func(Self::MOD_NAME, "shared_u64_transpose_pack_u64")
            .unwrap();

        Kernels {
            transpose_64x64,
            transpose_64x64_shared,
        }
    }
}

fn criterion_benchmark_transpose_64(
    c: &mut Criterion,
    dev: Arc<CudaDevice>,
    kernels: &Kernels,
    mat_size: usize,
) {
    let mut group = c.benchmark_group(format!(
        "Transpose 64x64 Field (buf_size = {} kMat)",
        mat_size / 1000
    ));
    group.throughput(criterion::Throughput::Elements(mat_size as u64));
    let buf_size = mat_size * 64;
    let in_a: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    let in_b: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    let mut out_a: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    let mut out_b: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    for log_num_threads in 5..=10 {
        let num_threads = 1 << log_num_threads;
        let config = lauch_config(num_threads, buf_size as u32);
        let kernel = kernels.transpose_64x64.clone();
        group.bench_function(format!("{} threads", num_threads), |b| {
            b.iter(|| unsafe {
                kernel
                    .clone()
                    .launch(
                        config.clone(),
                        (&mut out_a, &mut out_b, &in_a, &in_b, buf_size as i32, 64i32),
                    )
                    .unwrap()
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group(format!(
        "Transpose 64x64 Field Shared (buf_size = {} kMat)",
        mat_size / 1000
    ));
    group.throughput(criterion::Throughput::Elements(buf_size as u64));
    let buf_size = mat_size * 64;
    let in_a: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    let in_b: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    let mut out_a: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    let mut out_b: CudaSlice<u64> = dev.alloc_zeros(buf_size).unwrap();
    for log_num_threads in 5..=8 {
        let num_threads = 1 << log_num_threads;
        let config = lauch_config(num_threads, buf_size as u32);
        let kernel = kernels.transpose_64x64_shared.clone();
        group.bench_function(format!("{} threads", num_threads), |b| {
            b.iter(|| unsafe {
                kernel
                    .clone()
                    .launch(
                        config.clone(),
                        (&mut out_a, &mut out_b, &in_a, &in_b, buf_size as i32, 64i32),
                    )
                    .unwrap()
            })
        });
    }
    group.finish();
}
fn criterion_benchmark_transpose(c: &mut Criterion) {
    let dev = CudaDevice::new(0).unwrap();
    let pts = compile_ptx(include_str!("../src/threshold/cuda/kernel.cu")).unwrap();
    let kernels = Kernels::new(dev.clone(), pts);
    for log_buf_size in 10..=20 {
        let buf_size = 1usize << log_buf_size;
        criterion_benchmark_transpose_64(c, dev.clone(), &kernels, buf_size);
    }
}

criterion_group!(
    name = transpose_benches;
    config = Criterion::default();
    targets = criterion_benchmark_transpose
);
criterion_main!(transpose_benches);
