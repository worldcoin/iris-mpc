use crate::{setup::shamir::Shamir, threshold::cuda::kernel};
use cudarc::{
    driver::{CudaDevice, CudaFunction, DriverError, LaunchAsync, LaunchConfig},
    nvrtc::{self, Ptx},
};
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use std::sync::Arc;

use super::super::cuda::PTX_SRC;

fn random_vec<T, R: Rng>(n: usize, rng: &mut R) -> Vec<T>
where
    Standard: Distribution<T>,
{
    (0..n).map(|_| rng.gen()).collect()
}

fn random_vec_fp<R: Rng>(n: usize, rng: &mut R) -> Vec<u16> {
    (0..n).map(|_| Shamir::random_fp(rng)).collect()
}

fn launch_config_from_elements_and_threads(n: u32, t: u32) -> LaunchConfig {
    let num_blocks = (n + t - 1) / t;
    LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (t, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn setup_cuda(device_id: usize) -> Result<(Arc<CudaDevice>, Ptx), DriverError> {
    let dev = CudaDevice::new(device_id)?;
    let ptx = nvrtc::compile_ptx(PTX_SRC).unwrap();
    Ok((dev, ptx))
}

fn load_function(
    function: &'static str,
    dev: &Arc<CudaDevice>,
    ptx: Ptx,
) -> Result<CudaFunction, DriverError> {
    dev.load_ptx(ptx, function, &[function])?;
    let kernel = dev.get_func(function, function).unwrap();
    Ok(kernel)
}

const SIZE: usize = 1024;
const TESTRUNS: usize = 5;

#[test]
fn xor_test() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_xor", &dev, ptx).unwrap();
    let cfg = LaunchConfig::for_num_elems(SIZE as u32);

    // CPU
    let mut res_a = vec![0u64; SIZE];
    let mut res_b = vec![0u64; SIZE];

    // GPU
    let mut res_a_gpu = dev.alloc_zeros::<u64>(SIZE).unwrap();
    let mut res_b_gpu = dev.alloc_zeros::<u64>(SIZE).unwrap();
    let mut res_a_cpu = vec![0u64; SIZE];
    let mut res_b_cpu = vec![0u64; SIZE];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec(SIZE, &mut rng);
        let lhs_b = random_vec(SIZE, &mut rng);
        let rhs_a = random_vec(SIZE, &mut rng);
        let rhs_b = random_vec(SIZE, &mut rng);

        kernel::shared_xor(&mut res_a, &mut res_b, &lhs_a, &lhs_b, &rhs_a, &rhs_b);

        // GPU
        let lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();
        let rhs_a_cuda = dev.htod_copy(rhs_a).unwrap();
        let rhs_b_cuda = dev.htod_copy(rhs_b).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut res_a_gpu,
                        &mut res_b_gpu,
                        &lhs_a_cuda,
                        &lhs_b_cuda,
                        &rhs_a_cuda,
                        &rhs_b_cuda,
                        SIZE as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&res_a_gpu, &mut res_a_cpu).unwrap();
        dev.dtoh_sync_copy_into(&res_b_gpu, &mut res_b_cpu).unwrap();
        assert_eq!(res_a_cpu.len(), SIZE);
        assert_eq!(res_b_cpu.len(), SIZE);

        // Compare
        for i in 0..SIZE {
            assert_eq!(res_a[i], res_a_cpu[i]);
            assert_eq!(res_b[i], res_b_cpu[i]);
        }
    }
}

#[test]
fn xor_assign_test() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_xor_assign", &dev, ptx).unwrap();
    let cfg = LaunchConfig::for_num_elems(SIZE as u32);

    let mut res_a_cpu = vec![0u64; SIZE];
    let mut res_b_cpu = vec![0u64; SIZE];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let mut lhs_a = random_vec(SIZE, &mut rng);
        let mut lhs_b = random_vec(SIZE, &mut rng);
        let rhs_a = random_vec(SIZE, &mut rng);
        let rhs_b = random_vec(SIZE, &mut rng);

        let lhs_a_ = lhs_a.clone();
        let lhs_b_ = lhs_b.clone();

        kernel::shared_xor_assign(&mut lhs_a, &mut lhs_b, &rhs_a, &rhs_b);

        // GPU
        let mut lhs_a_cuda = dev.htod_copy(lhs_a_).unwrap();
        let mut lhs_b_cuda = dev.htod_copy(lhs_b_).unwrap();
        let rhs_a_cuda = dev.htod_copy(rhs_a).unwrap();
        let rhs_b_cuda = dev.htod_copy(rhs_b).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut lhs_a_cuda,
                        &mut lhs_b_cuda,
                        &rhs_a_cuda,
                        &rhs_b_cuda,
                        SIZE as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&lhs_a_cuda, &mut res_a_cpu)
            .unwrap();
        dev.dtoh_sync_copy_into(&lhs_b_cuda, &mut res_b_cpu)
            .unwrap();
        assert_eq!(res_a_cpu.len(), SIZE);
        assert_eq!(res_b_cpu.len(), SIZE);

        // Compare
        for i in 0..SIZE {
            assert_eq!(lhs_a[i], res_a_cpu[i]);
            assert_eq!(lhs_b[i], res_b_cpu[i]);
        }
    }
}

#[test]
fn and_pre_test() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_and_pre", &dev, ptx).unwrap();
    let cfg = LaunchConfig::for_num_elems(SIZE as u32);

    // CPU
    let mut res_a = vec![0u64; SIZE];

    // GPU
    let mut res_a_gpu = dev.alloc_zeros::<u64>(SIZE).unwrap();
    let mut res_a_cpu = vec![0u64; SIZE];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec(SIZE, &mut rng);
        let lhs_b = random_vec(SIZE, &mut rng);
        let rhs_a = random_vec(SIZE, &mut rng);
        let rhs_b = random_vec(SIZE, &mut rng);
        let rand = random_vec(SIZE, &mut rng); // This should in a production also happen on the GPU

        kernel::shared_and_pre(&mut res_a, &lhs_a, &lhs_b, &rhs_a, &rhs_b, &rand);

        // GPU
        let lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();
        let rhs_a_cuda = dev.htod_copy(rhs_a).unwrap();
        let rhs_b_cuda = dev.htod_copy(rhs_b).unwrap();
        let rand_cuda = dev.htod_copy(rand).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut res_a_gpu,
                        &lhs_a_cuda,
                        &lhs_b_cuda,
                        &rhs_a_cuda,
                        &rhs_b_cuda,
                        &rand_cuda,
                        SIZE as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&res_a_gpu, &mut res_a_cpu).unwrap();
        assert_eq!(res_a_cpu.len(), SIZE);

        // Compare
        for i in 0..SIZE {
            assert_eq!(res_a[i], res_a_cpu[i]);
        }
    }
}

#[test]
fn or_pre_test() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_or_pre_assign", &dev, ptx).unwrap();
    let cfg = LaunchConfig::for_num_elems(SIZE as u32);

    // CPU
    let mut res_a = vec![0u64; SIZE];

    // GPU
    let mut res_a_cpu = vec![0u64; SIZE];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec(SIZE, &mut rng);
        let lhs_b = random_vec(SIZE, &mut rng);
        let rhs_a = random_vec(SIZE, &mut rng);
        let rhs_b = random_vec(SIZE, &mut rng);
        let rand = random_vec(SIZE, &mut rng); // This should in a production also happen on the GPU

        kernel::shared_or_pre(&mut res_a, &lhs_a, &lhs_b, &rhs_a, &rhs_b, &rand);

        // GPU
        let mut lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();
        let rhs_a_cuda = dev.htod_copy(rhs_a).unwrap();
        let rhs_b_cuda = dev.htod_copy(rhs_b).unwrap();
        let rand_cuda = dev.htod_copy(rand).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut lhs_a_cuda,
                        &lhs_b_cuda,
                        &rhs_a_cuda,
                        &rhs_b_cuda,
                        &rand_cuda,
                        SIZE as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&lhs_a_cuda, &mut res_a_cpu)
            .unwrap();
        assert_eq!(res_a_cpu.len(), SIZE);

        // Compare
        for i in 0..SIZE {
            assert_eq!(res_a[i], res_a_cpu[i]);
        }
    }
}

#[test]
fn mul_lift_test() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_mul_lift_b", &dev, ptx).unwrap();
    let cfg = LaunchConfig::for_num_elems(SIZE as u32);

    // CPU
    let mut res_a = vec![0u64; SIZE];
    let mut res_b = vec![0u64; SIZE];

    // GPU
    let mut res_a_gpu = dev.alloc_zeros::<u64>(SIZE).unwrap();
    let mut res_b_gpu = dev.alloc_zeros::<u64>(SIZE).unwrap();
    let mut res_a_cpu = vec![0u64; SIZE];
    let mut res_b_cpu = vec![0u64; SIZE];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec_fp(SIZE, &mut rng);
        let lhs_b = random_vec_fp(SIZE, &mut rng);

        kernel::shared_mul_lift_b(&mut res_a, &mut res_b, &lhs_a, &lhs_b);

        // GPU
        let lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut res_a_gpu,
                        &mut res_b_gpu,
                        &lhs_a_cuda,
                        &lhs_b_cuda,
                        SIZE as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&res_a_gpu, &mut res_a_cpu).unwrap();
        dev.dtoh_sync_copy_into(&res_b_gpu, &mut res_b_cpu).unwrap();
        assert_eq!(res_a_cpu.len(), SIZE);
        assert_eq!(res_b_cpu.len(), SIZE);

        // Compare
        for i in 0..SIZE {
            assert_eq!(res_a[i], res_a_cpu[i]);
            assert_eq!(res_b[i], res_b_cpu[i]);
        }
    }
}

fn transpose_16x64_test_with_len<const L: usize>() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_u16_transpose_pack_u64", &dev, ptx).unwrap();

    #[allow(clippy::assertions_on_constants)]
    {
        assert!(SIZE % 64 == 0);
    }
    const N: usize = SIZE / 64;

    let cfg = LaunchConfig::for_num_elems(2 * N as u32);

    // GPU
    let mut res_a_gpu = dev.alloc_zeros::<u64>(N * L).unwrap();
    let mut res_b_gpu = dev.alloc_zeros::<u64>(N * L).unwrap();
    let mut res_a_cpu = vec![0u64; N * L];
    let mut res_b_cpu = vec![0u64; N * L];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec_fp(SIZE, &mut rng);
        let lhs_b = random_vec_fp(SIZE, &mut rng);

        let res_a = kernel::u16_transpose_pack_u64_with_len::<L>(&lhs_a);
        let res_b = kernel::u16_transpose_pack_u64_with_len::<L>(&lhs_b);

        // GPU
        let lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut res_a_gpu,
                        &mut res_b_gpu,
                        &lhs_a_cuda,
                        &lhs_b_cuda,
                        SIZE as u32,
                        L as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&res_a_gpu, &mut res_a_cpu).unwrap();
        dev.dtoh_sync_copy_into(&res_b_gpu, &mut res_b_cpu).unwrap();
        assert_eq!(res_a_cpu.len(), N * L);
        assert_eq!(res_b_cpu.len(), N * L);

        for (c, g) in res_a.iter().zip(res_a_cpu.chunks_exact(N)) {
            for i in 0..N {
                assert_eq!(c[i], g[i]);
            }
        }

        for (c, g) in res_b.iter().zip(res_b_cpu.chunks_exact(N)) {
            for i in 0..N {
                assert_eq!(c[i], g[i]);
            }
        }
    }
}

#[test]
fn transpose_16x64_test() {
    transpose_16x64_test_with_len::<16>();
    transpose_16x64_test_with_len::<6>();
    transpose_16x64_test_with_len::<2>();
}

fn transpose_32x64_test_with_len<const L: usize>() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_u32_transpose_pack_u64", &dev, ptx).unwrap();

    #[allow(clippy::assertions_on_constants)]
    {
        assert!(SIZE % 64 == 0);
    }
    const N: usize = SIZE / 64;

    let cfg = launch_config_from_elements_and_threads(2 * N as u32, 512);

    // GPU
    let mut res_a_gpu = dev.alloc_zeros::<u64>(N * L).unwrap();
    let mut res_b_gpu = dev.alloc_zeros::<u64>(N * L).unwrap();
    let mut res_a_cpu = vec![0u64; N * L];
    let mut res_b_cpu = vec![0u64; N * L];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec::<u32, _>(SIZE, &mut rng);
        let lhs_b = random_vec::<u32, _>(SIZE, &mut rng);

        let res_a = kernel::u32_transpose_pack_u64_with_len::<L>(&lhs_a);
        let res_b = kernel::u32_transpose_pack_u64_with_len::<L>(&lhs_b);

        // GPU
        let lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut res_a_gpu,
                        &mut res_b_gpu,
                        &lhs_a_cuda,
                        &lhs_b_cuda,
                        SIZE as u32,
                        L as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&res_a_gpu, &mut res_a_cpu).unwrap();
        dev.dtoh_sync_copy_into(&res_b_gpu, &mut res_b_cpu).unwrap();
        assert_eq!(res_a_cpu.len(), N * L);
        assert_eq!(res_b_cpu.len(), N * L);

        for (c, g) in res_a.iter().zip(res_a_cpu.chunks_exact(N)) {
            for i in 0..N {
                assert_eq!(c[i], g[i]);
            }
        }

        for (c, g) in res_b.iter().zip(res_b_cpu.chunks_exact(N)) {
            for i in 0..N {
                assert_eq!(c[i], g[i]);
            }
        }
    }
}

#[test]
fn transpose_32x64_test() {
    transpose_32x64_test_with_len::<32>();
    transpose_32x64_test_with_len::<18>();
    transpose_32x64_test_with_len::<2>();
}

fn transpose_64x64_test_with_len<const L: usize>() {
    // Setup cuda
    let (dev, ptx) = setup_cuda(0).unwrap();
    let function = load_function("shared_u64_transpose_pack_u64", &dev, ptx).unwrap();

    #[allow(clippy::assertions_on_constants)]
    {
        assert!(SIZE % 64 == 0);
    }
    const N: usize = SIZE / 64;

    let cfg = launch_config_from_elements_and_threads(2 * N as u32, 512);

    // GPU
    let mut res_a_gpu = dev.alloc_zeros::<u64>(N * L).unwrap();
    let mut res_b_gpu = dev.alloc_zeros::<u64>(N * L).unwrap();
    let mut res_a_cpu = vec![0u64; N * L];
    let mut res_b_cpu = vec![0u64; N * L];

    let mut rng = rand::thread_rng();
    for _ in 0..TESTRUNS {
        // CPU
        let lhs_a = random_vec::<u64, _>(SIZE, &mut rng);
        let lhs_b = random_vec::<u64, _>(SIZE, &mut rng);
        let mut lhs_a_copy = lhs_a.clone();
        let mut lhs_b_copy = lhs_b.clone();

        let res_a = kernel::u64_transpose_pack_u64_with_len::<L>(&mut lhs_a_copy);
        let res_b = kernel::u64_transpose_pack_u64_with_len::<L>(&mut lhs_b_copy);

        // GPU
        let lhs_a_cuda = dev.htod_copy(lhs_a).unwrap();
        let lhs_b_cuda = dev.htod_copy(lhs_b).unwrap();

        unsafe {
            function
                .to_owned()
                .launch(
                    cfg,
                    (
                        &mut res_a_gpu,
                        &mut res_b_gpu,
                        &lhs_a_cuda,
                        &lhs_b_cuda,
                        SIZE as u32,
                        L as u32,
                    ),
                )
                .unwrap();
        }
        dev.dtoh_sync_copy_into(&res_a_gpu, &mut res_a_cpu).unwrap();
        dev.dtoh_sync_copy_into(&res_b_gpu, &mut res_b_cpu).unwrap();
        assert_eq!(res_a_cpu.len(), N * L);
        assert_eq!(res_b_cpu.len(), N * L);

        for (c, g) in res_a.iter().zip(res_a_cpu.chunks_exact(N)) {
            for i in 0..N {
                assert_eq!(c[i], g[i]);
            }
        }

        for (c, g) in res_b.iter().zip(res_b_cpu.chunks_exact(N)) {
            for i in 0..N {
                assert_eq!(c[i], g[i]);
            }
        }
    }
}

#[test]
fn transpose_64x64_test() {
    transpose_64x64_test_with_len::<64>();
    transpose_64x64_test_with_len::<34>();
    transpose_64x64_test_with_len::<18>();
    transpose_64x64_test_with_len::<2>();
}
