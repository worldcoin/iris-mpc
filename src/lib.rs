use std::{ffi::c_void, sync::Arc, time::Instant};

use cudarc::{
    cublas::{result::gemm_ex, sys, CudaBlas},
    driver::{
        result, sys::lib, CudaDevice, CudaFunction, CudaSlice, DevicePtr, DevicePtrMut,
        LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};
use num_traits::FromPrimitive;
use rayon::prelude::*;

pub(crate) const P: u16 = (1 << 14) - 3;
const PTX_SRC: &str = include_str!("matmul.cu");
const IRIS_CODE_LENGTH: usize = 12800;
const QUERY_LENGTH: usize = 31 * 10;
const FUNCTION_NAME: &str = "matmul_f14";

pub fn gemm(
    handle: &sys::cublasHandle_t,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<i32>,
    a_offset: u64,
    b_offset: u64,
    c_offset: u64,
    m: usize,
    n: usize,
    k: usize,
    alpha: i32,
    beta: i32,
) {
    unsafe {
        gemm_ex(
            handle.clone(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const i32 as *const c_void,
            (*a.device_ptr() + a_offset) as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            (*b.device_ptr() + b_offset) as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            &beta as *const i32 as *const c_void,
            (*c.device_ptr_mut() + c_offset) as *mut _,
            sys::cublasDataType_t::CUDA_R_32I,
            m as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I_PEDANTIC,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .unwrap();
    }
}

pub struct IrisDB {
    db_length: usize,
    limbs: usize,
    blass: Vec<CudaBlas>,
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    db1: Vec<CudaSlice<u8>>,
    db0: Vec<CudaSlice<u8>>,
    db1_sums: Vec<CudaSlice<u32>>,
    db0_sums: Vec<CudaSlice<u32>>,
    query1_sums: Vec<CudaSlice<i32>>,
    query0_sums: Vec<CudaSlice<i32>>,
    ones: Vec<CudaSlice<u8>>,
    intermediate_results: Vec<CudaSlice<i32>>,
    results: Vec<CudaSlice<u16>>,
}

impl IrisDB {
    pub fn init(db_entries: &[u16]) -> Self {
        // TOOD: replace with a MAX_DB_SIZE to allow for insertions
        let db_length = db_entries.len() / IRIS_CODE_LENGTH;
        let limbs = 2;
        let ptx = compile_ptx(PTX_SRC).unwrap();

        // Start all devices
        let (devs, (blass, kernels)): (Vec<Arc<CudaDevice>>, (Vec<CudaBlas>, Vec<CudaFunction>)) =
            (0..CudaDevice::count().unwrap() as usize)
                .map(|i| {
                    let dev = CudaDevice::new(i).unwrap();
                    let blas = CudaBlas::new(dev.clone()).unwrap();
                    dev.load_ptx(ptx.clone(), FUNCTION_NAME, &[FUNCTION_NAME])
                        .unwrap();
                    let function = dev.get_func(FUNCTION_NAME, FUNCTION_NAME).unwrap();
                    (dev, (blas, function))
                })
                .unzip();

        let mut a1_host = db_entries
            .iter()
            .map(|&x: &u16| (x >> 8) as u8)
            .collect::<Vec<_>>();
        let mut a0_host = db_entries.iter().map(|&x| x as u8).collect::<Vec<_>>();

        // TODO: maybe use gemm here already to speed up loading (we'll need to correct the results as well)
        let a1_sums: Vec<u32> = a1_host
            .par_chunks(IRIS_CODE_LENGTH)
            .map(|row| row.iter().map(|&x| x as u32).sum::<u32>())
            .collect();

        let a0_sums: Vec<u32> = a0_host
            .par_chunks(IRIS_CODE_LENGTH)
            .map(|row| row.iter().map(|&x| x as u32).sum::<u32>())
            .collect();

        a1_host
            .par_iter_mut()
            .for_each(|x| *x = (*x as i32 - 128) as u8);

        a0_host
            .par_iter_mut()
            .for_each(|x| *x = (*x as i32 - 128) as u8);

        // Split up db and load to all devices
        let chunk_size = db_length / CudaDevice::count().unwrap() as usize;

        let db1_sums = a1_sums
            .chunks(chunk_size)
            .enumerate()
            .map(|(idx, chunk)| devs[idx].htod_sync_copy(&chunk).unwrap())
            .collect::<Vec<_>>();
        let db0_sums = a0_sums
            .chunks(chunk_size)
            .enumerate()
            .map(|(idx, chunk)| devs[idx].htod_sync_copy(&chunk).unwrap())
            .collect::<Vec<_>>();

        let db1 = a1_host
            .chunks(chunk_size * IRIS_CODE_LENGTH)
            .enumerate()
            .map(|(idx, chunk)| devs[idx].htod_sync_copy(&chunk).unwrap())
            .collect::<Vec<_>>();
        let db0 = a0_host
            .chunks(chunk_size * IRIS_CODE_LENGTH)
            .enumerate()
            .map(|(idx, chunk)| devs[idx].htod_sync_copy(&chunk).unwrap())
            .collect::<Vec<_>>();

        let (query1_sums, query0_sums): (Vec<CudaSlice<i32>>, Vec<CudaSlice<i32>>) = (0
            ..CudaDevice::count().unwrap() as usize)
            .map(|idx| {
                (
                    devs[idx].alloc_zeros(QUERY_LENGTH).unwrap(),
                    devs[idx].alloc_zeros(QUERY_LENGTH).unwrap(),
                )
            })
            .unzip();

        let ones = vec![1u8; IRIS_CODE_LENGTH];
        let ones = (0..CudaDevice::count().unwrap() as usize)
            .map(|idx| devs[idx].htod_sync_copy(&ones).unwrap())
            .collect::<Vec<_>>();

        //TODO: depending on the batch size, intermediate_results can get quite big, we can perform the gemm in chunks to limit this
        let (intermediate_results, results): (Vec<CudaSlice<i32>>, Vec<CudaSlice<u16>>) = (0
            ..CudaDevice::count().unwrap() as usize)
            .map(|idx| {
                (
                    devs[idx]
                        .alloc_zeros(db_length * IRIS_CODE_LENGTH * 4)
                        .unwrap(),
                    devs[idx].alloc_zeros(db_length * IRIS_CODE_LENGTH).unwrap(),
                )
            })
            .unzip();

        Self {
            db_length,
            limbs,
            blass,
            devs,
            kernels,
            db1,
            db0,
            db1_sums,
            db0_sums,
            query1_sums,
            query0_sums,
            intermediate_results,
            ones,
            results,
        }
    }

    pub fn preprocess_query(&self, query: &[u16]) -> Vec<Vec<u8>> {
        let mut result = vec![];
        for _ in 0..self.limbs {
            result.push(vec![0u8; query.len()]);
        }

        for (idx, &entry) in query.iter().enumerate() {
            for i in 0..self.limbs {
                let tmp = (entry as u32 >> (i * 8)) as u8;
                result[i][idx] = (tmp as i32 - 128) as u8;
            }
        }

        result.to_vec()
    }

    pub fn dot(&mut self, preprocessed_query: &Vec<Vec<u8>>, results_host: *mut u16) {
        // TODO
    }

}
