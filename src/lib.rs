use std::{ffi::c_void, str::FromStr, sync::Arc};

use cudarc::{
    cublas::{result::gemm_ex, sys, CudaBlas}, driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchAsync, LaunchConfig}, nccl::Id, nvrtc::compile_ptx
};
use once_cell::sync::Lazy;
use rayon::prelude::*;

static COMM_ID: Lazy<Vec<Id>> = Lazy::new(|| {
    (0..CudaDevice::count().unwrap())
        .map(|_| Id::new().unwrap())
        .collect::<Vec<_>>()
});

pub(crate) const P: u16 = (1 << 14) - 3;
const PTX_SRC: &str = include_str!("matmul.cu");
const IRIS_CODE_LENGTH: usize = 12800;
const QUERY_LENGTH: usize = 31 * 10;
const FUNCTION_NAME: &str = "matmul_f14";

struct IdWrapper(Id);

impl FromStr for IdWrapper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = hex::decode(s)
            .unwrap()
            .iter()
            .map(|&c| c as i8)
            .collect::<Vec<_>>();

        let mut id = [0i8; 128];
        id.copy_from_slice(&bytes);

        Ok(IdWrapper(Id::uninit(id)))
    }
}

impl ToString for IdWrapper {
    fn to_string(&self) -> String {
        hex::encode(
            self.0
                .internal()
                .iter()
                .map(|&c| c as u8)
                .collect::<Vec<_>>(),
        )
    }
}

pub fn gemm(
    handle: &CudaBlas,
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
            handle.handle().clone(),
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
    peer_id: usize,
    db_length: usize,
    limbs: usize,
    blass: Vec<CudaBlas>,
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    streams: Vec<CudaStream>,
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
    pub fn init(peer_id: usize, db_entries: &[u16], peer_url: Option<String>) -> Self {
        // TODO: replace with a MAX_DB_SIZE to allow for insertions
        let db_length = db_entries.len() / IRIS_CODE_LENGTH;
        let limbs = 2;
        let ptx = compile_ptx(PTX_SRC).unwrap();

        // Start all devices
        let (devs, (blass, (kernels, streams))): (
            Vec<Arc<CudaDevice>>,
            (Vec<CudaBlas>, (Vec<CudaFunction>, Vec<CudaStream>)),
        ) = (0..CudaDevice::count().unwrap() as usize)
            .map(|i| {
                let dev = CudaDevice::new(i).unwrap();
                let blas = CudaBlas::new(dev.clone()).unwrap();
                let stream = dev.fork_default_stream().unwrap();
                unsafe {
                    blas.set_stream(Some(&stream)).unwrap();
                }
                dev.load_ptx(ptx.clone(), FUNCTION_NAME, &[FUNCTION_NAME])
                    .unwrap();
                let function = dev.get_func(FUNCTION_NAME, FUNCTION_NAME).unwrap();
                (dev, (blas, (function, stream)))
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

        if peer_url.is_some() {
            for i in 0..CudaDevice::count().unwrap() as usize {
                let id = if peer_id == 0 {
                    COMM_ID[i]
                } else {
                    let res = reqwest::blocking::get(format!("http://{}/{}", peer_url.clone().unwrap(), i)).unwrap();
                    IdWrapper::from_str(&res.text().unwrap()).unwrap().0
                };
            }
        }

        Self {
            peer_id,
            db_length,
            limbs,
            blass,
            devs,
            kernels,
            streams,
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

    pub fn dot(&mut self, preprocessed_query: &Vec<Vec<u8>>) {
        let b_dev: Vec<Vec<CudaSlice<u8>>> = (0..CudaDevice::count().unwrap() as usize)
            .map(|idx| {
                preprocessed_query
                    .iter()
                    .map(|b| self.devs[idx].htod_sync_copy(b).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

            let num_elements = self.db_length * QUERY_LENGTH;
            let threads_per_block = 256;
            let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
            let cfg = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };

        for idx in 0..self.devs.len() {
            // TODO: check this is actually async

            // Calculate sums to correct output
            gemm(
                &self.blass[idx],
                &b_dev[idx][1],
                &self.ones[idx],
                &mut self.query1_sums[idx],
                0,
                0,
                0,
                QUERY_LENGTH,
                1,
                IRIS_CODE_LENGTH,
                1,
                0,
            );

            gemm(
                &self.blass[idx],
                &b_dev[idx][1],
                &self.ones[idx],
                &mut self.query1_sums[idx],
                0,
                0,
                0,
                QUERY_LENGTH,
                1,
                IRIS_CODE_LENGTH,
                1,
                0,
            );

            for (i, d) in [&self.db0[idx], &self.db1[idx]].iter().enumerate() {
                for (j, q) in b_dev[idx].iter().enumerate() {
                    gemm(
                        &self.blass[idx],
                        d,
                        q,
                        &mut self.intermediate_results[idx],
                        0,
                        0,
                        (self.db_length * QUERY_LENGTH * 4 * (i * 2 + j)) as u64,
                        self.db_length,
                        QUERY_LENGTH,
                        IRIS_CODE_LENGTH,
                        1,
                        0,
                    );
                }
            }

            unsafe {
                self.kernels[idx].clone().launch_on_stream(
                    &self.streams[idx],
                    cfg,
                    (
                        &self.intermediate_results[idx],
                        &mut self.results[idx],
                        &self.db0_sums[idx],
                        &self.db1_sums[idx],
                        &self.query0_sums[idx],
                        &self.query1_sums[idx],
                        self.db_length as u64,
                        (self.db_length * QUERY_LENGTH) as u64,
                        IRIS_CODE_LENGTH as u64,
                        P,
                    ),
                ).unwrap();
            }
        }
    }

    pub fn exchange() {



    }
}
