pub mod device_manager;
pub mod mmap;
pub mod rng;
pub mod setup;
pub mod sqs;

use std::{
    ffi::c_void,
    mem, ptr,
    str::FromStr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use axum::{extract::Path, routing::get, Router};
use cudarc::{
    cublas::{result::gemm_ex, sys, CudaBlas},
    driver::{
        result::{
            event, launch_kernel, malloc_async, memcpy_dtoh_async, memcpy_dtoh_sync,
            memcpy_htod_async, memset_d8_async,
            stream::{self, synchronize, wait_event},
        },
        sys::{CUevent, CUevent_flags},
        CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr,
        DeviceSlice, LaunchAsync, LaunchConfig,
    },
    nccl::{result, Comm, Id, NcclType},
    nvrtc::{compile_ptx, compile_ptx_with_opts, CompileOptions},
};
use device_manager::DeviceManager;
use rayon::prelude::*;
use rng::chacha_field::ChaChaCudaFeRng;

pub(crate) const P: u16 = 65519;
const PTX_SRC: &str = include_str!("kernel.cu");
const IRIS_CODE_LENGTH: usize = 12_800;
const CHACHA_BUFFER_SIZE: usize = 1000;
const MATCH_RATIO: f64 = 0.375;
const LIMBS: usize = 2;
const ROTATIONS: usize = 31;
const MATMUL_FUNCTION_NAME: &str = "matmul";
const DIST_FUNCTION_NAME: &str = "reconstructAndCompare";
const DEDUPAPPEND_FUNCTION_NAME: &str = "dedupAndAppend";

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

pub fn preprocess_query(query: &[u16]) -> Vec<Vec<u8>> {
    let mut result = vec![];
    for _ in 0..LIMBS {
        result.push(vec![0u8; query.len()]);
    }

    for (idx, &entry) in query.iter().enumerate() {
        for i in 0..LIMBS {
            let tmp = (entry as u32 >> (i * 8)) as u8;
            result[i][idx] = (tmp as i32 - 128) as u8;
        }
    }

    result.to_vec()
}

pub fn device_ptrs<T>(slice: &Vec<CudaSlice<T>>) -> Vec<u64> {
    slice.iter().map(|s| *s.device_ptr()).collect()
}

pub fn gemm(
    handle: &CudaBlas,
    a: u64,
    b: u64,
    c: u64,
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
            (a + a_offset) as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            (b + b_offset) as *const _,
            sys::cublasDataType_t::CUDA_R_8I,
            k as i32,
            &beta as *const i32 as *const c_void,
            (c + c_offset) as *mut _,
            sys::cublasDataType_t::CUDA_R_32I,
            m as i32,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32I_PEDANTIC,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
        .unwrap();
    }
}

fn broadcast_stream<T: NcclType>(
    sendbuff: &Option<&CudaSlice<T>>,
    recvbuff: &mut CudaSlice<T>,
    root: i32,
    len: usize,
    comm: &Comm,
    stream: &CudaStream,
) -> Result<result::NcclStatus, result::NcclError> {
    unsafe {
        let send_ptr = match sendbuff {
            Some(buffer) => *buffer.device_ptr() as *const _,
            None => ptr::null(),
        };
        result::broadcast(
            send_ptr,
            *recvbuff.device_ptr() as *mut c_void,
            len,
            T::as_nccl_type(),
            root,
            comm.comm,
            stream.stream as *mut _,
        )
    }
}

async fn http_root(ids: Vec<Id>, Path(device_id): Path<String>) -> String {
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(ids[device_id]).to_string()
}

pub struct DistanceComparator {
    pub devs: Vec<Arc<CudaDevice>>,
    pub dist_kernels: Vec<CudaFunction>,
    pub dedup_kernels: Vec<CudaFunction>,
    pub query_length: usize,
    pub n_devices: usize,
}

impl DistanceComparator {
    pub fn init(query_length: usize) -> Self {
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let mut dist_kernels = Vec::new();
        let mut dedup_kernels = Vec::new();
        let mut devs = Vec::new();
        let n_devices = CudaDevice::count().unwrap() as usize;

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            dev.load_ptx(ptx.clone(), DIST_FUNCTION_NAME, &[DIST_FUNCTION_NAME])
                .unwrap();
            let dist_function = dev
                .get_func(DIST_FUNCTION_NAME, DIST_FUNCTION_NAME)
                .unwrap();

            dev.load_ptx(
                ptx.clone(),
                DEDUPAPPEND_FUNCTION_NAME,
                &[DEDUPAPPEND_FUNCTION_NAME],
            )
            .unwrap();
            let dedup_function = dev
                .get_func(DEDUPAPPEND_FUNCTION_NAME, DEDUPAPPEND_FUNCTION_NAME)
                .unwrap();

            dist_kernels.push(dist_function);
            dedup_kernels.push(dedup_function);
            devs.push(dev);
        }

        Self {
            devs,
            dist_kernels,
            dedup_kernels,
            query_length,
            n_devices,
        }
    }

    pub fn prepare_results(&self) -> Vec<CudaSlice<u32>> {
        let results_uninit = vec![u32::MAX; self.query_length];
        (0..self.n_devices)
            .map(|i| self.devs[i].htod_copy(results_uninit.clone()).unwrap())
            .collect::<Vec<_>>()
    }

    pub fn reconstruct_and_compare(
        &mut self,
        codes_result_peers: &Vec<Vec<CudaSlice<u8>>>,
        masks_result_peers: &Vec<Vec<CudaSlice<u8>>>,
        db_sizes: &Vec<usize>,
        streams: &Vec<CudaStream>,
        results: Vec<u64>,
    ) {
        for i in 0..self.n_devices {
            let num_elements = db_sizes[i] * self.query_length;
            let threads_per_block = 256;
            let blocks_per_grid = num_elements.div_ceil(threads_per_block);
            let cfg = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.dist_kernels[i]
                    .clone()
                    .launch_on_stream(
                        &streams[i],
                        cfg,
                        (
                            &codes_result_peers[i][0],
                            &codes_result_peers[i][1],
                            &codes_result_peers[i][2],
                            &masks_result_peers[i][0],
                            &masks_result_peers[i][1],
                            &masks_result_peers[i][2],
                            results[i],
                            MATCH_RATIO,
                            db_sizes[i] as u64,
                            self.query_length as u64,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn dedup_and_append(
        &self,
        match_results_self: &Vec<u64>,
        match_results: &Vec<u64>,
        queries1: &Vec<u64>,
        queries2: &Vec<u64>,
        queries1_new: &Vec<u64>,
        queries2_new: &Vec<u64>,
        query_sums1: &Vec<u64>,
        query_sums2: &Vec<u64>,
        query_sums_new1: &Vec<u64>,
        query_sums_new2: &Vec<u64>,
        db_size_ptrs: &Vec<u64>,
        streams: &Vec<CudaStream>,
    ) {
        let num_elements = self.query_length / ROTATIONS / self.n_devices;
        let threads_per_block = 256;
        let blocks_per_grid = num_elements.div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        for i in 0..self.n_devices {
            self.devs[i].bind_to_thread().unwrap();

            let params = vec![
                match_results_self[i],
                match_results[i],
                queries1[i],
                queries2[i],
                queries1_new[i],
                queries2_new[i],
                query_sums1[i],
                query_sums2[i],
                query_sums_new1[i],
                query_sums_new2[i],
                db_size_ptrs[i],
                (self.query_length / ROTATIONS / self.n_devices) as u64,
                i as u64,
            ];

            unsafe {
                let mut params = params
                    .iter()
                    .map(|x| x.as_kernel_param())
                    .collect::<Vec<_>>();
                launch_kernel(
                    self.dedup_kernels[i].cu_function,
                    cfg.grid_dim,
                    cfg.block_dim,
                    cfg.shared_mem_bytes,
                    streams[i].stream,
                    &mut params,
                )
                .unwrap();
            }
        }
    }

    pub fn fetch_results(&self, dev_results: &Vec<u64>, streams: &Vec<u64>) -> Vec<Vec<u32>> {
        let mut results = vec![];
        for i in 0..self.n_devices {
            self.devs[i].bind_to_thread().unwrap();
            let mut tmp = vec![0u32; self.query_length];
            unsafe {
                memcpy_dtoh_async(&mut tmp, dev_results[i], streams[i] as *mut _).unwrap();
                synchronize(streams[i] as *mut _).unwrap();
            }
            results.push(tmp);
        }
        results
    }
}

pub struct ShareDB {
    is_remote: bool,
    lagrange_coeff: u16,
    query_length: usize,
    device_manager: DeviceManager,
    kernels: Vec<CudaFunction>,
    rngs: Vec<(ChaChaCudaFeRng, ChaChaCudaFeRng)>,
    comms: Vec<Arc<Comm>>,
    ones: Vec<CudaSlice<u8>>,
    intermediate_results: Vec<CudaSlice<i32>>,
    pub results: Vec<CudaSlice<u8>>,
    pub results_peers: Vec<Vec<CudaSlice<u8>>>,
}

impl ShareDB {
    pub fn init(
        peer_id: usize,
        device_manager: DeviceManager,
        lagrange_coeff: u16,
        max_db_length: usize,
        query_length: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        peer_url: Option<String>,
        is_remote: Option<bool>,
        server_port: Option<u16>,
    ) -> Self {
        let n_devices = device_manager.device_count();
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let is_remote = is_remote.unwrap_or(false);

        let mut kernels = Vec::new();

        for i in 0..n_devices {
            let dev = device_manager.device(i);
            dev.load_ptx(ptx.clone(), MATMUL_FUNCTION_NAME, &[MATMUL_FUNCTION_NAME])
                .unwrap();
            let function = dev
                .get_func(MATMUL_FUNCTION_NAME, MATMUL_FUNCTION_NAME)
                .unwrap();

            kernels.push(function);
        }

        let ones = vec![1u8; IRIS_CODE_LENGTH];
        let ones = (0..n_devices)
            .map(|idx| device_manager.device(idx).htod_sync_copy(&ones).unwrap())
            .collect::<Vec<_>>();

        //TODO: depending on the batch size, intermediate_results can get quite big, we can perform the gemm in chunks to limit this
        let mut intermediate_results = vec![];
        let mut results = vec![];
        let mut results_peers = vec![];
        let results_len = max_db_length / n_devices * query_length;

        for idx in 0..n_devices {
            unsafe {
                intermediate_results
                    .push(device_manager.device(idx).alloc(results_len * 4).unwrap());
                results.push(device_manager.device(idx).alloc(results_len * 2).unwrap());
                results_peers.push(vec![
                    device_manager.device(idx).alloc(results_len * 2).unwrap(),
                    device_manager.device(idx).alloc(results_len * 2).unwrap(),
                    device_manager.device(idx).alloc(results_len * 2).unwrap(),
                ]);
            }
        }

        // Init RNGs
        let rng_buf_size: usize = (max_db_length / n_devices * query_length)
            .div_ceil(CHACHA_BUFFER_SIZE)
            * CHACHA_BUFFER_SIZE;
        let mut rngs = vec![];
        for idx in 0..n_devices {
            let (seed0, seed1) = chacha_seeds;
            let mut chacha1 =
                ChaChaCudaFeRng::init(rng_buf_size, device_manager.device(idx).clone(), seed0);
            chacha1.get_mut_chacha().set_nonce(idx as u64);
            let mut chacha2 =
                ChaChaCudaFeRng::init(rng_buf_size, device_manager.device(idx).clone(), seed1);
            chacha2.get_mut_chacha().set_nonce(idx as u64);
            rngs.push((chacha1, chacha2));
        }

        // Init NCCL comms
        let mut comms = vec![];
        if is_remote {
            let mut ids = vec![];
            for _ in 0..n_devices {
                ids.push(Id::new().unwrap());
            }

            // Start HTTP server to exchange NCCL commIds
            if peer_id == 0 {
                let ids = ids.clone();
                tokio::spawn(async move {
                    println!("Starting server on port {}...", server_port.unwrap());
                    let app =
                        Router::new().route("/:device_id", get(move |req| http_root(ids, req)));
                    let listener =
                        tokio::net::TcpListener::bind(format!("0.0.0.0:{}", server_port.unwrap()))
                            .await
                            .unwrap();
                    axum::serve(listener, app).await.unwrap();
                });
            } else {
                thread::sleep(Duration::from_secs(10));
            }

            for i in 0..n_devices {
                let id = if peer_id == 0 {
                    ids[i]
                } else {
                    let res = reqwest::blocking::get(format!(
                        "http://{}:{}/{}",
                        peer_url.clone().unwrap(),
                        server_port.unwrap(),
                        i
                    ))
                    .unwrap();
                    IdWrapper::from_str(&res.text().unwrap()).unwrap().0
                };
                ids.push(id);

                // Bind to thread (important!)
                device_manager.device(i).bind_to_thread().unwrap();
                comms.push(Arc::new(
                    Comm::from_rank(device_manager.device(i), peer_id, 3, id).unwrap(),
                ));
            }
        }

        Self {
            is_remote,
            lagrange_coeff,
            query_length,
            device_manager,
            kernels,
            rngs,
            comms,
            intermediate_results,
            ones,
            results,
            results_peers,
        }
    }

    pub fn load_db(
        &self,
        db_entries: &[u16],
        db_length: usize, //TODO: should handle different sizes for each device
        max_db_length: usize,
    ) -> (
        (Vec<CudaSlice<i8>>, Vec<CudaSlice<i8>>),
        (Vec<CudaSlice<u32>>, Vec<CudaSlice<u32>>),
    ) {
        let mut a1_host = db_entries
            .par_iter()
            .map(|&x: &u16| (x >> 8) as i8)
            .collect::<Vec<_>>();
        let mut a0_host = db_entries.par_iter().map(|&x| x as i8).collect::<Vec<_>>();

        // TODO: maybe use gemm here already to speed up loading (we'll need to correct the results as well)
        a1_host
            .par_iter_mut()
            .for_each(|x| *x = (*x as i32 - 128) as i8);

        a0_host
            .par_iter_mut()
            .for_each(|x| *x = (*x as i32 - 128) as i8);

        let a1_sums: Vec<u32> = a1_host
            .par_chunks(IRIS_CODE_LENGTH)
            .map(|row| row.par_iter().map(|&x| x as u32).sum::<u32>())
            .collect();

        let a0_sums: Vec<u32> = a0_host
            .par_chunks(IRIS_CODE_LENGTH)
            .map(|row| row.par_iter().map(|&x| x as u32).sum::<u32>())
            .collect();

        // Split up db and load to all devices
        let chunk_size = db_length / self.device_manager.device_count();
        let max_size = max_db_length / self.device_manager.device_count();

        let db1_sums = a1_sums
            .chunks(chunk_size)
            .enumerate()
            .map(|(idx, chunk)| {
                let mut slice = unsafe {
                    self.device_manager
                        .device(idx)
                        .alloc(max_size)
                        .unwrap()
                };
                self.device_manager
                    .htod_copy_into(chunk.to_vec(), &mut slice, idx)
                    .unwrap();
                slice
            })
            .collect::<Vec<_>>();
        let db0_sums = a0_sums
            .chunks(chunk_size)
            .enumerate()
            .map(|(idx, chunk)| {
                let mut slice = unsafe {
                    self.device_manager
                        .device(idx)
                        .alloc(max_size)
                        .unwrap()
                };
                self.device_manager
                    .htod_copy_into(chunk.to_vec(), &mut slice, idx)
                    .unwrap();
                slice
            })
            .collect::<Vec<_>>();

        let db1 = a1_host
            .chunks(chunk_size * IRIS_CODE_LENGTH)
            .enumerate()
            .map(|(idx, chunk)| {
                let mut slice = unsafe {
                    self.device_manager
                        .device(idx)
                        .alloc(max_size * IRIS_CODE_LENGTH)
                        .unwrap()
                };
                self.device_manager
                    .htod_copy_into(chunk.to_vec(), &mut slice, idx)
                    .unwrap();
                slice
            })
            .collect::<Vec<_>>();
        let db0 = a0_host
            .chunks(chunk_size * IRIS_CODE_LENGTH)
            .enumerate()
            .map(|(idx, chunk)| {
                let mut slice = unsafe {
                    self.device_manager
                        .device(idx)
                        .alloc(max_size * IRIS_CODE_LENGTH)
                        .unwrap()
                };
                self.device_manager
                    .htod_copy_into(chunk.to_vec(), &mut slice, idx)
                    .unwrap();
                slice
            })
            .collect::<Vec<_>>();

        ((db0, db1), (db0_sums, db1_sums))
    }

    pub fn query_sums(
        &self,
        query_ptrs: &(Vec<u64>, Vec<u64>),
        streams: &Vec<CudaStream>,
        blass: &Vec<CudaBlas>,
    ) -> (Vec<u64>, Vec<u64>) {
        let mut query1_sums = vec![];
        let mut query0_sums = vec![];

        for idx in 0..self.device_manager.device_count() {
            self.device_manager.device(idx).bind_to_thread().unwrap();

            let query0 = query_ptrs.0[idx];
            let query1 = query_ptrs.1[idx];

            let query0_sum = unsafe {
                malloc_async(
                    streams[idx].stream,
                    self.query_length * mem::size_of::<u32>(),
                )
                .unwrap()
            };

            let query1_sum = unsafe {
                malloc_async(
                    streams[idx].stream,
                    self.query_length * mem::size_of::<u32>(),
                )
                .unwrap()
            };

            gemm(
                &blass[idx],
                query0,
                *self.ones[idx].device_ptr(),
                query0_sum,
                0,
                0,
                0,
                self.query_length,
                1,
                IRIS_CODE_LENGTH,
                1,
                0,
            );
            gemm(
                &blass[idx],
                query1,
                *self.ones[idx].device_ptr(),
                query1_sum,
                0,
                0,
                0,
                self.query_length,
                1,
                IRIS_CODE_LENGTH,
                1,
                0,
            );

            query0_sums.push(query0_sum);
            query1_sums.push(query1_sum);
        }
        (query0_sums, query1_sums)
    }

    pub fn dot(
        &mut self,
        query_ptrs: &(Vec<u64>, Vec<u64>),
        db: &(Vec<u64>, Vec<u64>),
        db_sizes: &Vec<usize>,
        streams: &Vec<CudaStream>,
        blass: &Vec<CudaBlas>,
    ) {
        for idx in 0..self.device_manager.device_count() {
            self.device_manager.device(idx).bind_to_thread().unwrap();
            let query0 = query_ptrs.0[idx];
            let query1 = query_ptrs.1[idx];

            // Prepare randomness to mask results
            if self.is_remote {
                let rng_buf_size: usize = (db_sizes[idx] * self.query_length)
                    .div_ceil(CHACHA_BUFFER_SIZE)
                    * CHACHA_BUFFER_SIZE;

                self.rngs[idx]
                    .0
                    .fill_rng_no_host_copy(rng_buf_size, &streams[idx]);
                self.rngs[idx]
                    .1
                    .fill_rng_no_host_copy(rng_buf_size, &streams[idx]);
            }

            for (i, d) in [db.0[idx], db.1[idx]].iter().enumerate() {
                for (j, q) in [query0, query1].iter().enumerate() {
                    gemm(
                        &blass[idx],
                        *d,
                        *q,
                        *self.intermediate_results[idx].device_ptr(),
                        0,
                        0,
                        (db_sizes[idx] * self.query_length * 4 * (i * 2 + j)) as u64,
                        db_sizes[idx],
                        self.query_length,
                        IRIS_CODE_LENGTH,
                        1,
                        0,
                    );
                }
            }
        }
    }

    pub fn dot_reduce(
        &mut self,
        query_sums: &(Vec<u64>, Vec<u64>),
        db_sums: &(Vec<u64>, Vec<u64>),
        db_sizes: &Vec<usize>,
        streams: &Vec<CudaStream>,
    ) {
        for idx in 0..self.device_manager.device_count() {
            let num_elements = db_sizes[idx] * self.query_length;
            let threads_per_block = 256;
            let blocks_per_grid = num_elements.div_ceil(threads_per_block);
            let cfg = LaunchConfig {
                block_dim: (threads_per_block as u32, 1, 1),
                grid_dim: (blocks_per_grid as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.kernels[idx]
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &self.intermediate_results[idx],
                            &mut self.results[idx],
                            db_sums.0[idx],
                            db_sums.1[idx],
                            query_sums.0[idx],
                            query_sums.1[idx],
                            db_sizes[idx] as u64,
                            (db_sizes[idx] * self.query_length) as u64,
                            IRIS_CODE_LENGTH as u64,
                            self.lagrange_coeff as u64,
                            self.rngs[idx].0.cuda_slice(),
                            self.rngs[idx].1.cuda_slice(),
                        ),
                    )
                    .unwrap();
            }
        }
    }

    /// Broadcasts the results to all other peers.
    /// Calls are async to host, but sync to device.
    pub fn exchange_results(&mut self, db_sizes: &Vec<usize>, streams: &Vec<CudaStream>) {
        for idx in 0..self.device_manager.device_count() {
            for i in 0..3 {
                broadcast_stream(
                    &Some(&self.results[idx]),
                    &mut self.results_peers[idx][i],
                    i as i32,
                    db_sizes[idx] * self.query_length,
                    &self.comms[idx],
                    &streams[idx],
                )
                .unwrap();
            }
        }
    }

    pub fn fetch_results(&self, results: &mut [u16], db_sizes: &Vec<usize>, device_id: usize) {
        unsafe {
            let res_trans =
                self.results[device_id].transmute(db_sizes[device_id] * self.query_length);

            self.device_manager
                .device(device_id)
                .dtoh_sync_copy_into(&res_trans.unwrap(), results)
                .unwrap();
        }
    }

    pub fn fetch_results_peer(
        &self,
        results: &mut [u16],
        db_sizes: &Vec<usize>,
        device_id: usize,
        peer_id: usize,
    ) {
        unsafe {
            let res_trans = self.results_peers[device_id][peer_id]
                .transmute(db_sizes[device_id] * self.query_length);

            self.device_manager
                .device(device_id)
                .dtoh_sync_copy_into(&res_trans.unwrap(), results)
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::sys::lib;
    use float_eq::assert_float_eq;
    use ndarray::Array2;
    use num_traits::FromPrimitive;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        device_manager::DeviceManager,
        device_ptrs, preprocess_query,
        setup::{
            id::PartyID,
            iris_db::{db::IrisDB, shamir_db::ShamirIrisDB, shamir_iris::ShamirIris},
            shamir::Shamir,
        },
        DistanceComparator, ShareDB, P,
    };
    const WIDTH: usize = 12_800;
    const QUERY_SIZE: usize = 31;
    const DB_SIZE: usize = 8 * 1000;
    const RNG_SEED: u64 = 42;
    const N_DEVICES: usize = 8;

    /// Helper to generate random ndarray
    fn random_ndarray<T>(array: Vec<u16>, n: usize, m: usize) -> Array2<T>
    where
        T: FromPrimitive,
    {
        Array2::from_shape_vec(
            (n as usize, m as usize),
            array
                .into_iter()
                .map(|x| T::from_u16(x).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    /// Helper to generate random vec
    fn random_vec(n: usize, m: usize, max_value: u32) -> Vec<u16> {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        (0..n * m)
            .map(|_| rng.gen_range(0..max_value) as u16)
            .collect()
    }

    /// Test to verify the matmul operation for random matrices in the field
    #[test]
    fn check_matmul_p16() {
        let db = random_vec(DB_SIZE, WIDTH, P as u32);
        let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
        let mut gpu_result = vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE];
        let device_manager = DeviceManager::init();
        let db_sizes = vec![DB_SIZE / device_manager.device_count(); device_manager.device_count()];

        let mut engine = ShareDB::init(
            0,
            device_manager.clone(),
            1,
            DB_SIZE,
            QUERY_SIZE,
            ([0u32; 8], [0u32; 8]),
            None,
            None,
            None,
        );
        let preprocessed_query = preprocess_query(&query);
        let streams = device_manager.fork_streams();
        let blass = device_manager.create_cublas(&streams);
        let preprocessed_query = device_manager.htod_transfer_query(&preprocessed_query, &streams);
        let query_sums = engine.query_sums(&preprocessed_query, &streams, &blass);
        let db_slices = engine.load_db(&db, DB_SIZE, DB_SIZE + 100);

        for _ in 0..5 {
            engine.dot(
                &preprocessed_query,
                &(device_ptrs(&db_slices.0 .0), device_ptrs(&db_slices.0 .1)),
                &db_sizes,
                &streams,
                &blass,
            );
            engine.dot_reduce(
                &query_sums,
                &(device_ptrs(&db_slices.1 .0), device_ptrs(&db_slices.1 .1)),
                &db_sizes,
                &streams,
            );
            device_manager.await_streams(&streams);

            let a_nda = random_ndarray::<u64>(db.clone(), DB_SIZE, WIDTH);
            let b_nda = random_ndarray::<u64>(query.clone(), QUERY_SIZE, WIDTH);
            let c_nda = a_nda.dot(&b_nda.t());

            let mut vec_column_major: Vec<u16> = Vec::new();
            for col in 0..c_nda.ncols() {
                for row in c_nda.column(col) {
                    vec_column_major.push((*row % (P as u64)) as u16);
                }
            }

            for device_idx in 0..N_DEVICES {
                engine.fetch_results(&mut gpu_result, &db_sizes, device_idx);
                let selected_elements: Vec<u16> = vec_column_major
                    .chunks(DB_SIZE)
                    .flat_map(|chunk| {
                        chunk
                            .iter()
                            .skip(DB_SIZE / N_DEVICES * device_idx)
                            .take(DB_SIZE / N_DEVICES)
                    })
                    .cloned()
                    .collect();

                assert_eq!(selected_elements, gpu_result);
            }
        }
    }

    /// Checks that the result of a matmul of the original data equals the
    /// reconstructed result of individual matmuls on the shamir shares.
    #[test]
    fn check_shared_matmul() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = random_vec(DB_SIZE, WIDTH, P as u32);
        let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
        let mut gpu_result = vec![
            vec![0u16; DB_SIZE * QUERY_SIZE / N_DEVICES],
            vec![0u16; DB_SIZE * QUERY_SIZE / N_DEVICES],
            vec![0u16; DB_SIZE * QUERY_SIZE / N_DEVICES],
        ];
        let db_sizes = vec![DB_SIZE / N_DEVICES; N_DEVICES];

        // Calculate non-shared
        let a_nda = random_ndarray::<u64>(db.clone(), DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u64>(query.clone(), QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push((*row % (P as u64)) as u16);
            }
        }

        let mut dbs = vec![vec![], vec![], vec![]];
        let mut querys = vec![vec![], vec![], vec![]];

        // Calculate shared
        for i in 0..db.len() {
            let shares = Shamir::share_d1(db[i], &mut rng);
            dbs[0].push(shares[0]);
            dbs[1].push(shares[1]);
            dbs[2].push(shares[2]);
        }

        for i in 0..query.len() {
            let shares = Shamir::share_d1(query[i], &mut rng);
            querys[0].push(shares[0]);
            querys[1].push(shares[1]);
            querys[2].push(shares[2]);
        }

        for i in 0..3 {
            let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(i as u8).unwrap());
            let device_manager = DeviceManager::init();

            let mut engine = ShareDB::init(
                0,
                device_manager.clone(),
                l_coeff,
                DB_SIZE,
                QUERY_SIZE,
                ([0u32; 8], [0u32; 8]),
                None,
                None,
                None,
            );
            let preprocessed_query = preprocess_query(&querys[i]);
            let streams = device_manager.fork_streams();
            let blass = device_manager.create_cublas(&streams);
            let preprocessed_query =
                device_manager.htod_transfer_query(&preprocessed_query, &streams);
            let query_sums = engine.query_sums(&preprocessed_query, &streams, &blass);
            let db_slices = engine.load_db(&db, DB_SIZE, DB_SIZE);
            engine.dot(
                &preprocessed_query,
                &(device_ptrs(&db_slices.0 .0), device_ptrs(&db_slices.0 .1)),
                &db_sizes,
                &streams,
                &blass,
            );
            engine.dot_reduce(
                &query_sums,
                &(device_ptrs(&db_slices.1 .0), device_ptrs(&db_slices.1 .1)),
                &db_sizes,
                &streams,
            );
            device_manager.await_streams(&streams);
            engine.fetch_results(&mut gpu_result[i], &db_sizes, 0);
        }

        // TODO: we should check for all devices
        for i in 0..DB_SIZE / N_DEVICES {
            assert_eq!(
                (gpu_result[0][i] as u32 + gpu_result[1][i] as u32 + gpu_result[2][i] as u32)
                    % P as u32,
                vec_column_major[i] as u32
            );
        }
    }

    /// Calculates the distances between a query and a shamir secret shared db and
    /// checks the result against reference plain implementation.
    #[test]
    fn check_shared_distances() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);
        let shamir_db = ShamirIrisDB::share_db_par(&db, &mut rng);
        let db_sizes = vec![DB_SIZE / N_DEVICES; N_DEVICES];

        // Prepare query
        let query_template = db.db[0].get_similar_iris(&mut rng);
        let random_query = ShamirIris::share_iris(&query_template, &mut rng);
        let mut code_queries = vec![vec![], vec![], vec![]];
        let mut mask_queries = vec![vec![], vec![], vec![]];

        for i in 0..QUERY_SIZE {
            // TODO: rotate
            let tmp: [ShamirIris; 3] = random_query.clone();
            code_queries[0].push(tmp[0].code.to_vec());
            code_queries[1].push(tmp[1].code.to_vec());
            code_queries[2].push(tmp[2].code.to_vec());

            mask_queries[0].push(tmp[0].mask.to_vec());
            mask_queries[1].push(tmp[1].mask.to_vec());
            mask_queries[2].push(tmp[2].mask.to_vec());
        }

        let mut results_codes = vec![
            vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE],
            vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE],
            vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE],
        ];

        let mut results_masks = vec![
            vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE],
            vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE],
            vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE],
        ];

        for party_id in 0..3 {
            let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(party_id as u8).unwrap());

            let codes_db = shamir_db[party_id]
                .db
                .iter()
                .flat_map(|entry| entry.code)
                .collect::<Vec<_>>();

            let masks_db = shamir_db[party_id]
                .db
                .iter()
                .flat_map(|entry| entry.mask)
                .collect::<Vec<_>>();

            let device_manager = DeviceManager::init();

            let mut codes_engine = ShareDB::init(
                party_id,
                device_manager.clone(),
                l_coeff,
                DB_SIZE,
                QUERY_SIZE,
                ([0u32; 8], [0u32; 8]),
                None,
                None,
                None,
            );
            let mut masks_engine = ShareDB::init(
                party_id,
                device_manager.clone(),
                l_coeff,
                DB_SIZE,
                QUERY_SIZE,
                ([0u32; 8], [0u32; 8]),
                None,
                None,
                None,
            );

            let code_query = preprocess_query(
                &code_queries[party_id]
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            );
            let mask_query = preprocess_query(
                &mask_queries[party_id]
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>(),
            );

            let streams = device_manager.fork_streams();
            let blass = device_manager.create_cublas(&streams);
            let code_query = device_manager.htod_transfer_query(&code_query, &streams);
            let mask_query = device_manager.htod_transfer_query(&mask_query, &streams);
            let code_query_sums = codes_engine.query_sums(&code_query, &streams, &blass);
            let mask_query_sums = masks_engine.query_sums(&mask_query, &streams, &blass);
            let code_db_slices = codes_engine.load_db(&codes_db, DB_SIZE, DB_SIZE);
            let mask_db_slices = codes_engine.load_db(&masks_db, DB_SIZE, DB_SIZE);

            codes_engine.dot(
                &code_query,
                &(
                    device_ptrs(&code_db_slices.0 .0),
                    device_ptrs(&code_db_slices.0 .1),
                ),
                &db_sizes,
                &streams,
                &blass,
            );
            masks_engine.dot(
                &mask_query,
                &(
                    device_ptrs(&mask_db_slices.0 .0),
                    device_ptrs(&mask_db_slices.0 .1),
                ),
                &db_sizes,
                &streams,
                &blass,
            );

            codes_engine.dot_reduce(
                &code_query_sums,
                &(
                    device_ptrs(&code_db_slices.1 .0),
                    device_ptrs(&code_db_slices.1 .1),
                ),
                &db_sizes,
                &streams,
            );
            masks_engine.dot_reduce(
                &mask_query_sums,
                &(
                    device_ptrs(&mask_db_slices.1 .0),
                    device_ptrs(&mask_db_slices.1 .1),
                ),
                &db_sizes,
                &streams,
            );

            device_manager.await_streams(&streams);

            // TODO: fetch results also for other devices
            codes_engine.fetch_results(&mut results_codes[party_id], &db_sizes, 0);
            masks_engine.fetch_results(&mut results_masks[party_id], &db_sizes, 0);
        }

        // Reconstruct the results
        let mut reconstructed_codes = vec![];
        let mut reconstructed_masks = vec![];

        for i in 0..results_codes[0].len() {
            let code = (results_codes[0][i] as u32
                + results_codes[1][i] as u32
                + results_codes[2][i] as u32)
                % P as u32;
            let mask = (results_masks[0][i] as u32
                + results_masks[1][i] as u32
                + results_masks[2][i] as u32)
                % P as u32;

            reconstructed_codes.push(code);
            reconstructed_masks.push(mask);
        }

        // Calculate the distance in plain
        let dists = reconstructed_codes
            .into_iter()
            .zip(reconstructed_masks)
            .map(|(code, mask)| {
                const OFFSET: u32 = 32759;
                let offset_nom = (code as u32 + OFFSET) % (P as u32);
                0.5f64 - offset_nom as f64 / (2f64 * mask as f64)
                    + OFFSET as f64 / (2f64 * mask as f64)
            })
            .collect::<Vec<_>>();

        // Compare against plain reference implementation
        let reference_dists = db.calculate_distances(&query_template);

        // TODO: check for all devices and the whole query
        for i in 0..DB_SIZE / N_DEVICES {
            assert_float_eq!(dists[i], reference_dists[i], abs <= 1e-6);
        }
    }

    #[test]
    fn test_dedup_query() {
        const ROTATIONS: usize = 31;
        const DB_SIZE: usize = 100;
        const QUERIES: usize = 248;
        const WIDTH: usize = 12800;
        const QUERY_LEN: usize = QUERIES * WIDTH;
        let query1 = [1u8; QUERY_LEN].to_vec();
        let query2 = [1u8; QUERY_LEN].to_vec();
        let query_sums1 = [1u8; QUERY_LEN].to_vec();
        let query_sums2 = [1u8; QUERY_LEN].to_vec();

        let query1_new = [0u8; QUERY_LEN / ROTATIONS].to_vec();
        let query2_new = [0u8; QUERY_LEN / ROTATIONS].to_vec();
        let query_sums1_new = [0u8; QUERY_LEN / ROTATIONS].to_vec();
        let query_sums2_new = [0u8; QUERY_LEN / ROTATIONS].to_vec();

        // set all to matches
        let mut match_results = [u32::MAX; QUERIES].to_vec();
        // set 2 to match
        match_results[32] = 0;

        let mut match_results_self = [u32::MAX; QUERIES].to_vec();
        // set 1 to match
        match_results_self[15] = 0;
        match_results_self[100] = 0;

        let device_manager = DeviceManager::init();
        let distance_comparator = DistanceComparator::init(QUERIES);

        let streams = device_manager.fork_streams();
        let query_ptrs = device_manager.htod_transfer_query(&vec![query1, query2], &streams);
        let query_sum_ptrs =
            device_manager.htod_transfer_query(&vec![query_sums1, query_sums2], &streams);
        let query_ptrs_new =
            device_manager.htod_transfer_query(&vec![query1_new, query2_new], &streams);
        let query_sum_ptrs_new =
            device_manager.htod_transfer_query(&vec![query_sums1_new, query_sums2_new], &streams);

        let mut result_ptrs = vec![];
        let mut result_ptrs_self = vec![];
        let mut db_sizes = vec![];

        for i in 0..device_manager.device_count() {
            result_ptrs.push(
                device_manager
                    .device(i)
                    .htod_copy(match_results.clone())
                    .unwrap(),
            );

            result_ptrs_self.push(
                device_manager
                    .device(i)
                    .htod_copy(match_results_self.clone())
                    .unwrap(),
            );

            db_sizes.push(device_manager.device(i).htod_copy(vec![0u32; 1]).unwrap());
        }

        device_manager.await_streams(&streams);

        distance_comparator.dedup_and_append(
            &device_ptrs(&result_ptrs_self),
            &device_ptrs(&result_ptrs),
            &query_ptrs.0,
            &query_ptrs.1,
            &query_ptrs_new.0,
            &query_ptrs_new.1,
            &query_sum_ptrs.0,
            &query_sum_ptrs.1,
            &query_sum_ptrs_new.0,
            &query_sum_ptrs_new.1,
            &device_ptrs(&db_sizes),
            &streams,
        );

        device_manager.await_streams(&streams);

        for i in 0..8 {
            let mut new_query1 = [3u8; QUERY_LEN / ROTATIONS / 8].to_vec();
            unsafe {
                device_manager.device(i).bind_to_thread().unwrap();
                lib().cuMemcpyDtoH_v2(
                    new_query1.as_mut_ptr() as *mut _,
                    query_ptrs_new.0[i],
                    new_query1.len(),
                );
            }
        }
    }
}
