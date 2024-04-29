pub mod setup;

use std::{ffi::c_void, str::FromStr, sync::Arc, thread, time::Duration};

use axum::{extract::Path, routing::get, Router};
use cudarc::{
    cublas::{result::gemm_ex, sys, CudaBlas},
    driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchAsync,
        LaunchConfig,
    },
    nccl::{Comm, Id},
    nvrtc::compile_ptx,
};
use rayon::prelude::*;
use setup::id;

pub(crate) const P: u16 = ((1u32 << 16) - 17) as u16;
const PTX_SRC: &str = include_str!("kernel.cu");
const IRIS_CODE_LENGTH: usize = 12_800;
const QUERY_LENGTH: usize = 930;
const MATMUL_FUNCTION_NAME: &str = "matmul";
const DIST_FUNCTION_NAME: &str = "reconstructDistance";

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

async fn http_root(ids: Vec<Id>, Path(device_id): Path<String>) -> String {
    let device_id: usize = device_id.parse().unwrap();
    IdWrapper(ids[device_id]).to_string()
}

pub struct DistanceComparator {
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    results: Vec<CudaSlice<f32>>,
    db_length: usize,
    n_devices: usize,
}

impl DistanceComparator {
    pub fn init(n_devices: usize, db_length: usize) -> Self {
        let ptx = compile_ptx(PTX_SRC).unwrap();
        let mut kernels = Vec::new();
        let mut devs = Vec::new();
        let mut results = Vec::new();

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            dev.load_ptx(ptx.clone(), DIST_FUNCTION_NAME, &[DIST_FUNCTION_NAME])
                .unwrap();
            let function = dev
                .get_func(DIST_FUNCTION_NAME, DIST_FUNCTION_NAME)
                .unwrap();

            let result = dev
                .alloc_zeros(db_length / n_devices * QUERY_LENGTH)
                .unwrap();

            kernels.push(function);
            devs.push(dev);
            results.push(result);
        }

        Self {
            devs,
            kernels,
            results,
            db_length,
            n_devices,
        }
    }

    pub fn reconstruct(
        &mut self,
        codes_result: &Vec<CudaSlice<u8>>,
        codes_result_peers: &Vec<Vec<CudaSlice<u8>>>,
        masks_result: &Vec<CudaSlice<u8>>,
        masks_result_peers: &Vec<Vec<CudaSlice<u8>>>,
    ) {
        let num_elements = self.db_length / self.n_devices * QUERY_LENGTH;
        let threads_per_block = 256;
        let blocks_per_grid = num_elements.div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        for i in 0..self.n_devices {
            unsafe {
                self.kernels[i]
                    .clone()
                    .launch(
                        cfg,
                        (
                            &codes_result[i],
                            &codes_result_peers[i][0],
                            &codes_result_peers[i][1],
                            &masks_result[i],
                            &masks_result_peers[i][0],
                            &masks_result_peers[i][1],
                            &mut self.results[i],
                            P,
                            (self.db_length / self.n_devices * QUERY_LENGTH) as u64,
                        ),
                    )
                    .unwrap();
            }
        }

        for i in 0..self.n_devices {
            self.devs[i].synchronize().unwrap();
        }
    }

    pub fn fetch_results(&self, device_id: usize) -> Vec<f32> {
        self.devs[device_id]
            .dtoh_sync_copy(&self.results[device_id])
            .unwrap()
    }
}

pub struct ShareDB {
    peer_id: usize,
    lagrange_coeff: u16,
    db_length: usize,
    limbs: usize,
    n_devices: usize,
    blass: Vec<CudaBlas>,
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    streams: Vec<CudaStream>,
    comms: Vec<Arc<Comm>>,
    db1: Vec<CudaSlice<u8>>,
    db0: Vec<CudaSlice<u8>>,
    db1_sums: Vec<CudaSlice<u32>>,
    db0_sums: Vec<CudaSlice<u32>>,
    query1_sums: Vec<CudaSlice<i32>>,
    query0_sums: Vec<CudaSlice<i32>>,
    ones: Vec<CudaSlice<u8>>,
    intermediate_results: Vec<CudaSlice<i32>>,
    pub results: Vec<CudaSlice<u8>>,
    pub results_peers: Vec<Vec<CudaSlice<u8>>>,
}

impl ShareDB {
    pub fn init(
        peer_id: usize,
        lagrange_coeff: u16,
        db_entries: &[u16],
        peer_url: Option<&String>,
        is_local: bool,
        server_port: Option<u16>,
    ) -> Self {
        // TODO: replace with a MAX_DB_SIZE to allow for insertions
        let db_length = db_entries.len() / IRIS_CODE_LENGTH;
        let n_devices = CudaDevice::count().unwrap() as usize;
        let limbs = 2;
        let ptx = compile_ptx(PTX_SRC).unwrap();

        let mut devs = Vec::new();
        let mut blass = Vec::new();
        let mut kernels = Vec::new();
        let mut streams = Vec::new();

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let blas = CudaBlas::new(dev.clone()).unwrap();
            // let stream = dev.fork_default_stream().unwrap();
            // unsafe {
            //     blas.set_stream(Some(&stream)).unwrap();
            // }
            dev.load_ptx(ptx.clone(), MATMUL_FUNCTION_NAME, &[MATMUL_FUNCTION_NAME])
                .unwrap();
            let function = dev
                .get_func(MATMUL_FUNCTION_NAME, MATMUL_FUNCTION_NAME)
                .unwrap();

            // streams.push(stream);
            blass.push(blas);
            devs.push(dev);
            kernels.push(function);
        }

        let mut a1_host = db_entries
            .par_iter()
            .map(|&x: &u16| (x >> 8) as u8)
            .collect::<Vec<_>>();
        let mut a0_host = db_entries.par_iter().map(|&x| x as u8).collect::<Vec<_>>();

        // TODO: maybe use gemm here already to speed up loading (we'll need to correct the results as well)
        let a1_sums: Vec<u32> = a1_host
            .par_chunks(IRIS_CODE_LENGTH)
            .map(|row| row.par_iter().map(|&x| x as u32).sum::<u32>())
            .collect();

        let a0_sums: Vec<u32> = a0_host
            .par_chunks(IRIS_CODE_LENGTH)
            .map(|row| row.par_iter().map(|&x| x as u32).sum::<u32>())
            .collect();

        a1_host
            .par_iter_mut()
            .for_each(|x| *x = (*x as i32 - 128) as u8);

        a0_host
            .par_iter_mut()
            .for_each(|x| *x = (*x as i32 - 128) as u8);

        // Split up db and load to all devices
        let chunk_size = db_length / n_devices;

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

        let ones = vec![1u8; IRIS_CODE_LENGTH];
        let ones = (0..n_devices)
            .map(|idx| devs[idx].htod_sync_copy(&ones).unwrap())
            .collect::<Vec<_>>();

        //TODO: depending on the batch size, intermediate_results can get quite big, we can perform the gemm in chunks to limit this
        let mut intermediate_results = vec![];
        let mut results = vec![];
        let mut results_peers = vec![];
        let mut query1_sums = vec![];
        let mut query0_sums = vec![];
        let results_len = chunk_size * QUERY_LENGTH;

        for idx in 0..n_devices {
            unsafe {
                intermediate_results.push(devs[idx].alloc(results_len * 4).unwrap());
                results.push(devs[idx].alloc(results_len * 2).unwrap());
                results_peers.push(vec![
                    devs[idx].alloc(results_len * 2).unwrap(),
                    devs[idx].alloc(results_len * 2).unwrap(),
                    devs[idx].alloc(results_len * 2).unwrap(),
                ]);
                query1_sums.push(devs[idx].alloc(QUERY_LENGTH).unwrap());
                query0_sums.push(devs[idx].alloc(QUERY_LENGTH).unwrap());
            }
        }

        let mut comms = vec![];
        if !is_local {
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
            }

            for i in 0..n_devices {
                let id = if peer_id == 0 {
                    ids[i]
                } else {
                    // If not the server, give it a few secs to start
                    thread::sleep(Duration::from_secs(5));

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
                devs[i].bind_to_thread().unwrap();
                comms.push(Arc::new(
                    Comm::from_rank(devs[i].clone(), peer_id, 3, id).unwrap(),
                ));
            }
        }

        Self {
            peer_id,
            lagrange_coeff,
            db_length,
            limbs,
            n_devices,
            blass,
            devs,
            kernels,
            streams,
            comms,
            db1,
            db0,
            db1_sums,
            db0_sums,
            query1_sums,
            query0_sums,
            intermediate_results,
            ones,
            results,
            results_peers,
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
        let num_elements = self.db_length / self.n_devices * QUERY_LENGTH;
        let threads_per_block = 256;
        let blocks_per_grid = num_elements.div_ceil(threads_per_block);
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        for idx in 0..self.n_devices {
            let query1 = self.devs[idx]
                .htod_sync_copy(&preprocessed_query[1])
                .unwrap();
            let query0 = self.devs[idx]
                .htod_sync_copy(&preprocessed_query[0])
                .unwrap();

            // Calculate sums to correct output
            gemm(
                &self.blass[idx],
                &query1,
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
                &query0,
                &self.ones[idx],
                &mut self.query0_sums[idx],
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
                for (j, q) in [&query0, &query1].iter().enumerate() {
                    gemm(
                        &self.blass[idx],
                        d,
                        q,
                        &mut self.intermediate_results[idx],
                        0,
                        0,
                        (self.db_length / self.n_devices * QUERY_LENGTH * 4 * (i * 2 + j)) as u64,
                        self.db_length / self.n_devices,
                        QUERY_LENGTH,
                        IRIS_CODE_LENGTH,
                        1,
                        0,
                    );
                }
            }

            unsafe {
                self.kernels[idx]
                    .clone()
                    .launch(
                        // &self.streams[idx],
                        cfg,
                        (
                            &self.intermediate_results[idx],
                            &mut self.results[idx],
                            &self.db0_sums[idx],
                            &self.db1_sums[idx],
                            &self.query0_sums[idx],
                            &self.query1_sums[idx],
                            self.db_length as u64 / self.n_devices as u64,
                            (self.db_length / self.n_devices * QUERY_LENGTH) as u64,
                            IRIS_CODE_LENGTH as u64,
                            P as u64,
                            self.lagrange_coeff as u64,
                        ),
                    )
                    .unwrap();
            }
        }

        for idx in 0..self.n_devices {
            self.devs[idx].synchronize().unwrap();
        }
    }

    /// Broadcasts the results to all other peers.
    /// Calls are async to host, but sync to device.
    pub fn exchange_results(&mut self) {
        for idx in 0..self.n_devices {
            self.comms[idx].broadcast(
                &Some(&self.results[idx]),
                &mut self.results_peers[idx][0],
                0,
            );
            self.comms[idx].broadcast(
                &Some(&self.results[idx]),
                &mut self.results_peers[idx][1],
                1,
            );
            self.comms[idx].broadcast(
                &Some(&self.results[idx]),
                &mut self.results_peers[idx][2],
                2,
            );
        }
        for idx in 0..self.n_devices {
            self.devs[idx].synchronize().unwrap();
        }
    }

    pub fn fetch_results(&self, results: &mut [u16], device_id: usize) {
        unsafe {
            let res_trans =
                self.results[device_id].transmute(self.db_length * QUERY_LENGTH / self.n_devices);

            self.devs[device_id]
                .dtoh_sync_copy_into(&res_trans.unwrap(), results)
                .unwrap();

            // TODO: async transfer
            // lib()
            //     .cuMemcpyDtoHAsync_v2(
            //         results,
            //         *self.results[device_id].device_ptr(),
            //         self.results[device_id].len(),
            //         self.streams[device_id].stream,
            //     )
            //     .result().unwrap();

            // result::stream::synchronize(self.streams[device_id].stream).unwrap();
        }
    }

    pub fn fetch_results_peer(&self, results: &mut [u16], device_id: usize, peer_id: usize) {
        unsafe {
            let res_trans = self.results_peers[device_id][peer_id]
                .transmute(self.db_length * QUERY_LENGTH / self.n_devices);

            self.devs[device_id]
                .dtoh_sync_copy_into(&res_trans.unwrap(), results)
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use num_traits::FromPrimitive;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        setup::{id::PartyID, iris_db::iris::IrisCode, shamir::Shamir},
        ShareDB, P,
    };
    const WIDTH: usize = 12_800;
    const QUERY_SIZE: usize = 31;
    const DB_SIZE: usize = 8 * 256;
    const RNG_SEED: u64 = 1337;
    const N_DEVICES: usize = 8;

    /// Helpers
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

    fn random_vec(n: usize, m: usize, max_value: u32) -> Vec<u16> {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        (0..n * m)
            .map(|_| rng.gen_range(0..max_value) as u16)
            .collect()
    }

    // #[test]
    // fn check_matmul_p16() {
    //     let db = random_vec(DB_SIZE, WIDTH, P as u32);
    //     let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
    //     let mut gpu_result = vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE];

    //     let mut engine = ShareDB::init(0, 1, &db, None, true, None);
    //     let preprocessed_query = engine.preprocess_query(&query);
    //     engine.dot(&preprocessed_query);

    //     let a_nda = random_ndarray::<u64>(db, DB_SIZE, WIDTH);
    //     let b_nda = random_ndarray::<u64>(query, QUERY_SIZE, WIDTH);
    //     let c_nda = a_nda.dot(&b_nda.t());

    //     let mut vec_column_major: Vec<u16> = Vec::new();
    //     for col in 0..c_nda.ncols() {
    //         for row in c_nda.column(col) {
    //             vec_column_major.push((*row % (P as u64)) as u16);
    //         }
    //     }

    //     for device_idx in 0..N_DEVICES {
    //         engine.fetch_results(&mut gpu_result, device_idx);
    //         let selected_elements: Vec<u16> = vec_column_major
    //             .chunks(DB_SIZE)
    //             .flat_map(|chunk| {
    //                 chunk
    //                     .iter()
    //                     .skip(DB_SIZE / N_DEVICES * device_idx)
    //                     .take(DB_SIZE / N_DEVICES)
    //             })
    //             .cloned()
    //             .collect();

    //         assert_eq!(selected_elements, gpu_result);
    //     }
    // }

    #[test]
    fn check_shared_matmul() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = random_vec(DB_SIZE, WIDTH, 65535 as u32);
        let query = random_vec(QUERY_SIZE, WIDTH, P as u32);
        let mut gpu_result = vec![
            vec![0u16; DB_SIZE * QUERY_SIZE / N_DEVICES],
            vec![0u16; DB_SIZE * QUERY_SIZE / N_DEVICES],
            vec![0u16; DB_SIZE * QUERY_SIZE / N_DEVICES],
        ];

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

            let mut engine = ShareDB::init(0, l_coeff, &dbs[i], None, true, None);
            let preprocessed_query = engine.preprocess_query(&querys[i]);
            engine.dot(&preprocessed_query);

            engine.fetch_results(&mut gpu_result[i], 0);
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

    // #[test]
    // fn check_matmul_xxx() {
    //     let mut rng = StdRng::seed_from_u64(RNG_SEED);
    //     let c1 = IrisCode::random_rng(&mut rng);
    //     let c2 = IrisCode::random_rng(&mut rng);

    //     let mut db = vec![];

    //     for i in 0..DB_SIZE {
    //         for j in 0..WIDTH {
    //             db.push(c1.code.get_bit(j) as u16);
    //         }
    //     }

    //     let mut query = vec![];
    //     for i in 0..QUERY_SIZE {
    //         for j in 0..WIDTH {
    //             query.push(c2.code.get_bit(j) as u16);
    //         }
    //     }

    //     let mut gpu_result = vec![0u16; DB_SIZE / N_DEVICES * QUERY_SIZE];

    //     let mut engine = ShareDB::init(0, 1, &db, None, true, None);
    //     let preprocessed_query = engine.preprocess_query(&query);
    //     engine.dot(&preprocessed_query);

    //     // let a_nda = random_ndarray::<u64>(db, DB_SIZE, WIDTH);
    //     // let b_nda = random_ndarray::<u64>(query, QUERY_SIZE, WIDTH);
    //     // let c_nda = a_nda.dot(&b_nda.t());

    //     for device_idx in 0..2 {
    //         engine.fetch_results(&mut gpu_result, device_idx);

    //         println!("{:?}", gpu_result[0..10].to_vec());

    //         // let selected_elements: Vec<u16> = vec_column_major
    //         //     .chunks(DB_SIZE)
    //         //     .flat_map(|chunk| {
    //         //         chunk
    //         //             .iter()
    //         //             .skip(DB_SIZE / N_DEVICES * device_idx)
    //         //             .take(DB_SIZE / N_DEVICES)
    //         //     })
    //         //     .cloned()
    //         //     .collect();

    //         // assert_eq!(selected_elements, gpu_result);
    //     }
    // }
}
