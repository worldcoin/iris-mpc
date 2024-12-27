use crate::{
    helpers::{
        comm::NcclComm,
        device_manager::DeviceManager,
        launch_config_from_elements_and_threads,
        query_processor::{
            CudaVec2DSlicer, CudaVec2DSlicerRawPointer, CudaVec2DSlicerU32, CudaVec2DSlicerU8,
            StreamAwareCudaSlice,
        },
        DEFAULT_LAUNCH_CONFIG_THREADS,
    },
    rng::chacha::ChaChaCudaRng,
    threshold_ring::protocol::ChunkShareView,
};
use core::panic;
use cudarc::{
    cublas::{
        result::gemm_ex,
        sys::{self, lib},
        CudaBlas,
    },
    driver::{
        result::{self, malloc_async},
        sys::{CUdeviceptr, CU_MEMHOSTALLOC_PORTABLE},
        CudaFunction, CudaSlice, CudaStream, CudaView, DevicePtr, DeviceSlice, LaunchAsync,
    },
    nccl,
    nvrtc::compile_ptx,
};
use itertools::{izip, Itertools};
use memmap2::MmapMut;
use rayon::prelude::*;
use std::{
    ffi::{c_void, CStr},
    mem::{self, forget},
    sync::Arc,
};

const PTX_SRC: &str = include_str!("kernel.cu");
const REDUCE_FUNCTION_NAME: &str = "matmul_correct_and_reduce";
const XOR_ASSIGN_U8_NAME: &str = "xor_assign_u8";
const LIMBS: usize = 2;

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

#[allow(clippy::too_many_arguments)]
pub fn gemm(
    handle: &CudaBlas,
    a: CUdeviceptr,
    b: CUdeviceptr,
    c: CUdeviceptr,
    a_offset: u64,
    b_offset: u64,
    c_offset: u64,
    m: usize,
    n: usize,
    k: usize,
    alpha: i32,
    beta: i32,
) {
    // https://docs.nvidia.com/cuda/cublas/#cublasgemmex:
    // "CUBLAS_COMPUTE_32I and CUBLAS_COMPUTE_32I_PEDANTIC compute types are only supported with A, B being 4-byte aligned and lda, ldb being multiples of 4."
    assert!(m % 4 == 0, "m must be a multiple of 4");
    // We don't enforce the following, since we use it for n=1 and emperial testing
    // shows that it works. assert!(n % 4 == 0, "n must be a multiple of 4");
    assert!(a % 4 == 0, "a must be aligned to 4 bytes");
    assert!(b % 4 == 0, "b must be aligned to 4 bytes");
    unsafe {
        let status = gemm_ex(
            *handle.handle(),
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
        );

        // Try to fetch more information in case of an error
        if let Err(e) = status {
            let c_str = CStr::from_ptr(lib().cublasGetStatusString(e.0));
            panic!("CUBLAS error: {:?}", c_str.to_str());
        }
    }
}

pub struct SlicedProcessedDatabase {
    pub code_gr:      CudaVec2DSlicerRawPointer,
    pub code_sums_gr: CudaVec2DSlicerU32,
}

#[derive(Clone)]
pub struct DBChunkBuffers {
    pub limb_0: Vec<CudaSlice<u8>>,
    pub limb_1: Vec<CudaSlice<u8>>,
}

pub struct ShareDB {
    peer_id:               usize,
    is_remote:             bool,
    query_length:          usize,
    device_manager:        Arc<DeviceManager>,
    kernels:               Vec<CudaFunction>,
    xor_assign_u8_kernels: Vec<CudaFunction>,
    rngs:                  Vec<(ChaChaCudaRng, ChaChaCudaRng)>,
    comms:                 Vec<Arc<NcclComm>>,
    ones:                  Vec<CudaSlice<u8>>,
    intermediate_results:  Vec<CudaSlice<i32>>,
    pub results:           Vec<CudaSlice<u8>>,
    pub results_peer:      Vec<CudaSlice<u8>>,
    code_length:           usize,
}

impl ShareDB {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn init(
        peer_id: usize,
        device_manager: Arc<DeviceManager>,
        max_db_length: usize,
        query_length: usize,
        code_length: usize,
        chacha_seeds: ([u32; 8], [u32; 8]),
        comms: Vec<Arc<NcclComm>>,
    ) -> Self {
        let n_devices = device_manager.device_count();
        let ptx = compile_ptx(PTX_SRC).unwrap();

        let mut kernels = Vec::new();

        for i in 0..n_devices {
            let dev = device_manager.device(i);
            dev.load_ptx(ptx.clone(), REDUCE_FUNCTION_NAME, &[REDUCE_FUNCTION_NAME])
                .unwrap();
            let function = dev
                .get_func(REDUCE_FUNCTION_NAME, REDUCE_FUNCTION_NAME)
                .unwrap();

            kernels.push(function);
        }

        let xor_assign_u8_kernels = (0..n_devices)
            .map(|i| {
                let dev = device_manager.device(i);
                dev.load_ptx(ptx.clone(), XOR_ASSIGN_U8_NAME, &[XOR_ASSIGN_U8_NAME])
                    .unwrap();
                dev.get_func(XOR_ASSIGN_U8_NAME, XOR_ASSIGN_U8_NAME)
                    .unwrap()
            })
            .collect_vec();

        let ones = vec![1u8; code_length];
        let ones = (0..n_devices)
            .map(|idx| device_manager.device(idx).htod_sync_copy(&ones).unwrap())
            .collect::<Vec<_>>();

        // TODO: depending on the batch size, intermediate_results can get quite big, we
        // can perform the gemm in chunks to limit this
        let mut intermediate_results = vec![];
        let mut results = vec![];
        let mut results_peer = vec![];
        let results_len = (max_db_length * query_length).div_ceil(64) * 64;

        for idx in 0..n_devices {
            unsafe {
                intermediate_results.push(device_manager.device(idx).alloc(results_len).unwrap());
                results.push(
                    device_manager
                        .device(idx)
                        .alloc(results_len * std::mem::size_of::<u16>())
                        .unwrap(),
                );
                results_peer.push(
                    device_manager
                        .device(idx)
                        .alloc(results_len * std::mem::size_of::<u16>())
                        .unwrap(),
                );
            }
        }

        // Init RNGs
        let rng_buf_size: usize =
            (max_db_length * query_length * mem::size_of::<u16>()).div_ceil(64) * 64;
        let mut rngs = vec![];
        for idx in 0..n_devices {
            let (seed0, seed1) = chacha_seeds;
            let mut chacha1 =
                ChaChaCudaRng::init(rng_buf_size, device_manager.device(idx).clone(), seed0);
            chacha1.get_mut_chacha().set_nonce(idx as u64);
            let mut chacha2 =
                ChaChaCudaRng::init(rng_buf_size, device_manager.device(idx).clone(), seed1);
            chacha2.get_mut_chacha().set_nonce(idx as u64);
            rngs.push((chacha1, chacha2));
        }

        Self {
            peer_id,
            query_length,
            device_manager,
            kernels,
            xor_assign_u8_kernels,
            rngs,
            is_remote: !comms.is_empty(),
            comms,
            intermediate_results,
            ones,
            results,
            results_peer,
            code_length,
        }
    }

    pub fn alloc_db(&self, max_db_length: usize) -> SlicedProcessedDatabase {
        let max_size = max_db_length / self.device_manager.device_count();
        let (db0_sums, (db1_sums, (db0, db1))) = self
            .device_manager
            .devices()
            .iter()
            .map(|device| unsafe {
                let host_mem0 = MmapMut::map_anon(max_size * self.code_length).unwrap();
                let host_mem1 = MmapMut::map_anon(max_size * self.code_length).unwrap();

                let host_mem0_ptr = host_mem0.as_ptr() as u64;
                let host_mem1_ptr = host_mem1.as_ptr() as u64;

                // Make sure to not drop the memory, even though we only use the pointers
                // afterwards. This also has the effect that this memory is never freed, which
                // is fine for the db.
                forget(host_mem0);
                forget(host_mem1);

                (
                    StreamAwareCudaSlice::from(device.alloc(max_size).unwrap()),
                    (
                        StreamAwareCudaSlice::from(device.alloc(max_size).unwrap()),
                        (host_mem0_ptr, host_mem1_ptr),
                    ),
                )
            })
            .unzip();

        for dev in self.device_manager.devices() {
            dev.synchronize().unwrap();
        }

        SlicedProcessedDatabase {
            code_gr:      CudaVec2DSlicerRawPointer {
                limb_0: db0,
                limb_1: db1,
            },
            code_sums_gr: CudaVec2DSlicerU32 {
                limb_0: db0_sums,
                limb_1: db1_sums,
            },
        }
    }

    pub fn register_host_memory(&self, db: &SlicedProcessedDatabase, max_db_length: usize) {
        let max_size = max_db_length / self.device_manager.device_count();
        for (device_index, device) in self.device_manager.devices().iter().enumerate() {
            device.bind_to_thread().unwrap();
            unsafe {
                let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                    db.code_gr.limb_0[device_index] as *mut _,
                    max_size * self.code_length,
                    CU_MEMHOSTALLOC_PORTABLE,
                );

                let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                    db.code_gr.limb_1[device_index] as *mut _,
                    max_size * self.code_length,
                    CU_MEMHOSTALLOC_PORTABLE,
                );
            }
        }
    }

    pub fn load_single_record(
        index: usize,
        db: &CudaVec2DSlicerRawPointer,
        record: &[u16],
        n_shards: usize,
        code_length: usize,
    ) {
        assert_eq!(record.len(), code_length);

        let a0_host = record
            .iter()
            .map(|&x| ((x as i8) as i32 - 128) as i8)
            .collect::<Vec<_>>();

        let a1_host = record
            .iter()
            .map(|&x: &u16| ((x >> 8) as i32 - 128) as i8)
            .collect::<Vec<_>>();

        let device_index = index % n_shards;
        let device_db_index = index / n_shards;

        unsafe {
            std::ptr::copy(
                a0_host.as_ptr() as *const _,
                (db.limb_0[device_index] + (device_db_index * code_length) as u64) as *mut _,
                code_length,
            );

            std::ptr::copy(
                a1_host.as_ptr() as *const _,
                (db.limb_1[device_index] + (device_db_index * code_length) as u64) as *mut _,
                code_length,
            );
        };
    }

    pub fn preprocess_db(&self, db: &mut SlicedProcessedDatabase, db_lens: &[usize]) {
        let code_len = self.code_length;
        for device_index in 0..self.device_manager.device_count() {
            for (limbs, sum_slices) in [
                (&db.code_gr.limb_0, &mut db.code_sums_gr.limb_0),
                (&db.code_gr.limb_1, &mut db.code_sums_gr.limb_1),
            ] {
                let sums = (0..db_lens[device_index])
                    .into_par_iter()
                    .map(|idx| {
                        let slice: &[i8] = unsafe {
                            std::slice::from_raw_parts(
                                (limbs[device_index] + (idx * code_len) as u64) as *const _,
                                code_len,
                            )
                        };
                        slice.iter().map(|&x| x as u32).sum::<u32>()
                    })
                    .collect::<Vec<_>>();

                self.device_manager
                    .device(device_index)
                    .bind_to_thread()
                    .unwrap();
                unsafe {
                    result::memcpy_htod_sync(sum_slices[device_index].cu_device_ptr, &sums)
                        .unwrap();
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn load_full_db(&self, db: &mut SlicedProcessedDatabase, db_entries: &[u16]) -> Vec<usize> {
        assert!(db_entries.len() % self.code_length == 0);

        let code_length = self.code_length;
        let n_shards = self.device_manager.device_count();
        db_entries
            .par_chunks(self.code_length)
            .enumerate()
            .for_each(|(idx, chunk)| {
                Self::load_single_record(idx, &db.code_gr, chunk, n_shards, code_length);
            });

        // Calculate the number of entries per shard
        let mut db_lens = vec![db_entries.len() / self.code_length / n_shards; n_shards];
        for i in 0..db_lens.len() {
            if i < (db_entries.len() / self.code_length) % n_shards {
                db_lens[i] += 1;
            }
        }

        self.preprocess_db(db, &db_lens);

        db_lens
    }

    pub fn query_sums(
        &self,
        query_ptrs: &CudaVec2DSlicerU8,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) -> CudaVec2DSlicerU32 {
        let mut query1_sums = vec![];
        let mut query0_sums = vec![];

        for idx in 0..self.device_manager.device_count() {
            let device = self.device_manager.device(idx);
            device.bind_to_thread().unwrap();

            let query0 = &query_ptrs.limb_0[idx];
            let query1 = &query_ptrs.limb_1[idx];

            let query0_sum = unsafe {
                malloc_async(
                    streams[idx].stream,
                    self.query_length * mem::size_of::<u32>(),
                )
                .unwrap()
            };
            let slice0_sum = StreamAwareCudaSlice::<u32>::upgrade_ptr_stream(
                query0_sum,
                streams[idx].stream,
                self.query_length,
            );

            let query1_sum = unsafe {
                malloc_async(
                    streams[idx].stream,
                    self.query_length * mem::size_of::<u32>(),
                )
                .unwrap()
            };

            let slice1_sum = StreamAwareCudaSlice::<u32>::upgrade_ptr_stream(
                query1_sum,
                streams[idx].stream,
                self.query_length,
            );

            gemm(
                &blass[idx],
                *query0.device_ptr(),
                *self.ones[idx].device_ptr(),
                query0_sum,
                0,
                0,
                0,
                self.query_length,
                1,
                self.code_length,
                1,
                0,
            );
            gemm(
                &blass[idx],
                *query1.device_ptr(),
                *self.ones[idx].device_ptr(),
                query1_sum,
                0,
                0,
                0,
                self.query_length,
                1,
                self.code_length,
                1,
                0,
            );

            query0_sums.push(slice0_sum);
            query1_sums.push(slice1_sum);
        }
        CudaVec2DSlicer {
            limb_0: query0_sums,
            limb_1: query1_sums,
        }
    }

    pub fn alloc_db_chunk_buffer(&self, max_chunk_size: usize) -> DBChunkBuffers {
        let mut limb_0 = vec![];
        let mut limb_1 = vec![];
        for device in self.device_manager.devices() {
            unsafe {
                limb_0.push(device.alloc(max_chunk_size * self.code_length).unwrap());
                limb_1.push(device.alloc(max_chunk_size * self.code_length).unwrap());
            }
        }
        DBChunkBuffers { limb_0, limb_1 }
    }

    pub fn prefetch_db_chunk(
        &self,
        db: &SlicedProcessedDatabase,
        buffers: &DBChunkBuffers,
        chunk_sizes: &[usize],
        offset: &[usize],
        db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        for idx in 0..self.device_manager.device_count() {
            let device = self.device_manager.device(idx);
            device.bind_to_thread().unwrap();

            if offset[idx] >= db_sizes[idx] || offset[idx] + chunk_sizes[idx] > db_sizes[idx] {
                continue;
            }

            unsafe {
                cudarc::driver::sys::lib()
                    .cuMemcpyHtoDAsync_v2(
                        *buffers.limb_0[idx].device_ptr(),
                        (db.code_gr.limb_0[idx] as usize + offset[idx] * self.code_length)
                            as *mut _,
                        chunk_sizes[idx] * self.code_length,
                        streams[idx].stream,
                    )
                    .result()
                    .unwrap();

                cudarc::driver::sys::lib()
                    .cuMemcpyHtoDAsync_v2(
                        *buffers.limb_1[idx].device_ptr(),
                        (db.code_gr.limb_1[idx] as usize + offset[idx] * self.code_length)
                            as *mut _,
                        chunk_sizes[idx] * self.code_length,
                        streams[idx].stream,
                    )
                    .result()
                    .unwrap();
            }
        }
    }

    pub fn dot<T>(
        &mut self,
        queries: &CudaVec2DSlicer<T>,
        db: &CudaVec2DSlicerRawPointer,
        chunk_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        for idx in 0..self.device_manager.device_count() {
            self.device_manager.device(idx).bind_to_thread().unwrap();
            let query0 = &queries.limb_0[idx];
            let query1 = &queries.limb_1[idx];

            // Prepare randomness to mask results
            if self.is_remote {
                let len: usize = (chunk_sizes[idx] * self.query_length).div_ceil(64) * 64;
                self.rngs[idx].0.fill_rng_no_host_copy(len, &streams[idx]);
                self.rngs[idx].1.fill_rng_no_host_copy(len, &streams[idx]);
            }

            for (i, d) in [db.limb_0[idx], db.limb_1[idx]].into_iter().enumerate() {
                for (j, q) in [query0, query1].iter().enumerate() {
                    if i + j >= LIMBS {
                        continue;
                    }
                    gemm(
                        &blass[idx],
                        d,
                        *q.device_ptr(),
                        *self.intermediate_results[idx].device_ptr(),
                        (offset * self.code_length) as u64,
                        0,
                        0,
                        chunk_sizes[idx],
                        self.query_length,
                        self.code_length,
                        1 << (8 * (i + j)),
                        if i + j == 0 { 0 } else { 1 },
                    );
                }
            }
        }
    }

    pub fn dot_reduce_and_multiply(
        &mut self,
        query_sums: &CudaVec2DSlicerU32,
        db_sums: &CudaVec2DSlicerU32,
        chunk_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
        multiplier: u16,
    ) {
        for idx in 0..self.device_manager.device_count() {
            assert!(
                self.rngs[idx].0.cuda_slice().is_some() && self.rngs[idx].1.cuda_slice().is_some()
            );

            let num_elements = chunk_sizes[idx] * self.query_length;
            let threads_per_block = DEFAULT_LAUNCH_CONFIG_THREADS; // ON CHANGE: sync with kernel
            let cfg = launch_config_from_elements_and_threads(
                num_elements as u32,
                threads_per_block,
                &self.device_manager.devices()[idx],
            );

            unsafe {
                self.kernels[idx]
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &self.intermediate_results[idx],
                            &mut self.results[idx],
                            *db_sums.limb_0[idx].device_ptr(),
                            *db_sums.limb_1[idx].device_ptr(),
                            *query_sums.limb_0[idx].device_ptr(),
                            *query_sums.limb_1[idx].device_ptr(),
                            chunk_sizes[idx] as u64,
                            (chunk_sizes[idx] * self.query_length) as u64,
                            offset as u64,
                            multiplier,
                            self.rngs[idx].0.cuda_slice().unwrap(),
                            self.rngs[idx].1.cuda_slice().unwrap(),
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn dot_reduce(
        &mut self,
        query_sums: &CudaVec2DSlicerU32,
        db_sums: &CudaVec2DSlicerU32,
        chunk_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
    ) {
        self.dot_reduce_and_multiply(query_sums, db_sums, chunk_sizes, offset, streams, 1);
    }

    fn single_xor_assign_u8(
        &self,
        x1: &mut CudaView<u8>,
        x2: &CudaView<u8>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let threads_per_block = DEFAULT_LAUNCH_CONFIG_THREADS; // ON CHANGE: sync with kernel
        let cfg = launch_config_from_elements_and_threads(
            size as u32,
            threads_per_block,
            &self.device_manager.devices()[idx],
        );

        unsafe {
            self.xor_assign_u8_kernels[idx]
                .clone()
                .launch_on_stream(&streams[idx], cfg, (&*x1, x2, size))
                .unwrap();
        }
    }

    // Fill randomness using my RNG
    fn fill_my_rng_into_u8<'a>(
        &mut self,
        rand: &'a mut CudaSlice<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaView<'a, u8> {
        self.rngs[idx]
            .0
            .fill_rng_into(&mut rand.slice_mut(..), &streams[idx]);
        let rand_trans: CudaView<u8> =
    // the transmute_mut is safe because we know that one u32 is 4 u8s, and the buffer is aligned properly for the transmute
        unsafe { rand.transmute(rand.len() * 4).unwrap() };
        rand_trans
    }

    // Fill randomness using the their RNG
    fn fill_their_rng_into_u8<'a>(
        &mut self,
        rand: &'a mut CudaSlice<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaView<'a, u8> {
        self.rngs[idx]
            .1
            .fill_rng_into(&mut rand.slice_mut(..), &streams[idx]);
        let rand_trans: CudaView<u8> =
        // the transmute_mut is safe because we know that one u32 is 4 u8s, and the buffer is aligned properly for the transmute
            unsafe { rand.transmute(rand.len() * 4).unwrap() };
        rand_trans
    }

    fn otp_encrypt_rng_result(
        &mut self,
        len: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u32> {
        assert_eq!(len & 3, 0);
        let mut rand = unsafe {
            self.device_manager
                .device(idx)
                .alloc::<u32>(len >> 2)
                .unwrap()
        };
        let mut rand_u8 = self.fill_my_rng_into_u8(&mut rand, idx, streams);
        self.single_xor_assign_u8(
            &mut rand_u8,
            &self.results[idx].slice(..),
            idx,
            len,
            streams,
        );
        rand
    }

    fn otp_decrypt_rng_result(&mut self, len: usize, idx: usize, streams: &[CudaStream]) {
        assert_eq!(len & 3, 0);
        let mut rand = unsafe {
            self.device_manager
                .device(idx)
                .alloc::<u32>(len >> 2)
                .unwrap()
        };
        let rand_u8 = self.fill_their_rng_into_u8(&mut rand, idx, streams);
        self.single_xor_assign_u8(
            &mut self.results_peer[idx].slice(..),
            &rand_u8,
            idx,
            len,
            streams,
        );
    }

    pub fn reshare_results(&mut self, db_sizes: &[usize], streams: &[CudaStream]) {
        let next_peer = (self.peer_id + 1) % 3;
        let prev_peer = (self.peer_id + 2) % 3;

        let send_bufs = (0..self.device_manager.device_count())
            .map(|idx| {
                let len = db_sizes[idx] * self.query_length * 2;
                self.otp_encrypt_rng_result(len, idx, streams)
            })
            .collect_vec();

        let send = &send_bufs;

        nccl::group_start().unwrap();
        for idx in 0..self.device_manager.device_count() {
            let len = db_sizes[idx] * self.query_length * 2;
            let send_len = len >> 2;
            let send_view = send[idx].slice(..send_len);
            self.comms[idx]
                .send_view(&send_view, next_peer, &streams[idx])
                .unwrap();

            let mut recv_view = self.results_peer[idx].slice(..len);
            self.comms[idx]
                .receive_view(&mut recv_view, prev_peer, &streams[idx])
                .unwrap();
        }
        nccl::group_end().unwrap();
        for idx in 0..self.device_manager.device_count() {
            let len = db_sizes[idx] * self.query_length * 2;
            self.otp_decrypt_rng_result(len, idx, streams);
        }
    }

    pub fn fetch_results(&self, results: &mut [u16], db_sizes: &[usize], device_id: usize) {
        unsafe {
            let res_trans =
                self.results[device_id].transmute(db_sizes[device_id] * self.query_length);

            self.device_manager
                .device(device_id)
                .dtoh_sync_copy_into(&res_trans.unwrap(), results)
                .unwrap();
        }
    }

    pub fn result_chunk_shares<'a>(&'a self, db_sizes: &[usize]) -> Vec<ChunkShareView<'a, u16>> {
        izip!(db_sizes, self.results.iter(), self.results_peer.iter())
            .map(|(&len, xa, xb)| {
                // SAFETY: All bit patterns are valid u16 values
                let xa_view = unsafe {
                    xa.transmute(len * self.query_length)
                        .expect("len is correct")
                };
                // SAFETY: All bit patterns are valid u16 values
                let xb_view = unsafe {
                    xb.transmute(len * self.query_length)
                        .expect("len is correct")
                };
                ChunkShareView {
                    a: xa_view,
                    b: xb_view,
                }
            })
            .collect()
    }
}

#[cfg(test)]
#[cfg(feature = "gpu_dependent")]
mod tests {
    use super::{preprocess_query, ShareDB};
    use crate::{
        dot::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH},
        helpers::device_manager::DeviceManager,
    };
    use float_eq::assert_float_eq;
    use iris_mpc_common::{
        galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
        iris_db::db::IrisDB,
    };
    use itertools::Itertools;
    use ndarray::Array2;
    use num_traits::FromPrimitive;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::sync::Arc;

    const WIDTH: usize = 12_800;
    const QUERY_SIZE: usize = 32;
    const DB_SIZE: usize = 8 * 1000;
    const RNG_SEED: u64 = 42;

    /// Helper to generate random ndarray
    fn random_ndarray<T>(array: Vec<u16>, n: usize, m: usize) -> Array2<T>
    where
        T: FromPrimitive,
    {
        Array2::from_shape_vec(
            (n, m),
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

    fn shard_db(db: &[u16], n_shards: usize) -> Vec<u16> {
        let mut res: Vec<Vec<u16>> = vec![vec![]; n_shards];
        db.iter()
            .chunks(WIDTH)
            .into_iter()
            .enumerate()
            .for_each(|(i, chunk)| {
                res[i % n_shards].extend(chunk);
            });
        res.into_iter().flatten().collect::<Vec<_>>()
    }

    /// Test to verify the matmul operation for random matrices in the field
    #[test]
    fn check_matmul() {
        let db = random_vec(DB_SIZE, WIDTH, u16::MAX as u32);
        let query = random_vec(QUERY_SIZE, WIDTH, u16::MAX as u32);
        let device_manager = Arc::new(DeviceManager::init());
        let n_devices = device_manager.device_count();

        let mut gpu_result = vec![0u16; DB_SIZE / n_devices * QUERY_SIZE];

        let mut engine = ShareDB::init(
            0,
            device_manager.clone(),
            DB_SIZE,
            QUERY_SIZE,
            IRIS_CODE_LENGTH,
            ([0u32; 8], [0u32; 8]),
            vec![],
        );
        let preprocessed_query = preprocess_query(&query);
        let streams = device_manager.fork_streams();
        let blass = device_manager.create_cublas(&streams);
        let preprocessed_query = device_manager
            .htod_transfer_query(&preprocessed_query, &streams, QUERY_SIZE, IRIS_CODE_LENGTH)
            .unwrap();
        let query_sums = engine.query_sums(&preprocessed_query, &streams, &blass);
        let mut db_slices = engine.alloc_db(DB_SIZE);
        let db_sizes = engine.load_full_db(&mut db_slices, &db);

        engine.dot(
            &preprocessed_query,
            &db_slices.code_gr,
            &db_sizes,
            0,
            &streams,
            &blass,
        );
        engine.dot_reduce(&query_sums, &db_slices.code_sums_gr, &db_sizes, 0, &streams);
        device_manager.await_streams(&streams);

        let a_nda = random_ndarray::<u16>(shard_db(&db, n_devices), DB_SIZE, WIDTH);
        let b_nda = random_ndarray::<u16>(query.clone(), QUERY_SIZE, WIDTH);
        let c_nda = a_nda.dot(&b_nda.t());

        let mut vec_column_major: Vec<u16> = Vec::new();
        for col in 0..c_nda.ncols() {
            for row in c_nda.column(col) {
                vec_column_major.push(*row);
            }
        }

        for device_idx in 0..n_devices {
            engine.fetch_results(&mut gpu_result, &db_sizes, device_idx);
            let selected_elements: Vec<u16> = vec_column_major
                .chunks(DB_SIZE)
                .flat_map(|chunk| {
                    chunk
                        .iter()
                        .skip(DB_SIZE / n_devices * device_idx)
                        .take(DB_SIZE / n_devices)
                })
                .cloned()
                .collect();

            assert_eq!(selected_elements, gpu_result);
        }
    }

    /// Checks that the result of a matmul of the original data equals the
    /// reconstructed result of individual matmuls on the shamir shares.
    #[test]
    fn check_shared_matmul() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let device_manager = Arc::new(DeviceManager::init());
        let n_devices = device_manager.device_count();

        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        let mut gpu_result = [
            vec![0u16; DB_SIZE * QUERY_SIZE / n_devices],
            vec![0u16; DB_SIZE * QUERY_SIZE / n_devices],
            vec![0u16; DB_SIZE * QUERY_SIZE / n_devices],
        ];

        for i in 0..3 {
            let device_manager = Arc::clone(&device_manager);

            let codes_db = db
                .db
                .iter()
                .flat_map(|iris| {
                    GaloisRingIrisCodeShare::encode_mask_code(
                        &iris.mask,
                        &mut StdRng::seed_from_u64(RNG_SEED),
                    )[i]
                        .coefs
                })
                .collect::<Vec<_>>();

            let querys = db.db[0..QUERY_SIZE]
                .iter()
                .flat_map(|iris| {
                    let mut shares = GaloisRingIrisCodeShare::encode_mask_code(
                        &iris.mask,
                        &mut StdRng::seed_from_u64(RNG_SEED),
                    );
                    shares[i].preprocess_iris_code_query_share();
                    shares[i].coefs
                })
                .collect::<Vec<_>>();

            let mut engine = ShareDB::init(
                0,
                device_manager.clone(),
                DB_SIZE,
                QUERY_SIZE,
                IRIS_CODE_LENGTH,
                ([0u32; 8], [0u32; 8]),
                vec![],
            );
            let preprocessed_query = preprocess_query(&querys);
            let streams = device_manager.fork_streams();
            let blass = device_manager.create_cublas(&streams);
            let preprocessed_query = device_manager
                .htod_transfer_query(&preprocessed_query, &streams, QUERY_SIZE, IRIS_CODE_LENGTH)
                .unwrap();
            let query_sums = engine.query_sums(&preprocessed_query, &streams, &blass);
            let mut db_slices = engine.alloc_db(DB_SIZE);
            let db_sizes = engine.load_full_db(&mut db_slices, &codes_db);

            engine.dot(
                &preprocessed_query,
                &db_slices.code_gr,
                &db_sizes,
                0,
                &streams,
                &blass,
            );
            engine.dot_reduce(&query_sums, &db_slices.code_sums_gr, &db_sizes, 0, &streams);
            device_manager.await_streams(&streams);
            engine.fetch_results(&mut gpu_result[i], &db_sizes, 0);
        }

        for i in 0..DB_SIZE * QUERY_SIZE / n_devices {
            assert_eq!(
                (gpu_result[0][i] + gpu_result[1][i] + gpu_result[2][i]),
                (db.db[i / (DB_SIZE / n_devices)].mask
                    & db.db[(i % (DB_SIZE / n_devices)) * n_devices].mask)
                    .count_ones() as u16
            );
        }
    }

    /// Calculates the distances between a query and a shamir secret shared db
    /// and checks the result against reference plain implementation.
    #[test]
    fn check_shared_distances() {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let device_manager = Arc::new(DeviceManager::init());
        let n_devices = device_manager.device_count();

        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        let mut results_codes = vec![vec![0u16; DB_SIZE / n_devices * QUERY_SIZE]; 3];
        let mut results_masks = vec![vec![0u16; DB_SIZE / n_devices * QUERY_SIZE]; 3];

        for party_id in 0..3 {
            // DBs
            let codes_db = db
                .db
                .iter()
                .flat_map(|iris| {
                    GaloisRingIrisCodeShare::encode_iris_code(
                        &iris.code,
                        &iris.mask,
                        &mut StdRng::seed_from_u64(RNG_SEED),
                    )[party_id]
                        .coefs
                })
                .collect::<Vec<_>>();

            let masks_db = db
                .db
                .iter()
                .flat_map(|iris| {
                    let mask: GaloisRingTrimmedMaskCodeShare =
                        GaloisRingIrisCodeShare::encode_mask_code(
                            &iris.mask,
                            &mut StdRng::seed_from_u64(RNG_SEED),
                        )[party_id]
                            .clone()
                            .into();
                    mask.coefs
                })
                .collect::<Vec<_>>();

            // Queries
            let code_queries = db.db[0..QUERY_SIZE]
                .iter()
                .flat_map(|iris| {
                    let mut shares = GaloisRingIrisCodeShare::encode_iris_code(
                        &iris.code,
                        &iris.mask,
                        &mut StdRng::seed_from_u64(RNG_SEED),
                    );
                    shares[party_id].preprocess_iris_code_query_share();
                    shares[party_id].coefs
                })
                .collect::<Vec<_>>();

            let mask_queries = db.db[0..QUERY_SIZE]
                .iter()
                .flat_map(|iris| {
                    let mut shares = GaloisRingIrisCodeShare::encode_mask_code(
                        &iris.mask,
                        &mut StdRng::seed_from_u64(RNG_SEED),
                    );
                    shares[party_id].preprocess_iris_code_query_share();
                    let mask: GaloisRingTrimmedMaskCodeShare = shares[party_id].clone().into();
                    mask.coefs
                })
                .collect::<Vec<_>>();

            let device_manager = Arc::new(DeviceManager::init());

            let mut codes_engine = ShareDB::init(
                party_id,
                device_manager.clone(),
                DB_SIZE,
                QUERY_SIZE,
                IRIS_CODE_LENGTH,
                ([0u32; 8], [0u32; 8]),
                vec![],
            );
            let mut masks_engine = ShareDB::init(
                party_id,
                device_manager.clone(),
                DB_SIZE,
                QUERY_SIZE,
                MASK_CODE_LENGTH,
                ([0u32; 8], [0u32; 8]),
                vec![],
            );

            let code_query = preprocess_query(&code_queries);
            let mask_query = preprocess_query(&mask_queries);

            let streams = device_manager.fork_streams();
            let blass = device_manager.create_cublas(&streams);
            let code_query = device_manager
                .htod_transfer_query(&code_query, &streams, QUERY_SIZE, IRIS_CODE_LENGTH)
                .unwrap();
            let mask_query = device_manager
                .htod_transfer_query(&mask_query, &streams, QUERY_SIZE, MASK_CODE_LENGTH)
                .unwrap();
            let code_query_sums = codes_engine.query_sums(&code_query, &streams, &blass);
            let mask_query_sums = masks_engine.query_sums(&mask_query, &streams, &blass);
            let mut code_db_slices = codes_engine.alloc_db(DB_SIZE);
            let db_sizes = codes_engine.load_full_db(&mut code_db_slices, &codes_db);
            let mut mask_db_slices = masks_engine.alloc_db(DB_SIZE);
            let mask_db_sizes = masks_engine.load_full_db(&mut mask_db_slices, &masks_db);

            assert_eq!(db_sizes, mask_db_sizes);

            codes_engine.dot(
                &code_query,
                &code_db_slices.code_gr,
                &db_sizes,
                0,
                &streams,
                &blass,
            );
            masks_engine.dot(
                &mask_query,
                &mask_db_slices.code_gr,
                &db_sizes,
                0,
                &streams,
                &blass,
            );

            codes_engine.dot_reduce(
                &code_query_sums,
                &code_db_slices.code_sums_gr,
                &db_sizes,
                0,
                &streams,
            );
            masks_engine.dot_reduce_and_multiply(
                &mask_query_sums,
                &mask_db_slices.code_sums_gr,
                &db_sizes,
                0,
                &streams,
                2,
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
            let code = results_codes[0][i] + results_codes[1][i] + results_codes[2][i];
            let mask = results_masks[0][i] + results_masks[1][i] + results_masks[2][i];

            reconstructed_codes.push(code);
            reconstructed_masks.push(mask);
        }

        // Calculate the distance in plain
        let dists = reconstructed_codes
            .into_iter()
            .zip(reconstructed_masks)
            .map(|(code, mask)| 0.5f64 - (code as i16) as f64 / (2f64 * mask as f64))
            .collect::<Vec<_>>();

        // Compare against plain reference implementation
        let reference_dists = db.calculate_distances(&db.db[0]);

        // TODO: check for all devices and the whole query
        for i in 0..DB_SIZE / n_devices {
            assert_float_eq!(dists[i], reference_dists[i * n_devices], abs <= 1e-6);
        }
    }
}
