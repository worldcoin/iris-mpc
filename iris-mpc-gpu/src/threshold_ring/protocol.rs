use crate::{
    helpers::{
        comm::NcclComm, device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync,
        launch_config_from_elements_and_threads, DEFAULT_LAUNCH_CONFIG_THREADS,
    },
    rng::chacha_corr::ChaChaCudaCorrRng,
    threshold_ring::cuda::PTX_SRC,
};
use cudarc::{
    driver::{
        result::stream, CudaDevice, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut,
        DeviceSlice, LaunchAsync,
    },
    nccl::result,
    nvrtc::{self, Ptx},
};
use itertools::{izip, Itertools};
use std::{ops::Range, sync::Arc};

pub(crate) const B_BITS: usize = 16;
const SHARE_RING_BITSIZE: usize = 16;

pub struct ChunkShare<T> {
    pub a: CudaSlice<T>,
    pub b: CudaSlice<T>,
}

pub struct ChunkShareView<'a, T> {
    pub a: CudaView<'a, T>,
    pub b: CudaView<'a, T>,
}

impl<T> ChunkShare<T> {
    pub fn new(a: CudaSlice<T>, b: CudaSlice<T>) -> Self {
        ChunkShare { a, b }
    }

    pub fn as_view(&self) -> ChunkShareView<T> {
        ChunkShareView {
            a: self.a.slice(..),
            b: self.b.slice(..),
        }
    }
    pub fn as_view_mut(&mut self) -> ChunkShareViewMut<T> {
        ChunkShareViewMut {
            a: self.a.slice_mut(..),
            b: self.b.slice_mut(..),
        }
    }

    pub fn get_offset(&self, i: usize, chunk_size: usize) -> ChunkShareView<T> {
        ChunkShareView {
            a: self.a.slice(i * chunk_size..(i + 1) * chunk_size),
            b: self.b.slice(i * chunk_size..(i + 1) * chunk_size),
        }
    }
    pub fn get_offset_mut(&mut self, i: usize, chunk_size: usize) -> ChunkShareViewMut<T> {
        ChunkShareViewMut {
            a: self.a.slice_mut(i * chunk_size..(i + 1) * chunk_size),
            b: self.b.slice_mut(i * chunk_size..(i + 1) * chunk_size),
        }
    }

    pub fn get_range(&self, start: usize, end: usize) -> ChunkShareView<T> {
        ChunkShareView {
            a: self.a.slice(start..end),
            b: self.b.slice(start..end),
        }
    }

    pub fn is_empty(&self) -> bool {
        assert_eq!(self.a.is_empty(), self.b.is_empty());
        self.a.is_empty()
    }

    pub fn len(&self) -> usize {
        assert_eq!(self.a.len(), self.b.len());
        self.a.len()
    }
}

pub struct ChunkShareViewMut<'a, T> {
    pub a: CudaViewMut<'a, T>,
    pub b: CudaViewMut<'a, T>,
}

#[allow(clippy::needless_lifetimes)]
impl<'a, T> ChunkShareView<'a, T> {
    pub fn get_offset(&self, i: usize, chunk_size: usize) -> ChunkShareView<T> {
        ChunkShareView {
            a: self.a.slice(i * chunk_size..(i + 1) * chunk_size),
            b: self.b.slice(i * chunk_size..(i + 1) * chunk_size),
        }
    }

    pub fn get_range(&self, start: usize, end: usize) -> ChunkShareView<T> {
        ChunkShareView {
            a: self.a.slice(start..end),
            b: self.b.slice(start..end),
        }
    }

    pub fn is_empty(&self) -> bool {
        assert_eq!(self.a.is_empty(), self.b.is_empty());
        self.a.is_empty()
    }

    pub fn len(&self) -> usize {
        assert_eq!(self.a.len(), self.b.len());
        self.a.len()
    }
}

impl<T> Clone for ChunkShare<T>
where
    T: cudarc::driver::DeviceRepr,
{
    fn clone(&self) -> Self {
        ChunkShare {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }
}

struct Kernels {
    pub(crate) and: CudaFunction,
    pub(crate) or_assign: CudaFunction,
    pub(crate) xor: CudaFunction,
    pub(crate) xor_assign: CudaFunction,
    pub(crate) single_xor_assign_u16: CudaFunction,
    pub(crate) single_xor_assign_u32: CudaFunction,
    pub(crate) single_xor_assign_u64: CudaFunction,
    pub(crate) split: CudaFunction,
    pub(crate) lift_split: CudaFunction,
    pub(crate) lift_mul_sub: CudaFunction,
    pub(crate) lifted_sub: CudaFunction,
    pub(crate) finalize_lift: CudaFunction,
    pub(crate) transpose_32x64: CudaFunction,
    pub(crate) transpose_16x64: CudaFunction,
    pub(crate) ot_sender: CudaFunction,
    pub(crate) ot_receiver: CudaFunction,
    pub(crate) ot_helper: CudaFunction,
    pub(crate) split_arithmetic_xor: CudaFunction,
    pub(crate) arithmetic_xor_assign: CudaFunction,
    pub(crate) assign_u32: CudaFunction,
    pub(crate) assign_u64: CudaFunction,
    pub(crate) collapse_u64_helper: CudaFunction,
    pub(crate) collapse_sum_assign: CudaFunction,
    pub(crate) collapse_sum: CudaFunction,
    pub(crate) rotate_bitvec: CudaFunction,
    pub(crate) mask_bitvec: CudaFunction,
    pub(crate) cross_mul_pre: CudaFunction,
    pub(crate) conditional_select_pre: CudaFunction,
    pub(crate) conditional_select_post: CudaFunction,
    pub(crate) prelifted_sub_ab: CudaFunction,
    pub(crate) pre_lift_u16_u32_signed: CudaFunction,
    pub(crate) finalize_lift_u16_u32_signed: CudaFunction,
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
                "xor_assign_u16",
                "xor_assign_u32",
                "xor_assign_u64",
                "shared_and_pre",
                "shared_or_pre_assign",
                "split",
                "lift_split",
                "shared_lift_mul_sub",
                "shared_finalize_lift",
                "shared_lifted_sub",
                "shared_u32_transpose_pack_u64",
                "shared_u16_transpose_pack_u64",
                "packed_ot_sender",
                "packed_ot_receiver",
                "packed_ot_helper",
                "split_for_arithmetic_xor",
                "shared_arithmetic_xor_pre_assign_u32",
                "shared_assign",
                "shared_assign_u32",
                "collapse_u64_helper",
                "collapse_sum_assign",
                "collapse_sum",
                "rotate_bitvec",
                "mask_bitvec",
                "cross_mul_pre",
                "conditional_select_pre",
                "conditional_select_post",
                "shared_prelifted_sub_ab",
                "shared_pre_lift_u16_u32_signed",
                "shared_finalize_lift_u16_u32_signed",
            ],
        )
        .unwrap();
        let and = dev.get_func(Self::MOD_NAME, "shared_and_pre").unwrap();
        let or_assign = dev
            .get_func(Self::MOD_NAME, "shared_or_pre_assign")
            .unwrap();
        let xor = dev.get_func(Self::MOD_NAME, "shared_xor").unwrap();
        let xor_assign = dev.get_func(Self::MOD_NAME, "shared_xor_assign").unwrap();
        let single_xor_assign_u16 = dev.get_func(Self::MOD_NAME, "xor_assign_u16").unwrap();
        let single_xor_assign_u32 = dev.get_func(Self::MOD_NAME, "xor_assign_u32").unwrap();
        let single_xor_assign_u64 = dev.get_func(Self::MOD_NAME, "xor_assign_u64").unwrap();
        let split = dev.get_func(Self::MOD_NAME, "split").unwrap();
        let lift_split = dev.get_func(Self::MOD_NAME, "lift_split").unwrap();
        let lift_mul_sub = dev.get_func(Self::MOD_NAME, "shared_lift_mul_sub").unwrap();
        let finalize_lift = dev
            .get_func(Self::MOD_NAME, "shared_finalize_lift")
            .unwrap();
        let lifted_sub = dev.get_func(Self::MOD_NAME, "shared_lifted_sub").unwrap();
        let transpose_32x64 = dev
            .get_func(Self::MOD_NAME, "shared_u32_transpose_pack_u64")
            .unwrap();
        let transpose_16x64 = dev
            .get_func(Self::MOD_NAME, "shared_u16_transpose_pack_u64")
            .unwrap();
        let ot_sender = dev.get_func(Self::MOD_NAME, "packed_ot_sender").unwrap();
        let ot_receiver = dev.get_func(Self::MOD_NAME, "packed_ot_receiver").unwrap();
        let ot_helper = dev.get_func(Self::MOD_NAME, "packed_ot_helper").unwrap();
        let split_arithmetic_xor = dev
            .get_func(Self::MOD_NAME, "split_for_arithmetic_xor")
            .unwrap();
        let arithmetic_xor_assign = dev
            .get_func(Self::MOD_NAME, "shared_arithmetic_xor_pre_assign_u32")
            .unwrap();
        let assign_u64 = dev.get_func(Self::MOD_NAME, "shared_assign").unwrap();
        let assign_u32 = dev.get_func(Self::MOD_NAME, "shared_assign_u32").unwrap();
        let collapse_u64_helper = dev.get_func(Self::MOD_NAME, "collapse_u64_helper").unwrap();
        let collapse_sum_assign = dev.get_func(Self::MOD_NAME, "collapse_sum_assign").unwrap();
        let collapse_sum = dev.get_func(Self::MOD_NAME, "collapse_sum").unwrap();
        let rotate_bitvec = dev.get_func(Self::MOD_NAME, "rotate_bitvec").unwrap();
        let mask_bitvec = dev.get_func(Self::MOD_NAME, "mask_bitvec").unwrap();
        let cross_mul_pre = dev.get_func(Self::MOD_NAME, "cross_mul_pre").unwrap();
        let conditional_select_pre = dev
            .get_func(Self::MOD_NAME, "conditional_select_pre")
            .unwrap();
        let conditional_select_post = dev
            .get_func(Self::MOD_NAME, "conditional_select_post")
            .unwrap();
        let prelifted_sub_ab = dev
            .get_func(Self::MOD_NAME, "shared_prelifted_sub_ab")
            .unwrap();
        let finalize_lift_u16_u32_signed = dev
            .get_func(Self::MOD_NAME, "shared_finalize_lift_u16_u32_signed")
            .unwrap();
        let pre_lift_u16_u32_signed = dev
            .get_func(Self::MOD_NAME, "shared_pre_lift_u16_u32_signed")
            .unwrap();

        Kernels {
            and,
            or_assign,
            xor,
            xor_assign,
            single_xor_assign_u16,
            single_xor_assign_u32,
            single_xor_assign_u64,
            split,
            lift_split,
            lift_mul_sub,
            lifted_sub,
            finalize_lift,
            transpose_32x64,
            transpose_16x64,
            ot_sender,
            ot_receiver,
            ot_helper,
            split_arithmetic_xor,
            arithmetic_xor_assign,
            assign_u32,
            assign_u64,
            collapse_u64_helper,
            collapse_sum_assign,
            collapse_sum,
            rotate_bitvec,
            mask_bitvec,
            cross_mul_pre,
            conditional_select_pre,
            conditional_select_post,
            prelifted_sub_ab,
            pre_lift_u16_u32_signed,
            finalize_lift_u16_u32_signed,
        }
    }
}

struct Buffers {
    lifted_shares: Option<Vec<ChunkShare<u32>>>,
    lifted_shares_buckets1: Option<Vec<ChunkShare<u32>>>,
    lifted_shares_buckets2: Option<Vec<ChunkShare<u32>>>,
    lifting_corrections: Option<Vec<ChunkShare<u16>>>,
    buckets_recv_buffer: Option<Vec<ChunkShare<u32>>>,
    // This is also the buffer where the result is stored:
    lifted_shares_split1_result: Option<Vec<ChunkShare<u64>>>,
    lifted_shares_split2: Option<Vec<ChunkShare<u64>>>,
    lifted_shares_split3: Option<Vec<ChunkShare<u64>>>,
    binary_adder_s: Option<Vec<ChunkShare<u64>>>,
    binary_adder_c: Option<Vec<ChunkShare<u64>>>,
    ot_m0: Option<Vec<CudaSlice<u16>>>,
    ot_m1: Option<Vec<CudaSlice<u16>>>,
    ot_wc: Option<Vec<CudaSlice<u16>>>,
    chunk_size: usize,
}

impl Buffers {
    fn new(devices: &[Arc<CudaDevice>], chunk_size: usize, n_buckets: Option<usize>) -> Self {
        let lifted_shares = Some(Self::allocate_buffer(chunk_size * 64, devices));
        let lifted_shares_buckets1 = Some(Self::allocate_buffer(chunk_size * 64, devices));
        let lifted_shares_buckets2 = Some(Self::allocate_buffer(chunk_size * 64, devices));
        let lifted_shares_split1_result = Some(Self::allocate_buffer(chunk_size * 32, devices));
        let lifted_shares_split2 = Some(Self::allocate_buffer(chunk_size * 32, devices));
        let lifted_shares_split3 = Some(Self::allocate_buffer(chunk_size * 32, devices));

        let binary_adder_s = Some(Self::allocate_buffer(chunk_size * 31, devices));
        let binary_adder_c = Some(Self::allocate_buffer(chunk_size * 31, devices));

        let lifting_corrections = Some(Self::allocate_buffer(chunk_size * 128, devices));

        let ot_m0 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));
        let ot_m1 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));
        let ot_wc = Some(Self::allocate_single_buffer(chunk_size * 128, devices));

        let buckets_recv_buffer =
            n_buckets.map(|n_buckets| Self::allocate_buffer(n_buckets, devices));

        Buffers {
            lifted_shares,
            lifted_shares_buckets1,
            lifted_shares_buckets2,
            lifting_corrections,
            lifted_shares_split1_result,
            lifted_shares_split2,
            lifted_shares_split3,
            buckets_recv_buffer,
            binary_adder_s,
            binary_adder_c,
            ot_m0,
            ot_m1,
            ot_wc,
            chunk_size,
        }
    }

    fn allocate_single_buffer<T>(size: usize, devices: &[Arc<CudaDevice>]) -> Vec<CudaSlice<T>>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        let mut res = Vec::with_capacity(devices.len());

        for dev in devices.iter() {
            res.push(dev.alloc_zeros::<T>(size).unwrap());
        }
        res
    }

    fn allocate_buffer<T>(size: usize, devices: &[Arc<CudaDevice>]) -> Vec<ChunkShare<T>>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        let mut res = Vec::with_capacity(devices.len());

        for dev in devices.iter() {
            let a = dev.alloc_zeros::<T>(size).unwrap();
            let b = dev.alloc_zeros::<T>(size).unwrap();
            res.push(ChunkShare::new(a, b));
            dev.synchronize().unwrap();
        }
        res
    }

    fn take_buffer<T>(inp: &mut Option<Vec<ChunkShare<T>>>) -> Vec<ChunkShare<T>> {
        assert!(inp.is_some());
        std::mem::take(inp).unwrap()
    }

    fn get_buffer_chunk<T>(inp: &[ChunkShare<T>], size: usize) -> Vec<ChunkShareView<T>> {
        let mut res = Vec::with_capacity(inp.len());
        for inp in inp {
            res.push(inp.get_range(0, size));
        }
        res
    }

    fn return_buffer<T>(des: &mut Option<Vec<ChunkShare<T>>>, src: Vec<ChunkShare<T>>) {
        assert!(des.is_none());
        *des = Some(src);
    }

    fn take_single_buffer<T>(inp: &mut Option<Vec<CudaSlice<T>>>) -> Vec<CudaSlice<T>> {
        assert!(inp.is_some());
        std::mem::take(inp).unwrap()
    }

    fn get_single_buffer_chunk<T>(inp: &[CudaSlice<T>], size: usize) -> Vec<CudaView<T>> {
        let mut res = Vec::with_capacity(inp.len());
        for inp in inp {
            res.push(inp.slice(..size));
        }
        res
    }

    fn return_single_buffer<T>(des: &mut Option<Vec<CudaSlice<T>>>, src: Vec<CudaSlice<T>>) {
        assert!(des.is_none());
        *des = Some(src);
    }

    fn check_buffers(&self) {
        assert!(self.lifted_shares.is_some());
        assert!(self.lifted_shares_buckets1.is_some());
        assert!(self.lifted_shares_buckets2.is_some());
        assert!(self.lifted_shares_split1_result.is_some());
        assert!(self.lifted_shares_split2.is_some());
        assert!(self.lifted_shares_split3.is_some());
        assert!(self.binary_adder_s.is_some());
        assert!(self.binary_adder_c.is_some());
        assert!(self.lifting_corrections.is_some());
        assert!(self.ot_m0.is_some());
        assert!(self.ot_m1.is_some());
        assert!(self.ot_wc.is_some());
    }
}

pub struct Circuits {
    peer_id: usize,
    next_id: usize,
    prev_id: usize,
    chunk_size: usize,
    n_devices: usize,
    devs: Vec<Arc<CudaDevice>>,
    comms: Vec<Arc<NcclComm>>,
    kernels: Vec<Kernels>,
    buffers: Buffers,
    rngs: Vec<ChaChaCudaCorrRng>,
}

impl Circuits {
    const BITS: usize = SHARE_RING_BITSIZE + B_BITS;

    pub fn synchronize_all(&self) {
        for dev in self.devs.iter() {
            dev.synchronize().unwrap();
        }
    }

    pub fn synchronize_streams(&self, streams: &[CudaStream]) {
        for (dev, stream) in izip!(self.devs.iter(), streams.iter()) {
            dev.bind_to_thread().unwrap();
            unsafe { stream::synchronize(stream.stream).unwrap() }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        peer_id: usize,
        input_size: usize, // per GPU
        alloc_size: usize,
        n_buckets: Option<usize>,
        chacha_seeds: ([u32; 8], [u32; 8]),
        device_manager: Arc<DeviceManager>,
        comms: Vec<Arc<NcclComm>>,
    ) -> Self {
        // For the transpose, inputs should be multiple of 64 bits
        assert!(input_size % 64 == 0);
        // Chunk size is the number of u64 elements per bit in the binary circuits
        let chunk_size = input_size / 64;
        assert!(alloc_size >= chunk_size);
        let n_devices = device_manager.device_count();

        let mut devs = Vec::with_capacity(n_devices);
        let mut kernels = Vec::with_capacity(n_devices);
        let mut rngs = Vec::with_capacity(n_devices);

        let ptx = nvrtc::compile_ptx(PTX_SRC).unwrap();
        for i in 0..n_devices {
            let dev = device_manager.device(i);
            let kernel = Kernels::new(dev.clone(), ptx.clone());
            let rng = ChaChaCudaCorrRng::init(dev.clone(), chacha_seeds.0, chacha_seeds.1);

            devs.push(dev);
            kernels.push(kernel);
            rngs.push(rng);
        }

        let buffers = Buffers::new(&devs, alloc_size, n_buckets);

        Circuits {
            peer_id,
            next_id: (peer_id + 1) % 3,
            prev_id: (peer_id + 2) % 3,
            chunk_size,
            n_devices,
            devs,
            comms,
            kernels,
            buffers,
            rngs,
        }
    }

    // TODO: have different chunk sizes for each gpu
    pub fn set_chunk_size(&mut self, chunk_size: usize) {
        assert!(chunk_size <= self.buffers.chunk_size);
        self.chunk_size = chunk_size;
    }

    // TODO: have different chunk sizes for each gpu
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn take_result_buffer(&mut self) -> Vec<ChunkShare<u64>> {
        Buffers::take_buffer(&mut self.buffers.lifted_shares_split1_result)
    }

    pub fn return_result_buffer(&mut self, src: Vec<ChunkShare<u64>>) {
        Buffers::return_buffer(&mut self.buffers.lifted_shares_split1_result, src);
    }

    pub fn peer_id(&self) -> usize {
        self.peer_id
    }

    pub fn next_id(&self) -> usize {
        self.next_id
    }
    pub fn prev_id(&self) -> usize {
        self.prev_id
    }

    pub fn get_devices(&self) -> Vec<Arc<CudaDevice>> {
        self.devs.clone()
    }

    pub fn comms(&self) -> &[Arc<NcclComm>] {
        &self.comms
    }

    // Fill randomness using the correlated RNG
    fn fill_rand_u64(&mut self, rand: &mut CudaSlice<u64>, idx: usize, streams: &[CudaStream]) {
        let rng = &mut self.rngs[idx];
        let mut rand_trans: CudaViewMut<u32> =
        // the transmute_mut is safe because we know that one u64 is 2 u32s, and the buffer is aligned properly for the transmute
            unsafe { rand.transmute_mut(rand.len() * 2).unwrap() };
        rng.fill_rng_into(&mut rand_trans, &streams[idx]);
    }

    // Fill randomness using the correlated RNG
    fn fill_my_rand_u64(&mut self, rand: &mut CudaSlice<u64>, idx: usize, streams: &[CudaStream]) {
        let rng = &mut self.rngs[idx];
        let mut rand_trans: CudaViewMut<u32> =
              // the transmute_mut is safe because we know that one u64 is 2 u32s, and the buffer is aligned properly for the transmute
                  unsafe { rand.transmute_mut(rand.len() * 2).unwrap() };
        rng.fill_my_rng_into(&mut rand_trans, &streams[idx]);
    }

    // Fill randomness using the correlated RNG
    fn fill_their_rand_u64(
        &mut self,
        rand: &mut CudaSlice<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let rng = &mut self.rngs[idx];
        let mut rand_trans: CudaViewMut<u32> =
                  // the transmute_mut is safe because we know that one u64 is 2 u32s, and the buffer is aligned properly for the transmute
                      unsafe { rand.transmute_mut(rand.len() * 2).unwrap() };
        rng.fill_their_rng_into(&mut rand_trans, &streams[idx]);
    }

    // Fill randomness using the correlated RNG
    fn fill_my_rng_into_u16<'a>(
        &mut self,
        rand: &'a mut CudaSlice<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaView<'a, u16> {
        let rng = &mut self.rngs[idx];
        rng.fill_my_rng_into(&mut rand.slice_mut(..), &streams[idx]);
        let rand_trans: CudaView<u16> =
        // the transmute_mut is safe because we know that one u32 is 2 u16s, and the buffer is aligned properly for the transmute
            unsafe { rand.transmute(rand.len() * 2).unwrap() };
        rand_trans
    }

    // Fill randomness using the correlated RNG
    fn fill_their_rng_into_u16<'a>(
        &mut self,
        rand: &'a mut CudaSlice<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaView<'a, u16> {
        let rng = &mut self.rngs[idx];
        rng.fill_their_rng_into(&mut rand.slice_mut(..), &streams[idx]);
        let rand_trans: CudaView<u16> =
        // the transmute_mut is safe because we know that one u32 is 2 u16s, and the buffer is aligned properly for the transmute
            unsafe { rand.transmute(rand.len() * 2).unwrap() };
        rand_trans
    }

    fn packed_and_many_pre(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        bits: usize,
        idx: usize,
        streams: &[CudaStream],
    ) {
        // SAFETY: Only unsafe because memory is not initialized. But, we fill
        // afterwards.
        let size = (self.chunk_size * bits + 7) / 8;
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(size * 8).unwrap() };
        self.fill_rand_u64(&mut rand, idx, streams);

        let cfg = launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .and
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg,
                    (
                        &res.a,
                        &x1.a,
                        &x1.b,
                        &x2.a,
                        &x2.b,
                        &rand,
                        self.chunk_size * bits,
                    ),
                )
                .unwrap();
        }
    }

    fn assign_view(
        &mut self,
        des: &mut ChunkShareView<u64>,
        src: &ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(src.len(), des.len());
        let cfg = launch_config_from_elements_and_threads(
            src.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .assign_u64
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg,
                    (&des.a, &des.b, &src.a, &src.b, src.len()),
                )
                .unwrap();
        }
    }

    fn assign_view_u32(
        &mut self,
        des: &mut ChunkShareView<u32>,
        src: &ChunkShareView<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(src.len(), des.len());
        let cfg = launch_config_from_elements_and_threads(
            src.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .assign_u32
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg,
                    (&des.a, &des.b, &src.a, &src.b, src.len()),
                )
                .unwrap();
        }
    }

    fn and_many_pre(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        // SAFETY: Only unsafe because memory is not initialized. But, we fill
        // afterwards.
        let size = (self.chunk_size + 7) / 8;
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(size * 8).unwrap() };
        self.fill_rand_u64(&mut rand, idx, streams);

        unsafe {
            self.kernels[idx]
                .and
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg,
                    (&res.a, &x1.a, &x1.b, &x2.a, &x2.b, &rand, self.chunk_size),
                )
                .unwrap();
        }
    }

    fn or_many_pre_assign(
        &mut self,
        x1: &mut ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        // SAFETY: Only unsafe because memory is not initialized. But, we fill
        // afterwards.
        let size = (x1.len() + 7) / 8;
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(size * 8).unwrap() };
        self.fill_rand_u64(&mut rand, idx, streams);

        let cfg = launch_config_from_elements_and_threads(
            x1.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .or_assign
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg,
                    (&x1.a, &x1.b, &x2.a, &x2.b, &rand, x1.len()),
                )
                .unwrap();
        }
    }

    fn arithmetic_xor_many_pre_assign(
        &mut self,
        x1: &mut ChunkShareView<u32>,
        x2: &ChunkShareView<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            x1.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        let rng = &mut self.rngs[idx];
        let stream = &streams[idx];
        let dev = &self.devs[idx];

        // SAFETY: Only unsafe because memory is not initialized. But, we fill
        // afterwards.
        let size = (x1.len() + 15) / 16;
        let size = size * 16;
        let mut my_rand = unsafe { dev.alloc::<u32>(size).unwrap() };
        let mut their_rand = unsafe { dev.alloc::<u32>(size).unwrap() };

        rng.fill_my_rng_into(&mut my_rand.slice_mut(..), stream);
        rng.fill_their_rng_into(&mut their_rand.slice_mut(..), stream);

        unsafe {
            self.kernels[idx]
                .arithmetic_xor_assign
                .clone()
                .launch_on_stream(
                    stream,
                    cfg,
                    (&x1.a, &x1.b, &x2.a, &x2.b, &my_rand, &their_rand, x1.len()),
                )
                .unwrap();
        }
    }

    // Encrypt using chacha in my_rng
    fn chacha1_encrypt_u16(
        &mut self,
        input: &CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u32> {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut keystream = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let mut keystream_u16 = self.fill_my_rng_into_u16(&mut keystream, idx, streams);
        self.single_xor_assign_u16(&mut keystream_u16, input, idx, data_len, streams);
        keystream
    }

    // Encrypt using chacha in their_rng
    fn chacha2_encrypt_u16(
        &mut self,
        input: &CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u32> {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut keystream = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let mut keystream_u16 = self.fill_their_rng_into_u16(&mut keystream, idx, streams);
        self.single_xor_assign_u16(&mut keystream_u16, input, idx, data_len, streams);
        keystream
    }

    // Decrypt using chacha in my_rng
    fn chacha1_decrypt_u16(
        &mut self,
        input: &mut CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut keystream = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let keystream_u16 = self.fill_my_rng_into_u16(&mut keystream, idx, streams);
        self.single_xor_assign_u16(input, &keystream_u16, idx, data_len, streams);
    }

    // Decrypt using chacha in their_rng
    fn chacha2_decrypt_u16(
        &mut self,
        input: &mut CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut keystream = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let keystream_u16 = self.fill_their_rng_into_u16(&mut keystream, idx, streams);
        self.single_xor_assign_u16(input, &keystream_u16, idx, data_len, streams);
    }

    // Encrypt using chacha in my_rng
    fn chacha1_encrypt_u64(
        &mut self,
        input: &ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u64> {
        let data_len = input.len();
        let keystream_size = (data_len + 7) / 8; // Multiple of 16 u32
        let mut keystream = unsafe { self.devs[idx].alloc::<u64>(keystream_size * 8).unwrap() };
        self.fill_my_rand_u64(&mut keystream, idx, streams);
        self.single_xor_assign_u64(
            &mut keystream.slice(..data_len),
            &input.a,
            idx,
            data_len,
            streams,
        );
        keystream
    }

    // Decrypt using chacha in their_rng
    fn chacha2_decrypt_u64(
        &mut self,
        inout: &mut ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = inout.len();
        let keystream_size = (data_len + 7) / 8; // Multiple of 16 u32
        let mut keystream = unsafe { self.devs[idx].alloc::<u64>(keystream_size * 8).unwrap() };
        self.fill_their_rand_u64(&mut keystream, idx, streams);
        self.single_xor_assign_u64(
            &mut inout.b,
            &keystream.slice(..data_len),
            idx,
            data_len,
            streams,
        );
    }

    // Encrypt using chacha in my_rng
    fn chacha1_encrypt_u32(
        &mut self,
        input: &ChunkShareView<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u32> {
        let data_len = input.len();
        let keystream_size = (data_len + 15) / 16; // Multiple of 16 u32
        let mut keystream = unsafe { self.devs[idx].alloc::<u32>(keystream_size * 16).unwrap() };
        self.rngs[idx].fill_my_rng_into(&mut keystream.slice_mut(..), &streams[idx]);
        self.single_xor_assign_u32(
            &mut keystream.slice(..data_len),
            &input.a,
            idx,
            data_len,
            streams,
        );
        keystream
    }

    // Decrypt using chacha in their_rng
    fn chacha2_decrypt_u32(
        &mut self,
        inout: &mut ChunkShareView<u32>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = inout.len();
        let keystream_size = (data_len + 15) / 16; // Multiple of 16 u32
        let mut keystream = unsafe { self.devs[idx].alloc::<u32>(keystream_size * 16).unwrap() };
        self.rngs[idx].fill_their_rng_into(&mut keystream.slice_mut(..), &streams[idx]);
        self.single_xor_assign_u32(
            &mut inout.b,
            &keystream.slice(..data_len),
            idx,
            data_len,
            streams,
        );
    }

    fn packed_send_receive_view(
        &mut self,
        res: &mut [ChunkShareView<u64>],
        bits: usize,
        streams: &[CudaStream],
    ) {
        self.send_receive_view_with_offset(res, 0..bits * self.chunk_size, streams)
    }

    fn send_receive_view_with_offset_single_gpu(
        &mut self,
        res: &mut ChunkShareView<u64>,
        range: Range<usize>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let send_bufs =
            self.chacha1_encrypt_u64(&res.get_range(range.start, range.end), idx, streams);

        result::group_start().unwrap();
        self.comms[idx]
            .send_view(
                &send_bufs.slice(0..range.len()),
                self.next_id,
                &streams[idx],
            )
            .unwrap();
        let mut rcv = res.b.slice(range.to_owned());
        self.comms[idx]
            .receive_view(&mut rcv, self.prev_id, &streams[idx])
            .unwrap();
        result::group_end().unwrap();
        self.chacha2_decrypt_u64(&mut res.get_range(range.start, range.end), idx, streams);
    }

    fn send_receive_view_with_offset(
        &mut self,
        res: &mut [ChunkShareView<u64>],
        range: Range<usize>,
        streams: &[CudaStream],
    ) {
        assert_eq!(res.len(), self.n_devices);

        let send_bufs = res
            .iter()
            .enumerate()
            .map(|(idx, res)| {
                self.chacha1_encrypt_u64(&res.get_range(range.start, range.end), idx, streams)
            })
            .collect_vec();

        result::group_start().unwrap();
        for (idx, res) in send_bufs.iter().enumerate() {
            self.comms[idx]
                .send_view(&res.slice(0..range.len()), self.next_id, &streams[idx])
                .unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            let mut rcv = res.b.slice(range.to_owned());
            self.comms[idx]
                .receive_view(&mut rcv, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
        for (idx, res) in res.iter_mut().enumerate() {
            self.chacha2_decrypt_u64(&mut res.get_range(range.start, range.end), idx, streams);
        }
    }

    fn send_receive_view(&mut self, res: &mut [ChunkShareView<u64>], streams: &[CudaStream]) {
        assert_eq!(res.len(), self.n_devices);

        let send_bufs = res
            .iter()
            .enumerate()
            .map(|(idx, res)| self.chacha1_encrypt_u64(res, idx, streams))
            .collect_vec();

        result::group_start().unwrap();
        for (idx, r) in send_bufs.iter().enumerate() {
            self.comms[idx]
                .send_view(&r.slice(0..res[idx].len()), self.next_id, &streams[idx])
                .unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            self.comms[idx]
                .receive_view(&mut res.b, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
        for (idx, res) in res.iter_mut().enumerate() {
            self.chacha2_decrypt_u64(res, idx, streams);
        }
    }

    fn send_receive_view_u32(&mut self, res: &mut [ChunkShareView<u32>], streams: &[CudaStream]) {
        assert_eq!(res.len(), self.n_devices);

        let send_bufs = res
            .iter()
            .enumerate()
            .map(|(idx, res)| self.chacha1_encrypt_u32(res, idx, streams))
            .collect_vec();

        result::group_start().unwrap();
        for (idx, r) in send_bufs.iter().enumerate() {
            self.comms[idx]
                .send_view(&r.slice(0..res[idx].len()), self.next_id, &streams[idx])
                .unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            self.comms[idx]
                .receive_view(&mut res.b, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
        for (idx, res) in res.iter_mut().enumerate() {
            self.chacha2_decrypt_u32(res, idx, streams);
        }
    }

    fn send_receive_view_single_gpu(
        &mut self,
        res: &mut ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let send_bufs = self.chacha1_encrypt_u64(res, idx, streams);

        result::group_start().unwrap();
        self.comms[idx]
            .send_view(&send_bufs.slice(0..res.len()), self.next_id, &streams[idx])
            .unwrap();
        self.comms[idx]
            .receive_view(&mut res.b, self.prev_id, &streams[idx])
            .unwrap();
        result::group_end().unwrap();
        self.chacha2_decrypt_u64(res, idx, streams);
    }

    fn single_xor_assign_u16(
        &self,
        x1: &mut CudaView<u16>,
        x2: &CudaView<u16>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .single_xor_assign_u16
                .clone()
                .launch_on_stream(&streams[idx], cfg, (&*x1, x2, size))
                .unwrap();
        }
    }

    fn single_xor_assign_u32(
        &self,
        x1: &mut CudaView<u32>,
        x2: &CudaView<u32>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .single_xor_assign_u32
                .clone()
                .launch_on_stream(&streams[idx], cfg, (&*x1, x2, size))
                .unwrap();
        }
    }

    fn single_xor_assign_u64(
        &self,
        x1: &mut CudaView<u64>,
        x2: &CudaView<u64>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .single_xor_assign_u64
                .clone()
                .launch_on_stream(&streams[idx], cfg, (&*x1, x2, size))
                .unwrap();
        }
    }

    fn xor_assign_u64(
        &self,
        x1: &mut ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .xor_assign
                .clone()
                .launch_on_stream(&streams[idx], cfg, (&x1.a, &x1.b, &x2.a, &x2.b, size))
                .unwrap();
        }
    }

    fn xor_assign_many(
        &self,
        x1: &mut ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        self.xor_assign_u64(x1, x2, idx, self.chunk_size, streams);
    }

    fn packed_xor_assign_many(
        &self,
        x1: &mut ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        bits: usize,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .xor_assign
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg.to_owned(),
                    (&x1.a, &x1.b, &x2.a, &x2.b, self.chunk_size * bits),
                )
                .unwrap();
        }
    }
    fn packed_xor_many(
        &self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        bits: usize,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let cfg = launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[idx],
        );

        unsafe {
            self.kernels[idx]
                .xor
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg.to_owned(),
                    (
                        &res.a,
                        &res.b,
                        &x1.a,
                        &x1.b,
                        &x2.a,
                        &x2.b,
                        self.chunk_size * bits,
                    ),
                )
                .unwrap();
        }
    }

    pub fn allocate_buffer<T>(&self, size: usize) -> Vec<ChunkShare<T>>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        Buffers::allocate_buffer(size, &self.devs)
    }

    #[allow(clippy::too_many_arguments)]
    fn split_for_arithmetic_xor(
        &mut self,
        inp: &[ChunkShareView<u64>],
        x1: &mut [ChunkShareView<u32>],
        x2: &mut [ChunkShareView<u32>],
        x3: &mut [ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        debug_assert_eq!(self.n_devices, inp.len());
        debug_assert_eq!(self.n_devices, x1.len());
        debug_assert_eq!(self.n_devices, x2.len());
        debug_assert_eq!(self.n_devices, x3.len());

        for (idx, (inp, x1, x2, x3)) in izip!(inp, x1, x2, x3).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .split_arithmetic_xor
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &x1.a,
                            &x1.b,
                            &x2.a,
                            &x2.b,
                            &x3.a,
                            &x3.b,
                            &inp.a,
                            &inp.b,
                            self.chunk_size,
                            self.peer_id as u32,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    fn bit_inject_arithmetic_xor(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        debug_assert_eq!(self.n_devices, inp.len());
        debug_assert_eq!(self.n_devices, outp.len());

        let x1_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_split2);
        let x2_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_split3);

        // Reuse the existing buffers to have less memory
        // the transmute_mut is safe because we know that one u64 is 2 u32s, and the
        // buffer is aligned properly for the transmute
        let mut x1 = Vec::with_capacity(x1_.len());
        for x in x1_.iter() {
            let a: CudaView<u32> = unsafe { x.a.transmute(64 * self.chunk_size).unwrap() };
            let b: CudaView<u32> = unsafe { x.b.transmute(64 * self.chunk_size).unwrap() };
            let view = ChunkShareView { a, b };
            x1.push(view);
        }
        let mut x2 = Vec::with_capacity(x2_.len());
        for x in x2_.iter() {
            let a: CudaView<u32> = unsafe { x.a.transmute(64 * self.chunk_size).unwrap() };
            let b: CudaView<u32> = unsafe { x.b.transmute(64 * self.chunk_size).unwrap() };
            let view = ChunkShareView { a, b };
            x2.push(view);
        }

        // Split to x1, x2, x3
        self.split_for_arithmetic_xor(inp, &mut x1, &mut x2, outp, streams);

        // First arithmetic xor: x3 ^= x1
        for (idx, (x3, x1)) in izip!(outp.iter_mut(), x1.iter()).enumerate() {
            self.arithmetic_xor_many_pre_assign(x3, x1, idx, streams);
        }
        // Send/Receive
        self.send_receive_view_u32(outp, streams);

        // Second arithmetic xor: x3 ^= x2
        for (idx, (x3, x2)) in izip!(outp.iter_mut(), x2.iter()).enumerate() {
            self.arithmetic_xor_many_pre_assign(x3, x2, idx, streams);
        }
        // Send/Receive
        self.send_receive_view_u32(outp, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares_split2, x1_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_split3, x2_);
    }

    fn bit_inject_ot_sender(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let m0_ = Buffers::take_single_buffer(&mut self.buffers.ot_m0);
        let m1_ = Buffers::take_single_buffer(&mut self.buffers.ot_m1);
        let m0 = Buffers::get_single_buffer_chunk(&m0_, self.chunk_size * 128);
        let m1 = Buffers::get_single_buffer_chunk(&m1_, self.chunk_size * 128);

        for (idx, (inp, res, m0, m1)) in izip!(inp, outp, &m0, &m1).enumerate() {
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_ca_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_ca = self.fill_my_rng_into_u16(&mut rand_ca_alloc, idx, streams);
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_cb_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_cb = self.fill_their_rng_into_u16(&mut rand_cb_alloc, idx, streams);
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_wa1_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_wa1 = self.fill_my_rng_into_u16(&mut rand_wa1_alloc, idx, streams);
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_wa2_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_wa2 = self.fill_my_rng_into_u16(&mut rand_wa2_alloc, idx, streams);

            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .ot_sender
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &res.a,
                            &res.b,
                            &inp.a,
                            &inp.b,
                            m0,
                            m1,
                            &rand_ca,
                            &rand_cb,
                            &rand_wa1,
                            &rand_wa2,
                            2 * self.chunk_size,
                        ),
                    )
                    .unwrap();
            }
        }

        // OTP encrypt
        let m0 = m0
            .into_iter()
            .enumerate()
            .map(|(idx, m0)| self.chacha2_encrypt_u16(&m0, idx, streams))
            .collect_vec();
        let m1 = m1
            .into_iter()
            .enumerate()
            .map(|(idx, m1)| self.chacha2_encrypt_u16(&m1, idx, streams))
            .collect_vec();

        result::group_start().unwrap();
        for (idx, (m0, m1)) in izip!(&m0, &m1).enumerate() {
            self.comms[idx]
                .send(m0, self.prev_id, &streams[idx])
                .unwrap();
            self.comms[idx]
                .send(m1, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();

        Buffers::return_single_buffer(&mut self.buffers.ot_m0, m0_);
        Buffers::return_single_buffer(&mut self.buffers.ot_m1, m1_);
    }

    fn bit_inject_ot_receiver(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let m0_ = Buffers::take_single_buffer(&mut self.buffers.ot_m0);
        let m1_ = Buffers::take_single_buffer(&mut self.buffers.ot_m1);
        let wc_ = Buffers::take_single_buffer(&mut self.buffers.ot_wc);
        let mut m0 = Buffers::get_single_buffer_chunk(&m0_, self.chunk_size * 128);
        let mut m1 = Buffers::get_single_buffer_chunk(&m1_, self.chunk_size * 128);
        let mut wc = Buffers::get_single_buffer_chunk(&wc_, self.chunk_size * 128);

        let mut send = Vec::with_capacity(inp.len());

        result::group_start().unwrap();
        for (idx, (m0, m1, wc)) in izip!(&mut m0, &mut m1, &mut wc).enumerate() {
            self.comms[idx]
                .receive_view_u16(m0, self.next_id, &streams[idx])
                .unwrap();
            self.comms[idx]
                .receive_view_u16(wc, self.prev_id, &streams[idx])
                .unwrap();
            self.comms[idx]
                .receive_view_u16(m1, self.next_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();

        for (idx, (inp, res, m0, m1, wc)) in izip!(
            inp,
            outp.iter_mut(),
            m0.iter_mut(),
            m1.iter_mut(),
            wc.iter_mut()
        )
        .enumerate()
        {
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_ca_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_ca = self.fill_my_rng_into_u16(&mut rand_ca_alloc, idx, streams);

            // ChaCha decrypt
            {
                self.chacha1_decrypt_u16(m0, idx, streams);
                self.chacha2_decrypt_u16(wc, idx, streams);
                self.chacha1_decrypt_u16(m1, idx, streams);
            }

            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .ot_receiver
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &res.a,
                            &res.b,
                            &inp.b,
                            &*m0,
                            &*m1,
                            &rand_ca,
                            &*wc,
                            2 * self.chunk_size,
                        ),
                    )
                    .unwrap();
            }
            // OTP encrypt
            send.push(self.chacha2_encrypt_u16(&res.b, idx, streams));
        }

        // Reshare to Helper
        result::group_start().unwrap();
        for (idx, send) in send.iter().enumerate() {
            self.comms[idx]
                .send(send, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();

        Buffers::return_single_buffer(&mut self.buffers.ot_m0, m0_);
        Buffers::return_single_buffer(&mut self.buffers.ot_m1, m1_);
        Buffers::return_single_buffer(&mut self.buffers.ot_wc, wc_);
    }

    fn bit_inject_ot_helper(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let wc_ = Buffers::take_single_buffer(&mut self.buffers.ot_wc);
        let wc = Buffers::get_single_buffer_chunk(&wc_, self.chunk_size * 128);

        let mut send = Vec::with_capacity(inp.len());

        for (idx, (inp, res, wc)) in izip!(inp, outp.iter_mut(), &wc).enumerate() {
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_cb_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_cb = self.fill_their_rng_into_u16(&mut rand_cb_alloc, idx, streams);
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_wb1_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_wb1 = self.fill_their_rng_into_u16(&mut rand_wb1_alloc, idx, streams);
            // SAFETY: Only unsafe because memory is not initialized. But, we fill
            // afterwards.
            let mut rand_wb2_alloc =
                unsafe { self.devs[idx].alloc::<u32>(self.chunk_size * 64).unwrap() };
            let rand_wb2 = self.fill_their_rng_into_u16(&mut rand_wb2_alloc, idx, streams);

            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .ot_helper
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &res.b,
                            &inp.a,
                            &rand_cb,
                            &rand_wb1,
                            &rand_wb2,
                            wc,
                            2 * self.chunk_size,
                        ),
                    )
                    .unwrap();
            }

            // OTP encrypt
            send.push(self.chacha1_encrypt_u16(wc, idx, streams));
        }

        result::group_start().unwrap();
        for (idx, send) in send.iter().enumerate() {
            self.comms[idx]
                .send(send, self.next_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
        result::group_start().unwrap();
        for (idx, res) in outp.iter_mut().enumerate() {
            self.comms[idx]
                .receive_view_u16(&mut res.a, self.next_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
        // OTP decrypt
        {
            for (idx, res) in outp.iter_mut().enumerate() {
                self.chacha1_decrypt_u16(&mut res.a, idx, streams);
            }
        }

        Buffers::return_single_buffer(&mut self.buffers.ot_wc, wc_);
    }

    pub fn bit_inject_ot(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        match self.peer_id {
            0 => self.bit_inject_ot_helper(inp, outp, streams),
            1 => self.bit_inject_ot_receiver(inp, outp, streams),
            2 => self.bit_inject_ot_sender(inp, outp, streams),
            _ => unreachable!(),
        }
    }

    fn pre_lift_correction_u16_u32_signed(
        &mut self,
        inout: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, inout.len());

        for (idx, inp) in izip!(inout).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                inp.len() as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .pre_lift_u16_u32_signed
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (&inp.a, &inp.b, self.peer_id as u32, inp.len()),
                    )
                    .unwrap();
            }
        }
    }

    fn transpose_pack_u16_with_len(
        &mut self,
        inp: &[ChunkShareView<u16>],
        outp: &mut [ChunkShareView<u64>],
        bitlen: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, inp.len());
        assert_eq!(self.n_devices, outp.len());

        for (idx, (inp, outp)) in izip!(inp, outp).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .transpose_16x64
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &outp.a,
                            &outp.b,
                            &inp.a,
                            &inp.b,
                            self.chunk_size * 64,
                            bitlen,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    fn transpose_pack_u32_with_len(
        &mut self,
        inp: &[ChunkShareView<u32>],
        outp: &mut [ChunkShareView<u64>],
        bitlen: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, inp.len());
        assert_eq!(self.n_devices, outp.len());

        for (idx, (inp, outp)) in izip!(inp, outp).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );
            unsafe {
                self.kernels[idx]
                    .transpose_32x64
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &outp.a,
                            &outp.b,
                            &inp.a,
                            &inp.b,
                            self.chunk_size * 64,
                            bitlen,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    fn split(
        &mut self,
        inout1: &mut [ChunkShareView<u64>],
        out2: &mut [ChunkShareView<u64>],
        out3: &mut [ChunkShareView<u64>],
        bits: usize,
        streams: &[CudaStream],
    ) {
        // K = 16 is hardcoded in the kernel
        for (idx, (x1, x2, x3)) in izip!(inout1, out2, out3).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .split
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &x1.a,
                            &x1.b,
                            &x2.a,
                            &x2.b,
                            &x3.a,
                            &x3.b,
                            self.chunk_size * bits,
                            self.peer_id as u32,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    fn lift_split(
        &mut self,
        inp: &[ChunkShareView<u16>],
        lifted: &mut [ChunkShareView<u32>],
        inout1: &mut [ChunkShareView<u64>],
        out2: &mut [ChunkShareView<u64>],
        out3: &mut [ChunkShareView<u64>],
        streams: &[CudaStream],
    ) {
        // K = 16 is hardcoded in the kernel
        for (idx, (inp, lifted, x1, x2, x3)) in izip!(inp, lifted, inout1, out2, out3).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .lift_split
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &inp.a,
                            &inp.b,
                            &lifted.a,
                            &lifted.b,
                            &x1.a,
                            &x1.b,
                            &x2.a,
                            &x2.b,
                            &x3.a,
                            &x3.b,
                            self.chunk_size,
                            self.peer_id as u32,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn lift_mul_sub(
        &mut self,
        mask_lifted: &mut [ChunkShareView<u32>],
        mask_correction: &[ChunkShareView<u16>],
        code: &[ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, mask_lifted.len());
        assert_eq!(self.n_devices, mask_correction.len());
        assert_eq!(self.n_devices, code.len());

        for (idx, (m, mc, c)) in izip!(mask_lifted, mask_correction, code).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .lift_mul_sub
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &m.a,
                            &m.b,
                            &mc.a,
                            &mc.b,
                            &c.a,
                            &c.b,
                            self.peer_id as u32,
                            self.chunk_size * 64,
                        ),
                    )
                    .unwrap();
            }
        }
    }
    fn finalize_lift_u16_u32_signed(
        &mut self,
        lifted: &mut [ChunkShareView<u32>],
        corrections: &[ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        assert!(self.n_devices >= lifted.len());
        assert_eq!(corrections.len(), lifted.len());

        for (idx, (lift, corr)) in izip!(lifted, corrections).enumerate() {
            assert_eq!(2 * lift.len(), corr.len(), "Two correction values per lift");
            let len = lift.len();
            let cfg = launch_config_from_elements_and_threads(
                len as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .finalize_lift_u16_u32_signed
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (&lift.a, &lift.b, &corr.a, &corr.b, self.peer_id as u32, len),
                    )
                    .unwrap();
            }
        }
    }

    pub fn finalize_lifts(
        &mut self,
        mask_lifted: &mut [ChunkShareView<u32>],
        code_lifted: &mut [ChunkShareView<u32>],
        mask_correction: &[ChunkShareView<u16>],
        code: &[ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, mask_lifted.len());
        assert_eq!(self.n_devices, code_lifted.len());
        assert_eq!(self.n_devices, mask_correction.len());
        assert_eq!(self.n_devices, code.len());

        for (idx, (m, cl, mc, c)) in
            izip!(mask_lifted, code_lifted, mask_correction, code).enumerate()
        {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .finalize_lift
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &m.a,
                            &m.b,
                            &cl.a,
                            &cl.b,
                            &mc.a,
                            &mc.b,
                            &c.a,
                            &c.b,
                            self.chunk_size * 64,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn lifted_sub(
        &mut self,
        output: &mut [ChunkShareView<u32>],
        mask_lifted: &[ChunkShareView<u32>],
        code_lifted: &[ChunkShareView<u32>],
        a: u32,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, mask_lifted.len());
        assert_eq!(self.n_devices, code_lifted.len());
        assert_eq!(self.n_devices, output.len());

        for (idx, (m, c, o)) in izip!(mask_lifted, code_lifted, output).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .lifted_sub
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &m.a,
                            &m.b,
                            &c.a,
                            &c.b,
                            &o.a,
                            &o.b,
                            a,
                            self.peer_id as u32,
                            m.len(),
                        ),
                    )
                    .unwrap();
            }
        }
    }

    /// Performs the subtraction of mask_lifted * a - code_lifted * b. In contrast to lifted_sub, this also multiplies the B factor.
    pub fn pre_lifted_sub_ab(
        &mut self,
        output: &mut [ChunkShareView<u32>],
        mask_lifted: &[ChunkShareView<u32>],
        code_lifted: &[ChunkShareView<u32>],
        a: u32,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, mask_lifted.len());
        assert_eq!(self.n_devices, code_lifted.len());
        assert_eq!(self.n_devices, output.len());

        for (idx, (m, c, o)) in izip!(mask_lifted, code_lifted, output).enumerate() {
            assert!(m.len() == c.len());
            let cfg = launch_config_from_elements_and_threads(
                m.len() as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .prelifted_sub_ab
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &m.a,
                            &m.b,
                            &c.a,
                            &c.b,
                            &o.a,
                            &o.b,
                            a,
                            self.peer_id as u32,
                            m.len(),
                        ),
                    )
                    .unwrap();
            }
        }
    }

    // input should be of size: n_devices * input_size
    // outputs the uncorrected lifted shares and the injected correction values
    pub fn lift_mpc(
        &mut self,
        shares: &[ChunkShareView<u16>],
        xa: &mut [ChunkShareView<u32>],
        injected: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        const K: usize = SHARE_RING_BITSIZE;
        let mut x1 = Vec::with_capacity(self.n_devices);
        let mut x2 = Vec::with_capacity(self.n_devices);
        let mut x3 = Vec::with_capacity(self.n_devices);
        let mut c = Vec::with_capacity(self.n_devices);
        // No subbuffer taken here, since we extract it manually
        let buffer1 = Buffers::take_buffer(&mut self.buffers.lifted_shares_split1_result);
        let buffer2 = Buffers::take_buffer(&mut self.buffers.lifted_shares_split2);
        for (b1, b2) in izip!(&buffer1, &buffer2) {
            let a = b1.get_offset(0, K * self.chunk_size);
            let b = b1.get_offset(1, K * self.chunk_size);
            let c_ = b2.get_offset(0, K * self.chunk_size);
            let d = b2.get_range(K * self.chunk_size, (K + 2) * self.chunk_size);
            x1.push(a);
            x2.push(b);
            x3.push(c_);
            c.push(d);
        }

        self.transpose_pack_u16_with_len(shares, &mut x1, K, streams);
        self.lift_split(shares, xa, &mut x1, &mut x2, &mut x3, streams);
        self.binary_add_3_get_two_carries(&mut c, &mut x1, &mut x2, &mut x3, streams);
        self.bit_inject_ot(&c, injected, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares_split1_result, buffer1);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_split2, buffer2);
    }

    /// Lifts u16 shares to u32 shares, using the same method as lift_mpc, but also already corrects the output and injects the correction values, and does not multiply with B.
    /// This lifts the shares in signed representation, so -1000 in u16 = 2^16-1000, becomes -1000 in u32 = 2^32-1000.
    pub fn lift_u16_to_u32_signed(
        &mut self,
        shares: &mut [ChunkShareView<u16>],
        out: &mut [ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        let buf: Vec<ChunkShare<u32>> = Buffers::allocate_buffer(128 * self.chunk_size, &self.devs);
        self.synchronize_streams(streams);
        const K: usize = SHARE_RING_BITSIZE;
        let mut x1 = Vec::with_capacity(self.n_devices);
        let mut x2 = Vec::with_capacity(self.n_devices);
        let mut x3 = Vec::with_capacity(self.n_devices);
        let mut c = Vec::with_capacity(self.n_devices);
        // No subbuffer taken here, since we extract it manually
        let buffer1 = Buffers::take_buffer(&mut self.buffers.lifted_shares_split2);
        let buffer2 = Buffers::take_buffer(&mut self.buffers.lifted_shares_split1_result);
        for (b1, b2) in izip!(&buffer1, &buffer2) {
            let a = b1.get_offset(0, K * self.chunk_size);
            let b = b1.get_offset(1, K * self.chunk_size);
            let c_ = b2.get_offset(0, K * self.chunk_size);
            let d = b2.get_range(K * self.chunk_size, (K + 2) * self.chunk_size);
            x1.push(a);
            x2.push(b);
            x3.push(c_);
            c.push(d);
        }

        self.pre_lift_correction_u16_u32_signed(shares, streams);
        self.transpose_pack_u16_with_len(shares, &mut x1, K, streams);
        self.lift_split(shares, out, &mut x1, &mut x2, &mut x3, streams);
        self.binary_add_3_get_two_carries(&mut c, &mut x1, &mut x2, &mut x3, streams);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_split2, buffer1);

        let corr1 = c
            .iter()
            .map(|x| x.get_offset(0, self.chunk_size))
            .collect_vec();
        let corr2 = c
            .iter()
            .map(|x| x.get_offset(1, self.chunk_size))
            .collect_vec();
        let mut inj1 = buf
            .iter()
            .map(|x| x.get_offset(0, 64 * self.chunk_size))
            .collect_vec();
        let mut inj2 = buf
            .iter()
            .map(|x| x.get_offset(1, 64 * self.chunk_size))
            .collect_vec();
        self.bit_inject_arithmetic_xor(&corr1, &mut inj1, streams);
        self.bit_inject_arithmetic_xor(&corr2, &mut inj2, streams);

        let buf_view = buf.iter().map(|x| x.as_view()).collect_vec();
        self.finalize_lift_u16_u32_signed(out, &buf_view, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares_split1_result, buffer2);
    }

    // K is 16 in our case
    fn binary_add_3_get_two_carries(
        &mut self,
        c: &mut [ChunkShareView<u64>],
        x1: &mut [ChunkShareView<u64>],
        x2: &mut [ChunkShareView<u64>],
        x3: &mut [ChunkShareView<u64>],
        streams: &[CudaStream],
    ) {
        const K: usize = SHARE_RING_BITSIZE;
        assert_eq!(self.n_devices, c.len());
        assert_eq!(self.n_devices, x1.len());
        assert_eq!(self.n_devices, x2.len());
        assert_eq!(self.n_devices, x3.len());

        // Reuse buffer
        let mut s = Vec::with_capacity(self.n_devices);
        let mut carry = Vec::with_capacity(self.n_devices);
        // No subbuffer taken here, since we extract it manually
        let buffer1 = Buffers::take_buffer(&mut self.buffers.lifted_shares_split3);

        for (idx, (x1, x2, x3, b)) in izip!(x1, x2, x3.iter_mut(), &buffer1).enumerate() {
            let mut s_ = b.get_offset(0, (K - 1) * self.chunk_size);
            let mut c = b.get_offset(1, K * self.chunk_size);

            // First full adder to get 2 * c1 and s1
            let x2x3 = x2;
            self.packed_xor_assign_many(x2x3, x3, K, idx, streams);
            // Don't need first bit for s
            self.packed_xor_many(
                &x1.get_range(self.chunk_size, K * self.chunk_size),
                &x2x3.get_range(self.chunk_size, K * self.chunk_size),
                &mut s_,
                K - 1,
                idx,
                streams,
            );
            // 2 * c1
            let x1x3 = x1;
            self.packed_xor_assign_many(x1x3, x3, K, idx, streams);
            self.packed_and_many_pre(x1x3, x2x3, &mut c, K, idx, streams);
            s.push(s_);
            carry.push(c);
        }
        // Send/Receive full adders
        self.packed_send_receive_view(&mut carry, K, streams);
        // Postprocess xor
        for (idx, (c, x3)) in izip!(&mut carry, x3).enumerate() {
            self.packed_xor_assign_many(c, x3, K, idx, streams);
        }

        // Add 2c + s via a ripple carry adder
        // LSB of c is 0
        // First round: half adder can be skipped due to LSB of c being 0
        let mut a = s;
        let mut b = carry;

        // The first part of the result is used as the carry
        let mut carry = Vec::with_capacity(self.n_devices);

        // First full adder (carry is 0)
        for (idx, (a, b, c)) in izip!(&a, &b, c.iter_mut()).enumerate() {
            let mut c = c.get_offset(0, self.chunk_size);
            let a = a.get_offset(0, self.chunk_size);
            let b = b.get_offset(0, self.chunk_size);
            self.and_many_pre(&a, &b, &mut c, idx, streams);
            carry.push(c);
        }
        // Send/Receive
        self.send_receive_view(&mut carry, streams);

        for k in 1..K - 1 {
            for (idx, (a, b, c)) in izip!(&mut a, &mut b, carry.iter_mut()).enumerate() {
                // Unused space used for temparary storage
                let mut tmp_c = a.get_offset(0, self.chunk_size);

                let mut a = a.get_offset(k, self.chunk_size);
                let mut b = b.get_offset(k, self.chunk_size);

                self.xor_assign_many(&mut a, c, idx, streams);
                self.xor_assign_many(&mut b, c, idx, streams);
                self.and_many_pre(&a, &b, &mut tmp_c, idx, streams);
            }
            // Send/Receive
            self.send_receive_view_with_offset(&mut a, 0..self.chunk_size, streams);
            // Postprocess xor
            for (idx, (c, a)) in izip!(carry.iter_mut(), &a).enumerate() {
                // Unused space used for temparary storage
                let tmp_c = a.get_offset(0, self.chunk_size);
                self.xor_assign_many(c, &tmp_c, idx, streams);
            }
        }

        // Finally, last bit of a is 0
        for (idx, (b, c)) in izip!(&mut b, c.iter_mut()).enumerate() {
            let mut c1 = c.get_offset(0, self.chunk_size);
            let mut c2 = c.get_offset(1, self.chunk_size);
            let b = b.get_offset(K - 1, self.chunk_size);
            self.and_many_pre(&b, &c1, &mut c2, idx, streams);
            self.xor_assign_many(&mut c1, &b, idx, streams);
        }
        // Send/Receive
        self.send_receive_view_with_offset(c, self.chunk_size..2 * self.chunk_size, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares_split3, buffer1);
    }

    pub fn extract_msb(&mut self, x: &mut [ChunkShareView<u32>], streams: &[CudaStream]) {
        let x1_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_split1_result);
        let x2_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_split2);
        let x3_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_split3);
        let mut x1 = Buffers::get_buffer_chunk(&x1_, 32 * self.chunk_size);
        let mut x2 = Buffers::get_buffer_chunk(&x2_, 32 * self.chunk_size);
        let mut x3 = Buffers::get_buffer_chunk(&x3_, 32 * self.chunk_size);

        self.transpose_pack_u32_with_len(x, &mut x1, Self::BITS, streams);
        self.split(&mut x1, &mut x2, &mut x3, Self::BITS, streams);
        self.binary_add_3_get_msb(&mut x1, &mut x2, &mut x3, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares_split1_result, x1_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_split2, x2_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_split3, x3_);
    }

    // K is Self::BITS = SHARE_RING_BITSIZE + B_BITS in our case
    // The result is located in the first bit of x1
    fn binary_add_3_get_msb(
        &mut self,
        x1: &mut [ChunkShareView<u64>],
        x2: &mut [ChunkShareView<u64>],
        x3: &mut [ChunkShareView<u64>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, x1.len());
        assert_eq!(self.n_devices, x2.len());
        assert_eq!(self.n_devices, x3.len());

        let s_ = Buffers::take_buffer(&mut self.buffers.binary_adder_s);
        let carry_ = Buffers::take_buffer(&mut self.buffers.binary_adder_c);
        let mut s = Buffers::get_buffer_chunk(&s_, self.chunk_size * (Self::BITS - 1));
        let mut carry = Buffers::get_buffer_chunk(&carry_, self.chunk_size * (Self::BITS - 1));

        for (idx, (x1, x2, x3, s, c)) in
            izip!(x1.iter_mut(), x2, x3.iter_mut(), &mut s, &mut carry).enumerate()
        {
            // First full adder to get 2 * c1 and s1
            let x2x3 = x2;
            self.packed_xor_assign_many(x2x3, x3, Self::BITS, idx, streams);
            // Don't need first bit for s
            self.packed_xor_many(
                &x1.get_range(self.chunk_size, Self::BITS * self.chunk_size),
                &x2x3.get_range(self.chunk_size, Self::BITS * self.chunk_size),
                s,
                Self::BITS - 1,
                idx,
                streams,
            );
            // 2 * c1
            let x1x3 = x1;
            self.packed_xor_assign_many(x1x3, x3, Self::BITS - 1, idx, streams);
            self.packed_and_many_pre(x1x3, x2x3, c, Self::BITS - 1, idx, streams);
        }
        // Send/Receive full adders
        self.packed_send_receive_view(&mut carry, Self::BITS - 1, streams);
        // Postprocess xor
        for (idx, (c, x3)) in izip!(&mut carry, x3).enumerate() {
            self.packed_xor_assign_many(c, x3, Self::BITS - 1, idx, streams);
        }

        // Add 2c + s via a ripple carry adder
        // LSB of c is 0
        // First round: half adder can be skipped due to LSB of c being 0
        let mut a = s;
        let mut b = carry;

        // The first part of x1 is used as the carry and the result
        let mut carry = Vec::with_capacity(self.n_devices);

        // First full adder (carry is 0)
        for (idx, (a, b, c)) in izip!(&a, &b, x1.iter_mut()).enumerate() {
            let mut c = c.get_offset(0, self.chunk_size);
            let a = a.get_offset(0, self.chunk_size);
            let b = b.get_offset(0, self.chunk_size);
            self.and_many_pre(&a, &b, &mut c, idx, streams);
            carry.push(c);
        }
        // Send/Receive
        self.send_receive_view(&mut carry, streams);

        for k in 1..Self::BITS - 2 {
            for (idx, (a, b, c)) in izip!(&mut a, &mut b, carry.iter_mut()).enumerate() {
                // Unused space used for temparary storage
                let mut tmp_c = a.get_offset(0, self.chunk_size);

                let mut a = a.get_offset(k, self.chunk_size);
                let mut b = b.get_offset(k, self.chunk_size);

                self.xor_assign_many(&mut a, c, idx, streams);
                self.xor_assign_many(&mut b, c, idx, streams);
                self.and_many_pre(&a, &b, &mut tmp_c, idx, streams);
            }
            // Send/Receive
            self.send_receive_view_with_offset(&mut a, 0..self.chunk_size, streams);
            // Postprocess xor
            for (idx, (c, a)) in izip!(carry.iter_mut(), &a).enumerate() {
                // Unused space used for temparary storage
                let tmp_c = a.get_offset(0, self.chunk_size);
                self.xor_assign_many(c, &tmp_c, idx, streams);
            }
        }

        // Las round: just caclculate the output
        for (idx, (a, b, c)) in izip!(&a, &b, &mut carry).enumerate() {
            let a = a.get_offset(Self::BITS - 2, self.chunk_size);
            let b = b.get_offset(Self::BITS - 2, self.chunk_size);
            self.xor_assign_many(c, &a, idx, streams);
            self.xor_assign_many(c, &b, idx, streams);
        }

        Buffers::return_buffer(&mut self.buffers.binary_adder_s, s_);
        Buffers::return_buffer(&mut self.buffers.binary_adder_c, carry_);

        // Result is in the first bit of x1
    }

    // Input has size ChunkSize
    // Result is in lowest u64 of the input
    fn or_tree_on_gpus(&mut self, bits: &mut [ChunkShareView<u64>], streams: &[CudaStream]) {
        assert_eq!(self.n_devices, bits.len());
        assert!(self.chunk_size <= bits[0].len());

        let mut num = self.chunk_size;
        while num > 1 {
            let mod_ = num & 1;
            num >>= 1;

            for (idx, bit) in bits.iter().enumerate() {
                let mut a = bit.get_offset(0, num);
                let b = bit.get_offset(1, num);
                self.or_many_pre_assign(&mut a, &b, idx, streams);
                if mod_ != 0 {
                    let src = bit.get_offset(2 * num, 1);
                    let mut des = bit.get_offset(num, 1);
                    self.assign_view(&mut des, &src, idx, streams);
                }
            }

            // Reshare
            self.send_receive_view_with_offset(bits, 0..num, streams);

            num += mod_;
        }
    }

    // Same as or_tree_on_gpus, but on one GPU only
    // Result is in lowest u64 of the input
    fn or_tree_on_gpu(
        &mut self,
        bits: &mut [ChunkShareView<u64>],
        size: usize,
        idx: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, bits.len());
        assert!(size <= bits[idx].len());

        let bit = &mut bits[idx];

        let mut num = size;
        while num > 1 {
            let mod_ = num & 1;
            num >>= 1;

            let mut a = bit.get_offset(0, num);
            let b = bit.get_offset(1, num);
            self.or_many_pre_assign(&mut a, &b, idx, streams);
            if mod_ != 0 {
                let src = bit.get_offset(2 * num, 1);
                let mut des = bit.get_offset(num, 1);
                self.assign_view(&mut des, &src, idx, streams);
            }

            // Reshare
            self.send_receive_view_with_offset_single_gpu(bit, 0..num, idx, streams);

            num += mod_;
        }
    }

    fn collect_graphic_result(&mut self, data: &mut [ChunkShareView<u64>], streams: &[CudaStream]) {
        assert!(self.n_devices <= self.chunk_size);
        let dev0 = &self.devs[0];
        let stream0 = &streams[0];
        let data0 = &data[0];

        // Get results onto CPU
        let mut a = Vec::with_capacity(self.n_devices - 1);
        let mut b = Vec::with_capacity(self.n_devices - 1);
        for (dev, stream, d) in izip!(self.get_devices(), streams, data.iter()).skip(1) {
            let src = d.get_range(0, 1);

            let mut a_ = dtoh_on_stream_sync(&src.a, &dev, stream).unwrap();
            let mut b_ = dtoh_on_stream_sync(&src.b, &dev, stream).unwrap();

            a.push(a_.pop().unwrap());
            b.push(b_.pop().unwrap());
        }

        // Put results onto first GPU
        let mut des = data0.get_range(1, self.n_devices);
        let a = htod_on_stream_sync(&a, dev0, stream0).unwrap();
        let b = htod_on_stream_sync(&b, dev0, stream0).unwrap();
        let c = ChunkShare::new(a, b);

        self.assign_view(&mut des, &c.as_view(), 0, streams);
    }

    fn collect_graphic_result_u32(
        &mut self,
        data: &mut [ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        assert!(self.n_devices <= self.chunk_size);
        let dev0 = &self.devs[0];
        let stream0 = &streams[0];
        let data0 = &data[0];

        // Get results onto CPU
        let mut a = Vec::with_capacity(self.n_devices - 1);
        let mut b = Vec::with_capacity(self.n_devices - 1);
        for (dev, stream, d) in izip!(self.get_devices(), streams, data.iter()).skip(1) {
            let src = d.get_range(0, 1);

            let mut a_ = dtoh_on_stream_sync(&src.a, &dev, stream).unwrap();
            let mut b_ = dtoh_on_stream_sync(&src.b, &dev, stream).unwrap();

            a.push(a_.pop().unwrap());
            b.push(b_.pop().unwrap());
        }

        // Put results onto first GPU
        let mut des = data0.get_range(1, self.n_devices);
        let a = htod_on_stream_sync(&a, dev0, stream0).unwrap();
        let b = htod_on_stream_sync(&b, dev0, stream0).unwrap();
        let c = ChunkShare::new(a, b);

        self.assign_view_u32(&mut des, &c.as_view(), 0, streams);
    }

    fn collapse_u64(&mut self, input: &mut ChunkShare<u64>, streams: &[CudaStream]) {
        let mut res = input.get_offset(0, 1);
        let helper = input.get_offset(1, 1);

        let cfg = launch_config_from_elements_and_threads(
            1,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[0],
        );

        // SAFETY: Only unsafe because memory is not initialized. But, we fill
        // afterwards.
        let mut rand = unsafe { self.devs[0].alloc::<u64>(16).unwrap() }; // minimum size is 16 for RNG, need only 10 though
        self.fill_rand_u64(&mut rand, 0, streams);

        let mut rand_offset = rand.slice(..);

        let mut current_bitsize = 64;
        while current_bitsize > 1 {
            current_bitsize >>= 1;
            unsafe {
                self.kernels[0]
                    .collapse_u64_helper
                    .clone()
                    .launch_on_stream(
                        &streams[0],
                        cfg,
                        (
                            &res.a,
                            &res.b,
                            &helper.a,
                            &helper.b,
                            &rand_offset,
                            current_bitsize,
                        ),
                    )
                    .unwrap();
            }
            let bytes = (current_bitsize + 7) / 8;
            rand_offset = rand_offset.slice(bytes..); // Advance randomness
            self.send_receive_view_single_gpu(&mut res, 0, streams)
        }
    }

    fn rotate_bitvec(
        &mut self,
        bitvec_out: &mut [ChunkShareView<u64>],
        bitvec_in: &[ChunkShareView<u64>],
        rotation: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, bitvec_out.len());
        assert!(self.chunk_size <= bitvec_out[0].len());
        assert_eq!(self.n_devices, bitvec_in.len());
        assert_eq!(bitvec_in[0].len(), bitvec_out[0].len());

        for (idx, (out, inp)) in bitvec_out.iter_mut().zip(bitvec_in.iter()).enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .rotate_bitvec
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (&out.a, &out.b, &inp.a, &inp.b, rotation, out.len()),
                    )
                    .unwrap();
            }
        }
    }

    fn mask_bitvec(
        &mut self,
        bitvec_inout: &mut [ChunkShareView<u64>],
        mask_in: &[Vec<Vec<u64>>],
        mask_idx: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, bitvec_inout.len());
        assert!(self.chunk_size <= bitvec_inout[0].len());
        assert_eq!(self.n_devices, mask_in.len());
        assert_eq!(mask_in[0][mask_idx].len(), bitvec_inout[0].len());

        for (idx, (inout, inp)) in bitvec_inout.iter_mut().zip(mask_in.iter()).enumerate() {
            let bitvec =
                htod_on_stream_sync(&inp[mask_idx][..], &self.devs[idx], &streams[idx]).unwrap();

            let cfg = launch_config_from_elements_and_threads(
                self.chunk_size as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .mask_bitvec
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (&inout.a, &inout.b, &bitvec, inout.len()),
                    )
                    .unwrap();
            }
        }
    }

    // Result is in the first bit of the first GPU
    pub fn or_reduce_result(&mut self, result: &mut [ChunkShare<u64>], streams: &[CudaStream]) {
        let mut bits = Vec::with_capacity(self.n_devices);
        for r in result.iter() {
            // Result is in the first bit of the input
            bits.push(r.get_offset(0, self.chunk_size));
        }

        self.or_tree_on_gpus(&mut bits, streams);
        if self.n_devices > 1 {
            // We have to collaps to one GPU
            self.collect_graphic_result(&mut bits, streams);
            self.or_tree_on_gpu(&mut bits, self.n_devices, 0, streams);
        }

        // Result is in lowest u64 bits on the first GPU
        self.collapse_u64(&mut result[0], streams);
        // Result is in the first bit of the first GPU
    }

    fn collapse_sum(&mut self, injected_bits: &mut [ChunkShareView<u32>], streams: &[CudaStream]) {
        for (idx, data) in injected_bits.iter_mut().enumerate() {
            let cfg = launch_config_from_elements_and_threads(
                2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .collapse_sum_assign
                    .clone()
                    .launch_on_stream(&streams[idx], cfg, (&data.a, &data.b, self.chunk_size * 64))
                    .unwrap();
            }
        }
    }

    fn collapse_sum_on_gpu(
        &mut self,
        inout: &mut ChunkShare<u32>,
        inputs: &[ChunkShareView<u32>],
        size: usize,
        inout_idx: usize,
        inputs_idx: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, inputs.len());
        assert!(size <= inputs[inputs_idx].len());

        let cfg = launch_config_from_elements_and_threads(
            2,
            DEFAULT_LAUNCH_CONFIG_THREADS,
            &self.devs[inputs_idx],
        );

        unsafe {
            self.kernels[inputs_idx]
                .collapse_sum
                .clone()
                .launch_on_stream(
                    &streams[inputs_idx],
                    cfg,
                    (
                        &inout.a,
                        &inout.b,
                        &inputs[inputs_idx].a,
                        &inputs[inputs_idx].b,
                        inout_idx,
                        size,
                    ),
                )
                .unwrap();
        }
    }

    // input should be of size: n_devices * input_size
    // Result is in the first bit of the result buffer
    pub fn compare_threshold_masked_many(
        &mut self,
        code_dots: &[ChunkShareView<u16>],
        mask_dots: &[ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, code_dots.len());
        assert_eq!(self.n_devices, mask_dots.len());
        for chunk in code_dots.iter().chain(mask_dots.iter()) {
            assert!(chunk.len() % 64 == 0);
        }

        let x_ = Buffers::take_buffer(&mut self.buffers.lifted_shares);
        let corrections_ = Buffers::take_buffer(&mut self.buffers.lifting_corrections);
        let mut x = Buffers::get_buffer_chunk(&x_, 64 * self.chunk_size);
        let mut corrections = Buffers::get_buffer_chunk(&corrections_, 128 * self.chunk_size);

        self.lift_mpc(mask_dots, &mut x, &mut corrections, streams);
        self.lift_mul_sub(&mut x, &corrections, code_dots, streams);
        self.extract_msb(&mut x, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares, x_);
        Buffers::return_buffer(&mut self.buffers.lifting_corrections, corrections_);
        self.buffers.check_buffers();

        // Result is in the first bit of the result buffer
    }

    pub fn translate_threshold_a(a: f64) -> u64 {
        ((1. - 2. * a) * ((1u64 << B_BITS) as f64)) as u64
    }

    // same as compare_threshold_masked_many, just via the functions used in the
    // bucketing
    // Just here for testing
    pub fn compare_threshold_masked_many_bucket_functions(
        &mut self,
        code_dots: &[ChunkShareView<u16>],
        mask_dots: &[ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let a = Self::translate_threshold_a(iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO);

        assert_eq!(self.n_devices, code_dots.len());
        assert_eq!(self.n_devices, mask_dots.len());
        for chunk in code_dots.iter().chain(mask_dots.iter()) {
            assert!(chunk.len() % 64 == 0);
        }

        let x_ = Buffers::take_buffer(&mut self.buffers.lifted_shares);
        let x1_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_buckets1);
        let x2_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_buckets2);
        let corrections_ = Buffers::take_buffer(&mut self.buffers.lifting_corrections);
        let mut masks = Buffers::get_buffer_chunk(&x1_, 64 * self.chunk_size);
        let mut codes = Buffers::get_buffer_chunk(&x2_, 64 * self.chunk_size);
        let mut x = Buffers::get_buffer_chunk(&x_, 64 * self.chunk_size);
        let mut corrections = Buffers::get_buffer_chunk(&corrections_, 128 * self.chunk_size);

        self.lift_mpc(mask_dots, &mut masks, &mut corrections, streams);
        self.finalize_lifts(&mut masks, &mut codes, &corrections, code_dots, streams);
        self.lifted_sub(&mut x, &masks, &codes, a as u32, streams);
        self.extract_msb(&mut x, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares, x_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_buckets1, x1_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_buckets2, x2_);
        Buffers::return_buffer(&mut self.buffers.lifting_corrections, corrections_);
        self.buffers.check_buffers();

        // Result is in the first bit of the result buffer
    }

    // input should be of size: n_devices * input_size
    // Result is in the lowest bit of the result buffer on the first gpu
    pub fn compare_threshold_masked_many_with_or_tree(
        &mut self,
        code_dots: &[ChunkShareView<u16>],
        mask_dots: &[ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        self.compare_threshold_masked_many(code_dots, mask_dots, streams);
        let mut result = self.take_result_buffer();
        self.or_reduce_result(&mut result, streams);
        // Result is in the first bit of the first GPU

        self.return_result_buffer(result);
        self.buffers.check_buffers();

        // Result is in the lowest bit of the result buffer on the first gpu
    }

    pub fn compare_multiple_thresholds(
        &mut self,
        code_dots: &[ChunkShareView<u16>],
        mask_dots: &[ChunkShareView<u16>],
        streams: &[CudaStream],
        thresholds_a: &[u16], // Thresholds are given as a/b, where b=2^16
        buckets: &mut ChunkShare<u32>, // Each element in the chunkshares is one bucket
    ) {
        assert_eq!(self.n_devices, code_dots.len());
        assert_eq!(self.n_devices, mask_dots.len());
        assert_eq!(thresholds_a.len(), buckets.len());
        for chunk in code_dots.iter().chain(mask_dots.iter()) {
            assert!(chunk.len() % 64 == 0);
        }

        let x_ = Buffers::take_buffer(&mut self.buffers.lifted_shares);
        let x1_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_buckets1);
        let x2_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_buckets2);
        let corrections_ = Buffers::take_buffer(&mut self.buffers.lifting_corrections);
        let mut masks = Buffers::get_buffer_chunk(&x1_, 64 * self.chunk_size);
        let mut codes = Buffers::get_buffer_chunk(&x2_, 64 * self.chunk_size);
        let mut x = Buffers::get_buffer_chunk(&x_, 64 * self.chunk_size);
        let mut corrections = Buffers::get_buffer_chunk(&corrections_, 128 * self.chunk_size);

        // Start with lifting
        self.lift_mpc(mask_dots, &mut masks, &mut corrections, streams);
        self.finalize_lifts(&mut masks, &mut codes, &corrections, code_dots, streams);

        for (bucket_idx, a) in thresholds_a.iter().enumerate() {
            // Continue with threshold comparison
            self.lifted_sub(&mut x, &masks, &codes, *a as u32, streams);
            self.extract_msb(&mut x, streams);

            // Result is in the first bit of the result buffer
            let result = self.take_result_buffer();
            let mut bits = Vec::with_capacity(self.n_devices);
            for r in result.iter() {
                // Result is in the first bit of the input
                bits.push(r.get_offset(0, self.chunk_size));
            }

            // Expand the result buffer to the x buffer and perform arithmetic xor
            self.bit_inject_arithmetic_xor(&bits, &mut x, streams);
            // Sum all elements in x to get the result in the first 32 bit word on each GPU
            self.collapse_sum(&mut x, streams);
            // Get data onto the first GPU
            if self.n_devices > 1 {
                self.collect_graphic_result_u32(&mut x, streams);
            }
            // Accumulate first result onto bucket
            self.collapse_sum_on_gpu(buckets, &x, self.n_devices, bucket_idx, 0, streams);
            self.return_result_buffer(result);
        }

        Buffers::return_buffer(&mut self.buffers.lifted_shares, x_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_buckets1, x1_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_buckets2, x2_);
        Buffers::return_buffer(&mut self.buffers.lifting_corrections, corrections_);
        self.buffers.check_buffers();
    }

    pub fn compare_multiple_thresholds_while_aggregating_per_query(
        &mut self,
        code_dots: &[ChunkShareView<u16>],
        mask_dots: &[ChunkShareView<u16>],
        bitmask: &[Vec<u64>],
        streams: &[CudaStream],
        thresholds_a: &[u16], // Thresholds are given as a/b, where b=2^16
        buckets: &mut ChunkShare<u32>, // Each element in the chunkshares is one bucket
    ) {
        assert_eq!(self.n_devices, code_dots.len());
        assert_eq!(self.n_devices, mask_dots.len());
        assert_eq!(thresholds_a.len(), buckets.len());
        assert_eq!(bitmask.len(), self.n_devices);
        for chunk in code_dots.iter().chain(mask_dots.iter()) {
            assert!(chunk.len() % 64 == 0);
        }

        // prepare bitmasks for rotations
        let max_rotations_needed = 30;
        // incoming bitmask looks like this for given query_indices:
        // [1,1,2,3,3,3,4,4,5] -> [1,0,1,1,0,0,1,0,1]
        // i.e., it is one if it is the first match for a given query index
        // we negate it -> [0,1,0,0,1,1,0,1,0] this gives us the indices of the duplicates
        // STEP i: We rotate the bitmask to the right by 1, so that we can use it as a mask
        // -> [1,0,0,1,1,0,1,0,0], this is then used to MASK the OR of the original bit vector with the rotated one
        // after the rotation, we AND the bitmask with the negated original bitmask, to clear the current bit for accumulation
        // and then we repeat from STEP i
        let mut bitmasks = vec![vec![]; self.n_devices];
        for i in 0..self.n_devices {
            bitmasks[i].push(bitmask[i].clone());
            let mut bitmask_negated = bitmask[i].clone();
            detail::negate_bitvec(&mut bitmask_negated);
            let mut bitmask = bitmask_negated.clone();
            for _ in 0..max_rotations_needed {
                detail::rotate_bitvec_right(&mut bitmask);
                bitmasks[i].push(bitmask.clone());
                detail::bitvec_and(&mut bitmask, &bitmask_negated);
            }
        }

        let x_ = Buffers::take_buffer(&mut self.buffers.lifted_shares);
        let x1_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_buckets1);
        let x2_ = Buffers::take_buffer(&mut self.buffers.lifted_shares_buckets2);
        let corrections_ = Buffers::take_buffer(&mut self.buffers.lifting_corrections);
        let mut masks = Buffers::get_buffer_chunk(&x1_, 64 * self.chunk_size);
        let mut codes = Buffers::get_buffer_chunk(&x2_, 64 * self.chunk_size);
        let mut x = Buffers::get_buffer_chunk(&x_, 64 * self.chunk_size);
        let mut corrections = Buffers::get_buffer_chunk(&corrections_, 128 * self.chunk_size);

        // Start with lifting
        self.lift_mpc(mask_dots, &mut masks, &mut corrections, streams);
        self.finalize_lifts(&mut masks, &mut codes, &corrections, code_dots, streams);

        for (bucket_idx, a) in thresholds_a.iter().enumerate() {
            // Continue with threshold comparison
            self.lifted_sub(&mut x, &masks, &codes, *a as u32, streams);
            self.extract_msb(&mut x, streams);

            // Result is in the first bit of the result buffer
            let result = self.take_result_buffer();

            let mut bits = Vec::with_capacity(self.n_devices);
            for r in result.iter() {
                // Result is in the first bit of the input
                bits.push(r.get_offset(0, self.chunk_size));
            }
            let tmp_rotated = Buffers::take_buffer(&mut self.buffers.lifted_shares_split2);
            let mut rotated = Buffers::get_buffer_chunk(&tmp_rotated, self.chunk_size);
            // we now use the prepared bitmasks to aggregate the results with OR
            for rotation in 0..max_rotations_needed {
                // we need to rotate the bitvecs to the right by one
                self.rotate_bitvec(&mut rotated, &bits, rotation + 1, streams);
                self.mask_bitvec(&mut rotated, &bitmasks, rotation + 1, streams);
                for idx in 0..self.n_devices {
                    self.or_many_pre_assign(&mut bits[idx], &rotated[idx], idx, streams);
                }
                self.send_receive_view_with_offset(&mut bits, 0..self.chunk_size, streams);
            }
            Buffers::return_buffer(&mut self.buffers.lifted_shares_split2, tmp_rotated);
            // finally, mask out all of the bits that are not the first match, where we have accumulated
            self.mask_bitvec(&mut bits, &bitmasks, 0, streams);

            // Expand the result buffer to the x buffer and perform arithmetic xor
            self.bit_inject_arithmetic_xor(&bits, &mut x, streams);
            // Sum all elements in x to get the result in the first 32 bit word on each GPU
            self.collapse_sum(&mut x, streams);
            // Get data onto the first GPU
            if self.n_devices > 1 {
                self.collect_graphic_result_u32(&mut x, streams);
            }
            // Accumulate first result onto bucket
            self.collapse_sum_on_gpu(buckets, &x, self.n_devices, bucket_idx, 0, streams);
            self.return_result_buffer(result);
        }
        Buffers::return_buffer(&mut self.buffers.lifted_shares, x_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_buckets1, x1_);
        Buffers::return_buffer(&mut self.buffers.lifted_shares_buckets2, x2_);
        Buffers::return_buffer(&mut self.buffers.lifting_corrections, corrections_);
        self.buffers.check_buffers();
    }

    // 2D anon stats implementation plan.
    // 1. Align more with the CPU code, since it would make it easier overall, even though the MPC operations are more expensive
    // We want to have the following interface to the 2D bucketing function:
    // Inputs:
    //  * distances: Vec<code_dot_left, mask_dot_left, code_dot_right, mask_dot_right>, a vec of distance tuples that matched on both sides, already reduced in terms of min fhd
    //   * thresholds_a: Vec<u16>, thresholds are given as a/b, where b=2^16
    // Outputs:
    //   * buckets: Vec<u32>, of length thresholds_a^2, with layout: (left=0,right=0), (left=0,right=1), ..., (left=1,right=0), (left=1,right=1), ... (left = N-1, right = N-1)

    // This is step one, we will also need:
    // 2. A way to reduce the distances per query to the min ones on both left and right side
    // We will do this like we did on the CPU side:
    //   * We will have ids associated with each distance, such that we can group them into rotations for the same query
    //   * We will group the distances by query id, take the first one as the base, and compare the rest to it, doing conditional swaps to keep the minimum one
    //   * For the conditional swaps, we will need to lift the 16-bit shares to 32-bit shares, which we will do in a single batch
    //
    // 3. Finally, we will need to collect the distances for matching elements during the batch evaluation.
    // This is actually one of the more complex parts, as during the evaluation of the right/left hand side, we do not know yet if this partial match is actually a full match as well.
    // We will do this by:
    //   * Storing all partial matches on the left side in a buffer (which we already do, however, this might get cleared if we produce 1D anon stats...)
    //   * Storing all partial matches on the right side in a buffer (which we already do, however, this might get cleared if we produce 1D anon stats...)
    //   * So the above buffers also need to be completely new ones I guess
    //   * After each query is fully done, we have to copy this to CPU, sort the entries by global_id, and do a set union on the global_id%rotations of left and right side
    //   * Then we can add it to a temporary storage which is later on input into phase2 when it is large enough...
    pub fn compare_multiple_thresholds_2d(
        &mut self,
        code_dots_left: &[ChunkShareView<u32>],
        mask_dots_left: &[ChunkShareView<u32>],
        code_dots_right: &[ChunkShareView<u32>],
        mask_dots_right: &[ChunkShareView<u32>],
        streams: &[CudaStream],
        thresholds_a: &[u16], // Thresholds are given as a/b, where b=2^16
    ) -> Vec<u32> {
        assert_eq!(self.n_devices, code_dots_left.len());
        assert_eq!(self.n_devices, code_dots_right.len());
        assert_eq!(self.n_devices, mask_dots_left.len());
        assert_eq!(self.n_devices, mask_dots_right.len());
        for (cl, ml, cr, mr) in izip!(
            code_dots_left,
            mask_dots_left,
            code_dots_right,
            mask_dots_right
        ) {
            assert!(ml.len() == cl.len());
            assert!(cr.len() == cl.len());
            assert!(mr.len() == cl.len());
            assert!(cl.len() <= self.chunk_size * 64);
        }

        // allocate a temporary buffer for the lifted shares
        let mut lifted_left: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(thresholds_a.len());
        let mut lifted_right: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(thresholds_a.len());
        let x_ = Buffers::take_buffer(&mut self.buffers.lifted_shares);
        let mut x = Buffers::get_buffer_chunk(&x_, 64 * self.chunk_size);

        for (code, mask, buffer) in [
            (code_dots_left, mask_dots_left, &mut lifted_left),
            (code_dots_right, mask_dots_right, &mut lifted_right),
        ] {
            for a in thresholds_a.iter() {
                // Continue with threshold comparison
                self.pre_lifted_sub_ab(&mut x, mask, code, *a as u32, streams);
                self.extract_msb(&mut x, streams);

                // Result is in the first bit of the result buffer
                let result = self.take_result_buffer();

                let mut bits = Vec::with_capacity(self.n_devices);
                for r in result.iter() {
                    // Result is in the first bit of the input
                    bits.push(r.get_offset(0, self.chunk_size));
                }

                // Expand the result buffer to the x buffer and perform arithmetic xor
                self.bit_inject_arithmetic_xor(&bits, &mut x, streams);

                let mut a_buf = Vec::new();
                let mut b_buf = Vec::new();
                for (i, injected_bits) in x.iter().enumerate() {
                    let len = code[i].len();
                    let a = dtoh_on_stream_sync(
                        &injected_bits.a.slice(..len),
                        &self.devs[i],
                        &streams[i],
                    )
                    .unwrap();
                    let b = dtoh_on_stream_sync(
                        &injected_bits.b.slice(..len),
                        &self.devs[i],
                        &streams[i],
                    )
                    .unwrap();
                    a_buf.extend(a);
                    b_buf.extend(b);
                }
                buffer.push((a_buf, b_buf));
                self.return_result_buffer(result);
            }
        }
        Buffers::return_buffer(&mut self.buffers.lifted_shares, x_);
        // randomness, needs to be additively correlated, that is why we squeeze them separately and subtract them below
        let num_buckets = thresholds_a.len() * thresholds_a.len();
        let mut buckets = {
            // need to pad this temporarily to a multiple of 16 for the RNG
            let padded_size = num_buckets.div_ceil(16) * 16;
            let mut bucket_randomness = self.devs[0].alloc_zeros::<u32>(padded_size * 2).unwrap();
            {
                let rng = &mut self.rngs[0];
                let mut rand_view = bucket_randomness.slice_mut(..padded_size);
                rng.fill_my_rng_into(&mut rand_view, &streams[0]);
                let mut rand_view = bucket_randomness.slice_mut(padded_size..);
                rng.fill_their_rng_into(&mut rand_view, &streams[0]);
            }
            let mut buckets =
                dtoh_on_stream_sync(&bucket_randomness, &self.devs[0], &streams[0]).unwrap();
            for i in 0..num_buckets {
                // Make correlated randomness from the two randomness vectors
                buckets[i] -= buckets[i + padded_size];
            }
            buckets.truncate(num_buckets);
            buckets
        };

        // Now we have the lifted 0/1 shares in lifted_left and lifted_right, do an outer product + aggregation to get all of the results.
        // TODO: we do this on the CPU for now, let's keep an eye on the performance though since it scales quadratically with the number of thresholds
        let mut idx = 0;
        for left_results in lifted_left.iter() {
            for right_results in lifted_right.iter() {
                // We have to do the outer product of the left and right results
                // and aggregate them into the buckets_squared
                assert_eq!(left_results.0.len(), right_results.0.len());
                assert_eq!(left_results.1.len(), right_results.1.len());
                let local_aggregated = izip!(
                    left_results.0.iter(),
                    left_results.1.iter(),
                    right_results.0.iter(),
                    right_results.1.iter()
                )
                .map(|(&l_a, &l_b, &r_a, &r_b)| {
                    // local part of mul, without randomness yet, will be added in the end
                    l_a.wrapping_mul(r_a)
                        .wrapping_add(l_a.wrapping_mul(r_b))
                        .wrapping_add(l_b.wrapping_mul(r_a))
                })
                .fold(0u32, |acc, x| acc.wrapping_add(x));
                buckets[idx] += local_aggregated;
                idx += 1;
            }
        }

        // Move the results to the first GPU, and communicate to reveal the results
        let result_buckets = {
            // TODO: add randomness to the buckets
            let result_share = htod_on_stream_sync(&buckets, &self.devs[0], &streams[0]).unwrap();
            let mut buf0 = self.devs[0].alloc_zeros::<u32>(result_share.len()).unwrap();
            let mut buf1 = self.devs[0].alloc_zeros::<u32>(result_share.len()).unwrap();
            self.synchronize_streams(streams);

            result::group_start().unwrap();
            self.comms[0]
                .send(&result_share, self.next_id, &streams[0])
                .unwrap();
            self.comms[0]
                .receive(&mut buf0, self.prev_id, &streams[0])
                .unwrap();
            result::group_end().unwrap();
            self.synchronize_streams(streams);
            result::group_start().unwrap();
            self.comms[0]
                .send(&result_share, self.prev_id, &streams[0])
                .unwrap();
            self.comms[0]
                .receive(&mut buf1, self.next_id, &streams[0])
                .unwrap();
            result::group_end().unwrap();
            self.synchronize_streams(streams);
            let mut buckets1 = dtoh_on_stream_sync(&buf0, &self.devs[0], &streams[0]).unwrap();
            let buckets2 = dtoh_on_stream_sync(&buf1, &self.devs[0], &streams[0]).unwrap();

            for (b0, b1, b2) in izip!(buckets1.iter_mut(), buckets2.iter(), buckets.iter()) {
                *b0 = b0.wrapping_add(b1.wrapping_add(*b2));
            }
            buckets1
        };

        self.buffers.check_buffers();
        result_buckets
    }

    /// Cross compare and swap the codes and masks of two sets of shares.
    /// This function sets the codes and masks inputs/outputs to the lower threshold between code/mask and code_2/mask_2.
    ///
    /// This function is also one of the few ones that is allowed to be called with any number of device inputs, as long as there are at least that many devices.
    /// In particular, it is used to do the work for the 2D anon stats minimum FHD step on the first GPU only.
    pub fn cross_compare_and_swap(
        &mut self,
        codes: &mut [ChunkShareView<u32>],
        masks: &mut [ChunkShareView<u32>],
        codes_2: &[ChunkShareView<u32>],
        masks_2: &[ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        let used_devices = codes.len();
        assert!(used_devices <= self.n_devices);
        assert_eq!(codes.len(), masks.len());
        assert_eq!(codes_2.len(), masks_2.len());
        assert_eq!(used_devices, codes_2.len());
        for (c, m, c2, m2) in izip!(codes.iter(), masks.iter(), codes_2.iter(), masks_2.iter()) {
            assert!(c.len() % 64 == 0);
            assert!(m.len() == c.len());
            assert!(c2.len() == c.len());
            assert!(m2.len() == c2.len());
            assert!(c.len() <= self.chunk_size * 64);
        }

        let x_ = Buffers::take_buffer(&mut self.buffers.lifted_shares);
        let mut xvec = Buffers::get_buffer_chunk(&x_, self.chunk_size * 64);
        {
            let x = &mut xvec[..used_devices];

            self.cross_mul(x, codes, masks, codes_2, masks_2, streams);
        }
        self.extract_msb(&mut xvec, streams);
        // Result is in the first bit of the result buffer
        let result = self.take_result_buffer();

        let mut bits = Vec::with_capacity(self.n_devices);
        for r in result.iter() {
            // Result is in the first bit of the input
            bits.push(r.get_offset(0, self.chunk_size));
        }

        // Expand the result buffer to the x buffer and perform arithmetic xor
        self.bit_inject_arithmetic_xor(&bits, &mut xvec, streams);
        self.return_result_buffer(result);

        let x = &mut xvec[..used_devices];

        self.conditionally_select_distance(x, codes, masks, codes_2, masks_2, streams);

        Buffers::return_buffer(&mut self.buffers.lifted_shares, x_);
        self.buffers.check_buffers();
    }

    /// Computes the cross product of distances shares represented as a fraction (code_dist, mask_dist).
    /// The cross product is computed as (d2.code_dist * d1.mask_dist - d1.code_dist * d2.mask_dist) and the result is shared.
    ///
    /// This is also one of the few functions that is allowed to be called with any number of device inputs, as long as there are at least that many devices.
    /// In particular, it is used to do the work for the 2D anon stats minimum FHD step on the first GPU only.
    pub fn cross_mul(
        &mut self,
        out: &mut [ChunkShareView<u32>],
        codes: &[ChunkShareView<u32>],
        masks: &[ChunkShareView<u32>],
        codes_2: &[ChunkShareView<u32>],
        masks_2: &[ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        let num_devices = out.len();
        assert!(num_devices <= self.n_devices);
        assert_eq!(out.len(), masks.len());
        assert_eq!(codes.len(), masks.len());
        assert_eq!(codes_2.len(), masks.len());
        assert_eq!(masks_2.len(), masks.len());

        for (idx, (out, code, mask, code_2, mask_2)) in izip!(
            out.iter(),
            codes.iter(),
            masks.iter(),
            codes_2.iter(),
            masks_2.iter()
        )
        .enumerate()
        {
            let len = code.len();
            assert_eq!(code.len(), mask.len());
            assert_eq!(code_2.len(), mask_2.len());
            assert_eq!(code.len(), code_2.len());

            // randomness, needs to be additively correlated, that is why we squeeze them separately and
            // combine them later in the kernel with - instead of xor
            let mut rand = unsafe { self.devs[idx].alloc::<u32>(len * 2).unwrap() };
            {
                let rng = &mut self.rngs[idx];
                let mut rand_view = rand.slice_mut(..len);
                rng.fill_my_rng_into(&mut rand_view, &streams[idx]);
                let mut rand_view = rand.slice_mut(len..);
                rng.fill_their_rng_into(&mut rand_view, &streams[idx]);
            }

            let cfg = launch_config_from_elements_and_threads(
                code.len() as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .cross_mul_pre
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &out.a, &code.a, &code.b, &mask.a, &mask.b, &code_2.a, &code_2.b,
                            &mask_2.a, &mask_2.b, &rand, len,
                        ),
                    )
                    .unwrap();
            }
        }
        // reshare the results
        result::group_start().unwrap();
        for (idx, send) in out.iter().enumerate() {
            self.comms[idx]
                .send_view(&send.a, self.next_id, &streams[idx])
                .unwrap();
        }
        for (idx, recv) in out.iter_mut().enumerate() {
            self.comms[idx]
                .receive_view(&mut recv.b, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
    }

    /// Conditionally select (code, masks), (codes_2, masks_2) based on (conds), if conds = 0 select *_2, otherwise select the first one.
    ///
    /// This is also one of the few functions that is allowed to be called with any number of device inputs, as long as there are at least that many devices.
    /// In particular, it is used to do the work for the 2D anon stats minimum FHD step on the first GPU only.
    pub fn conditionally_select_distance(
        &mut self,
        conds: &[ChunkShareView<u32>],
        codes: &mut [ChunkShareView<u32>],
        masks: &mut [ChunkShareView<u32>],
        codes_2: &[ChunkShareView<u32>],
        masks_2: &[ChunkShareView<u32>],
        streams: &[CudaStream],
    ) {
        let num_devices = conds.len();
        assert!(num_devices <= self.n_devices);
        assert_eq!(conds.len(), masks.len());
        assert_eq!(codes.len(), masks.len());
        assert_eq!(codes_2.len(), masks.len());
        assert_eq!(masks_2.len(), masks.len());

        for (idx, (cond, code, mask, code_2, mask_2)) in izip!(
            conds.iter(),
            codes.iter_mut(),
            masks.iter_mut(),
            codes_2.iter(),
            masks_2.iter()
        )
        .enumerate()
        {
            let len = code.len();
            assert_eq!(code.len(), mask.len());
            assert_eq!(code_2.len(), mask_2.len());
            assert_eq!(code.len(), code_2.len());

            // randomness, needs to be additively correlated, that is why we squeeze them separately and
            // combine them later in the kernel with - instead of xor
            let mut rand = unsafe { self.devs[idx].alloc::<u32>(len * 4).unwrap() };
            {
                let rng = &mut self.rngs[idx];
                let mut rand_view = rand.slice_mut(..len * 2);
                rng.fill_my_rng_into(&mut rand_view, &streams[idx]);
                let mut rand_view = rand.slice_mut(len * 2..);
                rng.fill_their_rng_into(&mut rand_view, &streams[idx]);
            }

            let cfg = launch_config_from_elements_and_threads(
                len as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .conditional_select_pre
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &cond.a, &cond.b, &code.a, &code.b, &mask.a, &mask.b, &code_2.a,
                            &code_2.b, &mask_2.a, &mask_2.b, &rand, len,
                        ),
                    )
                    .unwrap();
            }
        }
        // reshare the results
        result::group_start().unwrap();
        for (idx, send) in codes.iter().enumerate() {
            self.comms[idx]
                .send_view(&send.a, self.next_id, &streams[idx])
                .unwrap();
        }
        for (idx, recv) in codes.iter_mut().enumerate() {
            self.comms[idx]
                .receive_view(&mut recv.b, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();
        result::group_start().unwrap();
        for (idx, send) in masks.iter().enumerate() {
            self.comms[idx]
                .send_view(&send.a, self.next_id, &streams[idx])
                .unwrap();
        }
        for (idx, recv) in masks.iter_mut().enumerate() {
            self.comms[idx]
                .receive_view(&mut recv.b, self.prev_id, &streams[idx])
                .unwrap();
        }
        result::group_end().unwrap();

        for (idx, (code, mask, code_2, mask_2)) in izip!(
            codes.iter_mut(),
            masks.iter_mut(),
            codes_2.iter(),
            masks_2.iter()
        )
        .enumerate()
        {
            let len = code.len();
            let cfg = launch_config_from_elements_and_threads(
                code.len() as u32,
                DEFAULT_LAUNCH_CONFIG_THREADS,
                &self.devs[idx],
            );

            unsafe {
                self.kernels[idx]
                    .conditional_select_post
                    .clone()
                    .launch_on_stream(
                        &streams[idx],
                        cfg,
                        (
                            &code.a, &code.b, &mask.a, &mask.b, &code_2.a, &code_2.b, &mask_2.a,
                            &mask_2.b, len,
                        ),
                    )
                    .unwrap();
            }
        }
    }

    pub fn open_buckets(&mut self, buckets: &ChunkShare<u32>, streams: &[CudaStream]) -> Vec<u32> {
        let a = dtoh_on_stream_sync(&buckets.a, &self.devs[0], &streams[0]).unwrap();
        let b = dtoh_on_stream_sync(&buckets.b, &self.devs[0], &streams[0]).unwrap();
        let res = buckets.as_view();

        let rcv_buffer_ = Buffers::take_buffer(&mut self.buffers.buckets_recv_buffer);
        let mut rcv_buffer = rcv_buffer_[0].get_range(0, buckets.len());

        result::group_start().unwrap();
        self.comms[0]
            .send_view(&res.b, self.next_id, &streams[0])
            .unwrap();
        self.comms[0]
            .receive_view(&mut rcv_buffer.a, self.prev_id, &streams[0])
            .unwrap();
        result::group_end().unwrap();

        let c = dtoh_on_stream_sync(&rcv_buffer.a, &self.devs[0], &streams[0]).unwrap();

        Buffers::return_buffer(&mut self.buffers.buckets_recv_buffer, rcv_buffer_);
        self.buffers.check_buffers();

        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&a, &b), &c)| a.wrapping_add(b).wrapping_add(c))
            .collect()
    }
}

mod detail {
    // rotate a bit vector to the right by one (bit 1 -> bit 0)
    pub fn rotate_bitvec_right(vec: &mut [u64]) {
        let mut last = vec[0] & 1;
        for i in (0..vec.len()).rev() {
            let tmp = vec[i] & 1;
            vec[i] = (last << 63) | (vec[i] >> 1);
            last = tmp;
        }
    }

    pub fn negate_bitvec(vec: &mut [u64]) {
        for i in 0..vec.len() {
            vec[i] = !vec[i];
        }
    }
    pub fn bitvec_and(vec: &mut [u64], vec2: &[u64]) {
        assert!(vec.len() == vec2.len());
        for i in 0..vec.len() {
            vec[i] &= vec2[i];
        }
    }
}
