use crate::{
    helpers::{
        device_manager::DeviceManager,
        dtoh_on_stream_sync, htod_on_stream_sync,
        id_wrapper::{http_root, IdWrapper},
        task_monitor::TaskMonitor,
    },
    rng::chacha_corr::ChaChaCudaCorrRng,
    threshold_ring::cuda::PTX_SRC,
};
use axum::{routing::get, Router};
use cudarc::{
    driver::{
        result::stream, CudaDevice, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut,
        DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig,
    },
    nccl::{result, Comm, Id},
    nvrtc::{self, Ptx},
};
use itertools::izip;
#[cfg(feature = "otp_encrypt")]
use itertools::Itertools;
use std::{ops::Range, str::FromStr, sync::Arc, thread, time::Duration};
use tokio::task::AbortHandle;

pub(crate) const B_BITS: usize = 16;

const DEFAULT_LAUNCH_CONFIG_THREADS: u32 = 64;

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
    pub(crate) and:                   CudaFunction,
    pub(crate) or_assign:             CudaFunction,
    pub(crate) xor:                   CudaFunction,
    pub(crate) xor_assign:            CudaFunction,
    #[cfg(feature = "otp_encrypt")]
    pub(crate) single_xor_assign_u16: CudaFunction,
    #[cfg(feature = "otp_encrypt")]
    pub(crate) single_xor_assign_u64: CudaFunction,
    pub(crate) split:                 CudaFunction,
    pub(crate) lift_split:            CudaFunction,
    pub(crate) lift_mul_sub:          CudaFunction,
    pub(crate) transpose_32x64:       CudaFunction,
    pub(crate) transpose_16x64:       CudaFunction,
    pub(crate) ot_sender:             CudaFunction,
    pub(crate) ot_receiver:           CudaFunction,
    pub(crate) ot_helper:             CudaFunction,
    pub(crate) assign:                CudaFunction,
    pub(crate) collapse_u64_helper:   CudaFunction,
}

impl Kernels {
    const MOD_NAME: &'static str = "TComp";

    pub(crate) fn new(dev: Arc<CudaDevice>, ptx: Ptx) -> Kernels {
        dev.load_ptx(ptx.clone(), Self::MOD_NAME, &[
            "shared_xor",
            "shared_xor_assign",
            "xor_assign_u16",
            "xor_assign_u64",
            "shared_and_pre",
            "shared_or_pre_assign",
            "split",
            "lift_split",
            "shared_lift_mul_sub",
            "shared_u32_transpose_pack_u64",
            "shared_u16_transpose_pack_u64",
            "packed_ot_sender",
            "packed_ot_receiver",
            "packed_ot_helper",
            "shared_assign",
            "collapse_u64_helper",
        ])
        .unwrap();
        let and = dev.get_func(Self::MOD_NAME, "shared_and_pre").unwrap();
        let or_assign = dev
            .get_func(Self::MOD_NAME, "shared_or_pre_assign")
            .unwrap();
        let xor = dev.get_func(Self::MOD_NAME, "shared_xor").unwrap();
        let xor_assign = dev.get_func(Self::MOD_NAME, "shared_xor_assign").unwrap();
        #[cfg(feature = "otp_encrypt")]
        let single_xor_assign_u16 = dev.get_func(Self::MOD_NAME, "xor_assign_u16").unwrap();
        #[cfg(feature = "otp_encrypt")]
        let single_xor_assign_u64 = dev.get_func(Self::MOD_NAME, "xor_assign_u64").unwrap();
        let split = dev.get_func(Self::MOD_NAME, "split").unwrap();
        let lift_split = dev.get_func(Self::MOD_NAME, "lift_split").unwrap();
        let lift_mul_sub = dev.get_func(Self::MOD_NAME, "shared_lift_mul_sub").unwrap();
        let transpose_32x64 = dev
            .get_func(Self::MOD_NAME, "shared_u32_transpose_pack_u64")
            .unwrap();
        let transpose_16x64 = dev
            .get_func(Self::MOD_NAME, "shared_u16_transpose_pack_u64")
            .unwrap();
        let ot_sender = dev.get_func(Self::MOD_NAME, "packed_ot_sender").unwrap();
        let ot_receiver = dev.get_func(Self::MOD_NAME, "packed_ot_receiver").unwrap();
        let ot_helper = dev.get_func(Self::MOD_NAME, "packed_ot_helper").unwrap();
        let assign = dev.get_func(Self::MOD_NAME, "shared_assign").unwrap();
        let collapse_u64_helper = dev.get_func(Self::MOD_NAME, "collapse_u64_helper").unwrap();

        Kernels {
            and,
            or_assign,
            xor,
            xor_assign,
            #[cfg(feature = "otp_encrypt")]
            single_xor_assign_u16,
            #[cfg(feature = "otp_encrypt")]
            single_xor_assign_u64,
            split,
            lift_split,
            lift_mul_sub,
            transpose_32x64,
            transpose_16x64,
            ot_sender,
            ot_receiver,
            ot_helper,
            assign,
            collapse_u64_helper,
        }
    }
}

struct Buffers {
    u32_64c_1:         Option<Vec<ChunkShare<u32>>>,
    u64_32c_1:         Option<Vec<ChunkShare<u64>>>,
    u64_32c_2:         Option<Vec<ChunkShare<u64>>>,
    u64_32c_3:         Option<Vec<ChunkShare<u64>>>,
    u64_31c_1:         Option<Vec<ChunkShare<u64>>>,
    u64_31c_2:         Option<Vec<ChunkShare<u64>>>,
    u16_128c_1:        Option<Vec<ChunkShare<u16>>>,
    single_u16_128c_1: Option<Vec<CudaSlice<u16>>>,
    single_u16_128c_2: Option<Vec<CudaSlice<u16>>>,
    single_u16_128c_3: Option<Vec<CudaSlice<u16>>>,
    chunk_size:        usize,
}

impl Buffers {
    fn new(devices: &[Arc<CudaDevice>], chunk_size: usize) -> Self {
        let u32_64c_1 = Some(Self::allocate_buffer(chunk_size * 64, devices));
        let u64_32c_1 = Some(Self::allocate_buffer(chunk_size * 32, devices));
        let u64_32c_2 = Some(Self::allocate_buffer(chunk_size * 32, devices));
        let u64_32c_3 = Some(Self::allocate_buffer(chunk_size * 32, devices));

        let u64_31c_1 = Some(Self::allocate_buffer(chunk_size * 31, devices));
        let u64_31c_2 = Some(Self::allocate_buffer(chunk_size * 31, devices));

        let u16_128c_1 = Some(Self::allocate_buffer(chunk_size * 128, devices));

        let single_u16_128c_1 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));
        let single_u16_128c_2 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));
        let single_u16_128c_3 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));

        Buffers {
            u32_64c_1,
            u64_32c_1,
            u64_32c_2,
            u64_32c_3,
            u64_31c_1,
            u64_31c_2,
            u16_128c_1,
            single_u16_128c_1,
            single_u16_128c_2,
            single_u16_128c_3,
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
        assert!(self.u32_64c_1.is_some());
        assert!(self.u64_32c_1.is_some());
        assert!(self.u64_32c_2.is_some());
        assert!(self.u64_32c_3.is_some());
        assert!(self.u64_31c_1.is_some());
        assert!(self.u64_31c_2.is_some());
        assert!(self.u16_128c_1.is_some());
        assert!(self.single_u16_128c_1.is_some());
        assert!(self.single_u16_128c_2.is_some());
        assert!(self.single_u16_128c_3.is_some());
    }
}

pub struct Circuits {
    peer_id:          usize,
    next_id:          usize,
    prev_id:          usize,
    chunk_size:       usize,
    n_devices:        usize,
    devs:             Vec<Arc<CudaDevice>>,
    comms:            Vec<Arc<Comm>>,
    kernels:          Vec<Kernels>,
    buffers:          Buffers,
    rngs:             Vec<ChaChaCudaCorrRng>,
    pub server_abort: Option<AbortHandle>,
}

impl Circuits {
    const BITS: usize = 16 + B_BITS;

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
        chacha_seeds: ([u32; 8], [u32; 8]),
        peer_url: Option<String>,
        server_port: Option<u16>,
        server_task_set: Option<&mut TaskMonitor>,
        device_manager: Arc<DeviceManager>,
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

        let (comms, server_abort) =
            Self::instantiate_network(peer_id, peer_url, server_port, &devs, server_task_set);

        let buffers = Buffers::new(&devs, alloc_size);

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
            server_abort,
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
        Buffers::take_buffer(&mut self.buffers.u64_32c_1)
    }

    pub fn return_result_buffer(&mut self, src: Vec<ChunkShare<u64>>) {
        Buffers::return_buffer(&mut self.buffers.u64_32c_1, src);
    }

    #[allow(clippy::arc_with_non_send_sync)]
    pub fn instantiate_network(
        peer_id: usize,
        peer_url: Option<String>,
        server_port: Option<u16>,
        devices: &[Arc<CudaDevice>],
        server_task_set: Option<&mut TaskMonitor>,
    ) -> (Vec<Arc<Comm>>, Option<AbortHandle>) {
        let n_devices = devices.len();
        let mut comms = Vec::with_capacity(n_devices);
        let mut ids = Vec::with_capacity(n_devices);
        for _ in 0..n_devices {
            ids.push(Id::new().unwrap());
        }

        // Start HTTP server to exchange NCCL commIds
        let mut server_abort = None;
        if peer_id == 0 {
            let server_task_set = server_task_set
                .expect("task set must be supplied to peer_id 0 for remote connection monitoring");

            let ids = ids.clone();
            server_abort = Some(server_task_set.spawn(async move {
                println!("Starting server on port {}...", server_port.unwrap());
                let app = Router::new().route("/:device_id", get(move |req| http_root(ids, req)));
                let listener =
                    tokio::net::TcpListener::bind(format!("0.0.0.0:{}", server_port.unwrap()))
                        .await?;
                axum::serve(listener, app).await?;
                Ok(())
            }));
        }

        if peer_id != 0 {
            thread::sleep(Duration::from_secs(1));
        }

        for i in 0..n_devices {
            let id = if peer_id == 0 {
                ids[i]
            } else {
                let peer_url = peer_url.clone().unwrap();
                std::thread::spawn(move || {
                    let res = reqwest::blocking::get(format!(
                        "http://{}:{}/{}",
                        peer_url,
                        server_port.unwrap(),
                        i
                    ))
                    .unwrap();
                    IdWrapper::from_str(&res.text().unwrap()).unwrap().0
                })
                .join()
                .unwrap()
            };
            ids.push(id);

            // Bind to thread (important!)
            devices[i].bind_to_thread().unwrap();
            comms.push(Arc::new(
                Comm::from_rank(devices[i].clone(), peer_id, 3, id).unwrap(),
            ));
        }
        (comms, server_abort)
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

    fn launch_config_from_elements_and_threads(n: u32, t: u32) -> LaunchConfig {
        let num_blocks = (n + t - 1) / t;
        LaunchConfig {
            grid_dim:         (num_blocks, 1, 1),
            block_dim:        (t, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn send_u16(
        &mut self,
        send: &CudaSlice<u16>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not sendable
        let send_trans: CudaView<u8> = // the transmute_mut is safe because we
        // know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
         unsafe { send.transmute(send.len() * 2).unwrap() };
        self.send_view(&send_trans, peer_id, idx, streams)
    }

    pub fn receive_u16(
        &mut self,
        receive: &mut CudaSlice<u16>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not receivable
        let mut receive_trans: CudaView<u8> = // the transmute_mut is safe
    // because we know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
    unsafe { receive.transmute(receive.len() * 2).unwrap()
    };
        self.receive_view(&mut receive_trans, peer_id, idx, streams)
    }

    pub fn send_view_u16(
        &mut self,
        send: &CudaView<u16>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not sendable
        let send_trans: CudaView<u8> = // the transmute_mut is safe because we
    // know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
         unsafe { send.transmute(send.len() * 2).unwrap() };
        self.send_view(&send_trans, peer_id, idx, streams)
    }

    pub fn receive_view_u16(
        &mut self,
        receive: &mut CudaView<u16>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError> {
        // We have to transmute since u16 is not receivable
        let mut receive_trans: CudaView<u8> = // the transmute_mut
    // is safe because we know that one u16 is 2 u8s, and the buffer is aligned properly for the transmute
        unsafe { receive.transmute(receive.len() *
    2).unwrap() };
        self.receive_view(&mut receive_trans, peer_id, idx, streams)
    }

    pub fn send_view<T>(
        &mut self,
        send: &CudaView<T>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::send(
                *send.device_ptr() as *mut _,
                send.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comms[idx].comm.0,
                streams[idx].stream as *mut _,
            )
        }
    }

    pub fn receive_view<T>(
        &mut self,
        receive: &mut CudaView<T>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::recv(
                *receive.device_ptr() as *mut _,
                receive.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comms[idx].comm.0,
                streams[idx].stream as *mut _,
            )
        }
    }

    pub fn send<T>(
        &mut self,
        send: &CudaSlice<T>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::send(
                *send.device_ptr() as *mut _,
                send.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comms[idx].comm.0,
                streams[idx].stream as *mut _,
            )
        }
    }

    pub fn receive<T>(
        &mut self,
        receive: &mut CudaSlice<T>,
        peer_id: usize,
        idx: usize,
        streams: &[CudaStream],
    ) -> Result<result::NcclStatus, result::NcclError>
    where
        T: cudarc::nccl::NcclType,
    {
        unsafe {
            result::recv(
                *receive.device_ptr() as *mut _,
                receive.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comms[idx].comm.0,
                streams[idx].stream as *mut _,
            )
        }
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
    #[cfg(feature = "otp_encrypt")]
    fn fill_my_rand_u64(&mut self, rand: &mut CudaSlice<u64>, idx: usize, streams: &[CudaStream]) {
        let rng = &mut self.rngs[idx];
        let mut rand_trans: CudaViewMut<u32> =
              // the transmute_mut is safe because we know that one u64 is 2 u32s, and the buffer is aligned properly for the transmute
                  unsafe { rand.transmute_mut(rand.len() * 2).unwrap() };
        rng.fill_my_rng_into(&mut rand_trans, &streams[idx]);
    }

    // Fill randomness using the correlated RNG
    #[cfg(feature = "otp_encrypt")]
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

        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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
        let cfg = Self::launch_config_from_elements_and_threads(
            src.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .assign
                .clone()
                .launch_on_stream(
                    &streams[idx],
                    cfg,
                    (&des.a, &des.b, &src.a, &src.b, src.len() as i32),
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
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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

        let cfg = Self::launch_config_from_elements_and_threads(
            x1.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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

    #[cfg(feature = "otp_encrypt")]
    fn otp_encrypt_my_rng_u16(
        &mut self,
        input: &CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u32> {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut rand = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let mut rand_u16 = self.fill_my_rng_into_u16(&mut rand, idx, streams);
        self.single_xor_assign_u16(&mut rand_u16, input, idx, data_len, streams);
        rand
    }

    #[cfg(feature = "otp_encrypt")]
    fn otp_encrypt_their_rng_u16(
        &mut self,
        input: &CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u32> {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut rand = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let mut rand_u16 = self.fill_their_rng_into_u16(&mut rand, idx, streams);
        self.single_xor_assign_u16(&mut rand_u16, input, idx, data_len, streams);
        rand
    }

    #[cfg(feature = "otp_encrypt")]
    fn otp_decrypt_my_rng_u16(
        &mut self,
        input: &mut CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut rand = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let rand_u16 = self.fill_my_rng_into_u16(&mut rand, idx, streams);
        self.single_xor_assign_u16(input, &rand_u16, idx, data_len, streams);
    }

    #[cfg(feature = "otp_encrypt")]
    fn otp_decrypt_their_rng_u16(
        &mut self,
        input: &mut CudaView<u16>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = input.len();
        assert_eq!(data_len & 1, 0);
        let mut rand = unsafe { self.devs[idx].alloc::<u32>(data_len >> 1).unwrap() };
        let rand_u16 = self.fill_their_rng_into_u16(&mut rand, idx, streams);
        self.single_xor_assign_u16(input, &rand_u16, idx, data_len, streams);
    }

    #[cfg(feature = "otp_encrypt")]
    fn otp_encrypt_my_rng_u64(
        &mut self,
        input: &ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) -> CudaSlice<u64> {
        let data_len = input.len();
        let rand_size = (data_len + 7) / 8; // Multiple of 16 u32
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(rand_size * 8).unwrap() };
        self.fill_my_rand_u64(&mut rand, idx, streams);
        self.single_xor_assign_u64(
            &mut rand.slice(..data_len),
            &input.a,
            idx,
            data_len,
            streams,
        );
        rand
    }

    #[cfg(feature = "otp_encrypt")]
    fn otp_decrypt_their_rng_u64(
        &mut self,
        inout: &mut ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let data_len = inout.len();
        let rand_size = (data_len + 7) / 8; // Multiple of 16 u32
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(rand_size * 8).unwrap() };
        self.fill_their_rand_u64(&mut rand, idx, streams);
        self.single_xor_assign_u64(
            &mut inout.b,
            &rand.slice(..data_len),
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

    #[cfg(feature = "otp_encrypt")]
    fn send_receive_view_with_offset_single_gpu(
        &mut self,
        res: &mut ChunkShareView<u64>,
        range: Range<usize>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let send_bufs =
            self.otp_encrypt_my_rng_u64(&res.get_range(range.start, range.end), idx, streams);

        result::group_start().unwrap();
        self.send(&send_bufs, self.next_id, idx, streams).unwrap();
        let mut rcv = res.b.slice(range.to_owned());
        self.receive_view(&mut rcv, self.prev_id, idx, streams)
            .unwrap();
        result::group_end().unwrap();
        self.otp_decrypt_their_rng_u64(&mut res.get_range(range.start, range.end), idx, streams);
    }

    #[cfg(not(feature = "otp_encrypt"))]
    fn send_receive_view_with_offset_single_gpu(
        &mut self,
        res: &mut ChunkShareView<u64>,
        range: Range<usize>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        result::group_start().unwrap();
        let send = res.a.slice(range.to_owned());
        self.send_view(&send, self.next_id, idx, streams).unwrap();

        let mut rcv = res.b.slice(range.to_owned());
        self.receive_view(&mut rcv, self.prev_id, idx, streams)
            .unwrap();

        result::group_end().unwrap();
    }

    #[cfg(feature = "otp_encrypt")]
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
                self.otp_encrypt_my_rng_u64(&res.get_range(range.start, range.end), idx, streams)
            })
            .collect_vec();

        result::group_start().unwrap();
        for (idx, res) in send_bufs.iter().enumerate() {
            self.send(res, self.next_id, idx, streams).unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            let mut rcv = res.b.slice(range.to_owned());
            self.receive_view(&mut rcv, self.prev_id, idx, streams)
                .unwrap();
        }
        result::group_end().unwrap();
        for (idx, res) in res.iter_mut().enumerate() {
            self.otp_decrypt_their_rng_u64(
                &mut res.get_range(range.start, range.end),
                idx,
                streams,
            );
        }
    }

    #[cfg(not(feature = "otp_encrypt"))]
    fn send_receive_view_with_offset(
        &mut self,
        res: &mut [ChunkShareView<u64>],
        range: Range<usize>,
        streams: &[CudaStream],
    ) {
        assert_eq!(res.len(), self.n_devices);

        result::group_start().unwrap();
        for (idx, res) in res.iter().enumerate() {
            let send = res.a.slice(range.to_owned());
            self.send_view(&send, self.next_id, idx, streams).unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            let mut rcv = res.b.slice(range.to_owned());
            self.receive_view(&mut rcv, self.prev_id, idx, streams)
                .unwrap();
        }
        result::group_end().unwrap();
    }

    #[cfg(feature = "otp_encrypt")]
    fn send_receive_view(&mut self, res: &mut [ChunkShareView<u64>], streams: &[CudaStream]) {
        assert_eq!(res.len(), self.n_devices);

        let send_bufs = res
            .iter()
            .enumerate()
            .map(|(idx, res)| self.otp_encrypt_my_rng_u64(res, idx, streams))
            .collect_vec();

        result::group_start().unwrap();
        for (idx, res) in send_bufs.iter().enumerate() {
            self.send(res, self.next_id, idx, streams).unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            self.receive_view(&mut res.b, self.prev_id, idx, streams)
                .unwrap();
        }
        result::group_end().unwrap();
        for (idx, res) in res.iter_mut().enumerate() {
            self.otp_decrypt_their_rng_u64(res, idx, streams);
        }
    }

    #[cfg(not(feature = "otp_encrypt"))]
    fn send_receive_view(&mut self, res: &mut [ChunkShareView<u64>], streams: &[CudaStream]) {
        assert_eq!(res.len(), self.n_devices);

        result::group_start().unwrap();
        for (idx, res) in res.iter().enumerate() {
            self.send_view(&res.a, self.next_id, idx, streams).unwrap();
        }
        for (idx, res) in res.iter_mut().enumerate() {
            self.receive_view(&mut res.b, self.prev_id, idx, streams)
                .unwrap();
        }
        result::group_end().unwrap();
    }

    #[cfg(feature = "otp_encrypt")]
    fn send_receive_view_single_gpu(
        &mut self,
        res: &mut ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        let send_bufs = self.otp_encrypt_my_rng_u64(res, idx, streams);

        result::group_start().unwrap();
        self.send(&send_bufs, self.next_id, idx, streams).unwrap();
        self.receive_view(&mut res.b, self.prev_id, idx, streams)
            .unwrap();
        result::group_end().unwrap();
        self.otp_decrypt_their_rng_u64(res, idx, streams);
    }

    #[cfg(not(feature = "otp_encrypt"))]
    fn send_receive_view_single_gpu(
        &mut self,
        res: &mut ChunkShareView<u64>,
        idx: usize,
        streams: &[CudaStream],
    ) {
        result::group_start().unwrap();
        self.send_view(&res.a, self.next_id, idx, streams).unwrap();
        self.receive_view(&mut res.b, self.prev_id, idx, streams)
            .unwrap();
        result::group_end().unwrap();
    }

    #[cfg(feature = "otp_encrypt")]
    fn single_xor_assign_u16(
        &self,
        x1: &mut CudaView<u16>,
        x2: &CudaView<u16>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .single_xor_assign_u16
                .clone()
                .launch_on_stream(&streams[idx], cfg, (&*x1, x2, size))
                .unwrap();
        }
    }

    #[cfg(feature = "otp_encrypt")]
    fn single_xor_assign_u64(
        &self,
        x1: &mut CudaView<u64>,
        x2: &CudaView<u64>,
        idx: usize,
        size: usize,
        streams: &[CudaStream],
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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
        let cfg = Self::launch_config_from_elements_and_threads(
            size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
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

    fn bit_inject_ot_sender(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let m0_ = Buffers::take_single_buffer(&mut self.buffers.single_u16_128c_1);
        let m1_ = Buffers::take_single_buffer(&mut self.buffers.single_u16_128c_2);
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

            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
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
        #[cfg(feature = "otp_encrypt")]
        {
            let m0 = m0
                .into_iter()
                .enumerate()
                .map(|(idx, m0)| self.otp_encrypt_their_rng_u16(&m0, idx, streams))
                .collect_vec();
            let m1 = m1
                .into_iter()
                .enumerate()
                .map(|(idx, m1)| self.otp_encrypt_their_rng_u16(&m1, idx, streams))
                .collect_vec();

            result::group_start().unwrap();
            for (idx, (m0, m1)) in izip!(&m0, &m1).enumerate() {
                self.send(m0, self.prev_id, idx, streams).unwrap();
                self.send(m1, self.prev_id, idx, streams).unwrap();
            }
            result::group_end().unwrap();
        }
        #[cfg(not(feature = "otp_encrypt"))]
        {
            result::group_start().unwrap();
            for (idx, (m0, m1)) in izip!(&m0, &m1).enumerate() {
                self.send_view_u16(m0, self.prev_id, idx, streams).unwrap();
                self.send_view_u16(m1, self.prev_id, idx, streams).unwrap();
            }
            result::group_end().unwrap();
        }

        Buffers::return_single_buffer(&mut self.buffers.single_u16_128c_1, m0_);
        Buffers::return_single_buffer(&mut self.buffers.single_u16_128c_2, m1_);
    }

    fn bit_inject_ot_receiver(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let m0_ = Buffers::take_single_buffer(&mut self.buffers.single_u16_128c_1);
        let m1_ = Buffers::take_single_buffer(&mut self.buffers.single_u16_128c_2);
        let wc_ = Buffers::take_single_buffer(&mut self.buffers.single_u16_128c_3);
        let mut m0 = Buffers::get_single_buffer_chunk(&m0_, self.chunk_size * 128);
        let mut m1 = Buffers::get_single_buffer_chunk(&m1_, self.chunk_size * 128);
        let mut wc = Buffers::get_single_buffer_chunk(&wc_, self.chunk_size * 128);

        #[cfg(feature = "otp_encrypt")]
        let mut send = Vec::with_capacity(inp.len());

        result::group_start().unwrap();
        for (idx, (m0, m1, wc)) in izip!(&mut m0, &mut m1, &mut wc).enumerate() {
            self.receive_view_u16(m0, self.next_id, idx, streams)
                .unwrap();
            self.receive_view_u16(wc, self.prev_id, idx, streams)
                .unwrap();
            self.receive_view_u16(m1, self.next_id, idx, streams)
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

            // OTP decrypt
            #[cfg(feature = "otp_encrypt")]
            {
                self.otp_decrypt_my_rng_u16(m0, idx, streams);
                self.otp_decrypt_their_rng_u16(wc, idx, streams);
                self.otp_decrypt_my_rng_u16(m1, idx, streams);
            }

            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
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
            #[cfg(feature = "otp_encrypt")]
            send.push(self.otp_encrypt_their_rng_u16(&res.b, idx, streams));
        }

        // Reshare to Helper
        result::group_start().unwrap();
        #[cfg(feature = "otp_encrypt")]
        {
            for (idx, send) in send.iter().enumerate() {
                self.send(send, self.prev_id, idx, streams).unwrap();
            }
        }
        #[cfg(not(feature = "otp_encrypt"))]
        {
            for (idx, res) in outp.iter().enumerate() {
                self.send_view_u16(&res.b, self.prev_id, idx, streams)
                    .unwrap();
            }
        }
        result::group_end().unwrap();

        Buffers::return_single_buffer(&mut self.buffers.single_u16_128c_1, m0_);
        Buffers::return_single_buffer(&mut self.buffers.single_u16_128c_2, m1_);
        Buffers::return_single_buffer(&mut self.buffers.single_u16_128c_3, wc_);
    }

    fn bit_inject_ot_helper(
        &mut self,
        inp: &[ChunkShareView<u64>],
        outp: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        let wc_ = Buffers::take_single_buffer(&mut self.buffers.single_u16_128c_3);
        let wc = Buffers::get_single_buffer_chunk(&wc_, self.chunk_size * 128);

        #[cfg(feature = "otp_encrypt")]
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

            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
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
            #[cfg(feature = "otp_encrypt")]
            send.push(self.otp_encrypt_my_rng_u16(wc, idx, streams));
        }

        result::group_start().unwrap();
        #[cfg(feature = "otp_encrypt")]
        {
            for (idx, send) in send.iter().enumerate() {
                self.send(send, self.next_id, idx, streams).unwrap();
            }
        }
        #[cfg(not(feature = "otp_encrypt"))]
        {
            for (idx, wc) in wc.iter().enumerate() {
                self.send_view_u16(wc, self.next_id, idx, streams).unwrap();
            }
        }
        result::group_end().unwrap();
        result::group_start().unwrap();
        for (idx, res) in outp.iter_mut().enumerate() {
            self.receive_view_u16(&mut res.a, self.next_id, idx, streams)
                .unwrap();
        }
        result::group_end().unwrap();
        // OTP decrypt
        #[cfg(feature = "otp_encrypt")]
        {
            for (idx, res) in outp.iter_mut().enumerate() {
                self.otp_decrypt_my_rng_u16(&mut res.a, idx, streams);
            }
        }

        Buffers::return_single_buffer(&mut self.buffers.single_u16_128c_3, wc_);
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

    fn transpose_pack_u16_with_len(
        &mut self,
        inp: &[ChunkShare<u16>],
        outp: &mut [ChunkShareView<u64>],
        bitlen: usize,
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, inp.len());
        assert_eq!(self.n_devices, outp.len());

        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * 2,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        for (idx, (inp, outp)) in izip!(inp, outp).enumerate() {
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

        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * 2,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        for (idx, (inp, outp)) in izip!(inp, outp).enumerate() {
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
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * 64,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        // K = 16 is hardcoded in the kernel
        for (idx, (x1, x2, x3)) in izip!(inout1, out2, out3).enumerate() {
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
        inp: &[ChunkShare<u16>],
        lifted: &mut [ChunkShareView<u32>],
        inout1: &mut [ChunkShareView<u64>],
        out2: &mut [ChunkShareView<u64>],
        out3: &mut [ChunkShareView<u64>],
        streams: &[CudaStream],
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * 64,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        // K = 16 is hardcoded in the kernel
        for (idx, (inp, lifted, x1, x2, x3)) in izip!(inp, lifted, inout1, out2, out3).enumerate() {
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
        code: &[ChunkShare<u16>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, mask_lifted.len());
        assert_eq!(self.n_devices, mask_correction.len());
        assert_eq!(self.n_devices, code.len());

        for (idx, (m, mc, c)) in izip!(mask_lifted, mask_correction, code).enumerate() {
            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64,
                DEFAULT_LAUNCH_CONFIG_THREADS,
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
                            self.chunk_size as u32 * 64,
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
        shares: &[ChunkShare<u16>],
        xa: &mut [ChunkShareView<u32>],
        injected: &mut [ChunkShareView<u16>],
        streams: &[CudaStream],
    ) {
        const K: usize = 16;
        let mut x1 = Vec::with_capacity(self.n_devices);
        let mut x2 = Vec::with_capacity(self.n_devices);
        let mut x3 = Vec::with_capacity(self.n_devices);
        let mut c = Vec::with_capacity(self.n_devices);
        // No subbuffer taken here, since we extract it manually
        let buffer1 = Buffers::take_buffer(&mut self.buffers.u64_32c_1);
        let buffer2 = Buffers::take_buffer(&mut self.buffers.u64_32c_2);
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

        Buffers::return_buffer(&mut self.buffers.u64_32c_1, buffer1);
        Buffers::return_buffer(&mut self.buffers.u64_32c_2, buffer2);
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
        const K: usize = 16;
        assert_eq!(self.n_devices, c.len());
        assert_eq!(self.n_devices, x1.len());
        assert_eq!(self.n_devices, x2.len());
        assert_eq!(self.n_devices, x3.len());

        // Reuse buffer
        let mut s = Vec::with_capacity(self.n_devices);
        let mut carry = Vec::with_capacity(self.n_devices);
        // No subbuffer taken here, since we extract it manually
        let buffer1 = Buffers::take_buffer(&mut self.buffers.u64_32c_3);

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

        Buffers::return_buffer(&mut self.buffers.u64_32c_3, buffer1);
    }

    pub fn extract_msb(&mut self, x: &mut [ChunkShareView<u32>], streams: &[CudaStream]) {
        let x1_ = Buffers::take_buffer(&mut self.buffers.u64_32c_1);
        let x2_ = Buffers::take_buffer(&mut self.buffers.u64_32c_2);
        let x3_ = Buffers::take_buffer(&mut self.buffers.u64_32c_3);
        let mut x1 = Buffers::get_buffer_chunk(&x1_, 32 * self.chunk_size);
        let mut x2 = Buffers::get_buffer_chunk(&x2_, 32 * self.chunk_size);
        let mut x3 = Buffers::get_buffer_chunk(&x3_, 32 * self.chunk_size);

        self.transpose_pack_u32_with_len(x, &mut x1, Self::BITS, streams);
        self.split(&mut x1, &mut x2, &mut x3, Self::BITS, streams);
        self.binary_add_3_get_msb(&mut x1, &mut x2, &mut x3, streams);

        Buffers::return_buffer(&mut self.buffers.u64_32c_1, x1_);
        Buffers::return_buffer(&mut self.buffers.u64_32c_2, x2_);
        Buffers::return_buffer(&mut self.buffers.u64_32c_3, x3_);
    }

    // K is Self::BITS = 16 + B_BITS in our case
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

        let s_ = Buffers::take_buffer(&mut self.buffers.u64_31c_1);
        let carry_ = Buffers::take_buffer(&mut self.buffers.u64_31c_2);
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

        Buffers::return_buffer(&mut self.buffers.u64_31c_1, s_);
        Buffers::return_buffer(&mut self.buffers.u64_31c_2, carry_);

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

    fn collect_graphic_result(&mut self, bits: &mut [ChunkShareView<u64>], streams: &[CudaStream]) {
        assert!(self.n_devices <= self.chunk_size);
        let dev0 = &self.devs[0];
        let stream0 = &streams[0];
        let bit0 = &bits[0];

        // Get results onto CPU
        let mut a = Vec::with_capacity(self.n_devices - 1);
        let mut b = Vec::with_capacity(self.n_devices - 1);
        for (dev, stream, bit) in izip!(self.get_devices(), streams, bits.iter()).skip(1) {
            let src = bit.get_range(0, 1);

            let mut a_ = dtoh_on_stream_sync(&src.a, &dev, stream).unwrap();
            let mut b_ = dtoh_on_stream_sync(&src.b, &dev, stream).unwrap();

            a.push(a_.pop().unwrap());
            b.push(b_.pop().unwrap());
        }

        // Put results onto first GPU
        let mut des = bit0.get_range(1, self.n_devices);
        let a = htod_on_stream_sync(&a, dev0, stream0).unwrap();
        let b = htod_on_stream_sync(&b, dev0, stream0).unwrap();
        let c = ChunkShare::new(a, b);

        self.assign_view(&mut des, &c.as_view(), 0, streams);
    }

    fn collapse_u64(&mut self, input: &mut ChunkShare<u64>, streams: &[CudaStream]) {
        let mut res = input.get_offset(0, 1);
        let helper = input.get_offset(1, 1);

        let cfg = Self::launch_config_from_elements_and_threads(1, DEFAULT_LAUNCH_CONFIG_THREADS);

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

    // input should be of size: n_devices * input_size
    // Result is in the first bit of the result buffer
    pub fn compare_threshold_masked_many(
        &mut self,
        code_dots: &[ChunkShare<u16>],
        mask_dots: &[ChunkShare<u16>],
        streams: &[CudaStream],
    ) {
        assert_eq!(self.n_devices, code_dots.len());
        assert_eq!(self.n_devices, mask_dots.len());
        for chunk in code_dots.iter().chain(mask_dots.iter()) {
            assert!(chunk.len() % 64 == 0);
        }

        let x_ = Buffers::take_buffer(&mut self.buffers.u32_64c_1);
        let corrections_ = Buffers::take_buffer(&mut self.buffers.u16_128c_1);
        let mut x = Buffers::get_buffer_chunk(&x_, 64 * self.chunk_size);
        let mut corrections = Buffers::get_buffer_chunk(&corrections_, 128 * self.chunk_size);

        self.lift_mpc(mask_dots, &mut x, &mut corrections, streams);
        self.lift_mul_sub(&mut x, &corrections, code_dots, streams);
        self.extract_msb(&mut x, streams);

        Buffers::return_buffer(&mut self.buffers.u32_64c_1, x_);
        Buffers::return_buffer(&mut self.buffers.u16_128c_1, corrections_);
        self.buffers.check_buffers();

        // Result is in the first bit of the result buffer
    }

    // input should be of size: n_devices * input_size
    // Result is in the lowest bit of the result buffer on the first gpu
    pub fn compare_threshold_masked_many_with_or_tree(
        &mut self,
        code_dots: &[ChunkShare<u16>],
        mask_dots: &[ChunkShare<u16>],
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
}
