use crate::{
    http_root, rng::chacha_corr::ChaChaCudaCorrRng, setup::shamir::P,
    threshold_ring::cuda::PTX_SRC, IdWrapper,
};
use axum::{routing::get, Router};
use cudarc::{
    driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtr, DeviceSlice,
        LaunchAsync, LaunchConfig,
    },
    nccl::{result, Comm, Id},
    nvrtc::{self, Ptx},
};
use itertools::izip;
use std::{
    rc::Rc,
    str::FromStr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub(crate) const B_BITS: usize = 20;
pub(crate) const P2K: u64 = (P as u64) << B_BITS;

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

    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.a.is_empty(), self.b.is_empty());
        self.a.is_empty()
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.a.len(), self.b.len());
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
        debug_assert_eq!(self.a.is_empty(), self.b.is_empty());
        self.a.is_empty()
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.a.len(), self.b.len());
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
    pub(crate) not_inplace: CudaFunction,
    pub(crate) not: CudaFunction,
    pub(crate) transpose_64x64: CudaFunction,
    pub(crate) transpose_32x64: CudaFunction,
    pub(crate) ot_sender: CudaFunction,
    pub(crate) ot_receiver: CudaFunction,
    pub(crate) ot_helper: CudaFunction,
    pub(crate) assign: CudaFunction,
    pub(crate) collapse_u64_helper: CudaFunction,
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
                "shared_or_pre_assign",
                "shared_not_inplace",
                "shared_not",
                "shared_u64_transpose_pack_u64",
                "shared_u32_transpose_pack_u64",
                "packed_ot_sender",
                "packed_ot_receiver",
                "packed_ot_helper",
                "shared_assign",
                "collapse_u64_helper",
            ],
        )
        .unwrap();
        let and = dev.get_func(Self::MOD_NAME, "shared_and_pre").unwrap();
        let or_assign = dev
            .get_func(Self::MOD_NAME, "shared_or_pre_assign")
            .unwrap();
        let xor = dev.get_func(Self::MOD_NAME, "shared_xor").unwrap();
        let xor_assign = dev.get_func(Self::MOD_NAME, "shared_xor_assign").unwrap();
        let not_inplace = dev.get_func(Self::MOD_NAME, "shared_not_inplace").unwrap();
        let not = dev.get_func(Self::MOD_NAME, "shared_not").unwrap();
        let transpose_64x64 = dev
            .get_func(Self::MOD_NAME, "shared_u64_transpose_pack_u64")
            .unwrap();
        let transpose_32x64 = dev
            .get_func(Self::MOD_NAME, "shared_u32_transpose_pack_u64")
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
            not_inplace,
            not,
            transpose_64x64,
            transpose_32x64,
            ot_sender,
            ot_receiver,
            ot_helper,
            assign,
            collapse_u64_helper,
        }
    }
}

struct Buffers {
    single_u32_128c_1: Option<Vec<CudaSlice<u32>>>,
    single_u32_128c_2: Option<Vec<CudaSlice<u32>>>,
    single_u32_128c_3: Option<Vec<CudaSlice<u32>>>,
}

impl Buffers {
    fn new(devices: &[Arc<CudaDevice>], chunk_size: usize) -> Self {
        let single_u32_128c_1 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));
        let single_u32_128c_2 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));
        let single_u32_128c_3 = Some(Self::allocate_single_buffer(chunk_size * 128, devices));

        Buffers {
            single_u32_128c_1,
            single_u32_128c_2,
            single_u32_128c_3,
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
        debug_assert!(inp.is_some());
        std::mem::take(inp).unwrap()
    }

    fn return_buffer<T>(des: &mut Option<Vec<ChunkShare<T>>>, src: Vec<ChunkShare<T>>) {
        debug_assert!(des.is_none());
        *des = Some(src);
    }

    fn take_single_buffer<T>(inp: &mut Option<Vec<CudaSlice<T>>>) -> Vec<CudaSlice<T>> {
        debug_assert!(inp.is_some());
        std::mem::take(inp).unwrap()
    }

    fn return_single_buffer<T>(des: &mut Option<Vec<CudaSlice<T>>>, src: Vec<CudaSlice<T>>) {
        debug_assert!(des.is_none());
        *des = Some(src);
    }

    fn check_buffers(&self) {
        debug_assert!(self.single_u32_128c_1.is_some());
        debug_assert!(self.single_u32_128c_2.is_some());
        debug_assert!(self.single_u32_128c_3.is_some());
    }
}

pub struct Circuits {
    peer_id: usize,
    next_id: usize,
    prev_id: usize,
    chunk_size: usize,
    n_devices: usize,
    devs: Vec<Arc<CudaDevice>>,
    comms: Vec<Rc<Comm>>,
    kernels: Vec<Kernels>,
    buffers: Buffers,
    rngs: Vec<ChaChaCudaCorrRng>,
}

impl Circuits {
    const BITS: usize = 16 + B_BITS as usize;

    pub fn synchronize_all(&self) {
        for dev in self.devs.iter() {
            dev.synchronize().unwrap();
        }
    }

    pub fn new(
        peer_id: usize,
        input_size: usize, // per GPU
        peer_url: Option<&String>,
        server_port: Option<u16>,
    ) -> Self {
        // For the transpose, inputs should be multiple of 64 bits
        debug_assert!(input_size % 64 == 0);
        // Chunk size is the number of u64 elements per bit in the binary circuits
        let chunk_size = input_size / 64;

        debug_assert_eq!(Self::ceil_log2(P2K as usize), Self::BITS);
        let n_devices = CudaDevice::count().unwrap() as usize;

        let mut devs = Vec::with_capacity(n_devices);
        let mut kernels = Vec::with_capacity(n_devices);
        let mut rngs = Vec::with_capacity(n_devices);

        let ptx = nvrtc::compile_ptx(PTX_SRC).unwrap();
        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let kernel = Kernels::new(dev.clone(), ptx.clone());
            // TODO seeds are not random yet :)
            let rng = ChaChaCudaCorrRng::init(
                dev.clone(),
                [peer_id as u32; 8],
                [((peer_id + 2) % 3) as u32; 8],
            );

            devs.push(dev);
            kernels.push(kernel);
            rngs.push(rng);
        }

        let comms = Self::instantiate_network(peer_id, peer_url, server_port, &devs);

        let buffers = Buffers::new(&devs, chunk_size);

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

    // pub fn take_result_buffer(&mut self) -> Vec<ChunkShare<u64>> {
    //     Buffers::take_buffer(&mut self.buffers.u64_37c_1)
    // }

    // pub fn return_result_buffer(&mut self, src: Vec<ChunkShare<u64>>) {
    //     Buffers::return_buffer(&mut self.buffers.u64_37c_1, src);
    // }

    pub fn instantiate_network(
        peer_id: usize,
        peer_url: Option<&String>,
        server_port: Option<u16>,
        devices: &[Arc<CudaDevice>],
    ) -> Vec<Rc<Comm>> {
        let n_devices = devices.len();
        let mut comms = Vec::with_capacity(n_devices);
        let mut ids = Vec::with_capacity(n_devices);
        for _ in 0..n_devices {
            ids.push(Id::new().unwrap());
        }

        // Start HTTP server to exchange NCCL commIds
        if peer_id == 0 {
            let ids = ids.clone();
            tokio::spawn(async move {
                println!("Starting server on port {}...", server_port.unwrap());
                let app = Router::new().route("/:device_id", get(move |req| http_root(ids, req)));
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
                thread::sleep(Duration::from_secs(1));

                let res = reqwest::blocking::get(format!(
                    "http://{}:{}/{}",
                    peer_url.unwrap(),
                    server_port.unwrap(),
                    i
                ))
                .unwrap();
                IdWrapper::from_str(&res.text().unwrap()).unwrap().0
            };
            ids.push(id);

            // Bind to thread (important!)
            devices[i].bind_to_thread().unwrap();
            comms.push(Rc::new(
                Comm::from_rank(devices[i].clone(), peer_id, 3, id).unwrap(),
            ));
        }
        comms
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
            grid_dim: (num_blocks, 1, 1),
            block_dim: (t, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn send_view<T>(&mut self, send: &CudaView<T>, peer_id: usize, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        // Copied from cudarc nccl implementation
        unsafe {
            result::send(
                *send.device_ptr() as *mut _,
                send.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comms[idx].comm,
                *self.comms[idx].device.cu_stream() as *mut _,
            )
        }
        .unwrap();
    }

    pub fn receive_view<T>(&mut self, receive: &mut CudaView<T>, peer_id: usize, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        // Copied from cudarc nccl implementation
        unsafe {
            result::recv(
                *receive.device_ptr() as *mut _,
                receive.len(),
                T::as_nccl_type(),
                peer_id as i32,
                self.comms[idx].comm,
                *self.comms[idx].device.cu_stream() as *mut _,
            )
        }
        .unwrap();
    }

    pub fn send<T>(&mut self, send: &CudaSlice<T>, peer_id: usize, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        self.comms[idx].send(send, peer_id as i32).unwrap();
    }

    pub fn receive<T>(&mut self, receive: &mut CudaSlice<T>, peer_id: usize, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        self.comms[idx].recv(receive, peer_id as i32).unwrap();
    }

    // Fill randomness using the correlated RNG
    fn fill_rand_u64(&mut self, rand: &mut CudaSlice<u64>, idx: usize) {
        let rng = &mut self.rngs[idx];
        let mut rand_trans: CudaViewMut<u32> =
        // the transmute_mut is safe because we know that one u64 is 2 u32s, and the buffer is aligned properly for the transmute
            unsafe { rand.transmute_mut(rand.len() * 2).unwrap() };
        rng.fill_rng_into(&mut rand_trans);
    }

    fn packed_and_many_pre(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareViewMut<u64>,
        bits: usize,
        idx: usize,
    ) {
        // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(self.chunk_size * bits).unwrap() };
        self.fill_rand_u64(&mut rand, idx);

        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .and
                .clone()
                .launch(
                    cfg,
                    (
                        &mut res.a,
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

    // Keep in mind: group needs to be open!
    fn packed_and_many_send(&mut self, res: &ChunkShareView<u64>, bits: usize, idx: usize) {
        let send = res.a.slice(0..bits * self.chunk_size);
        self.send_view(&send, self.next_id, idx);
    }

    // Keep in mind: group needs to be open!
    fn packed_and_many_receive(&mut self, res: &mut ChunkShareView<u64>, bits: usize, idx: usize) {
        let mut rcv = res.b.slice(0..bits * self.chunk_size);
        self.receive_view(&mut rcv, self.prev_id, idx);
    }

    fn assign_view(
        &mut self,
        des: &mut ChunkShareView<u64>,
        src: &ChunkShareView<u64>,
        idx: usize,
    ) {
        debug_assert_eq!(src.len(), des.len());
        let cfg = Self::launch_config_from_elements_and_threads(
            src.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .assign
                .clone()
                .launch(cfg, (&des.a, &des.b, &src.a, &src.b, src.len() as i32))
                .unwrap();
        }
    }

    fn and_many_pre(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        idx: usize,
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(self.chunk_size).unwrap() };
        self.fill_rand_u64(&mut rand, idx);

        unsafe {
            self.kernels[idx]
                .and
                .clone()
                .launch(
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
    ) {
        // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
        let size = (x1.len() + 15) / 16;
        let mut rand = unsafe { self.devs[idx].alloc::<u64>(size * 16).unwrap() };
        self.fill_rand_u64(&mut rand, idx);

        let cfg = Self::launch_config_from_elements_and_threads(
            x1.len() as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .or_assign
                .clone()
                .launch(cfg, (&x1.a, &x1.b, &x2.a, &x2.b, &rand, x1.len()))
                .unwrap();
        }
    }

    fn send_receive(&mut self, res: &mut [ChunkShare<u64>]) {
        debug_assert_eq!(res.len(), self.n_devices);

        result::group_start().unwrap();
        for (idx, res) in res.iter().enumerate() {
            self.send(&res.a, self.next_id, idx);
        }
        for (idx, res) in res.iter_mut().enumerate() {
            self.receive(&mut res.b, self.prev_id, idx);
        }
        result::group_end().unwrap();
    }

    fn send_receive_view(&mut self, res: &mut [ChunkShareView<u64>]) {
        debug_assert_eq!(res.len(), self.n_devices);

        result::group_start().unwrap();
        for (idx, res) in res.iter().enumerate() {
            self.send_view(&res.a, self.next_id, idx);
        }
        for (idx, res) in res.iter_mut().enumerate() {
            self.receive_view(&mut res.b, self.prev_id, idx);
        }
        result::group_end().unwrap();
    }

    fn xor_assign_many(&self, x1: &mut ChunkShareView<u64>, x2: &ChunkShareView<u64>, idx: usize) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .xor_assign
                .clone()
                .launch(cfg, (&x1.a, &x1.b, &x2.a, &x2.b, self.chunk_size))
                .unwrap();
        }
    }

    fn packed_xor_assign_many(
        &self,
        x1: &mut ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        bits: usize,
        idx: usize,
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .xor_assign
                .clone()
                .launch(
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
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * bits as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .xor
                .clone()
                .launch(
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

    fn xor_many(
        &self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        idx: usize,
    ) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .xor
                .clone()
                .launch(
                    cfg,
                    (
                        &res.a,
                        &res.b,
                        &x1.a,
                        &x1.b,
                        &x2.a,
                        &x2.b,
                        self.chunk_size as i32,
                    ),
                )
                .unwrap();
        }
    }

    fn not_many(&self, x: &ChunkShareView<u64>, res: &mut ChunkShareView<u64>, idx: usize) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .not
                .clone()
                .launch(cfg, (&res.a, &res.b, &x.a, &x.b, self.chunk_size as i32))
                .unwrap();
        }
    }

    fn not_inplace_many(&self, x: &mut ChunkShareView<u64>, idx: usize) {
        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        unsafe {
            self.kernels[idx]
                .not_inplace
                .clone()
                .launch(cfg, (&x.a, &x.b, self.chunk_size as i32))
                .unwrap();
        }
    }

    pub fn allocate_buffer<T>(&self, size: usize) -> Vec<ChunkShare<T>>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        Buffers::allocate_buffer(size, &self.devs)
    }

    fn ceil_log2(x: usize) -> usize {
        let mut y = 0;
        let mut x = x - 1;
        while x > 0 {
            x >>= 1;
            y += 1;
        }
        y
    }

    fn bit_inject_ot_sender(&mut self, inp: &[ChunkShare<u64>], outp: &mut [ChunkShare<u32>]) {
        let mut m0 = Buffers::take_single_buffer(&mut self.buffers.single_u32_128c_1);
        let mut m1 = Buffers::take_single_buffer(&mut self.buffers.single_u32_128c_2);

        for (idx, (inp, res, m0, m1)) in izip!(inp, outp, &mut m0, &mut m1).enumerate() {
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_ca = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_my_rng_into(&mut rand_ca.slice_mut(..));
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_cb = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_their_rng_into(&mut rand_cb.slice_mut(..));
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_wa1 = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_my_rng_into(&mut rand_wa1.slice_mut(..));
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_wa2 = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_my_rng_into(&mut rand_wa2.slice_mut(..));

            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
            );

            unsafe {
                self.kernels[idx]
                    .ot_sender
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut res.a,
                            &mut res.b,
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

        result::group_start().unwrap();
        for (idx, (m0, m1)) in izip!(&m0, &m1).enumerate() {
            self.send(m0, self.prev_id, idx);
            self.send(m1, self.prev_id, idx);
        }
        result::group_end().unwrap();

        Buffers::return_single_buffer(&mut self.buffers.single_u32_128c_1, m0);
        Buffers::return_single_buffer(&mut self.buffers.single_u32_128c_2, m1);
    }

    fn bit_inject_ot_receiver(&mut self, inp: &[ChunkShare<u64>], outp: &mut [ChunkShare<u32>]) {
        let mut m0 = Buffers::take_single_buffer(&mut self.buffers.single_u32_128c_1);
        let mut m1 = Buffers::take_single_buffer(&mut self.buffers.single_u32_128c_2);
        let mut wc = Buffers::take_single_buffer(&mut self.buffers.single_u32_128c_3);

        result::group_start().unwrap();
        for (idx, (m0, m1, wc)) in izip!(&mut m0, &mut m1, &mut wc).enumerate() {
            self.receive(m0, self.next_id, idx);
            self.receive(wc, self.prev_id, idx);
            self.receive(m1, self.next_id, idx);
        }
        result::group_end().unwrap();

        for (idx, (inp, res, m0, m1, wc)) in izip!(inp, outp.iter_mut(), &m0, &m1, &wc).enumerate()
        {
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_ca = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_my_rng_into(&mut rand_ca.slice_mut(..));

            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
            );

            unsafe {
                self.kernels[idx]
                    .ot_receiver
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut res.a,
                            &mut res.b,
                            &inp.b,
                            m0,
                            m1,
                            &rand_ca,
                            wc,
                            2 * self.chunk_size,
                        ),
                    )
                    .unwrap();
            }
        }

        // Reshare to Helper
        result::group_start().unwrap();
        for (idx, res) in outp.iter().enumerate() {
            self.send(&res.b, self.prev_id, idx);
        }
        result::group_end().unwrap();

        Buffers::return_single_buffer(&mut self.buffers.single_u32_128c_1, m0);
        Buffers::return_single_buffer(&mut self.buffers.single_u32_128c_2, m1);
        Buffers::return_single_buffer(&mut self.buffers.single_u32_128c_3, wc);
    }

    fn bit_inject_ot_helper(&mut self, inp: &[ChunkShare<u64>], outp: &mut [ChunkShare<u32>]) {
        let mut wc = Buffers::take_single_buffer(&mut self.buffers.single_u32_128c_3);

        for (idx, (inp, res, wc)) in izip!(inp, outp.iter_mut(), &mut wc).enumerate() {
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_cb = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_their_rng_into(&mut rand_cb.slice_mut(..));
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_wb1 = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_their_rng_into(&mut rand_wb1.slice_mut(..));
            // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
            let mut rand_wb2 = unsafe {
                self.devs[idx]
                    .alloc::<u32>(self.chunk_size * 2 * 64)
                    .unwrap()
            };
            self.rngs[idx].fill_their_rng_into(&mut rand_wb2.slice_mut(..));

            let cfg = Self::launch_config_from_elements_and_threads(
                self.chunk_size as u32 * 64 * 2,
                DEFAULT_LAUNCH_CONFIG_THREADS,
            );

            unsafe {
                self.kernels[idx]
                    .ot_helper
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut res.b,
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
        }

        result::group_start().unwrap();
        for (idx, wc) in wc.iter().enumerate() {
            self.send(wc, self.next_id, idx);
        }
        result::group_end().unwrap();
        result::group_start().unwrap();
        for (idx, res) in outp.iter_mut().enumerate() {
            self.receive(&mut res.a, self.next_id, idx);
        }
        result::group_end().unwrap();

        Buffers::return_single_buffer(&mut self.buffers.single_u32_128c_3, wc);
    }

    pub fn bit_inject_ot(&mut self, inp: &[ChunkShare<u64>], outp: &mut [ChunkShare<u32>]) {
        match self.peer_id {
            0 => self.bit_inject_ot_helper(inp, outp),
            1 => self.bit_inject_ot_receiver(inp, outp),
            2 => self.bit_inject_ot_sender(inp, outp),
            _ => unreachable!(),
        }
    }

    fn transpose_pack_u32_with_len(
        &mut self,
        inp: &[ChunkShare<u32>],
        outp: &mut [ChunkShareView<u64>],
        bitlen: usize,
    ) {
        debug_assert_eq!(self.n_devices, inp.len());
        debug_assert_eq!(self.n_devices, outp.len());

        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * 2,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        for (idx, (inp, outp)) in izip!(inp, outp).enumerate() {
            unsafe {
                self.kernels[idx]
                    .transpose_32x64
                    .clone()
                    .launch(
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

    fn transpose_pack_u64_with_len(
        &mut self,
        inp: &[ChunkShare<u64>],
        outp: &mut [ChunkShare<u64>],
        bitlen: usize,
    ) {
        debug_assert_eq!(self.n_devices, inp.len());
        debug_assert_eq!(self.n_devices, outp.len());

        let cfg = Self::launch_config_from_elements_and_threads(
            self.chunk_size as u32 * 2,
            DEFAULT_LAUNCH_CONFIG_THREADS,
        );

        for (idx, (inp, outp)) in izip!(inp, outp).enumerate() {
            unsafe {
                self.kernels[idx]
                    .transpose_64x64
                    .clone()
                    .launch(
                        cfg,
                        (
                            &mut outp.a,
                            &mut outp.b,
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

    // Input has size ChunkSize
    // Result is in lowest u64 of the input
    fn or_tree_on_gpus(&mut self, bits: &mut [ChunkShareView<u64>]) {
        debug_assert_eq!(self.n_devices, bits.len());
        debug_assert!(self.chunk_size <= bits[0].len());

        let mut num = self.chunk_size;
        while num > 1 {
            let mod_ = num & 1;
            num >>= 1;

            for (idx, bit) in bits.iter().enumerate() {
                let mut a = bit.get_offset(0, num);
                let b = bit.get_offset(1, num);
                self.or_many_pre_assign(&mut a, &b, idx);
                if mod_ != 0 {
                    let src = bit.get_offset(2 * num, 1);
                    let mut des = bit.get_offset(num, 1);
                    self.assign_view(&mut des, &src, idx);
                }
            }

            // Reshare
            result::group_start().unwrap();
            for (idx, bit) in bits.iter().enumerate() {
                let a = bit.get_offset(0, num);
                self.send_view(&a.a, self.next_id, idx);
            }
            for (idx, bit) in bits.iter_mut().enumerate() {
                let mut a = bit.get_offset(0, num);
                self.receive_view(&mut a.b, self.prev_id, idx);
            }
            result::group_end().unwrap();

            num += mod_;
        }
    }

    // Same as or_tree_on_gpus, but on one GPU only
    // Result is in lowest u64 of the input
    fn or_tree_on_gpu(&mut self, bits: &mut [ChunkShareView<u64>], size: usize, idx: usize) {
        debug_assert_eq!(self.n_devices, bits.len());
        debug_assert!(size <= bits[idx].len());

        let bit = &mut bits[idx];

        let mut num = size;
        while num > 1 {
            let mod_ = num & 1;
            num >>= 1;

            let mut a = bit.get_offset(0, num);
            let b = bit.get_offset(1, num);
            self.or_many_pre_assign(&mut a, &b, idx);
            if mod_ != 0 {
                let src = bit.get_offset(2 * num, 1);
                let mut des = bit.get_offset(num, 1);
                self.assign_view(&mut des, &src, idx);
            }

            // Reshare
            result::group_start().unwrap();
            let mut a = bit.get_offset(0, num);
            self.send_view(&a.a, self.next_id, idx);
            self.receive_view(&mut a.b, self.prev_id, idx);
            result::group_end().unwrap();

            num += mod_;
        }
    }

    fn collect_graphic_result(&mut self, bits: &mut [ChunkShareView<u64>]) {
        debug_assert!(self.n_devices <= self.chunk_size);
        let dev0 = &self.devs[0];
        let bit0 = &bits[0];

        // Get results onto CPU
        let mut a = Vec::with_capacity(self.n_devices - 1);
        let mut b = Vec::with_capacity(self.n_devices - 1);
        for (dev, bit) in izip!(self.get_devices(), bits.iter()).skip(1) {
            let src = bit.get_range(0, 1);

            let mut a_ = dev.dtoh_sync_copy(&src.a).unwrap();
            let mut b_ = dev.dtoh_sync_copy(&src.b).unwrap();

            a.push(a_.pop().unwrap());
            b.push(b_.pop().unwrap());
        }

        // Put results onto first GPU
        let mut des = bit0.get_range(1, self.n_devices);
        let a = dev0.htod_sync_copy(&a).unwrap();
        let b = dev0.htod_sync_copy(&b).unwrap();
        let c = ChunkShare::new(a, b);

        self.assign_view(&mut des, &c.as_view(), 0);
    }

    fn collapse_u64(&mut self, input: &mut ChunkShare<u64>) {
        let mut res = input.get_offset(0, 1);
        let helper = input.get_offset(1, 1);

        let cfg = Self::launch_config_from_elements_and_threads(1, DEFAULT_LAUNCH_CONFIG_THREADS);

        // SAFETY: Only unsafe because memory is not initialized. But, we fill afterwards.
        let mut rand = unsafe { self.devs[0].alloc::<u64>(16).unwrap() }; // minimum size is 16 for RNG, need only 10 though
        self.fill_rand_u64(&mut rand, 0);

        let mut rand_offset = rand.slice(..);

        let mut current_bitsize = 64;
        while current_bitsize > 1 {
            current_bitsize >>= 1;
            unsafe {
                self.kernels[0]
                    .collapse_u64_helper
                    .clone()
                    .launch(
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
            result::group_start().unwrap();
            self.send_view(&res.a, self.next_id, 0);
            self.receive_view(&mut res.b, self.prev_id, 0);
            result::group_end().unwrap();
        }
    }

    // Result is in the first bit of the first GPU
    pub fn or_reduce_result(&mut self, result: &mut [ChunkShare<u64>]) {
        let mut bits = Vec::with_capacity(self.n_devices);
        for r in result.iter() {
            // Result is in the first bit of the input
            bits.push(r.get_offset(0, self.chunk_size));
        }

        self.or_tree_on_gpus(&mut bits);
        if self.n_devices > 1 {
            // We have to collaps to one GPU
            self.collect_graphic_result(&mut bits);
            self.or_tree_on_gpu(&mut bits, self.n_devices, 0);
        }

        // Result is in lowest u64 bits on the first GPU
        self.collapse_u64(&mut result[0]);
        // Result is in the first bit of the first GPU
    }

    // input should be of size: n_devices * input_size
    // Result is in the first bit of res u64_37c_1
    pub fn compare_threshold_masked_many(
        &mut self,
        code_dots: Vec<ChunkShare<u16>>,
        mask_dots: Vec<ChunkShare<u16>>,
    ) {
        // debug_assert_eq!(self.n_devices, code_dots.len());
        // debug_assert_eq!(self.n_devices, mask_dots.len());

        // let mut x2 = Buffers::take_buffer(&mut self.buffers.u64_64c_1);
        // let mut x01 = Buffers::take_buffer(&mut self.buffers.u64_64c_2);
        // let mut corrections = Buffers::take_buffer(&mut self.buffers.u32_128c_1);

        // self.lift_p2k(mask_dots, &mut x2, &mut corrections);
        // self.lift_mul_sub_split(&mut x2, &corrections, &mut x01, code_dots);

        // let mut res = Buffers::take_buffer(&mut self.buffers.u64_37c_1);
        // self.extract_msb_sum_mod(&x01, &x2, &mut res);

        // Buffers::return_buffer(&mut self.buffers.u64_64c_1, x2);
        // Buffers::return_buffer(&mut self.buffers.u64_64c_2, x01);
        // Buffers::return_buffer(&mut self.buffers.u32_128c_1, corrections);
        // Buffers::return_buffer(&mut self.buffers.u64_37c_1, res);
        // self.buffers.check_buffers();

        // Result is in the first bit of res u64_37c_1
    }

    // input should be of size: n_devices * input_size
    // Result is in the lowest bit of the result buffer on the first gpu
    pub fn compare_threshold_masked_many_with_or_tree(
        &mut self,
        code_dots: Vec<ChunkShare<u16>>,
        mask_dots: Vec<ChunkShare<u16>>,
    ) {
        // self.compare_threshold_masked_many(code_dots, mask_dots);
        // let mut result = self.take_result_buffer();
        // self.or_reduce_result(&mut result);
        // // Result is in the first bit of the first GPU

        // self.return_result_buffer(result);
        // self.buffers.check_buffers();

        // Result is in the lowest bit of the result buffer on the first gpu
    }
}
