use super::cuda::kernel::B_BITS;
use crate::{http_root, setup::shamir::P, threshold::cuda::PTX_SRC, IdWrapper};
use axum::{routing::get, Router};
use cudarc::{
    driver::{
        CudaDevice, CudaFunction, CudaSlice, CudaView, DevicePtr, DeviceSlice, LaunchAsync,
        LaunchConfig,
    },
    nccl::{result, Comm, Id},
    nvrtc::{self, Ptx},
};
use itertools::izip;
use std::{rc::Rc, str::FromStr, sync::Arc, thread, time::Duration};

pub(crate) const P2K: u64 = (P as u64) << B_BITS;

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

    pub fn get_offset(&self, i: usize, chunk_size: usize) -> ChunkShareView<T> {
        ChunkShareView {
            a: self.a.slice(i * chunk_size..(i + 1) * chunk_size),
            b: self.b.slice(i * chunk_size..(i + 1) * chunk_size),
        }
    }
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
    pub(crate) xor: CudaFunction,
    pub(crate) xor_assign: CudaFunction,
    pub(crate) not_inplace: CudaFunction,
    pub(crate) not: CudaFunction,
    pub(crate) lift_mul_sub_split: CudaFunction,
    pub(crate) transpose_64x64: CudaFunction,
}

impl Kernels {
    const MOD_NAME: &'static str = "TComp";

    pub(crate) fn new(dev: Arc<CudaDevice>, ptx: Ptx) -> Kernels {
        dev.load_ptx(
            ptx.clone(),
            Self::MOD_NAME,
            &[
                "shared_xor, shared_xor_assign, shared_and_pre, shared_not_inplace",
                "shared_not",
                "shared_lift_mul_sub_split",
                "shared_u64_transpose_pack_u64",
            ],
        )
        .unwrap();
        let and = dev.get_func(Self::MOD_NAME, "shared_and_pre").unwrap();
        let xor = dev.get_func(Self::MOD_NAME, "shared_xor").unwrap();
        let xor_assign = dev.get_func(Self::MOD_NAME, "shared_xor_assign").unwrap();
        let not_inplace = dev.get_func(Self::MOD_NAME, "shared_not_inplace").unwrap();
        let not = dev.get_func(Self::MOD_NAME, "shared_not").unwrap();
        let lift_mul_sub_split = dev
            .get_func(Self::MOD_NAME, "shared_lift_mul_sub_split")
            .unwrap();
        let transpose_64x64 = dev
            .get_func(Self::MOD_NAME, "shared_u64_transpose_pack_u64")
            .unwrap();

        Kernels {
            and,
            xor,
            xor_assign,
            not_inplace,
            not,
            lift_mul_sub_split,
            transpose_64x64,
        }
    }
}

pub struct Circuits {
    peer_id: usize,
    next_id: usize,
    cfg: LaunchConfig,
    chunk_size: usize,
    n_devices: usize,
    devs: Vec<Arc<CudaDevice>>,
    comms: Vec<Rc<Comm>>,
    kernels: Vec<Kernels>,
}

impl Circuits {
    const BITS: usize = 36;

    pub fn new(
        peer_id: usize,
        input_size: usize, // per GPU
        peer_url: Option<&String>,
        server_port: Option<u16>,
    ) -> Self {
        // TODO check this
        // For the transpose, inputs should be multiple of 64 bits
        debug_assert!(input_size % 64 == 0);
        // Chunk size is the number of u64 elements per bit in the binary circuits
        let chunk_size = input_size / 64;

        debug_assert_eq!(Self::ceil_log2(P2K as usize), Self::BITS);
        let n_devices = CudaDevice::count().unwrap() as usize;

        // TODO check this
        let cfg = Self::launch_config_from_elements_and_threads(chunk_size as u32, 1024);

        let mut devs = Vec::with_capacity(n_devices);
        let mut kernels = Vec::with_capacity(n_devices);

        let ptx = nvrtc::compile_ptx(PTX_SRC).unwrap();
        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let kernel = Kernels::new(dev.clone(), ptx.clone());

            devs.push(dev);
            kernels.push(kernel);
        }

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
                thread::sleep(Duration::from_secs(5));

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
            devs[i].bind_to_thread().unwrap();
            comms.push(Rc::new(
                Comm::from_rank(devs[i].clone(), peer_id, 3, id).unwrap(),
            ));
        }

        Circuits {
            peer_id,
            next_id: (peer_id + 1) % 3,
            cfg,
            chunk_size,
            n_devices,
            devs,
            comms,
            kernels,
        }
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

    pub fn send_view<T>(&mut self, send: &CudaView<T>, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        // Copied from cudarc nccl implementation
        unsafe {
            result::send(
                *send.device_ptr() as *mut _,
                send.len(),
                T::as_nccl_type(),
                idx as i32,
                self.comms[idx].comm,
                *self.comms[idx].device.cu_stream() as *mut _,
            )
        }
        .unwrap();

        self.devs[idx].synchronize().unwrap();
    }

    pub fn receive_view<T>(&mut self, receive: &mut CudaView<T>, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        // Copied from cudarc nccl implementation
        unsafe {
            result::recv(
                *receive.device_ptr() as *mut _,
                receive.len(),
                T::as_nccl_type(),
                idx as i32,
                self.comms[idx].comm,
                *self.comms[idx].device.cu_stream() as *mut _,
            )
        }
        .unwrap();

        self.devs[idx].synchronize().unwrap();
    }

    fn send<T>(&mut self, send: &CudaSlice<T>, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        self.comms[idx].send(send, self.next_id as i32).unwrap();
    }

    fn receive<T>(&mut self, receive: &mut CudaSlice<T>, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        self.comms[idx].recv(receive, self.next_id as i32).unwrap();
        self.devs[idx].synchronize().unwrap();
    }

    // TODO include randomness
    fn packed_and_many_send(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        // rand: &CudaView<u64>,
        bits: usize,
        idx: usize,
    ) {
        // TODO this is just a placeholder
        let rand = self.devs[idx].alloc_zeros::<u64>(self.chunk_size).unwrap();

        unsafe {
            self.kernels[idx]
                .and
                .clone()
                .launch(
                    self.cfg.to_owned(),
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

        // TODO check if this is the best we can do
        for i in 0..bits {
            let send = res.a.slice(i * self.chunk_size..(i + 1) * self.chunk_size);
            self.send_view(&send, idx);
        }
    }

    fn packed_and_many_receive(&mut self, res: &mut ChunkShareView<u64>, bits: usize, idx: usize) {
        // TODO check if this is the best we can do
        for i in 0..bits {
            let mut rcv = res.b.slice(i * self.chunk_size..(i + 1) * self.chunk_size);
            self.receive_view(&mut rcv, idx);
        }
    }

    // TODO include randomness
    fn packed_and_many(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        // rand: &CudaView<u64>,
        bits: usize,
        idx: usize,
    ) {
        self.packed_and_many_send(x1, x2, res, bits, idx);
        self.packed_and_many_receive(res, bits, idx);
    }

    // TODO include randomness
    fn and_many_send(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        // rand: &CudaView<u64>,
        idx: usize,
    ) {
        // TODO this is just a placeholder
        let rand = self.devs[idx].alloc_zeros::<u64>(self.chunk_size).unwrap();

        unsafe {
            self.kernels[idx]
                .and
                .clone()
                .launch(
                    self.cfg.to_owned(),
                    (&res.a, &x1.a, &x1.b, &x2.a, &x2.b, &rand, self.chunk_size),
                )
                .unwrap();
        }
        self.send_view(&res.a, idx);
    }

    fn and_many_receive(&mut self, res: &mut ChunkShareView<u64>, idx: usize) {
        self.receive_view(&mut res.b, idx);
    }

    // TODO include randomness
    fn and_many(
        &mut self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
        // rand: &CudaView<u64>,
        idx: usize,
    ) {
        self.and_many_send(x1, x2, res, idx);
        self.and_many_receive(res, idx);
    }

    pub fn xor_assign_many(
        &self,
        x1: &mut ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        idx: usize,
    ) {
        unsafe {
            self.kernels[idx]
                .xor_assign
                .clone()
                .launch(
                    self.cfg.to_owned(),
                    (&x1.a, &x1.b, &x2.a, &x2.b, self.chunk_size),
                )
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
        unsafe {
            self.kernels[idx]
                .xor_assign
                .clone()
                .launch(
                    self.cfg.to_owned(),
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
        unsafe {
            self.kernels[idx]
                .xor
                .clone()
                .launch(
                    self.cfg.to_owned(),
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
        unsafe {
            self.kernels[idx]
                .xor
                .clone()
                .launch(
                    self.cfg.to_owned(),
                    (&res.a, &res.b, &x1.a, &x1.b, &x2.a, &x2.b, self.chunk_size),
                )
                .unwrap();
        }
    }

    fn not_many(&self, x: &ChunkShareView<u64>, res: &mut ChunkShareView<u64>, idx: usize) {
        unsafe {
            self.kernels[idx]
                .not
                .clone()
                .launch(self.cfg.to_owned(), (&res.a, &res.b, &x.a, &x.b))
                .unwrap();
        }
    }

    fn not_inplace_many(&self, x: &mut ChunkShareView<u64>, idx: usize) {
        unsafe {
            self.kernels[idx]
                .not_inplace
                .clone()
                .launch(self.cfg.to_owned(), (&x.a, &x.b))
                .unwrap();
        }
    }

    fn allocate_single_buffer<T>(&self, size: usize) -> Vec<CudaSlice<T>>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        let mut res = Vec::with_capacity(self.n_devices);

        for dev in self.devs.iter() {
            res.push(dev.alloc_zeros::<T>(size).unwrap());
        }
        res
    }

    pub fn allocate_buffer<T>(&self, size: usize) -> Vec<ChunkShare<T>>
    where
        T: cudarc::driver::ValidAsZeroBits + cudarc::driver::DeviceRepr,
    {
        let mut res = Vec::with_capacity(self.n_devices);

        for dev in self.devs.iter() {
            let a = dev.alloc_zeros::<T>(size).unwrap();
            let b = dev.alloc_zeros::<T>(size).unwrap();
            res.push(ChunkShare::new(a, b));
        }
        res
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

    // input should be of size: n_devices * input_size
    fn lift_p2k(&mut self, shares: Vec<ChunkShare<u16>>) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(self.n_devices, shares.len());
        todo!()
    }

    // K is 18 in our case
    // requires 66 * n_devices * chunk_size random u64 elements
    fn binary_add_3_get_msb_twice<const K: usize>(
        &mut self,
        xa_1: Vec<ChunkShare<u64>>,
        xa_2: Vec<ChunkShare<u64>>,
        xa_3: Vec<ChunkShare<u64>>,
        xb_1: Vec<ChunkShare<u64>>,
        xb_2: Vec<ChunkShare<u64>>,
        xb_3: Vec<ChunkShare<u64>>,
    ) -> (Vec<ChunkShare<u64>>, Vec<ChunkShare<u64>>) {
        debug_assert_eq!(self.n_devices, xa_1.len());
        debug_assert_eq!(self.n_devices, xa_2.len());
        debug_assert_eq!(self.n_devices, xa_3.len());
        debug_assert_eq!(self.n_devices, xb_1.len());
        debug_assert_eq!(self.n_devices, xb_2.len());
        debug_assert_eq!(self.n_devices, xb_3.len());

        // TODO the buffers should probably already be allocated
        let mut s1 = self.allocate_buffer::<u64>(self.chunk_size * K);
        let mut s2 = self.allocate_buffer::<u64>(self.chunk_size * K);
        let mut c1 = self.allocate_buffer::<u64>(self.chunk_size * (K - 1));
        let mut c2 = self.allocate_buffer::<u64>(self.chunk_size * (K - 1));
        let mut ca = self.allocate_buffer::<u64>(self.chunk_size);
        let mut cb = self.allocate_buffer::<u64>(self.chunk_size);

        for (idx, (x1, x2, x3, y1, y2, y3, s1, s2, c1, c2)) in
            izip!(&xa_1, &xa_2, &xa_3, &xb_1, &xb_2, &xb_3, &mut s1, &mut s2, &mut c1, &mut c2)
                .enumerate()
        {
            // First full adder to get 2 * c1 and s1
            let x1 = x1.as_view();
            let x3 = x3.as_view();
            let mut x2x3 = x2.as_view();
            self.packed_xor_assign_many(&mut x2x3, &x3, K, idx);
            self.packed_xor_many(&x1, &x2x3, &mut s1.as_view(), K, idx);
            // 2 * c1 (No pop since we start at index 0 and the functions only compute whats required)
            let mut x1x3 = x1;
            self.packed_xor_assign_many(&mut x1x3, &x3, K - 1, idx);
            let mut c1 = c1.as_view();
            self.packed_and_many_send(&x1x3, &x2x3, &mut c1, K - 1, idx);

            // Second full adder to get 2 * c2 and s2
            let y1 = y1.as_view();
            let y3 = y3.as_view();
            let mut y2y3 = y2.as_view();
            self.packed_xor_assign_many(&mut y2y3, &y3, K, idx);
            self.packed_xor_many(&y1, &y2y3, &mut s2.as_view(), K, idx);
            // 2 * c2 (No pop since we start at index 0 and the functions only compute whats required)
            let mut y1y3 = y1;
            self.packed_xor_assign_many(&mut y1y3, &y3, K - 1, idx);
            let mut c2 = c2.as_view();
            self.packed_and_many_send(&y1y3, &y2y3, &mut c2, K - 1, idx);
        }

        // Receive full adders
        for (idx, (c1, c2, x3, y3)) in izip!(&mut c1, &mut c2, &xa_3, &xb_3).enumerate() {
            let mut c1 = c1.as_view();
            self.packed_and_many_receive(&mut c1, K - 1, idx);
            self.packed_xor_assign_many(&mut c1, &x3.as_view(), K - 1, idx);

            let mut c2 = c2.as_view();
            self.packed_and_many_receive(&mut c2, K - 1, idx);
            self.packed_xor_assign_many(&mut c2, &y3.as_view(), K - 1, idx);
        }

        // Add 2c + s via a ripple carry adder
        // LSB of c is 0
        // First round: half adder can be skipped due to LSB of c being 0
        let mut a1 = s1;
        let mut b1 = c1;
        let mut a2 = s2;
        let mut b2 = c2;

        // First full adder (carry is 0)
        for (idx, (a1, b1, a2, b2, ca, cb)) in
            izip!(&a1, &b1, &a2, &b2, &mut ca, &mut cb).enumerate()
        {
            let a1 = a1.get_offset(1, self.chunk_size);
            let b1 = b1.get_offset(0, self.chunk_size);
            self.and_many_send(&a1, &b1, &mut ca.as_view(), idx);
            let a2 = a2.get_offset(1, self.chunk_size);
            let b2 = b2.get_offset(0, self.chunk_size);
            self.and_many_send(&a2, &b2, &mut cb.as_view(), idx);
        }
        for (idx, (ca, cb)) in izip!(&mut ca, &mut cb).enumerate() {
            self.and_many_receive(&mut ca.as_view(), idx);
            self.and_many_receive(&mut cb.as_view(), idx);
        }

        for k in 1..K - 2 {
            for (idx, (a1, b1, a2, b2, ca, cb)) in
                izip!(&mut a1, &mut b1, &mut a2, &mut b2, &mut ca, &mut cb).enumerate()
            {
                // Unused space used for temparary storage
                let mut tmp_ca = a1.get_offset(0, self.chunk_size);
                let mut tmp_cb = a2.get_offset(0, self.chunk_size);

                let mut a1 = a1.get_offset(k + 1, self.chunk_size);
                let mut a2 = a2.get_offset(k + 1, self.chunk_size);
                let mut b1 = b1.get_offset(k, self.chunk_size);
                let mut b2 = b2.get_offset(k, self.chunk_size);

                let ca = ca.as_view();
                let cb = cb.as_view();

                self.xor_assign_many(&mut a1, &ca, idx);
                self.xor_assign_many(&mut b1, &ca, idx);
                self.and_many_send(&a1, &b1, &mut tmp_ca, idx);
                self.xor_assign_many(&mut a2, &cb, idx);
                self.xor_assign_many(&mut b2, &cb, idx);
                self.and_many_send(&a2, &b2, &mut tmp_cb, idx);
            }
            for (idx, (a1, a2, ca, cb)) in izip!(&a1, &a1, &mut ca, &mut cb).enumerate() {
                // Unused space used for temparary storage
                let mut tmp_ca = a1.get_offset(0, self.chunk_size);
                let mut tmp_cb = a2.get_offset(0, self.chunk_size);
                self.and_many_receive(&mut tmp_ca, idx);
                self.xor_assign_many(&mut ca.as_view(), &tmp_ca, idx);
                self.and_many_receive(&mut tmp_cb, idx);
                self.xor_assign_many(&mut cb.as_view(), &tmp_cb, idx);
            }
        }

        // Final outputs
        for (idx, (a1, a2, b1, b2, ca, cb)) in
            izip!(&a1, &a2, &b1, &b2, &mut ca, &mut cb).enumerate()
        {
            let mut ca = ca.as_view();
            self.xor_assign_many(&mut ca, &a1.get_offset(K - 1, self.chunk_size), idx);
            self.xor_assign_many(&mut ca, &b1.get_offset(K - 2, self.chunk_size), idx);

            let mut cb = cb.as_view();
            self.xor_assign_many(&mut cb, &a2.get_offset(K - 1, self.chunk_size), idx);
            self.xor_assign_many(&mut cb, &b2.get_offset(K - 2, self.chunk_size), idx);
        }

        // TODO no truncation and convert to bits yet!
        (ca, cb)
    }

    fn transpose_pack_u64_with_len(
        &mut self,
        inp: Vec<ChunkShare<u64>>,
        bitlen: usize,
    ) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(self.n_devices, inp.len());

        // TODO the buffers should probably already be allocated
        let mut res = self.allocate_buffer::<u64>(self.chunk_size * bitlen);

        for (idx, (inp, outp)) in izip!(inp, &mut res).enumerate() {
            unsafe {
                self.kernels[idx]
                    .transpose_64x64
                    .clone()
                    .launch(
                        self.cfg.to_owned(),
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

        res
    }

    pub fn extract_msb_sum_mod(
        &mut self,
        x01_send: Vec<CudaSlice<u64>>,
        x2: Vec<ChunkShare<u64>>,
    ) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(self.n_devices, x01_send.len());
        debug_assert_eq!(self.n_devices, x2.len());

        let x01_rec = self.allocate_single_buffer(self.chunk_size * 64);
        let mut x01 = Vec::with_capacity(self.n_devices);

        // First thing: Reshare x01
        for (idx, x01_send) in x01_send.iter().enumerate() {
            self.send(x01_send, idx)
        }
        for (idx, (x01_send, mut x01_rec)) in izip!(x01_send, x01_rec).enumerate() {
            self.receive(&mut x01_rec, idx);
            x01.push(ChunkShare::new(x01_send, x01_rec));
        }

        // Transpose
        let x01 = self.transpose_pack_u64_with_len(x01, Self::BITS);
        let x2 = self.transpose_pack_u64_with_len(x2, Self::BITS);

        let mut sum = self.binary_add_two(x01, x2);
        self.extraxt_msb_mod_p2k_of_sum(&mut sum);

        // Result is in the first bit of the input
        sum
    }

    // requires 36 * n_devices * chunk_size random u64 elements
    fn binary_add_two(
        &mut self,
        x1: Vec<ChunkShare<u64>>,
        x2: Vec<ChunkShare<u64>>,
    ) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(self.n_devices, x1.len());
        debug_assert_eq!(self.n_devices, x2.len());

        // TODO the buffers should probably already be allocated
        // Result with additional bit for overflow
        let mut s = self.allocate_buffer::<u64>(self.chunk_size * (Self::BITS + 1));
        let mut c = self.allocate_buffer::<u64>(self.chunk_size);
        let mut tmp_c = self.allocate_buffer::<u64>(self.chunk_size);

        // Add via a ripple carry adder
        let mut a = x1;
        let mut b = x2;

        // first half adder
        for (idx, (aa, bb, ss, cc)) in izip!(&a, &mut b, &s, &c).enumerate() {
            let a0 = aa.get_offset(0, self.chunk_size);
            let b0 = bb.get_offset(0, self.chunk_size);
            let mut s0 = ss.get_offset(0, self.chunk_size);
            let mut c_ = cc.as_view();

            self.and_many(&a0, &b0, &mut c_, idx);
            self.xor_many(&a0, &b0, &mut s0, idx);
        }

        // Full adders: 1->k
        for k in 1..Self::BITS {
            for (idx, (aa, bb, ss, cc, tmp_cc)) in
                izip!(&mut a, &mut b, &mut s, &mut c, &mut tmp_c).enumerate()
            {
                let mut ak = aa.get_offset(k, self.chunk_size);
                let mut bk = bb.get_offset(k, self.chunk_size);
                let mut sk = ss.get_offset(k, self.chunk_size);
                let mut c_ = cc.as_view();
                let mut tmp_cc_ = tmp_cc.as_view();

                self.xor_assign_many(&mut ak, &c_, idx);
                self.xor_many(&ak, &bk, &mut sk, idx);
                self.xor_assign_many(&mut bk, &c_, idx);
                self.and_many(&ak, &bk, &mut tmp_cc_, idx);
                self.xor_assign_many(&mut c_, &tmp_cc_, idx);
            }
        }

        // Copy the last carry to the last bit of s
        for (cc, ss) in izip!(&c, &s) {
            let c_ = cc.as_view();
            let mut s_ = ss.get_offset(Self::BITS, self.chunk_size);
            s_.a = c_.a;
            s_.b = c_.b;
        }

        s
    }

    // The result will be located in the first bit of x
    // requires 18 * n_devices * chunk_size random u64 elements
    fn extraxt_msb_mod_p2k_of_sum(&mut self, x: &mut [ChunkShare<u64>]) {
        debug_assert_eq!(self.n_devices, x.len());

        // x - P, where x has k+1 bits and P has k bits
        // skip half sub since LSB of P2K is not set
        // skip (B_BITS-1 FULL Subs since C = 0 and Bit of P2K is 0)
        // So in total we skip B_BITS
        for i in 0..B_BITS {
            debug_assert!(((P2K >> i) & 1) == 0);
        }
        #[allow(clippy::assertions_on_constants)]
        {
            debug_assert!(((P2K >> B_BITS) & 1) == 1);
        }

        // For the next full_sub, the carry is zero and the P2K bit is 1;
        // We also store buffers for intermediate values
        let mut c = Vec::with_capacity(self.n_devices);
        let mut buffer_x1 = Vec::with_capacity(self.n_devices);
        let mut buffer_x2 = Vec::with_capacity(self.n_devices);
        let mut not_msb = Vec::with_capacity(self.n_devices);
        for (idx, x) in x.iter().enumerate() {
            let mut x0 = x.get_offset(0, self.chunk_size);
            let x1 = x.get_offset(1, self.chunk_size);
            let x2 = x.get_offset(2, self.chunk_size);
            let mut xb = x.get_offset(B_BITS as usize, self.chunk_size);
            let xmsb = x.get_offset(Self::BITS - 1, self.chunk_size);
            self.not_inplace_many(&mut xb, idx);
            self.not_many(&xmsb, &mut x0, idx);
            c.push(xb);
            buffer_x1.push(x1);
            buffer_x2.push(x2);
            not_msb.push(x0);
        }

        let mut tmp_c = buffer_x1;
        let mut tmp_not = buffer_x2;

        // Finally, we have to do normal full subs up to k-1
        for k in B_BITS as usize + 1..Self::BITS - 1 {
            // Normal sub_adder only calculating the carry
            let p_bit = ((P2K >> k) & 1) == 1;

            for (idx, (x, cc, tmp_cc, tmp_not_)) in
                izip!(x.iter(), &mut c, &mut tmp_c, &mut tmp_not).enumerate()
            {
                let mut xk = x.get_offset(k, self.chunk_size);
                if p_bit {
                    self.not_many(cc, tmp_not_, idx);
                    self.not_inplace_many(&mut xk, idx);
                    self.and_many(&xk, tmp_not_, tmp_cc, idx);
                } else {
                    self.and_many(&xk, cc, tmp_cc, idx);
                }
                self.xor_assign_many(cc, tmp_cc, idx);
            }
        }

        // For the next round, we also have to calculate s (P2K bit is 1)
        #[allow(clippy::assertions_on_constants)]
        {
            debug_assert!(((P2K >> (Self::BITS - 1)) & 1) == 1);
        }
        for (idx, (not_msb, cc, tmp_cc, tmp_not_)) in
            izip!(&mut not_msb, &mut c, &mut tmp_c, &mut tmp_not).enumerate()
        {
            self.not_many(cc, tmp_not_, idx);
            self.and_many(not_msb, tmp_not_, tmp_cc, idx);
            self.xor_assign_many(cc, tmp_cc, idx);
            self.xor_assign_many(not_msb, cc, idx)
        }
        let mut res_msb = not_msb;

        // Finally, the overflow bit (P2K is zero)
        #[allow(clippy::assertions_on_constants)]
        {
            debug_assert!(((P2K >> Self::BITS) & 1) == 0);
        }
        for (idx, (xx, ov, res_msb)) in izip!(x.iter(), &mut c, &mut res_msb).enumerate() {
            let xmsb = xx.get_offset(Self::BITS - 1, self.chunk_size);
            let xb = xx.get_offset(Self::BITS, self.chunk_size);
            self.xor_assign_many(ov, &xb, idx);

            // We now have the ov bit (ov), the msb of the addition (xmsb) and the result of the subtraction (res_msb). We need to multiplex.
            // ov = 1 -> xmsb, ov = 0 -> res_msb
            let mut and = xb;
            let mut xor = xmsb;
            self.xor_assign_many(&mut xor, res_msb, idx);
            self.and_many(&xor, ov, &mut and, idx);
            self.xor_assign_many(res_msb, &and, idx);
        }

        // Result is in res_msb, which is the first bit of the input
    }

    // Puts the result (x2) into mask_lifted and returns x01
    // requires 64 * n_devices * chunk_size random u64 elements
    // TODO include randomness
    pub fn lift_mul_sub_split(
        &mut self,
        mask_lifted: &mut [ChunkShare<u64>],
        code: Vec<ChunkShare<u16>>,
    ) -> Vec<CudaSlice<u64>> {
        debug_assert_eq!(self.n_devices, mask_lifted.len());
        debug_assert_eq!(self.n_devices, code.len());

        let mut x01 = self.allocate_single_buffer(self.chunk_size * 64);

        // TODO check the config
        // TODO WIP: have to adapt to include split in the kernel as well
        for (idx, (m, c, x01)) in izip!(mask_lifted, &code, &mut x01).enumerate() {
            // TODO this is just a placeholder
            let rand = self.devs[idx]
                .alloc_zeros::<u64>(self.chunk_size * 64)
                .unwrap();
            unsafe {
                self.kernels[idx]
                    .lift_mul_sub_split
                    .clone()
                    .launch(
                        self.cfg.to_owned(),
                        (
                            x01,
                            &m.a,
                            &m.b,
                            &c.a,
                            &c.b,
                            &rand,
                            self.chunk_size as u32 * 64,
                            self.peer_id as u32,
                        ),
                    )
                    .unwrap();
            }
        }

        x01
    }

    // input should be of size: n_devices * input_size
    // Result is in res_msb, which is the first bit of the input
    pub fn compare_threshold_masked_many_fp(
        &mut self,
        code_dots: Vec<ChunkShare<u16>>,
        mask_dots: Vec<ChunkShare<u16>>,
    ) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(self.n_devices, code_dots.len());
        debug_assert_eq!(self.n_devices, mask_dots.len());

        let mut x2 = self.lift_p2k(mask_dots);
        let x01 = self.lift_mul_sub_split(&mut x2, code_dots);
        let result = self.extract_msb_sum_mod(x01, x2);
        // Result is in the first bit of the input

        // TODO the or tree is still missing as well
        result
    }
}
