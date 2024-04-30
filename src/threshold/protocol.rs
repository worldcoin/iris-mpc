use super::cuda::kernel::B_BITS;
use crate::{http_root, setup::shamir::P, threshold::cuda::PTX_SRC, IdWrapper};
use axum::{routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, CudaView, LaunchAsync, LaunchConfig},
    nccl::{Comm, Id},
    nvrtc::{self, Ptx},
};
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
}

struct Kernels {
    pub(crate) and: CudaFunction,
    pub(crate) xor: CudaFunction,
    pub(crate) xor_assign: CudaFunction,
    pub(crate) not_inplace: CudaFunction,
    pub(crate) not: CudaFunction,
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
            ],
        )
        .unwrap();
        let and = dev.get_func(Self::MOD_NAME, "shared_and_pre").unwrap();
        let xor = dev.get_func(Self::MOD_NAME, "shared_xor").unwrap();
        let xor_assign = dev.get_func(Self::MOD_NAME, "shared_xor_assign").unwrap();
        let not_inplace = dev.get_func(Self::MOD_NAME, "shared_not_inplace").unwrap();
        let not = dev.get_func(Self::MOD_NAME, "shared_not_inplace").unwrap();

        Kernels {
            and,
            xor,
            xor_assign,
            not_inplace,
            not,
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
        chunk_size: usize,
        peer_url: Option<&String>,
        is_local: bool,
        server_port: Option<u16>,
    ) -> Self {
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
        if !is_local {
            let mut ids = Vec::with_capacity(n_devices);
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
                comms.push(Rc::new(
                    Comm::from_rank(devs[i].clone(), peer_id, 3, id).unwrap(),
                ));
            }
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

    fn launch_config_from_elements_and_threads(n: u32, t: u32) -> LaunchConfig {
        let num_blocks = (n + t - 1) / t;
        LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (t, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    fn send_receive_view<T>(&mut self, send: &CudaView<T>, receive: &mut CudaView<T>, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        todo!("implement in comm (requires modifying cudarc)")
    }

    fn send_receive<T>(&mut self, send: &CudaSlice<T>, receive: &mut CudaSlice<T>, idx: usize)
    where
        T: cudarc::nccl::NcclType,
    {
        self.comms[idx].send(send, self.next_id as i32).unwrap();

        self.comms[idx].recv(receive, self.next_id as i32).unwrap();

        self.devs[idx].synchronize().unwrap();
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
                        // rand,
                        self.chunk_size,
                    ),
                )
                .unwrap();
        }
        self.send_receive_view(&res.a, &mut res.b, idx);
    }

    fn xor_assign_many(&self, x1: &mut ChunkShareView<u64>, x2: &ChunkShareView<u64>, idx: usize) {
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

    fn allocate_buffer<T>(&self, size: usize) -> Vec<ChunkShare<T>>
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
        for (idx, ((aa, bb), (ss, cc))) in a
            .iter()
            .zip(b.iter())
            .zip(s.iter_mut().zip(c.iter_mut()))
            .enumerate()
        {
            let a0 = aa.get_offset(0, self.chunk_size);
            let b0 = bb.get_offset(0, self.chunk_size);
            let mut s0 = ss.get_offset(0, self.chunk_size);
            let mut c_ = cc.as_view();

            self.and_many(&a0, &b0, &mut c_, idx);
            self.xor_many(&a0, &b0, &mut s0, idx);
        }

        // Full adders: 1->k
        for k in 1..Self::BITS {
            for (idx, (((aa, bb), (ss, cc)), tmp_cc)) in a
                .iter_mut()
                .zip(b.iter_mut())
                .zip(s.iter_mut().zip(c.iter_mut()))
                .zip(tmp_c.iter_mut())
                .enumerate()
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
        for (cc, ss) in c.iter().zip(s.iter_mut()) {
            let c_ = cc.as_view();
            let mut s_ = ss.get_offset(Self::BITS, self.chunk_size);
            s_.a = c_.a;
            s_.b = c_.b;
        }

        s
    }

    // The result will be located in the first bit of x
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

            for (idx, ((x, cc), (tmp_cc, tmp_not_))) in x
                .iter()
                .zip(c.iter_mut())
                .zip(tmp_c.iter_mut().zip(tmp_not.iter_mut()))
                .enumerate()
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
        for (idx, ((not_msb, cc), (tmp_cc, tmp_not_))) in not_msb
            .iter_mut()
            .zip(c.iter_mut())
            .zip(tmp_c.iter_mut().zip(tmp_not.iter_mut()))
            .enumerate()
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
        let mut ov = c;
        for (idx, (xx, (ov, res_msb))) in x
            .iter()
            .zip(ov.iter_mut().zip(res_msb.iter_mut()))
            .enumerate()
        {
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

    pub fn compare_threshold_masked_many_fp(
        &mut self,
        code_dots: Vec<ChunkShare<u16>>,
        mask_dots: Vec<ChunkShare<u16>>,
    ) -> Vec<ChunkShare<u64>> {
        // lift masks to x
        // do lift code to y, and x*a - y in one kernel
        todo!("implement")
    }
}
