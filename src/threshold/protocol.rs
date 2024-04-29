use axum::{routing::get, Router};
use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, CudaView, LaunchAsync, LaunchConfig},
    nccl::{Comm, Id},
    nvrtc,
};
use std::{rc::Rc, str::FromStr, sync::Arc, thread, time::Duration};

use crate::{http_root, threshold::cuda::PTX_SRC, IdWrapper};

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

pub struct Circuits {
    peer_id: usize,
    next_id: usize,
    cfg: LaunchConfig,
    chunk_size: usize,
    n_devices: usize,
    devs: Vec<Arc<CudaDevice>>,
    comms: Vec<Rc<Comm>>,
    and_kernel: Vec<CudaFunction>,
    xor_kernel: Vec<CudaFunction>,
    xor_assign_kernel: Vec<CudaFunction>,
}

impl Circuits {
    const MOD_NAME: &'static str = "TComp";

    pub fn new(
        peer_id: usize,
        chunk_size: usize,
        peer_url: Option<&String>,
        is_local: bool,
        server_port: Option<u16>,
    ) -> Self {
        let n_devices = CudaDevice::count().unwrap() as usize;

        // TODO check this
        let cfg = Self::launch_config_from_elements_and_threads(chunk_size as u32, 1024);

        let mut devs = Vec::with_capacity(n_devices);
        let mut and_kernel = Vec::with_capacity(n_devices);
        let mut xor_kernel = Vec::with_capacity(n_devices);
        let mut xor_assign_kernel = Vec::with_capacity(n_devices);

        let ptx = nvrtc::compile_ptx(PTX_SRC).unwrap();
        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let stream = dev.fork_default_stream().unwrap();

            dev.load_ptx(
                ptx.clone(),
                Self::MOD_NAME,
                &["shared_xor, shared_xor_assign, shared_and_pre"],
            )
            .unwrap();
            let and = dev.get_func(Self::MOD_NAME, "shared_and_pre").unwrap();
            let xor = dev.get_func(Self::MOD_NAME, "shared_xor").unwrap();
            let xor_assign = dev.get_func(Self::MOD_NAME, "shared_xor_assign").unwrap();

            devs.push(dev);
            and_kernel.push(and);
            xor_kernel.push(xor);
            xor_assign_kernel.push(xor_assign);
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
            and_kernel,
            xor_kernel,
            xor_assign_kernel,
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
            self.and_kernel[idx]
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
        todo!("communicate result")
    }

    fn xor_assign_many(&self, x1: &mut ChunkShareView<u64>, x2: &ChunkShareView<u64>, idx: usize) {
        unsafe {
            self.xor_assign_kernel[idx]
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
            self.xor_kernel[idx]
                .clone()
                .launch(
                    self.cfg.to_owned(),
                    (&res.a, &res.b, &x1.a, &x1.b, &x2.a, &x2.b, self.chunk_size),
                )
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

    pub fn binary_add_two(
        &mut self,
        x1: Vec<ChunkShare<u64>>,
        x2: Vec<ChunkShare<u64>>,
        bits: usize,
    ) -> Vec<ChunkShare<u64>> {
        debug_assert_eq!(self.n_devices, x1.len());
        debug_assert_eq!(self.n_devices, x2.len());

        // TODO the buffers should probably already be allocated
        // Result with additional bit for overflow
        let mut s = self.allocate_buffer::<u64>((self.chunk_size + 1) * bits);
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
        for k in 1..bits {
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
            let mut s_ = ss.get_offset(bits, self.chunk_size);
            s_.a = c_.a;
            s_.b = c_.b;
        }

        s
    }
}
