use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, CudaView};

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
    chunk_size: usize,
    and_kernel: CudaFunction,
    my_id: usize,
    n_devices: usize,
    devs: Vec<Arc<CudaDevice>>,
}

impl Circuits {
    pub fn new(my_id: usize, chunk_size: usize) -> Self {
        let n_devices = CudaDevice::count().unwrap() as usize;
        todo!("instantiate")
        // Circuits {
        //     chunk_size,
        //     my_id,
        //     n_devices,
        // }
    }

    fn and_many(
        &self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
    ) {
        todo!("Start kernel and communicate result")
    }

    fn xor_assign_many(&self, x1: &mut ChunkShareView<u64>, x2: &ChunkShareView<u64>) {
        todo!("Start kernel")
    }

    fn xor_many(
        &self,
        x1: &ChunkShareView<u64>,
        x2: &ChunkShareView<u64>,
        res: &mut ChunkShareView<u64>,
    ) {
        todo!("Start kernel")
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
        &self,
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
        for ((aa, bb), (ss, cc)) in a.iter().zip(b.iter()).zip(s.iter_mut().zip(c.iter_mut())) {
            let a0 = aa.get_offset(0, self.chunk_size);
            let b0 = bb.get_offset(0, self.chunk_size);
            let mut s0 = ss.get_offset(0, self.chunk_size);
            let mut c_ = cc.as_view();
            self.and_many(&a0, &b0, &mut c_);
            self.xor_many(&a0, &b0, &mut s0);
        }

        // Full adders: 1->k
        for k in 1..bits {
            for (((aa, bb), (ss, cc)), tmp_cc) in a
                .iter_mut()
                .zip(b.iter_mut())
                .zip(s.iter_mut().zip(c.iter_mut()))
                .zip(tmp_c.iter_mut())
            {
                let mut ak = aa.get_offset(k, self.chunk_size);
                let mut bk = bb.get_offset(k, self.chunk_size);
                let mut sk = ss.get_offset(k, self.chunk_size);
                let mut c_ = cc.as_view();
                let mut tmp_cc_ = tmp_cc.as_view();

                self.xor_assign_many(&mut ak, &c_);
                self.xor_many(&ak, &bk, &mut sk);
                self.xor_assign_many(&mut bk, &c_);
                self.and_many(&ak, &bk, &mut tmp_cc_);
                self.xor_assign_many(&mut c_, &tmp_cc_);
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
