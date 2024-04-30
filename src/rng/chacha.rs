use std::sync::Arc;

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

pub struct ChaChaCudaRng {
    buf_size: usize,
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    rng_chunks: Vec<CudaSlice<u32>>,
    output_buffer: Vec<u32>,
}

const CHACHA_PTX_SRC: &str = include_str!("chacha.cu");
const CHACHA_FUNCTION_NAME: &str = "chacha12";

impl ChaChaCudaRng {
    // takes number of u32 elements to produce
    pub fn init(buf_size: usize) -> Self {
        let n_devices = CudaDevice::count().unwrap() as usize;
        let mut devs = Vec::new();
        let mut kernels = Vec::new();
        let ptx = compile_ptx(CHACHA_PTX_SRC).unwrap();

        assert!(buf_size % 16 == 0, "buf_size must be a multiple of 16 atm");

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            dev.load_ptx(ptx.clone(), CHACHA_FUNCTION_NAME, &[CHACHA_FUNCTION_NAME])
                .unwrap();
            let function = dev
                .get_func(CHACHA_FUNCTION_NAME, CHACHA_FUNCTION_NAME)
                .unwrap();

            devs.push(dev);
            kernels.push(function);
        }

        let buf = vec![0u32; buf_size];
        let rng_chunks = vec![devs[0].htod_sync_copy(&buf).unwrap()]; // just do on device 0 for now

        Self {
            buf_size,
            devs,
            kernels,
            rng_chunks,
            output_buffer: buf,
        }
    }

    pub fn fill_rng(&mut self) {
        self.fill_rng_no_host_copy();

        self.devs[0]
            .dtoh_sync_copy_into(&self.rng_chunks[0], &mut self.output_buffer)
            .unwrap();
    }

    pub fn fill_rng_no_host_copy(&mut self) {
        let num_ks_calls = self.buf_size / 16;
        let threads_per_block = 256; // todo sync with kernel
        let blocks_per_grid = (num_ks_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0, // do we need this since we use __shared__ in kernel?
        };
        let ctx = ChaChaCtx::init([0u32; 8], 0, 0); // todo keep internal state
        let state_slice = self.devs[0].htod_sync_copy(&ctx.state).unwrap();
        unsafe {
            self.kernels[0]
                .clone()
                .launch(cfg, (&mut self.rng_chunks[0], &state_slice))
                .unwrap();
        }
    }
    pub fn data(&self) -> &[u32] {
        &self.output_buffer
    }
}

//
// struct chacha_ctx
// {
//     uint32_t keystream[16];
//     uint32_t state[16];
//     uint32_t *counter;
// };

const CHACONST: [u32; 4] = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574];

pub struct ChaChaCtx {
    // 12 32-bit words for the key
    // 2 32-bit words for the counter
    // 2 32-bit words for the nonce (stream id)
    pub(crate) state: [u32; 16],
}

impl ChaChaCtx {
    const COUNTER_START_IDX: usize = 12;
    const NONCE_START_IDX: usize = 14;
    pub fn init(key: [u32; 8], counter: u64, nonce: u64) -> Self {
        let mut state = [0u32; 16];
        state[0] = CHACONST[0];
        state[1] = CHACONST[1];
        state[2] = CHACONST[2];
        state[3] = CHACONST[3];
        state[4..12].copy_from_slice(&key);

        let mut res = Self { state };
        res.set_counter(counter);
        res.set_nonce(nonce);
        res
    }
    fn get_value(&self, idx: usize) -> u64 {
        self.state[idx] as u64 | ((self.state[idx + 1] as u64) << 32)
    }
    fn set_value(&mut self, idx: usize, value: u64) {
        self.state[idx] = value as u32;
        self.state[idx + 1] = (value >> 32) as u32;
    }

    pub fn set_counter(&mut self, counter: u64) {
        self.set_value(Self::COUNTER_START_IDX, counter)
    }
    pub fn get_counter(&self) -> u64 {
        self.get_value(Self::COUNTER_START_IDX)
    }
    pub fn set_nonce(&mut self, nonce: u64) {
        self.set_value(Self::NONCE_START_IDX, nonce)
    }
    pub fn get_nonce(&self) -> u64 {
        self.get_value(Self::NONCE_START_IDX)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_chacha_rng() {
        let mut rng = ChaChaCudaRng::init(1024 * 1024);
        rng.fill_rng();
        let zeros = rng.data().iter().filter(|x| x == &&0).count();
        // we would expect no 0s in the output buffer even 1 is 1/4096;
        assert!(zeros <= 1);
    }
}
