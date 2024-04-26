use std::{mem, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, LaunchAsync, LaunchConfig},
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

        assert!(buf_size % 64 == 0, "buf_size must be a multiple of 64 atm");

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let stream = dev.fork_default_stream().unwrap();
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
        let num_ks_calls = self.buf_size / 64;
        let threads_per_block = 256; // todo sync with kernel
        let blocks_per_grid = (num_ks_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0, // do we need this since we use __shared__ in kernel?
        };
        let ctx = ChaChaCtx::init([0u32; 8], 0, [0u32; 3]); // todo keep internal state
        let state_slice = self.devs[0].htod_sync_copy(&ctx.state).unwrap();
        unsafe {
            self.kernels[0]
                .clone()
                .launch(cfg, (&mut self.rng_chunks[0], &state_slice))
                .unwrap();
        }

        self.devs[0]
            .dtoh_sync_copy_into(&self.rng_chunks[0], &mut self.output_buffer)
            .unwrap();
    }

    pub fn fill_rng_no_host_copy(&mut self) {
        let num_ks_calls = self.buf_size / 64;
        let threads_per_block = 256; // todo sync with kernel
        let blocks_per_grid = (num_ks_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0, // do we need this since we use __shared__ in kernel?
        };
        let ctx = ChaChaCtx::init([0u32; 8], 0, [0u32; 3]); // todo keep internal state
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

struct ChaChaCtx {
    state: [u32; 16],
}

impl ChaChaCtx {
    fn init(key: [u32; 8], counter: u32, nonce: [u32; 3]) -> Self {
        let mut state = [0u32; 16];
        state[0] = CHACONST[0];
        state[1] = CHACONST[1];
        state[2] = CHACONST[2];
        state[3] = CHACONST[3];
        state[4..12].copy_from_slice(&key);
        state[12] = counter;
        state[13..16].copy_from_slice(&nonce);

        Self { state }
    }

    fn set_counter(&mut self, counter: u32) {
        self.state[12] = counter;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chacha_rng() {
        let mut rng = ChaChaCudaRng::init(1024 * 1024);
        rng.fill_rng();
        dbg!(&rng.data()[0..100]);
    }
}
