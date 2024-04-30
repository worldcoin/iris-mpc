use std::sync::Arc;

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

use super::chacha::ChaChaCtx;

pub struct ChaChaCudaRng {
    buf_size: usize,
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    rng_chunks: Vec<CudaSlice<u32>>,
    output_buffer: Vec<u32>,
}

const CHACHA_PTX_SRC: &str = include_str!("chacha_two.cu");
const CHACHA_FUNCTION_NAME: &str = "chacha12_two";
const CHACHA_FUNCTION_NAME_SEQ: &str = "chacha12_two_seq";

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
            dev.load_ptx(
                ptx.clone(),
                CHACHA_FUNCTION_NAME,
                &[CHACHA_FUNCTION_NAME, CHACHA_FUNCTION_NAME_SEQ],
            )
            .unwrap();
            let function = dev
                .get_func(CHACHA_FUNCTION_NAME, CHACHA_FUNCTION_NAME)
                .unwrap();
            let function2 = dev
                .get_func(CHACHA_FUNCTION_NAME, CHACHA_FUNCTION_NAME_SEQ)
                .unwrap();

            devs.push(dev);
            kernels.push(function);
            kernels.push(function2);
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

    pub fn fill_rng(&mut self, variant: usize) {
        self.fill_rng_no_host_copy(variant);

        self.devs[0]
            .dtoh_sync_copy_into(&self.rng_chunks[0], &mut self.output_buffer)
            .unwrap();
    }

    pub fn fill_rng_no_host_copy(&mut self, variant: usize) {
        let num_ks_calls = self.buf_size / 16;
        let threads_per_block = 256; // todo sync with kernel
        let blocks_per_grid = (num_ks_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0, // do we need this since we use __shared__ in kernel?
        };
        let ctx = ChaChaCtx::init([0u32; 8], 0, 0); // todo keep internal state
        let ctx2 = ChaChaCtx::init([1u32; 8], 0, 0); // todo keep internal state
        let state_slice = self.devs[0].htod_sync_copy(&ctx.state).unwrap();
        let state_slice2 = self.devs[0].htod_sync_copy(&ctx2.state).unwrap();
        unsafe {
            self.kernels[variant]
                .clone()
                .launch(cfg, (&mut self.rng_chunks[0], &state_slice, &state_slice2))
                .unwrap();
        }
    }
    pub fn data(&self) -> &[u32] {
        &self.output_buffer
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_chacha_rng() {
        let mut rng = ChaChaCudaRng::init(1024 * 1024);
        rng.fill_rng(0);
        let zeros = rng.data().iter().filter(|x| x == &&0).count();
        // we would expect no 0s in the output buffer even 1 is 1/4096;
        assert!(zeros <= 1);

        let data = rng.data().to_vec();
        rng.fill_rng(1);
        assert!(&data == rng.data(), "two variants must be the same");
    }
}
