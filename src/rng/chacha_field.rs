use std::sync::Arc;

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

use super::chacha::ChaChaCtx;

pub struct ChaChaCudaRng {
    // the total buffer size
    buf_size: usize,
    // the amount of valid values in the buffer
    valid_buffer_size: usize,
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    rng_chunks: Vec<CudaSlice<u32>>,
    output_buffer: Vec<u32>,
}

const CHACHA_PTX_SRC: &str = include_str!("chacha_field.cu");
const CHACHA_FUNCTION_NAME: &str = "chacha12";
const FIELD_FUNCTION_NAME: &str = "fix_fe";

// probability calculation says that prob that more than 24 values of 1024 are not in field is less than 1/2^128
const MIN_U16_BUF_ELEMENTS: usize = 1024;
const OK_U16_BUF_ELEMENTS: usize = 1000;

impl ChaChaCudaRng {
    // takes number of u16 elements to produce
    pub fn init(buf_size: usize) -> Self {
        let n_devices = CudaDevice::count().unwrap() as usize;
        let mut devs = Vec::new();
        let mut kernels = Vec::new();
        let ptx = compile_ptx(CHACHA_PTX_SRC).unwrap();

        assert!(
            buf_size % (MIN_U16_BUF_ELEMENTS) == 0,
            "buf_size must be a multiple of 1024 atm"
        );

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            dev.load_ptx(
                ptx.clone(),
                CHACHA_FUNCTION_NAME,
                &[CHACHA_FUNCTION_NAME, FIELD_FUNCTION_NAME],
            )
            .unwrap();
            let function = dev
                .get_func(CHACHA_FUNCTION_NAME, CHACHA_FUNCTION_NAME)
                .unwrap();
            let fe_fix_function = dev
                .get_func(CHACHA_FUNCTION_NAME, FIELD_FUNCTION_NAME)
                .unwrap();

            devs.push(dev);
            kernels.push(function);
            kernels.push(fe_fix_function);
        }

        let buf = vec![0u32; buf_size / std::mem::size_of::<u16>()];
        let rng_chunks = vec![devs[0].htod_sync_copy(&buf).unwrap()]; // just do on device 0 for now

        let valid_buffer_size = (buf_size / (MIN_U16_BUF_ELEMENTS)) * OK_U16_BUF_ELEMENTS;

        Self {
            buf_size,
            valid_buffer_size,
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
        // slice is now filled with u32s, we need to fix the contained u16 to be valid field elements
        let num_fix_calls = self.valid_buffer_size / 1000;
        let threads_per_block = 256; // this should be fine to be whatever
        let blocks_per_grid = (num_fix_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0, // do we need this since we use __shared__ in kernel?
        };
        unsafe {
            self.kernels[1]
                .clone()
                .launch(
                    cfg,
                    (
                        &mut self.rng_chunks[0],
                        u32::try_from(self.valid_buffer_size).unwrap(),
                    ),
                )
                .unwrap();
        }
    }
    pub fn data(&self) -> &[u16] {
        &bytemuck::cast_slice(self.output_buffer.as_slice())[0..self.valid_buffer_size]
    }
    pub fn num_valid(&self) -> usize {
        self.valid_buffer_size
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    const P: u16 = 65519;
    #[test]
    fn test_chacha_rng() {
        let mut rng = ChaChaCudaRng::init(1024 * 1024);
        rng.fill_rng();
        assert!(rng.data().iter().all(|&x| x < P));
    }
}
