use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};
use std::{mem, sync::Arc};

pub struct AesCudaRng {
    buf_size:   usize,
    devs:       Vec<Arc<CudaDevice>>,
    kernels:    Vec<CudaFunction>,
    rng_chunks: Vec<CudaSlice<u8>>,
    output_buf: Vec<u8>,
}

const AES_PTX_SRC: &str = include_str!("aes.cu");
const AES_FUNCTION_NAME: &str = "aes_128_rng";

impl AesCudaRng {
    // buf size in u8
    pub fn init(buf_size: usize) -> Self {
        let n_devices = CudaDevice::count().unwrap() as usize;
        let mut devs = Vec::new();
        let mut kernels = Vec::new();
        let ptx = compile_ptx(AES_PTX_SRC).unwrap();

        for i in 0..n_devices {
            // This call to CudaDevice::new is only used in context of a benchmark - not
            // used in the server binary
            let dev = CudaDevice::new(i).unwrap();
            dev.load_ptx(ptx.clone(), AES_FUNCTION_NAME, &[AES_FUNCTION_NAME])
                .unwrap();
            let function = dev.get_func(AES_FUNCTION_NAME, AES_FUNCTION_NAME).unwrap();

            devs.push(dev);
            kernels.push(function);
        }

        assert!(buf_size % 16 == 0, "buf_size must be a multiple of 16 atm");

        let buf = vec![0u8; buf_size];
        let rng_chunks = vec![devs[0].htod_sync_copy(&buf).unwrap()]; // just do on device 0 for now

        Self {
            buf_size,
            devs,
            kernels,
            rng_chunks,
            output_buf: buf,
        }
    }

    pub fn fill_rng(&mut self) {
        self.fill_rng_no_host_copy();
        self.devs[0]
            .dtoh_sync_copy_into(&self.rng_chunks[0], &mut self.output_buf[..])
            .unwrap();
    }
    pub fn fill_rng_no_host_copy(&mut self) {
        let num_kernel_calls = self.buf_size / 16;
        let threads_per_block = 256;
        let blocks_per_grid = (num_kernel_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim:        (threads_per_block as u32, 1, 1),
            grid_dim:         (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let key_bytes = [0u8; 16];
        let key_slice = self.devs[0].htod_sync_copy(&key_bytes[..]).unwrap();
        unsafe {
            self.kernels[0]
                .clone()
                .launch(
                    cfg,
                    (
                        &key_slice,
                        16,
                        &self.rng_chunks[0],
                        self.buf_size,
                        mem::size_of::<u8>(),
                    ),
                )
                .unwrap();
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.output_buf
    }
}

#[cfg(test)]
#[allow(unused)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu_dependent")]
    fn test_aes_rng() {
        let mut rng = AesCudaRng::init(1024 * 1024);
        rng.fill_rng();

        let zeros = rng.data().iter().filter(|x| x == &&0).count();
        let expected = 1024 * 1024 / 256;
        assert!(1.1 * expected as f64 > zeros as f64);
        assert!(1.1 * zeros as f64 > expected as f64);
    }
}
