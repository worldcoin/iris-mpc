use std::{mem, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaSlice, CudaStream, LaunchAsync, LaunchConfig},
    nvrtc::compile_ptx,
};

struct AesCudaRng {
    devs: Vec<Arc<CudaDevice>>,
    kernels: Vec<CudaFunction>,
    streams: Vec<CudaStream>,
    rng_chunks: Vec<CudaSlice<u8>>,
}

const PTX_SRC: &str = include_str!("block_cipher.cu");
const FUNCTION_NAME: &str = "aes_128_rng";
const NUM_ELEMENTS: usize = 1024 * 1024 * 1024;

impl AesCudaRng {
    fn init() -> Self {
        let n_devices = CudaDevice::count().unwrap() as usize;
        let mut devs = Vec::new();
        let mut kernels = Vec::new();
        let mut streams = Vec::new();
        let ptx = compile_ptx(PTX_SRC).unwrap();

        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let stream = dev.fork_default_stream().unwrap();
            dev.load_ptx(ptx.clone(), FUNCTION_NAME, &[FUNCTION_NAME])
                .unwrap();
            let function = dev.get_func(FUNCTION_NAME, FUNCTION_NAME).unwrap();

            streams.push(stream);
            devs.push(dev);
            kernels.push(function);
        }

        let buf = vec![0u8; NUM_ELEMENTS];
        let rng_chunks = vec![devs[0].htod_sync_copy(&buf).unwrap()]; // just do on device 0 for now

        Self {
            devs,
            kernels,
            streams,
            rng_chunks,
        }
    }

    fn rng(&self) -> Vec<u8> {
        let num_elements = NUM_ELEMENTS;
        let threads_per_block = 256;
        let blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let key_bytes = [0u8; 16];
        let key_slice = self.devs[0].htod_sync_copy(&key_bytes[..]).unwrap();
        unsafe {
            self.kernels[0]
                .clone()
                .launch_on_stream(
                    &self.streams[0],
                    cfg,
                    (
                        &key_slice,
                        16,
                        &self.rng_chunks[0],
                        num_elements,
                        mem::size_of::<u8>(),
                    ),
                )
                .unwrap();
        }
        let buf = vec![0u8; NUM_ELEMENTS];
        let rng_result = self.devs[0].dtoh_sync_copy(&self.rng_chunks[0]).unwrap();
        rng_result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes_rng() {
        let rng = AesCudaRng::init();
        let rng_result = rng.rng();
        dbg!(&rng_result[0..100]);
        assert_eq!(rng_result.len(), NUM_ELEMENTS);
    }
}
