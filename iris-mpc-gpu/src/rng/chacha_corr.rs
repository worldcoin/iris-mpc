use super::chacha::ChachaCommon;
use cudarc::{
    driver::{CudaDevice, CudaFunction, CudaStream, CudaViewMut},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

pub struct ChaChaCudaCorrRng {
    fill_kernel: CudaFunction,
    xor_kernel:  CudaFunction,
    chacha1:     ChachaCommon,
    chacha2:     ChachaCommon,
}

impl ChaChaCudaCorrRng {
    // takes number of bytes to produce, buffer has u32 datatype so will produce
    // buf_size/4 u32s
    pub fn init(dev: Arc<CudaDevice>, seed1: [u32; 8], seed2: [u32; 8]) -> Self {
        let ptx = compile_ptx(ChachaCommon::CHACHA_PTX_SRC).unwrap();

        dev.load_ptx(ptx.clone(), ChachaCommon::CHACHA_FILL_FUNCTION_NAME, &[
            ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
            ChachaCommon::CHACHA_XOR_FUNCTION_NAME,
        ])
        .unwrap();
        let fill_kernel = dev
            .get_func(
                ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
                ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
            )
            .unwrap();
        let xor_kernel = dev
            .get_func(
                ChachaCommon::CHACHA_FILL_FUNCTION_NAME,
                ChachaCommon::CHACHA_XOR_FUNCTION_NAME,
            )
            .unwrap();

        let chacha1 = ChachaCommon::init(&dev, seed1);
        let chacha2 = ChachaCommon::init(&dev, seed2);

        Self {
            fill_kernel,
            xor_kernel,
            chacha1,
            chacha2,
        }
    }

    pub fn fill_rng_into(&mut self, buf: &mut CudaViewMut<u32>, stream: &CudaStream) {
        self.chacha1.fill_rng_into(buf, stream, &self.fill_kernel);
        self.chacha2.fill_rng_into(buf, stream, &self.xor_kernel);
    }

    pub fn fill_my_rng_into(&mut self, buf: &mut CudaViewMut<u32>, stream: &CudaStream) {
        self.chacha1.fill_rng_into(buf, stream, &self.fill_kernel);
    }

    pub fn fill_their_rng_into(&mut self, buf: &mut CudaViewMut<u32>, stream: &CudaStream) {
        self.chacha2.fill_rng_into(buf, stream, &self.fill_kernel);
    }

    pub fn advance_by_bytes(&mut self, bytes: u64) {
        assert!(bytes % 64 == 0, "bytes must be a multiple of 64");
        let num_ks_calls = bytes / 64; // we produce 16 u32s per kernel call
                                       // one call to KS produces 16 u32s
        self.chacha1.advance_counter(num_ks_calls);
        self.chacha2.advance_counter(num_ks_calls);
    }
}

#[cfg(test)]
#[cfg(feature = "gpu_dependent")]
mod tests {

    use super::*;
    use crate::helpers::dtoh_on_stream_sync;
    use itertools::izip;

    #[test]
    fn test_chacha_rng() {
        // This call to CudaDevice::new is only used in context of a test - not used in
        // the server binary
        let dev = CudaDevice::new(0).unwrap();
        let stream = dev.fork_default_stream().unwrap();
        let mut rng = ChaChaCudaCorrRng::init(dev.clone(), [0u32; 8], [1u32; 8]);
        let mut buf = dev.alloc_zeros(1024 * 1024).unwrap();
        rng.fill_rng_into(&mut buf.slice_mut(..), &stream);
        let data = dtoh_on_stream_sync(&buf, &dev, &stream).unwrap();
        let zeros = data.iter().filter(|x| x == &&0).count();
        // we would expect no 0s in the output buffer even 1 is 1/4096;
        assert!(zeros <= 1);
        rng.fill_rng_into(&mut buf.slice_mut(..), &stream);
        let data2 = dtoh_on_stream_sync(&buf, &dev, &stream).unwrap();
        assert!(data != data2);
    }

    #[test]
    fn test_correlation() {
        // This call to CudaDevice::new is only used in context of a test - not used in
        // the server binary
        let dev = CudaDevice::new(0).unwrap();
        let stream = dev.fork_default_stream().unwrap();
        let seed1 = [0u32; 8];
        let seed2 = [1u32; 8];
        let seed3 = [2u32; 8];
        let mut rng1 = ChaChaCudaCorrRng::init(dev.clone(), seed1, seed2);
        let mut rng2 = ChaChaCudaCorrRng::init(dev.clone(), seed2, seed3);
        let mut rng3 = ChaChaCudaCorrRng::init(dev.clone(), seed3, seed1);

        let mut buf = dev.alloc_zeros(1024 * 1024).unwrap();
        rng1.fill_rng_into(&mut buf.slice_mut(..), &stream);
        let data1 = dtoh_on_stream_sync(&buf, &dev, &stream).unwrap();
        rng2.fill_rng_into(&mut buf.slice_mut(..), &stream);
        let data2 = dtoh_on_stream_sync(&buf, &dev, &stream).unwrap();
        rng3.fill_rng_into(&mut buf.slice_mut(..), &stream);
        let data3 = dtoh_on_stream_sync(&buf, &dev, &stream).unwrap();
        for (a, b, c) in izip!(data1, data2, data3) {
            assert_eq!(a ^ b ^ c, 0);
        }
    }
}
