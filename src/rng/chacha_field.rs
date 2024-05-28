use std::sync::Arc;

use cudarc::{
    driver::{
        result, CudaDevice, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
        DeviceSlice, LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};

use super::chacha::ChaChaCtx;

pub struct ChaChaCudaFeRng {
    // the total buffer size
    buf_size: usize,
    // the amount of valid values in the buffer
    valid_buffer_size: usize,
    /// the device to use
    dev: Arc<CudaDevice>,
    /// compiled and loaded kernels for our 2 functions
    kernels: Vec<CudaFunction>,
    /// a reference to the current chunk of the rng output in the cuda device
    rng_chunk: CudaSlice<u32>,
    /// a buffer to copy the output to in the host device
    output_buffer: Vec<u32>,
    /// the current state of the chacha rng
    chacha_ctx: ChaChaCtx,
}

const CHACHA_PTX_SRC: &str = include_str!("chacha_field.cu");
const CHACHA_FUNCTION_NAME: &str = "chacha12";
const FIELD_FUNCTION_NAME: &str = "fix_fe";

// probability calculation says that prob that more than 24 values of 1024 are not in field is less than 1/2^128
const MIN_U16_BUF_ELEMENTS: usize = 1024;
const OK_U16_BUF_ELEMENTS: usize = 1000;

impl ChaChaCudaFeRng {
    ///
    /// # Arguments
    /// `buf_size`: takes number of u16 elements to produce per call to rng(), needs to be a multiple of 1000
    /// `dev`: the cuda device to run the RNG on
    /// `key`: the seed to use for the RNG
    pub fn init(buf_size: usize, dev: Arc<CudaDevice>, seed: [u32; 8]) -> Self {
        let mut kernels = Vec::new();
        let ptx = compile_ptx(CHACHA_PTX_SRC).unwrap();

        assert!(
            buf_size % (OK_U16_BUF_ELEMENTS) == 0,
            "buf_size must be a multiple of 1000 atm"
        );

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

        kernels.push(function);
        kernels.push(fe_fix_function);

        let valid_buffer_size = buf_size;
        let buf_size = (valid_buffer_size / OK_U16_BUF_ELEMENTS) * MIN_U16_BUF_ELEMENTS;

        let buf = vec![0u32; (buf_size / std::mem::size_of::<u32>()) * std::mem::size_of::<u16>()];
        let rng_chunk = dev.htod_sync_copy(&buf).unwrap();

        let chacha_ctx = ChaChaCtx::init(seed, 0, 0);

        Self {
            buf_size,
            valid_buffer_size,
            dev,
            kernels,
            rng_chunk,
            output_buffer: buf,
            chacha_ctx,
        }
    }

    // pub fn fill_rng(&mut self) {
    //     self.fill_rng_no_host_copy(None);

    //     self.dev
    //         .dtoh_sync_copy_into(&self.rng_chunk, &mut self.output_buffer)
    //         .unwrap();
    // }

    pub fn fill_rng_no_host_copy(&mut self, buf_size: usize, stream: &CudaStream) {
        let num_ks_calls = self.buf_size / 32; // one call to KS produces 32 u16s
        let threads_per_block = 256; // todo sync with kernel
        let blocks_per_grid = (num_ks_calls + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            block_dim: (threads_per_block as u32, 1, 1),
            grid_dim: (blocks_per_grid as u32, 1, 1),
            shared_mem_bytes: 0, // do we need this since we use __shared__ in kernel?
        };

        self.dev.bind_to_thread().unwrap();
        let state_slice: CudaSlice<u32> = unsafe {
            self.dev
                .alloc(std::mem::size_of::<u32>() * self.chacha_ctx.state.len())
                .unwrap()
        };
        unsafe {
            result::memcpy_htod_async(*state_slice.device_ptr(), &self.chacha_ctx.state, stream.stream);
        }

        let buf_size = (buf_size / OK_U16_BUF_ELEMENTS) * MIN_U16_BUF_ELEMENTS;
        let len = (buf_size / std::mem::size_of::<u32>()) * std::mem::size_of::<u16>();
        unsafe {
            self.kernels[0]
                .clone()
                .launch_on_stream(stream, cfg, (&mut self.rng_chunk, &state_slice, len as u64))
                .unwrap();
        }

        // increment the state counter of the ChaChaRng with the number of produced blocks
        let mut counter = self.chacha_ctx.get_counter();
        counter += self.buf_size as u64 / 32; // one call to KS produces 32 u16s
        self.chacha_ctx.set_counter(counter);

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
                .launch_on_stream(
                    stream,
                    cfg,
                    (
                        &mut self.rng_chunk,
                        u32::try_from(self.valid_buffer_size).unwrap(),
                    ),
                )
                .unwrap();
        }
        // do we need this synchronize?
        // self.dev.synchronize().unwrap();
    }
    pub fn data(&self) -> &[u16] {
        &bytemuck::cast_slice(self.output_buffer.as_slice())[0..self.valid_buffer_size]
    }
    pub fn num_valid(&self) -> usize {
        self.valid_buffer_size
    }

    pub fn get_mut_chacha(&mut self) -> &mut ChaChaCtx {
        &mut self.chacha_ctx
    }

    pub fn cuda_slice(&self) -> &CudaSlice<u32> {
        &self.rng_chunk
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    const P: u16 = 65519;
    // #[test]
    // fn test_chacha_rng() {
    //     let mut rng =
    //         ChaChaCudaFeRng::init(1000 * 1000 * 50, CudaDevice::new(0).unwrap(), [0u32; 8]);
    //     rng.fill_rng();
    //     assert!(rng.data().iter().all(|&x| x < P));
    //     let data = rng.data().to_vec();
    //     rng.fill_rng();
    //     assert!(rng.data().iter().all(|&x| x < P));
    //     assert!(&data[..] != rng.data());
    // }
}
