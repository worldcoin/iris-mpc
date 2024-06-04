use cudarc::driver::{CudaSlice, DevicePtr};

pub mod mmap;
pub mod sqs;
pub mod id_wrapper;

pub fn device_ptrs<T>(slice: &Vec<CudaSlice<T>>) -> Vec<u64> {
    slice.iter().map(|s| *s.device_ptr()).collect()
}