use cudarc::driver::{CudaSlice, DevicePtr};

pub mod aws_sigv4;
pub mod id_wrapper;
pub mod kms_dh;
pub mod mmap;
pub mod sqs;

pub fn device_ptrs<T>(slice: &[CudaSlice<T>]) -> Vec<u64> {
    slice.iter().map(|s| *s.device_ptr()).collect()
}
