use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;

pub mod aws_sigv4;
pub mod id_wrapper;
pub mod kms_dh;
pub mod mmap;
pub mod sqs;

pub fn device_ptrs<T>(slice: &[CudaSlice<T>]) -> Vec<u64> {
    slice.iter().map(|s| *s.device_ptr()).collect()
}

pub fn device_ptrs_to_slices<T>(
    ptrs: &[u64],
    sizes: &[usize],
    devs: &[Arc<CudaDevice>],
) -> Vec<CudaSlice<T>> {
    ptrs.iter()
        .enumerate()
        .map(|(idx, &p)| CudaSlice {
            cu_device_ptr: p,
            len:           sizes[idx],
            device:        devs[idx].clone(),
            host_buf:      None,
        })
        .collect()
}
