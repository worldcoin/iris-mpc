use cudarc::driver::{
    result::{memcpy_dtoh_async, memcpy_htod_async},
    sys::CUdeviceptr,
    CudaDevice, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice,
    DriverError,
};
use std::sync::Arc;

pub mod aws_sigv4;
pub mod id_wrapper;
pub mod kms_dh;
pub mod mmap;
pub mod sqs;

pub fn device_ptrs<T>(slice: &[CudaSlice<T>]) -> Vec<CUdeviceptr> {
    slice.iter().map(|s| *s.device_ptr()).collect()
}

pub fn device_ptrs_to_slices<T>(
    ptrs: &[CUdeviceptr],
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

pub fn dtoh_async<T: Default + Clone, U: DevicePtr<T>>(
    // input: &CudaView<T>,
    input: &U,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> Result<Vec<T>, DriverError> {
    device.bind_to_thread()?;
    let mut buf = vec![T::default(); input.len()];
    unsafe {
        memcpy_dtoh_async(&mut buf, *input.device_ptr(), stream.stream)?;
    }

    Ok(buf)
}

pub fn htod_async<T: DeviceRepr>(
    input: &[T],
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> Result<CudaSlice<T>, DriverError> {
    device.bind_to_thread()?;
    let mut buf = unsafe { device.alloc(input.len()) }?;
    unsafe {
        memcpy_htod_async(*buf.device_ptr_mut(), input, stream.stream)?;
    }
    Ok(buf)
}
