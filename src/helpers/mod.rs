use cudarc::driver::{
    result::{memcpy_dtoh_async, memcpy_htod_async, stream},
    sys::CUdeviceptr,
    CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError,
};
use std::sync::Arc;

pub mod aws;
pub mod aws_sigv4;
pub mod id_wrapper;
pub mod kms_dh;
pub mod mmap;
pub mod sqs;
pub mod task_monitor;

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

pub fn dtoh_on_stream_sync<T: Default + Clone, U: DevicePtr<T>>(
    input: &U,
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> Result<Vec<T>, DriverError> {
    device.bind_to_thread()?;
    let mut buf = vec![T::default(); input.len()];
    // SAFETY: for the below async copy call, the `buf` needs to not be moved or
    // dropped until the copy is complete therefore, we synchronize the stream
    // before returning the buffer or wrapping it in `Ok`
    unsafe {
        memcpy_dtoh_async(&mut buf, *input.device_ptr(), stream.stream)?;
        stream::synchronize(stream.stream).unwrap()
    }

    Ok(buf)
}

pub fn htod_on_stream_sync<T: DeviceRepr>(
    input: &[T],
    device: &Arc<CudaDevice>,
    stream: &CudaStream,
) -> Result<CudaSlice<T>, DriverError> {
    device.bind_to_thread()?;
    // SAFETY: for the below async copy call, we need to allocate a buffer of the
    // input size. We can use alloc, which uses uninitialized memory, since we
    // are going to copy the input into it immediately afterwards.
    // The synchronization is done to ensure that the input slice is valid for the
    // duration of the otherwise async copy
    let buf = unsafe {
        let mut buf = device.alloc(input.len())?;
        memcpy_htod_async(*buf.device_ptr_mut(), input, stream.stream)?;
        stream::synchronize(stream.stream).unwrap();
        buf
    };
    Ok(buf)
}
