use crate::threshold_ring::protocol::ChunkShare;
use cudarc::driver::{
    result::{self, memcpy_dtoh_async, memcpy_htod_async, stream},
    sys::{CUdeviceptr, CUstream, CUstream_st},
    CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError,
};
use std::sync::Arc;

pub mod aws;
pub mod aws_sigv4;
pub mod device_manager;
pub mod id_wrapper;
pub mod key_pair;
pub mod kms_dh;
pub mod query_processor;
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

pub fn await_streams(streams: &mut [&mut CUstream_st]) {
    for i in 0..streams.len() {
        // SAFETY: these streams have already been created, and the caller holds a
        // reference to their CudaDevice, which makes sure they aren't dropped.
        unsafe {
            stream::synchronize(streams[i]).unwrap();
        }
    }
}

pub fn device_ptrs_to_shares<T>(
    a: &[CUdeviceptr],
    b: &[CUdeviceptr],
    lens: &[usize],
    devs: &[Arc<CudaDevice>],
) -> Vec<ChunkShare<T>> {
    let a = device_ptrs_to_slices(a, lens, devs);
    let b = device_ptrs_to_slices(b, lens, devs);

    a.into_iter()
        .zip(b)
        .map(|(a, b)| ChunkShare::new(a, b))
        .collect::<Vec<_>>()
}

/// Copy a slice between on-device buffers with respective offsets.
/// # Safety
///
/// The caller must ensure that the `dst` and `src` pointers are valid
/// with the respective offsets
pub unsafe fn dtod_at_offset(
    dst: CUdeviceptr,
    dst_offset: usize,
    src: CUdeviceptr,
    src_offset: usize,
    len: usize,
    stream_ptr: CUstream,
) {
    unsafe {
        result::memcpy_dtod_async(
            dst + dst_offset as CUdeviceptr,
            src + src_offset as CUdeviceptr,
            len,
            stream_ptr,
        )
        .unwrap();
    }
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
