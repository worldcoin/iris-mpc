use cudarc::driver::{
    result::{self, stream::synchronize},
    CudaDevice, CudaSlice,
};
use gpu_iris_mpc::{helpers::device_ptrs, threshold_ring::protocol::ChunkShare};
use std::sync::Arc;

/// Wrapper to copy device to device with offsets
pub fn dtod_at_offset(
    dst: u64,
    dst_offset: usize,
    src: u64,
    src_offset: usize,
    len: usize,
    stream_ptr: u64,
) {
    unsafe {
        result::memcpy_dtod_async(
            dst + dst_offset as u64,
            src + src_offset as u64,
            len,
            stream_ptr as *mut _,
        )
        .unwrap();
    }
}

/// Helper to convert from raw device pointers to `ChunkShare`
pub fn device_ptrs_to_shares<T>(
    a: &[u64],
    b: &[u64],
    lens: &[usize],
    devs: &[Arc<CudaDevice>],
) -> Vec<ChunkShare<T>> {
    let a = device_ptrs_to_slices(a, lens, devs);
    let b = device_ptrs_to_slices(b, lens, devs);

    a.into_iter()
        .zip(b.into_iter())
        .map(|(a, b)| ChunkShare::new(a, b))
        .collect::<Vec<_>>()
}

/// Reset device pointers to given host memory async
pub fn reset_device_ptrs(devs: &[Arc<CudaDevice>], dst: &[u64], src: &[u32], streams: &[u64]) {
    for i in 0..devs.len() {
        devs[i].bind_to_thread().unwrap();
        unsafe { result::memcpy_htod_async(dst[i], src, streams[i] as *mut _) }.unwrap();
    }
}

/// Synchronize all streams
pub fn await_streams(streams: &[u64]) {
    for i in 0..streams.len() {
        unsafe {
            synchronize(streams[i] as *mut _).unwrap();
        }
    }
}

/// Helper to convert CudaSlice to raw pointer tuples
#[allow(clippy::type_complexity)]
pub fn slice_tuples_to_ptrs(
    tuple: &(
        (Vec<CudaSlice<i8>>, Vec<CudaSlice<i8>>),
        (Vec<CudaSlice<u32>>, Vec<CudaSlice<u32>>),
    ),
) -> ((Vec<u64>, Vec<u64>), (Vec<u64>, Vec<u64>)) {
    (
        (device_ptrs(&tuple.0 .0), device_ptrs(&tuple.0 .1)),
        (device_ptrs(&tuple.1 .0), device_ptrs(&tuple.1 .1)),
    )
}

/// Helper to convert raw device ptrs to CudaSlices
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
