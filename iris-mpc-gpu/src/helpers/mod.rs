use crate::threshold_ring::protocol::ChunkShare;
use cudarc::driver::{
    result::{self, memcpy_dtoh_async, memcpy_htod_async, stream},
    sys::{lib, CUdeviceptr, CUstream, CUstream_st, CU_MEMHOSTALLOC_PORTABLE},
    CudaDevice, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError,
    LaunchConfig,
};
use device_manager::DeviceManager;
use query_processor::CudaVec2DSlicerRawPointer;
use std::sync::Arc;

pub mod comm;
pub mod device_manager;
pub mod id_wrapper;
pub mod query_processor;

pub(crate) const DEFAULT_LAUNCH_CONFIG_THREADS: u32 = 256;

pub fn check_max_grid_size(device: &Arc<CudaDevice>, size: usize) {
    let max_grid_dim_x = unsafe {
        cudarc::driver::result::device::get_attribute(
            *device.cu_device(),
            cudarc::driver::sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        )
    }
    .expect("Fetching CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X should work");
    assert!(size <= max_grid_dim_x as usize);
}

pub fn launch_config_from_elements_and_threads(
    num_elements: u32,
    threads: u32,
    device: &Arc<CudaDevice>,
) -> LaunchConfig {
    let num_blocks = num_elements.div_ceil(threads);
    // Check if kernel can be launched
    check_max_grid_size(device, num_blocks as usize);
    LaunchConfig {
        grid_dim:         (num_blocks, 1, 1),
        block_dim:        (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

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
        .map(|(idx, &p)| unsafe { devs[idx].upgrade_device_ptr(p, sizes[idx]) })
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

/// Copy a slice from device to host with respective offsets.
/// # Safety
///
/// The caller must ensure that the `dst` and `src` pointers are valid
/// with the respective offsets
pub unsafe fn dtoh_at_offset(
    dst: u64,
    dst_offset: usize,
    src: CUdeviceptr,
    src_offset: usize,
    len: usize,
    stream_ptr: CUstream,
) {
    unsafe {
        lib()
            .cuMemcpyDtoHAsync_v2(
                (dst + dst_offset as u64) as *mut _,
                (src + src_offset as u64) as CUdeviceptr,
                len,
                stream_ptr,
            )
            .result()
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

pub fn register_host_memory(
    device_manager: Arc<DeviceManager>,
    db: &CudaVec2DSlicerRawPointer,
    max_db_length: usize,
    code_length: usize,
) {
    let max_size = max_db_length / device_manager.device_count();
    for (device_index, device) in device_manager.devices().iter().enumerate() {
        device.bind_to_thread().unwrap();
        unsafe {
            let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                db.limb_0[device_index] as *mut _,
                max_size * code_length,
                CU_MEMHOSTALLOC_PORTABLE,
            );

            let _ = cudarc::driver::sys::lib().cuMemHostRegister_v2(
                db.limb_1[device_index] as *mut _,
                max_size * code_length,
                CU_MEMHOSTALLOC_PORTABLE,
            );
        }
    }
}
