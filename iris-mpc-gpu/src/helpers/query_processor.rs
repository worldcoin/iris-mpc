use crate::{
    dot::{
        share_db::{DBChunkBuffers, ShareDB, SlicedProcessedDatabase},
        IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
    },
    helpers::device_manager::DeviceManager,
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{free_async, stream},
        sys::{CUdeviceptr, CUstream},
        CudaSlice, CudaStream, DevicePtr, DeviceSlice,
    },
};
use eyre::Result;
use iris_mpc_common::galois_engine::CompactGaloisRingShares;
use std::marker::{Send, Sync};

pub struct StreamAwareCudaSlice<T> {
    pub cu_device_ptr: CUdeviceptr,
    pub len: usize,
    pub stream: CUstream,
    pub _phantom: std::marker::PhantomData<T>,
}

unsafe impl<T: Send> Send for StreamAwareCudaSlice<T> {}
unsafe impl<T: Sync> Sync for StreamAwareCudaSlice<T> {}

impl<T> StreamAwareCudaSlice<T> {
    pub fn device_ptr(&self) -> &CUdeviceptr {
        &self.cu_device_ptr
    }

    pub fn upgrade_ptr_stream(cu_device_ptr: CUdeviceptr, cu_stream: CUstream, len: usize) -> Self {
        StreamAwareCudaSlice {
            cu_device_ptr,
            len,
            stream: cu_stream,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> From<CudaSlice<T>> for StreamAwareCudaSlice<T> {
    fn from(value: CudaSlice<T>) -> Self {
        let res = {
            StreamAwareCudaSlice {
                stream: *value.device().cu_stream(),
                cu_device_ptr: *value.device_ptr(),
                len: value.len(),
                _phantom: std::marker::PhantomData,
            }
        };
        // forgetting the slice is ok since we are going to free up the memory using the
        // `StreamAwareCudaSlice` destructor.
        value.leak();
        res
    }
}

impl<T> Drop for StreamAwareCudaSlice<T> {
    fn drop(&mut self) {
        unsafe {
            // Synchronize the stream to ensure all pending GPU operations on this memory
            // have completed before freeing. This prevents CUDA_ERROR_ILLEGAL_ADDRESS
            // that occurs when memory is freed while still being accessed by GPU operations.
            stream::synchronize(self.stream).unwrap();
            free_async(self.cu_device_ptr, self.stream).unwrap();
        }
    }
}

/// Holds the raw memory pointers for the 2D slices.
/// Memory is not freed when the struct is dropped, but must be freed manually.
#[derive(Debug, Clone)]
pub struct CudaVec2DSlicerRawPointer {
    pub limb_0: Vec<u64>,
    pub limb_1: Vec<u64>,
}

impl<T> From<&CudaVec2DSlicer<T>> for CudaVec2DSlicerRawPointer {
    fn from(slicer: &CudaVec2DSlicer<T>) -> Self {
        CudaVec2DSlicerRawPointer {
            limb_0: slicer.limb_0.iter().map(|s| *s.device_ptr()).collect(),
            limb_1: slicer.limb_1.iter().map(|s| *s.device_ptr()).collect(),
        }
    }
}

impl From<&DBChunkBuffers> for CudaVec2DSlicerRawPointer {
    fn from(buffers: &DBChunkBuffers) -> Self {
        CudaVec2DSlicerRawPointer {
            limb_0: buffers.limb_0.iter().map(|s| *s.device_ptr()).collect(),
            limb_1: buffers.limb_1.iter().map(|s| *s.device_ptr()).collect(),
        }
    }
}

pub struct CudaVec2DSlicer<T> {
    pub limb_0: Vec<StreamAwareCudaSlice<T>>,
    pub limb_1: Vec<StreamAwareCudaSlice<T>>,
}

pub type CudaVec2DSlicerU32 = CudaVec2DSlicer<u32>;
pub type CudaVec2DSlicerU8 = CudaVec2DSlicer<u8>;
pub type CudaVec2DSlicerI8 = CudaVec2DSlicer<i8>;

pub struct CompactQuery {
    pub code_query: CompactGaloisRingShares,
    pub mask_query: CompactGaloisRingShares,
    pub code_query_insert: CompactGaloisRingShares,
    pub mask_query_insert: CompactGaloisRingShares,
}

impl CompactQuery {
    pub fn htod_transfer(
        &self,
        device: &DeviceManager,
        streams: &[CudaStream],
        batch_size: usize,
    ) -> Result<DeviceCompactQuery> {
        Ok(DeviceCompactQuery {
            code_query: device.htod_transfer_query(
                &self.code_query,
                streams,
                batch_size,
                IRIS_CODE_LENGTH,
            )?,
            mask_query: device.htod_transfer_query(
                &self.mask_query,
                streams,
                batch_size,
                MASK_CODE_LENGTH,
            )?,
            code_query_insert: device.htod_transfer_query(
                &self.code_query_insert,
                streams,
                batch_size,
                IRIS_CODE_LENGTH,
            )?,
            mask_query_insert: device.htod_transfer_query(
                &self.mask_query_insert,
                streams,
                batch_size,
                MASK_CODE_LENGTH,
            )?,
        })
    }
}

pub struct DeviceCompactQuery {
    code_query: CudaVec2DSlicerU8,
    mask_query: CudaVec2DSlicerU8,
    pub code_query_insert: CudaVec2DSlicerU8,
    pub mask_query_insert: CudaVec2DSlicerU8,
}

impl DeviceCompactQuery {
    pub fn query_sums(
        &self,
        code_engine: &ShareDB,
        mask_engine: &ShareDB,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) -> Result<DeviceCompactSums> {
        Ok(DeviceCompactSums {
            code_query: code_engine.query_sums(&self.code_query, streams, blass),
            mask_query: mask_engine.query_sums(&self.mask_query, streams, blass),
            code_query_insert: code_engine.query_sums(&self.code_query_insert, streams, blass),
            mask_query_insert: mask_engine.query_sums(&self.mask_query_insert, streams, blass),
        })
    }

    pub fn compute_dot_products(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        db_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        code_engine.dot(
            &self.code_query,
            &(&self.code_query_insert).into(),
            db_sizes,
            offset,
            streams,
            blass,
        );

        mask_engine.dot(
            &self.mask_query,
            &(&self.mask_query_insert).into(),
            db_sizes,
            offset,
            streams,
            blass,
        );
    }

    // TODO(Dragos) function signature can be compressed if there's a large refactor
    // of iris_mpc_gpu to place the 2 engines into one struct and DBs into a single
    // struct.
    #[allow(clippy::too_many_arguments)]
    pub fn dot_products_against_db(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        sliced_code_db: &CudaVec2DSlicerRawPointer,
        sliced_mask_db: &CudaVec2DSlicerRawPointer,
        database_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        code_engine.dot(
            &self.code_query,
            sliced_code_db,
            database_sizes,
            offset,
            streams,
            blass,
        );
        mask_engine.dot(
            &self.mask_query,
            sliced_mask_db,
            database_sizes,
            offset,
            streams,
            blass,
        );
    }
}
pub struct DeviceCompactSums {
    code_query: CudaVec2DSlicerU32,
    mask_query: CudaVec2DSlicerU32,
    pub code_query_insert: CudaVec2DSlicerU32,
    pub mask_query_insert: CudaVec2DSlicerU32,
}

impl DeviceCompactSums {
    pub fn compute_dot_reducers(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        db_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
    ) {
        code_engine.dot_reduce(
            &self.code_query,
            &self.code_query_insert,
            db_sizes,
            offset,
            streams,
        );
        mask_engine.dot_reduce_and_multiply(
            &self.mask_query,
            &self.mask_query_insert,
            db_sizes,
            offset,
            streams,
            2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_dot_reducer_against_db(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        sliced_code_db: &SlicedProcessedDatabase,
        sliced_mask_db: &SlicedProcessedDatabase,
        database_sizes: &[usize],
        offset: usize,
        streams: &[CudaStream],
    ) {
        code_engine.dot_reduce(
            &self.code_query,
            &sliced_code_db.code_sums_gr,
            database_sizes,
            offset,
            streams,
        );
        mask_engine.dot_reduce_and_multiply(
            &self.mask_query,
            &sliced_mask_db.code_sums_gr,
            database_sizes,
            offset,
            streams,
            2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_dot_reducer_against_prepared_db(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        code_sums_gr: &CudaVec2DSlicer<u32>,
        mask_sums_gr: &CudaVec2DSlicer<u32>,
        database_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        code_engine.dot_reduce(&self.code_query, code_sums_gr, database_sizes, 0, streams);
        mask_engine.dot_reduce_and_multiply(
            &self.mask_query,
            mask_sums_gr,
            database_sizes,
            0,
            streams,
            2,
        );
    }
}
