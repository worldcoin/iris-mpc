use crate::{
    dot::share_db::{ShareDB, SlicedProcessedDatabase},
    helpers::device_manager::DeviceManager,
    setup::galois_engine::CompactGaloisRingShares,
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::free_async,
        sys::{CUdeviceptr, CUstream},
        CudaSlice, CudaStream, DevicePtr,
    },
};
use std::{
    marker::{Send, Sync},
    pin::Pin,
};

pub struct StreamAwareCudaSlice<T> {
    pub cu_device_ptr: CUdeviceptr,
    pub len:           usize,
    pub stream:        CUstream,
    pub host_buf:      Option<Pin<Vec<T>>>,
}

unsafe impl<T: Send> Send for StreamAwareCudaSlice<T> {}
unsafe impl<T: Sync> Sync for StreamAwareCudaSlice<T> {}

impl<T> StreamAwareCudaSlice<T> {
    pub fn device_ptr(&self) -> &CUdeviceptr {
        &self.cu_device_ptr
    }

    pub fn upgrade_device_with_stream(
        cu_device_ptr: CUdeviceptr,
        cu_stream: CUstream,
        len: usize,
    ) -> Self {
        StreamAwareCudaSlice {
            cu_device_ptr,
            len,
            stream: cu_stream,
            host_buf: None,
        }
    }
}

impl<T> From<CudaSlice<T>> for StreamAwareCudaSlice<T> {
    fn from(value: CudaSlice<T>) -> Self {
        let res = StreamAwareCudaSlice {
            stream:        *value.device().cu_stream(),
            cu_device_ptr: *value.device_ptr(),
            len:           value.len,
            host_buf:      None,
        };
        // forgetting the slice is ok since we are going to free up the memory using the
        // `StreamAwareCudaSlice` destructor.
        std::mem::forget(value);
        res
    }
}

impl<T> Drop for StreamAwareCudaSlice<T> {
    fn drop(&mut self) {
        unsafe {
            free_async(self.cu_device_ptr, self.stream).unwrap();
        }
    }
}

pub struct CudaVec2DSlicer<T> {
    pub limb_0: Vec<StreamAwareCudaSlice<T>>,
    pub limb_1: Vec<StreamAwareCudaSlice<T>>,
}

impl<T> CudaVec2DSlicer<T> {
    pub fn get0_ref(&self) -> &Vec<StreamAwareCudaSlice<T>> {
        &self.limb_0
    }
    pub fn get1_ref(&self) -> &Vec<StreamAwareCudaSlice<T>> {
        &self.limb_1
    }

    pub fn as_tuple_refs(&self) -> (&Vec<StreamAwareCudaSlice<T>>, &Vec<StreamAwareCudaSlice<T>>) {
        (&self.limb_0, &self.limb_0)
    }
}

pub type CudaVec2DSlicerU32 = CudaVec2DSlicer<u32>;
pub type CudaVec2DSlicerU8 = CudaVec2DSlicer<u8>;
pub type CudaVec2DSlicerI8 = CudaVec2DSlicer<i8>;

pub struct CompactQuery {
    pub code_query:        CompactGaloisRingShares,
    pub mask_query:        CompactGaloisRingShares,
    pub code_query_insert: CompactGaloisRingShares,
    pub mask_query_insert: CompactGaloisRingShares,
}

impl CompactQuery {
    pub fn htod_transfer(
        &self,
        device: &DeviceManager,
        streams: &[CudaStream],
    ) -> eyre::Result<DeviceCompactQuery> {
        Ok(DeviceCompactQuery {
            code_query:        device.htod_transfer_query(&self.code_query, streams)?,
            mask_query:        device.htod_transfer_query(&self.mask_query, streams)?,
            code_query_insert: device.htod_transfer_query(&self.code_query_insert, streams)?,
            mask_query_insert: device.htod_transfer_query(&self.mask_query_insert, streams)?,
        })
    }
}

pub struct DeviceCompactQuery {
    code_query:            CudaVec2DSlicerU8,
    mask_query:            CudaVec2DSlicerU8,
    pub code_query_insert: CudaVec2DSlicerU8,
    pub mask_query_insert: CudaVec2DSlicerU8,
}

// TODO(Dragos) need to make query_sums allocate slices instead.
impl DeviceCompactQuery {
    pub fn query_sums(
        &self,
        code_engine: &ShareDB,
        mask_engine: &ShareDB,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) -> eyre::Result<DeviceCompactSums> {
        Ok(DeviceCompactSums {
            code_query:        code_engine.query_sums(&self.code_query, streams, blass),
            mask_query:        mask_engine.query_sums(&self.mask_query, streams, blass),
            code_query_insert: code_engine.query_sums(&self.code_query_insert, streams, blass),
            mask_query_insert: mask_engine.query_sums(&self.mask_query_insert, streams, blass),
        })
    }

    pub fn compute_dot_products(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        db_sizes: &[usize],
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        code_engine.dot(
            &self.code_query,
            &self.code_query_insert,
            db_sizes,
            streams,
            blass,
        );

        mask_engine.dot(
            &self.mask_query,
            &self.mask_query_insert,
            db_sizes,
            streams,
            blass,
        );
    }

    // TODO(Dragos) function signature can be compressed if there's a large refactor
    // of server.rs to place the 2 engines into one struct and DBs into a single
    // struct.
    #[allow(clippy::too_many_arguments)]
    pub fn dot_products_against_db(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        sliced_code_db: &SlicedProcessedDatabase,
        sliced_mask_db: &SlicedProcessedDatabase,
        database_sizes: &[usize],
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        code_engine.dot(
            &self.code_query,
            &sliced_code_db.code_gr,
            database_sizes,
            streams,
            blass,
        );
        mask_engine.dot(
            &self.mask_query,
            &sliced_mask_db.code_gr,
            database_sizes,
            streams,
            blass,
        );
    }
}
pub struct DeviceCompactSums {
    code_query:            CudaVec2DSlicerU32,
    mask_query:            CudaVec2DSlicerU32,
    pub code_query_insert: CudaVec2DSlicerU32,
    pub mask_query_insert: CudaVec2DSlicerU32,
}

impl DeviceCompactSums {
    pub fn compute_dot_reducers(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        code_engine.dot_reduce(&self.code_query, &self.code_query_insert, db_sizes, streams);
        mask_engine.dot_reduce(&self.mask_query, &self.mask_query_insert, db_sizes, streams);
    }

    pub fn compute_dot_reducer_against_db(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        sliced_code_db: &SlicedProcessedDatabase,
        sliced_mask_db: &SlicedProcessedDatabase,
        database_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        code_engine.dot_reduce(
            &self.code_query,
            &sliced_code_db.code_sums_gr,
            database_sizes,
            streams,
        );
        mask_engine.dot_reduce(
            &self.mask_query,
            &sliced_mask_db.code_sums_gr,
            database_sizes,
            streams,
        );
    }
}
