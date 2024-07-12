use crate::{
    dot::share_db::{ShareDB, SlicedProcessedDatabase},
    helpers::device_manager::DeviceManager,
    setup::galois_engine::CompactGaloisRingShares,
};
use cudarc::{
    cublas::CudaBlas,
    driver::{CudaSlice, CudaStream},
};

pub type CudaSliceMatrixTuple<T> = (Vec<CudaSlice<T>>, Vec<CudaSlice<T>>);
pub type RefCudaSliceMatrixTuple<'a, T> = (&'a Vec<CudaSlice<T>>, &'a Vec<CudaSlice<T>>);
pub type CudaSliceMatrixTupleU8 = CudaSliceMatrixTuple<u8>;
pub type CudaSliceMatrixTupleU32 = CudaSliceMatrixTuple<u32>;

// pub struct NgCudaVec1DSlicer<T> {
//     pub entry: Vec<CudaSlice<T>>,
// }

// impl<T> AsRef<NgCudaVec1DSlicer<T>> for NgCudaVec1DSlicer<T> {
//     fn as_ref(&self) -> &NgCudaVec1DSlicer<T> {
//         self
//     }
// }
pub struct NgCudaVec2DSlicer<T> {
    pub entry_0: Vec<CudaSlice<T>>,
    pub entry_1: Vec<CudaSlice<T>>,
}

impl<T> NgCudaVec2DSlicer<T> {
    pub fn get0_ref(&self) -> &Vec<CudaSlice<T>> {
        &self.entry_0
    }
    pub fn get1_ref(&self) -> &Vec<CudaSlice<T>> {
        &self.entry_1
    }

    pub fn as_tuple_refs(&self) -> (&Vec<CudaSlice<T>>, &Vec<CudaSlice<T>>) {
        (&self.entry_0, &self.entry_0)
    }
}

pub type NgCudaVec2DSlicerU32 = NgCudaVec2DSlicer<u32>;
pub type NgCudaVec2DSlicerU8 = NgCudaVec2DSlicer<u8>;
pub type NgCudaVec2DSlicerI8 = NgCudaVec2DSlicer<i8>;

pub type RefCudaSliceMatrixTupleU32<'a> = RefCudaSliceMatrixTuple<'a, u32>;

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
    code_query:            NgCudaVec2DSlicerU8,
    mask_query:            NgCudaVec2DSlicerU8,
    pub code_query_insert: NgCudaVec2DSlicerU8,
    pub mask_query_insert: NgCudaVec2DSlicerU8,
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
    code_query:            NgCudaVec2DSlicerU32,
    mask_query:            NgCudaVec2DSlicerU32,
    pub code_query_insert: NgCudaVec2DSlicerU32,
    pub mask_query_insert: NgCudaVec2DSlicerU32,
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
