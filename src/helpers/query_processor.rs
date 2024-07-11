use crate::{
    dot::share_db::{ShareDB, SlicedProcessedDatabase},
    helpers::{device_manager::DeviceManager, device_ptrs},
    setup::galois_engine::CompactGaloisRingShares,
};
use cudarc::{
    cublas::CudaBlas,
    driver::{
        result::{malloc_async, memcpy_htod_async},
        sys::CUdeviceptr,
        CudaSlice, CudaStream,
    },
};

pub type CudaSliceTuple = (Vec<CudaSlice<u8>>, Vec<CudaSlice<u8>>);

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
            code_query:        device.custom_htod_transfer_query(&self.code_query, streams)?,
            mask_query:        device.custom_htod_transfer_query(&self.mask_query, streams)?,
            code_query_insert: device
                .custom_htod_transfer_query(&self.code_query_insert, streams)?,
            mask_query_insert: device
                .custom_htod_transfer_query(&self.mask_query_insert, streams)?,
        })
    }
}

pub struct DeviceCompactQuery {
    code_query:            CudaSliceTuple,
    mask_query:            CudaSliceTuple,
    pub code_query_insert: CudaSliceTuple,
    pub mask_query_insert: CudaSliceTuple,
}

impl DeviceCompactQuery {
    pub fn query_sums(
        &self,
        code_engine: &ShareDB,
        mask_engine: &ShareDB,
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) -> eyre::Result<DeviceCompactSums> {
        Ok(DeviceCompactSums {
            code_query:        code_engine.custom_query_sums(&self.code_query, streams, blass),
            mask_query:        mask_engine.custom_query_sums(&self.mask_query, streams, blass),
            code_query_insert: code_engine.custom_query_sums(
                &self.code_query_insert,
                streams,
                blass,
            ),
            mask_query_insert: mask_engine.custom_query_sums(
                &self.mask_query_insert,
                streams,
                blass,
            ),
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
        code_engine.custom_dot(
            &self.code_query,
            &self.code_query_insert,
            &db_sizes,
            streams,
            blass,
        );

        mask_engine.custom_dot(
            &self.mask_query,
            &self.mask_query_insert,
            &db_sizes,
            streams,
            blass,
        );
    }

    pub fn code_dot_product_against_db(
        &self,
        engine: &mut ShareDB,
        sliced_database: &SlicedProcessedDatabase,
        database_sizes: &[usize],
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        engine.custom_dot2(
            &self.code_query,
            &(&sliced_database.code_gr0, &sliced_database.code_gr1),
            &database_sizes,
            streams,
            blass,
        );
    }

    pub fn mask_dot_product_against_db(
        &self,
        engine: &mut ShareDB,
        sliced_database: &SlicedProcessedDatabase,
        database_sizes: &[usize],
        streams: &[CudaStream],
        blass: &[CudaBlas],
    ) {
        engine.custom_dot2(
            &self.mask_query,
            &(&sliced_database.code_gr0, &sliced_database.code_gr1),
            &database_sizes,
            streams,
            blass,
        );
    }
}
pub struct DeviceCompactSums {
    code_query:            (Vec<CUdeviceptr>, Vec<CUdeviceptr>),
    mask_query:            (Vec<CUdeviceptr>, Vec<CUdeviceptr>),
    pub code_query_insert: (Vec<CUdeviceptr>, Vec<CUdeviceptr>),
    pub mask_query_insert: (Vec<CUdeviceptr>, Vec<CUdeviceptr>),
}

impl DeviceCompactSums {
    pub fn compute_dot_reducers(
        &self,
        code_engine: &mut ShareDB,
        mask_engine: &mut ShareDB,
        db_sizes: &[usize],
        streams: &[CudaStream],
    ) {
        code_engine.dot_reduce(
            &self.code_query,
            &self.code_query_insert,
            &db_sizes,
            &streams,
        );

        mask_engine.dot_reduce(
            &self.mask_query,
            &self.mask_query_insert,
            &db_sizes,
            &streams,
        );
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
            &(
                device_ptrs(&sliced_code_db.code_sums_gr0),
                device_ptrs(&sliced_code_db.code_sums_gr1),
            ),
            &database_sizes,
            streams,
        );
        mask_engine.dot_reduce(
            &self.mask_query,
            &(
                device_ptrs(&sliced_mask_db.code_sums_gr0),
                device_ptrs(&sliced_mask_db.code_sums_gr1),
            ),
            &database_sizes,
            streams,
        );
    }
}
