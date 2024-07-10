use crate::{
    helpers::device_manager::DeviceManager, setup::galois_engine::CompactGaloisRingShares,
};
use cudarc::driver::{
    result::{malloc_async, memcpy_htod_async},
    CudaSlice, CudaStream,
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
            code_query: device.custom_htod_transfer_query(&self.code_query, streams)?,
            mask_query: device.custom_htod_transfer_query(&self.mask_query, streams)?,
        })
    }
}

pub struct DeviceCompactQuery {
    code_query: CudaSliceTuple,
    mask_query: CudaSliceTuple,
    // code_query_insert: CompactGaloisRingShares,
    // mask_query_insert: CompactGaloisRingShares,
}
