use cudarc::driver::{result::memset_d8_sync, CudaSlice, DevicePtr, DeviceSlice};
use iris_mpc_common::job::Eye;

use crate::{helpers::device_manager::DeviceManager, threshold_ring::protocol::ChunkShare};

pub struct DistanceCache {
    pub(crate) match_distances_buffer_codes_left: Vec<ChunkShare<u16>>,
    pub(crate) match_distances_buffer_codes_right: Vec<ChunkShare<u16>>,
    pub(crate) match_distances_buffer_masks_left: Vec<ChunkShare<u16>>,
    pub(crate) match_distances_buffer_masks_right: Vec<ChunkShare<u16>>,
    pub(crate) match_distances_counter_left: Vec<CudaSlice<u32>>,
    pub(crate) match_distances_counter_right: Vec<CudaSlice<u32>>,
    pub(crate) match_distances_indices_left: Vec<CudaSlice<u64>>,
    pub(crate) match_distances_indices_right: Vec<CudaSlice<u64>>,
}

impl DistanceCache {
    /// Creates a new `DistanceCache` instance, initializing all buffers and counters.
    /// # Arguments
    /// * `device_manager` - The device manager to allocate buffers on.
    /// * `distance_buffer_len` - The length of the distance buffers to allocate.
    pub fn init(device_manager: &DeviceManager, distance_buffer_len: usize) -> Self {
        let match_distances_buffer_codes_left =
            Self::prepare_match_distances_buffer(device_manager, distance_buffer_len);
        let match_distances_buffer_codes_right =
            Self::prepare_match_distances_buffer(device_manager, distance_buffer_len);
        let match_distances_buffer_masks_left =
            Self::prepare_match_distances_buffer(device_manager, distance_buffer_len);
        let match_distances_buffer_masks_right =
            Self::prepare_match_distances_buffer(device_manager, distance_buffer_len);
        let match_distances_counter_left = Self::prepare_match_distances_counter(device_manager);
        let match_distances_counter_right = Self::prepare_match_distances_counter(device_manager);
        let match_distances_indices_left =
            Self::prepare_match_distances_index(device_manager, distance_buffer_len);
        let match_distances_indices_right =
            Self::prepare_match_distances_index(device_manager, distance_buffer_len);

        Self {
            match_distances_buffer_codes_left,
            match_distances_buffer_codes_right,
            match_distances_buffer_masks_left,
            match_distances_buffer_masks_right,
            match_distances_counter_left,
            match_distances_counter_right,
            match_distances_indices_left,
            match_distances_indices_right,
        }
    }

    /// Returns the buffers for the given eye. Resulting tuple is (codes, masks, counters, indices).
    #[allow(clippy::type_complexity)]
    pub fn get_buffers(
        &self,
        eye: Eye,
    ) -> (
        &[ChunkShare<u16>],
        &[ChunkShare<u16>],
        &[CudaSlice<u32>],
        &[CudaSlice<u64>],
    ) {
        match eye {
            Eye::Left => (
                &self.match_distances_buffer_codes_left,
                &self.match_distances_buffer_masks_left,
                &self.match_distances_counter_left,
                &self.match_distances_indices_left,
            ),
            Eye::Right => (
                &self.match_distances_buffer_codes_right,
                &self.match_distances_buffer_masks_right,
                &self.match_distances_counter_right,
                &self.match_distances_indices_right,
            ),
        }
    }

    fn prepare_match_distances_buffer(
        device_manager: &DeviceManager,
        max_size: usize,
    ) -> Vec<ChunkShare<u16>> {
        (0..device_manager.device_count())
            .map(|i| {
                let a = device_manager.device(i).alloc_zeros(max_size).unwrap();
                let b = device_manager.device(i).alloc_zeros(max_size).unwrap();

                device_manager.device(i).bind_to_thread().unwrap();
                unsafe {
                    memset_d8_sync(*a.device_ptr(), 0xff, a.num_bytes()).unwrap();
                    memset_d8_sync(*b.device_ptr(), 0xff, b.num_bytes()).unwrap();
                }

                ChunkShare::new(a, b)
            })
            .collect::<Vec<_>>()
    }

    fn prepare_match_distances_counter(device_manager: &DeviceManager) -> Vec<CudaSlice<u32>> {
        (0..device_manager.device_count())
            .map(|i| device_manager.device(i).alloc_zeros(1).unwrap())
            .collect::<Vec<_>>()
    }

    fn prepare_match_distances_index(
        device_manager: &DeviceManager,
        max_size: usize,
    ) -> Vec<CudaSlice<u64>> {
        (0..device_manager.device_count())
            .map(|i| {
                let a = device_manager.device(i).alloc_zeros(max_size).unwrap();
                unsafe {
                    memset_d8_sync(*a.device_ptr(), 0xff, a.num_bytes()).unwrap();
                }
                a
            })
            .collect::<Vec<_>>()
    }

    /// Prepares the match distances buckets for the given device manager and number of buckets.
    /// This only operates on the first device.
    ///
    /// # Arguments
    /// * `device_manager` - The device manager to allocate buffers on.
    /// * `n_buckets` - The number of buckets to prepare.
    pub fn prepare_match_distances_buckets(
        device_manager: &DeviceManager,
        n_buckets: usize,
    ) -> ChunkShare<u32> {
        let a = device_manager.device(0).alloc_zeros(n_buckets).unwrap();
        let b = device_manager.device(0).alloc_zeros(n_buckets).unwrap();
        ChunkShare::new(a, b)
    }
}
