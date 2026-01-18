use std::collections::HashMap;

use crate::{
    helpers::{device_manager::DeviceManager, dtoh_on_stream_sync},
    threshold_ring::protocol::ChunkShare,
};
use ampc_anon_stats::types::Eye;
use ampc_anon_stats::AnonStatsOperation;
use cudarc::driver::{result::memset_d8_sync, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use iris_mpc_common::ROTATIONS;
use itertools::{izip, Itertools};

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

    pub fn load_counters(&self, device_manager: &DeviceManager, eye: Eye) -> Vec<usize> {
        let (_, _, counters, _) = self.get_buffers(eye);
        device_manager
            .devices()
            .iter()
            .enumerate()
            .map(|(i, device)| device.dtoh_sync_copy(&counters[i]).unwrap()[0] as usize)
            .collect::<Vec<_>>()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_additions_since(
        &self,
        device_manager: &DeviceManager,
        eye: Eye,
        old_counters: Vec<usize>,
        max_internal_buffer_size: usize,
        streams: &[CudaStream],
        operations: &[AnonStatsOperation],
        max_query_length: u64,
    ) -> Vec<OneSidedDistanceCache> {
        let (codes, masks, counters, indices_gpu) = self.get_buffers(eye);
        let counters = device_manager
            .devices()
            .iter()
            .enumerate()
            .map(|(i, device)| device.dtoh_sync_copy(&counters[i]).unwrap()[0] as usize)
            .collect::<Vec<_>>();
        let mut indices: Vec<Vec<u64>> = indices_gpu
            .iter()
            .enumerate()
            .map(|(i, x)| dtoh_on_stream_sync(x, &device_manager.device(i), &streams[i]).unwrap())
            .collect::<Vec<Vec<u64>>>();

        let mut codes_a: Vec<Vec<u16>> = codes
            .iter()
            .enumerate()
            .map(|(i, x)| {
                dtoh_on_stream_sync(&x.a, &device_manager.device(i), &streams[i]).unwrap()
            })
            .collect::<Vec<Vec<u16>>>();
        let mut codes_b: Vec<Vec<u16>> = codes
            .iter()
            .enumerate()
            .map(|(i, x)| {
                dtoh_on_stream_sync(&x.b, &device_manager.device(i), &streams[i]).unwrap()
            })
            .collect::<Vec<Vec<u16>>>();
        let mut masks_a: Vec<Vec<u16>> = masks
            .iter()
            .enumerate()
            .map(|(i, x)| {
                dtoh_on_stream_sync(&x.a, &device_manager.device(i), &streams[i]).unwrap()
            })
            .collect::<Vec<Vec<u16>>>();
        let mut masks_b: Vec<Vec<u16>> = masks
            .iter()
            .enumerate()
            .map(|(i, x)| {
                dtoh_on_stream_sync(&x.b, &device_manager.device(i), &streams[i]).unwrap()
            })
            .collect::<Vec<Vec<u16>>>();

        // Stable permutation based on indices
        for i in 0..indices.len() {
            let mut perm: Vec<usize> = (0..indices[i].len()).collect();
            perm.sort_by_key(|&j| indices[i][j]);

            indices[i] = perm.iter().map(|&j| indices[i][j]).collect::<Vec<u64>>();
            codes_a[i] = perm.iter().map(|&j| codes_a[i][j]).collect::<Vec<u16>>();
            codes_b[i] = perm.iter().map(|&j| codes_b[i][j]).collect::<Vec<u16>>();
            masks_a[i] = perm.iter().map(|&j| masks_a[i][j]).collect::<Vec<u16>>();
            masks_b[i] = perm.iter().map(|&j| masks_b[i][j]).collect::<Vec<u16>>();
        }

        let mut maps = old_counters
            .iter()
            .zip(counters.iter())
            .map(|(old, new)| {
                assert!(new >= old);
                HashMap::<u64, Vec<CpuDistanceShare>>::with_capacity(new - old)
            })
            .collect_vec();

        for (map, ind, code_a_v, code_b_v, mask_a_v, mask_b_v, old, new) in izip!(
            &mut maps,
            &indices,
            &codes_a,
            &codes_b,
            &masks_a,
            &masks_b,
            old_counters,
            counters
        ) {
            if new >= max_internal_buffer_size {
                tracing::info!(
                    "While saving distances for 2d stats, internal buffer full: size {}, max size {}, eye {:?}. Skipping saving distances.",
                    new,
                    max_internal_buffer_size,
                    eye
                );
                continue;
            }
            for (&idx, &c_a, &c_b, &m_a, &m_b) in izip!(
                ind.iter(),
                code_a_v.iter(),
                code_b_v.iter(),
                mask_a_v.iter(),
                mask_b_v.iter()
            )
            .skip(old)
            .take(new - old)
            {
                let operation = operation_from_match_id(idx, operations, max_query_length);
                map.entry(idx / ROTATIONS as u64)
                    .or_insert_with(|| Vec::with_capacity(4))
                    .push(CpuDistanceShare {
                        idx,
                        code_a: c_a,
                        code_b: c_b,
                        mask_a: m_a,
                        mask_b: m_b,
                        operation,
                    });
            }
        }

        maps.into_iter()
            .map(|map| OneSidedDistanceCache { map })
            .collect::<Vec<_>>()
    }
}

#[derive(Copy, Clone, Debug)]
/// Represents a share of mask and code dot products, stored on the CPU.
pub struct CpuDistanceShare {
    pub idx: u64,
    pub code_a: u16,
    pub code_b: u16,
    pub mask_a: u16,
    pub mask_b: u16,
    pub operation: AnonStatsOperation,
}

fn operation_from_match_id(
    match_idx: u64,
    operations: &[AnonStatsOperation],
    max_query_length: u64,
) -> AnonStatsOperation {
    if operations.is_empty() || max_query_length == 0 {
        return AnonStatsOperation::Uniqueness;
    }
    let query_idx = (match_idx % max_query_length) as usize;
    let request_idx = query_idx / ROTATIONS;
    operations
        .get(request_idx)
        .copied()
        .unwrap_or(AnonStatsOperation::Uniqueness)
}

/// Represents a cache for one-sided distances, stored on the CPU.
/// This will be used to do a union with the one-sided distances from the other side, to arrive at the distances for actual matches.
#[derive(Debug, Default, Clone)]
pub struct OneSidedDistanceCache {
    map: HashMap<u64, Vec<CpuDistanceShare>>,
}

impl OneSidedDistanceCache {
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &Vec<CpuDistanceShare>)> {
        self.map.iter()
    }

    pub fn insert(
        &mut self,
        idx: u64,
        shares: Vec<CpuDistanceShare>,
    ) -> Option<Vec<CpuDistanceShare>> {
        self.map.insert(idx, shares)
    }
}

/// Represents a cache for match distances that matched on both sides.
#[derive(Debug, Default, Clone)]
pub struct TwoSidedDistanceCache {
    pub map: HashMap<u64, (Vec<CpuDistanceShare>, Vec<CpuDistanceShare>)>,
}

impl TwoSidedDistanceCache {
    /// Merges two one-sided distance caches into a two-sided distance cache.
    pub fn merge(
        left: OneSidedDistanceCache,
        mut right: OneSidedDistanceCache,
    ) -> TwoSidedDistanceCache {
        let mut map = HashMap::new();
        for (key, left_values) in left.map {
            if let Some(right_values) = right.map.remove(&key) {
                map.insert(key, (left_values, right_values));
            }
        }
        TwoSidedDistanceCache { map }
    }

    pub fn extend(&mut self, other: TwoSidedDistanceCache) {
        for (key, (left_values, right_values)) in other.map {
            let res = self.map.insert(key, (left_values, right_values));
            assert!(
                res.is_none(),
                "Key {} is not allowed to already exist in the distance cache",
                key
            );
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (&u64, &(Vec<CpuDistanceShare>, Vec<CpuDistanceShare>))> {
        self.map.iter()
    }
}
