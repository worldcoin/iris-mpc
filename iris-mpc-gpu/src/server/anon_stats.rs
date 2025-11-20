use std::collections::HashMap;

use ampc_server_utils::statistics::Eye;
use cudarc::driver::{result::memset_d8_sync, CudaSlice, CudaStream, DevicePtr, DeviceSlice};
use iris_mpc_common::ROTATIONS;
use itertools::{izip, Itertools};

use crate::{
    helpers::{device_manager::DeviceManager, dtoh_on_stream_sync, htod_on_stream_sync},
    threshold_ring::protocol::{ChunkShare, Circuits},
};

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

    pub fn load_counters(&self, device_manager: &DeviceManager, eye: Eye) -> Vec<usize> {
        let (_, _, counters, _) = self.get_buffers(eye);
        device_manager
            .devices()
            .iter()
            .enumerate()
            .map(|(i, device)| device.dtoh_sync_copy(&counters[i]).unwrap()[0] as usize)
            .collect::<Vec<_>>()
    }

    pub fn load_additions_since(
        &self,
        device_manager: &DeviceManager,
        eye: Eye,
        old_counters: Vec<usize>,
        max_internal_buffer_size: usize,
        streams: &[CudaStream],
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
                map.entry(idx / ROTATIONS as u64)
                    .or_insert_with(|| Vec::with_capacity(4))
                    .push(CpuDistanceShare {
                        idx,
                        code_a: c_a,
                        code_b: c_b,
                        mask_a: m_a,
                        mask_b: m_b,
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
}

pub struct CpuLiftedDistanceShare {
    code_a: u32,
    code_b: u32,
    mask_a: u32,
    mask_b: u32,
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

    fn sort_internal_groups(&mut self) {
        for (left_values, right_values) in self.map.values_mut() {
            left_values.sort_by_key(|x| x.idx);
            right_values.sort_by_key(|x| x.idx);
        }
    }

    // 2. A way to reduce the distances per query to the min ones on both left and right side
    // We will do this like we did on the CPU side:
    //   * We will have ids associated with each distance, such that we can group them into rotations for the same query
    //   * We will group the distances by query id, take the first one as the base, and compare the rest to it, doing conditional swaps to keep the minimum one
    //   * For the conditional swaps, we will need to lift the 16-bit shares to 32-bit shares, which we will do in a single batch
    pub fn into_min_distance_cache(
        mut caches: Vec<TwoSidedDistanceCache>,
        protocol: &mut Circuits,
        streams: &[CudaStream],
    ) -> TwoSidedMinDistanceCache {
        for cache in &mut caches {
            cache.sort_internal_groups();
        }

        let actual_len = caches.iter().map(|x| x.len()).sum::<usize>();
        let len = protocol.chunk_size() * 64;
        tracing::info!(
            "Computing min distance cache for {} entries, truncating to {}",
            actual_len,
            len
        );
        let mut left_code_a = Vec::with_capacity(len);
        let mut left_code_b = Vec::with_capacity(len);
        let mut left_mask_a = Vec::with_capacity(len);
        let mut left_mask_b = Vec::with_capacity(len);
        let mut right_code_a = Vec::with_capacity(len);
        let mut right_code_b = Vec::with_capacity(len);
        let mut right_mask_a = Vec::with_capacity(len);
        let mut right_mask_b = Vec::with_capacity(len);

        let mut sorted = caches
            .into_iter()
            .flat_map(|cache| cache.map.into_iter().sorted_by_key(|(key, _)| *key))
            .collect_vec();

        sorted.truncate(len);

        // flatten the values into a single vector to batch the lifting step
        let (flattened_a, flattened_b): (Vec<_>, Vec<_>) = sorted
            .iter()
            .flat_map(|(_, (left_values, right_values))| {
                left_values
                    .iter()
                    .flat_map(|x| [(x.code_a, x.code_b), (x.mask_a, x.mask_b)])
                    .chain(
                        right_values
                            .iter()
                            .flat_map(|x| [(x.code_a, x.code_b), (x.mask_a, x.mask_b)]),
                    )
            })
            .collect();

        let devices = protocol.get_devices();
        let (flattened_lifted_a, flattened_lifted_b) = {
            // the underlying protocol can only work on its chunk size, so we might need to call this multiple times
            let mut flattened_lifted_a = Vec::with_capacity(flattened_a.len());
            let mut flattened_lifted_b = Vec::with_capacity(flattened_b.len());

            let result_chunk_buf = protocol.allocate_buffer(len);
            let mut result = result_chunk_buf
                .iter()
                .map(|x| x.as_view())
                .collect::<Vec<_>>();

            for (chunk_a, chunk_b) in flattened_a.chunks(len).zip(flattened_b.chunks(len)) {
                let on_device_a = if chunk_a.len() == len {
                    htod_on_stream_sync(chunk_a, &devices[0], &streams[0]).unwrap()
                } else {
                    let mut chunk_pad = vec![0; len];
                    chunk_pad[..chunk_a.len()].copy_from_slice(chunk_a);
                    htod_on_stream_sync(&chunk_pad, &devices[0], &streams[0]).unwrap()
                };
                let on_device_b = if chunk_b.len() == len {
                    htod_on_stream_sync(chunk_b, &devices[0], &streams[0]).unwrap()
                } else {
                    let mut chunk_pad = vec![0; len];
                    chunk_pad[..chunk_b.len()].copy_from_slice(chunk_b);
                    htod_on_stream_sync(&chunk_pad, &devices[0], &streams[0]).unwrap()
                };
                let on_device = ChunkShare::new(on_device_a, on_device_b);

                protocol.lift_u16_to_u32(&[on_device.as_view()], &mut result, &streams[..1]);
                let result_a = dtoh_on_stream_sync(&result[0].a, &devices[0], &streams[0]).unwrap();
                let result_b = dtoh_on_stream_sync(&result[0].b, &devices[0], &streams[0]).unwrap();
                flattened_lifted_a.extend(result_a.into_iter().take(chunk_a.len()));
                flattened_lifted_b.extend(result_b.into_iter().take(chunk_b.len()));
            }
            (flattened_lifted_a, flattened_lifted_b)
        };

        // Build back the structure from the flattened lifted values
        let mut idx = 0;
        let mut sorted = sorted
            .into_iter()
            .map(|(key, (left, right))| {
                let left_values = left
                    .into_iter()
                    .map(|_| {
                        let code_a = flattened_lifted_a[idx];
                        let code_b = flattened_lifted_b[idx];
                        let mask_a = flattened_lifted_a[idx + 1];
                        let mask_b = flattened_lifted_b[idx + 1];
                        idx += 2;
                        CpuLiftedDistanceShare {
                            code_a,
                            code_b,
                            mask_a,
                            mask_b,
                        }
                    })
                    .collect_vec();
                let right_values = right
                    .into_iter()
                    .map(|_| {
                        let code_a = flattened_lifted_a[idx];
                        let code_b = flattened_lifted_b[idx];
                        let mask_a = flattened_lifted_a[idx + 1];
                        let mask_b = flattened_lifted_b[idx + 1];
                        idx += 2;
                        CpuLiftedDistanceShare {
                            code_a,
                            code_b,
                            mask_a,
                            mask_b,
                        }
                    })
                    .collect_vec();

                (key, (left_values, right_values))
            })
            .collect_vec();

        for (_, (left_values, right_values)) in sorted.iter_mut() {
            let left = left_values
                .pop()
                .expect("we have at least one value in our vec if it exists");
            let right = right_values
                .pop()
                .expect("we have at least one value in our vec if it exists");

            left_code_a.push(left.code_a);
            left_code_b.push(left.code_b);
            left_mask_a.push(left.mask_a);
            left_mask_b.push(left.mask_b);
            right_code_a.push(right.code_a);
            right_code_b.push(right.code_b);
            right_mask_a.push(right.mask_a);
            right_mask_b.push(right.mask_b);
        }

        tracing::info!("lifted values, now reducing them to min distances");

        // While we have more than one rotation left, we will keep popping values from the rotations, comparing them and keeping the minimum one.
        // If a specific one does not have more rotations to pop, we will just compare against the current one (essentially creating a dummy operation).
        while sorted.iter().any(|(_, (left_values, right_values))| {
            !left_values.is_empty() || !right_values.is_empty()
        }) {
            let mut left_code_a_2 = Vec::with_capacity(len);
            let mut left_code_b_2 = Vec::with_capacity(len);
            let mut left_mask_a_2 = Vec::with_capacity(len);
            let mut left_mask_b_2 = Vec::with_capacity(len);
            let mut right_code_a_2 = Vec::with_capacity(len);
            let mut right_code_b_2 = Vec::with_capacity(len);
            let mut right_mask_a_2 = Vec::with_capacity(len);
            let mut right_mask_b_2 = Vec::with_capacity(len);
            for (idx, (_, (left_values, right_values))) in sorted.iter_mut().enumerate() {
                if let Some(left) = left_values.pop() {
                    left_code_a_2.push(left.code_a);
                    left_code_b_2.push(left.code_b);
                    left_mask_a_2.push(left.mask_a);
                    left_mask_b_2.push(left.mask_b);
                } else {
                    left_code_a_2.push(left_code_a[idx]);
                    left_code_b_2.push(left_code_b[idx]);
                    left_mask_a_2.push(left_mask_a[idx]);
                    left_mask_b_2.push(left_mask_b[idx]);
                }
                if let Some(right) = right_values.pop() {
                    right_code_a_2.push(right.code_a);
                    right_code_b_2.push(right.code_b);
                    right_mask_a_2.push(right.mask_a);
                    right_mask_b_2.push(right.mask_b);
                } else {
                    right_code_a_2.push(right_code_a[idx]);
                    right_code_b_2.push(right_code_b[idx]);
                    right_mask_a_2.push(right_mask_a[idx]);
                    right_mask_b_2.push(right_mask_b[idx]);
                }
            }

            let mut swap = |ca: &mut Vec<_>,
                            cb: &mut Vec<_>,
                            ma: &mut Vec<_>,
                            mb: &mut Vec<_>,
                            ca2,
                            cb2,
                            ma2,
                            mb2| {
                let codes_a = htod_on_stream_sync(ca, &devices[0], &streams[0]).unwrap();
                let codes_b = htod_on_stream_sync(cb, &devices[0], &streams[0]).unwrap();
                let codes = ChunkShare::new(codes_a, codes_b);
                let codes_view = codes.as_view();
                let masks_a = htod_on_stream_sync(ma, &devices[0], &streams[0]).unwrap();
                let masks_b = htod_on_stream_sync(mb, &devices[0], &streams[0]).unwrap();
                let masks = ChunkShare::new(masks_a, masks_b);
                let masks_view = masks.as_view();

                let codes2_a = htod_on_stream_sync(ca2, &devices[0], &streams[0]).unwrap();
                let codes2_b = htod_on_stream_sync(cb2, &devices[0], &streams[0]).unwrap();
                let codes2 = ChunkShare::new(codes2_a, codes2_b);
                let codes2_view = codes2.as_view();
                let masks2_a = htod_on_stream_sync(ma2, &devices[0], &streams[0]).unwrap();
                let masks2_b = htod_on_stream_sync(mb2, &devices[0], &streams[0]).unwrap();
                let masks2 = ChunkShare::new(masks2_a, masks2_b);
                let masks2_view = masks2.as_view();

                protocol.cross_compare_and_swap(
                    &mut [codes_view],
                    &mut [masks_view],
                    &[codes2_view],
                    &[masks2_view],
                    &streams[..1],
                );

                let result_code_a =
                    dtoh_on_stream_sync(&codes.a, &devices[0], &streams[0]).unwrap();
                let result_code_b =
                    dtoh_on_stream_sync(&codes.b, &devices[0], &streams[0]).unwrap();
                let result_mask_a =
                    dtoh_on_stream_sync(&masks.a, &devices[0], &streams[0]).unwrap();
                let result_mask_b =
                    dtoh_on_stream_sync(&masks.b, &devices[0], &streams[0]).unwrap();

                *ca = result_code_a;
                *cb = result_code_b;
                *ma = result_mask_a;
                *mb = result_mask_b;
            };

            swap(
                &mut left_code_a,
                &mut left_code_b,
                &mut left_mask_a,
                &mut left_mask_b,
                &left_code_a_2,
                &left_code_b_2,
                &left_mask_a_2,
                &left_mask_b_2,
            );
            swap(
                &mut right_code_a,
                &mut right_code_b,
                &mut right_mask_a,
                &mut right_mask_b,
                &right_code_a_2,
                &right_code_b_2,
                &right_mask_a_2,
                &right_mask_b_2,
            );
        }

        TwoSidedMinDistanceCache {
            left_code_a,
            left_code_b,
            left_mask_a,
            left_mask_b,
            right_code_a,
            right_code_b,
            right_mask_a,
            right_mask_b,
        }
    }
}

pub struct TwoSidedMinDistanceCache {
    pub left_code_a: Vec<u32>,
    pub left_code_b: Vec<u32>,
    pub left_mask_a: Vec<u32>,
    pub left_mask_b: Vec<u32>,
    pub right_code_a: Vec<u32>,
    pub right_code_b: Vec<u32>,
    pub right_mask_a: Vec<u32>,
    pub right_mask_b: Vec<u32>,
}

impl TwoSidedMinDistanceCache {
    pub fn compute_buckets(
        self,
        protocol: &mut Circuits,
        streams: &[CudaStream],
        thresholds: &[u16],
    ) -> Vec<u32> {
        for v in [
            &self.left_code_a,
            &self.left_code_b,
            &self.left_mask_a,
            &self.left_mask_b,
            &self.right_code_a,
            &self.right_code_b,
            &self.right_mask_a,
            &self.right_mask_b,
        ] {
            assert_eq!(v.len(), protocol.chunk_size() * 64);
        }
        let devices = protocol.get_devices();

        let left_code_a = htod_on_stream_sync(&self.left_code_a, &devices[0], &streams[0]).unwrap();
        let left_code_b = htod_on_stream_sync(&self.left_code_b, &devices[0], &streams[0]).unwrap();
        let left_mask_a = htod_on_stream_sync(&self.left_mask_a, &devices[0], &streams[0]).unwrap();
        let left_mask_b = htod_on_stream_sync(&self.left_mask_b, &devices[0], &streams[0]).unwrap();
        let left_code = ChunkShare::new(left_code_a, left_code_b);
        let left_mask = ChunkShare::new(left_mask_a, left_mask_b);

        let right_code_a =
            htod_on_stream_sync(&self.right_code_a, &devices[0], &streams[0]).unwrap();
        let right_code_b =
            htod_on_stream_sync(&self.right_code_b, &devices[0], &streams[0]).unwrap();
        let right_mask_a =
            htod_on_stream_sync(&self.right_mask_a, &devices[0], &streams[0]).unwrap();
        let right_mask_b =
            htod_on_stream_sync(&self.right_mask_b, &devices[0], &streams[0]).unwrap();
        let right_code = ChunkShare::new(right_code_a, right_code_b);
        let right_mask = ChunkShare::new(right_mask_a, right_mask_b);

        let results = protocol.compare_multiple_thresholds_2d(
            &[left_code.as_view()],
            &[left_mask.as_view()],
            &[right_code.as_view()],
            &[right_mask.as_view()],
            &streams[..1],
            thresholds,
        );
        protocol.synchronize_streams(streams);
        results
    }
}
