use crate::protocol::shared_iris::ArcIris;
use ampc_actor_utils::fast_metrics::FastHistogram;
pub use ampc_actor_utils::protocol::ops::{
    conditionally_select_distances_with_plain_ids, conditionally_select_distances_with_shared_ids,
    conditionally_swap_distances, conditionally_swap_distances_plain_ids, galois_ring_to_rep3,
    lt_zero_and_open_u16, open_ring, setup_replicated_prf, setup_shared_seed, sub_pub,
    DistancePair, IdDistance, B, B_BITS,
};
pub use ampc_secret_sharing::shares::{
    bit::Bit,
    ring_impl::{RingElement, VecRingElement},
    share::{reconstruct_distance_vector, DistanceShare, Share},
    vecshare::VecShare,
};
use iris_mpc_common::{
    galois_engine::degree4::{IrisRotation, SHARE_OF_MAX_DISTANCE},
    ROTATIONS as ALL_ROTATIONS,
};
use std::time::Instant;

use std::cell::RefCell;

thread_local! {
    static PAIRWISE_DISTANCE_METRICS: RefCell<[FastHistogram; 2]> = RefCell::new([
        FastHistogram::new("pairwise_distance.batch_size"),
        FastHistogram::new("pairwise_distance.per_pair_duration"),
    ]);
}

/// See pairwise_distance.
/// This variant takes as input a Vec of Arc.
pub fn galois_ring_pairwise_distance(
    pairs: Vec<Option<(ArcIris, ArcIris)>>,
) -> Vec<RingElement<u16>> {
    pairwise_distance(pairs.iter().map(|opt| opt.as_ref().map(|(x, y)| (x, y))))
}

/// Computes the dot product between the iris pairs; for both the code and the
/// mask of the irises. We pack the dot products of the code and mask into one
/// vector to be able to reshare it later.
/// This function takes an iterator of known size.
pub fn pairwise_distance<'a, I>(pairs: I) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<(&'a ArcIris, &'a ArcIris)>> + ExactSizeIterator,
{
    let start = Instant::now();
    let mut count = 0;
    let mut additive_shares = Vec::with_capacity(2 * pairs.len());

    for pair in pairs {
        let (code_dist, mask_dist) = if let Some((x, y)) = pair {
            count += 1;
            let (a, b) = (x.code.trick_dot(&y.code), x.mask.trick_dot(&y.mask));
            // When applying the trick dot on trimmed masks, we have to multiply
            // the result by 2 because a GaloisRingTrimmedMask is encoded using
            // half the elements of a full GaloisRingMask, representing that
            // real/imaginary bits at an index are either both masked or both
            // unmasked.
            (RingElement(a), RingElement(2) * RingElement(b))
        } else {
            // Non-existent vectors get the largest relative distance of 100%.
            let (a, b) = SHARE_OF_MAX_DISTANCE;
            (RingElement(a), RingElement(b))
        };
        additive_shares.push(code_dist);
        additive_shares.push(mask_dist);
    }

    let batch_size = count as f64;
    let duration = start.elapsed().as_secs_f64() / batch_size;
    PAIRWISE_DISTANCE_METRICS.with_borrow_mut(|[metric_batch_size, metric_per_pair_duration]| {
        metric_batch_size.record(batch_size);
        metric_per_pair_duration.record(duration);
    });

    additive_shares
}

/// This is similar to `pairwise_distance`, but performs dot products on all rotations of the query.
pub fn rotation_aware_pairwise_distance<'a, const ROTATIONS: usize, I>(
    query: &'a ArcIris,
    targets: I,
) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<&'a ArcIris>> + ExactSizeIterator,
{
    let start = Instant::now();
    let mut count = 0;
    let mut additive_shares = Vec::with_capacity(2 * ROTATIONS * targets.len());

    for target in targets {
        for rotation in IrisRotation::all()
            .skip((ALL_ROTATIONS - ROTATIONS) / 2)
            .take(ROTATIONS)
        {
            let (code_dist, mask_dist) = if let Some(y) = target {
                count += 1;
                let (a, b) = (
                    query.code.rotation_aware_trick_dot(&y.code, &rotation),
                    query.mask.rotation_aware_trick_dot(&y.mask, &rotation),
                );
                (RingElement(a), RingElement(2) * RingElement(b))
            } else {
                // Non-existent vectors get the largest relative distance of 100%.
                let (a, b) = SHARE_OF_MAX_DISTANCE;
                (RingElement(a), RingElement(b))
            };
            additive_shares.push(code_dist);
            additive_shares.push(mask_dist);
        }
    }

    let batch_size = count as f64;
    let duration = start.elapsed().as_secs_f64() / batch_size;
    PAIRWISE_DISTANCE_METRICS.with_borrow_mut(|[metric_batch_size, metric_per_pair_duration]| {
        metric_batch_size.record(batch_size);
        metric_per_pair_duration.record(duration);
    });
    additive_shares
}

/// Row-major prerotated query for L1 cache efficiency.
/// Layout: `[row0_rot0..rot30][row1_rot0..rot30]...`
pub struct PrerotatedQueryRowMajor {
    /// code: 16 rows × 31 rotations × 800 elements = 396,800 u16s (~793KB)
    code_data: Vec<u16>,
    /// mask: 8 rows × 31 rotations × 800 elements = 198,400 u16s (~397KB)
    mask_data: Vec<u16>,
    /// Keep the last query alive so pointer equality cannot be fooled by an
    /// allocator reusing its address after query eviction.
    cached_query: Option<ArcIris>,
    cached_rotations: usize,
}

impl PrerotatedQueryRowMajor {
    pub const ROW_SIZE: usize = 800;
    pub const CODE_ROWS: usize = 16;
    pub const MASK_ROWS: usize = 8;
    const CODE_SIZE: usize = Self::CODE_ROWS * ALL_ROTATIONS * Self::ROW_SIZE;
    const MASK_SIZE: usize = Self::MASK_ROWS * ALL_ROTATIONS * Self::ROW_SIZE;

    /// Create a new buffer (allocates memory).
    fn new_buffer() -> Self {
        Self {
            code_data: vec![0u16; Self::CODE_SIZE],
            mask_data: vec![0u16; Self::MASK_SIZE],
            cached_query: None,
            cached_rotations: 0,
        }
    }

    fn matches<const ROTATIONS: usize>(&self, query: &ArcIris) -> bool {
        self.cached_rotations == ROTATIONS
            && self
                .cached_query
                .as_ref()
                .is_some_and(|cached| std::sync::Arc::ptr_eq(cached, query))
    }
}

pub struct PrerotatedQueryRowMajorView<'a, const ROTATIONS: usize> {
    storage: &'a mut PrerotatedQueryRowMajor,
}

impl<const ROTATIONS: usize> PrerotatedQueryRowMajorView<'_, ROTATIONS> {
    pub const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;
    pub const CODE_ROWS: usize = PrerotatedQueryRowMajor::CODE_ROWS;
    pub const MASK_ROWS: usize = PrerotatedQueryRowMajor::MASK_ROWS;

    /// Rotation amounts for each rotation.
    const ROTATION_AMOUNTS: [usize; ROTATIONS] = {
        let mut amounts = [0usize; ROTATIONS];
        // Left rotations: rotate left by 60, 56, ..., 4 elements
        let half = ROTATIONS / 2;
        let mut i = 0;
        while i < half {
            amounts[i] = (half - i) * 4;
            i += 1;
        }
        // Center: no rotation
        amounts[half] = 0;
        // Right rotations: rotate right by 4, 8, ..., 60 (i.e., left by 796, 792, ..., 740)
        let mut i = 1;
        while i <= half {
            amounts[half + i] = 800 - i * 4;
            i += 1;
        }
        amounts
    };

    /// Rotate row directly into destination buffer (zero allocations).
    #[inline]
    fn rotate_row_into(src: &[u16], dst: &mut [u16], left_amount: usize) {
        let len = src.len();
        debug_assert_eq!(dst.len(), len);
        debug_assert!(left_amount < len);
        // Rotate left by `left_amount`: [left_amount..] ++ [..left_amount]
        let (first, second) = src.split_at(left_amount);
        dst[..second.len()].copy_from_slice(second);
        dst[second.len()..].copy_from_slice(first);
    }

    /// Fill the buffer with rotated query data (reuses existing allocation).
    fn fill(&mut self, query: &ArcIris) {
        // Process code rows - write directly into destination (no intermediate allocations)
        for row_idx in 0..Self::CODE_ROWS {
            let src_row =
                &query.code.coefs[row_idx * Self::ROW_SIZE..(row_idx + 1) * Self::ROW_SIZE];
            for rot_idx in 0..ROTATIONS {
                let dst_offset = (row_idx * ROTATIONS + rot_idx) * Self::ROW_SIZE;
                let dst = &mut self.storage.code_data[dst_offset..dst_offset + Self::ROW_SIZE];
                Self::rotate_row_into(src_row, dst, Self::ROTATION_AMOUNTS[rot_idx]);
            }
        }

        // Process mask rows - write directly into destination (no intermediate allocations)
        for row_idx in 0..Self::MASK_ROWS {
            let src_row =
                &query.mask.coefs[row_idx * Self::ROW_SIZE..(row_idx + 1) * Self::ROW_SIZE];
            for rot_idx in 0..ROTATIONS {
                let dst_offset = (row_idx * ROTATIONS + rot_idx) * Self::ROW_SIZE;
                let dst = &mut self.storage.mask_data[dst_offset..dst_offset + Self::ROW_SIZE];
                Self::rotate_row_into(src_row, dst, Self::ROTATION_AMOUNTS[rot_idx]);
            }
        }
    }

    /// Reuse the prerotated representation across consecutive work items for
    /// the same query on this worker thread. A full scan submits many chunks
    /// carrying clones of one `ArcIris`; rebuilding its ~1.19 MiB rotation
    /// matrix for every chunk is redundant.
    fn fill_if_changed(&mut self, query: &ArcIris) {
        if self.storage.matches::<ROTATIONS>(query) {
            return;
        }

        self.fill(query);
        self.storage.cached_query = Some(query.clone());
        self.storage.cached_rotations = ROTATIONS;
    }

    /// Get all rotations of a code row (contiguous in memory, at most 50KB)
    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn code_row_rotations(&self, row_idx: usize) -> &[u16] {
        let start = row_idx * ROTATIONS * Self::ROW_SIZE;
        let end = start + ROTATIONS * Self::ROW_SIZE;
        &self.storage.code_data[start..end]
    }

    /// Get all rotations of a mask row (contiguous in memory, at most 50KB)
    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn mask_row_rotations(&self, row_idx: usize) -> &[u16] {
        let start = row_idx * ROTATIONS * Self::ROW_SIZE;
        let end = start + ROTATIONS * Self::ROW_SIZE;
        &self.storage.mask_data[start..end]
    }
}

const PREROTATED_CACHE_CAPACITY: usize = 2;

/// A small per-worker cache for prerotated queries.
///
/// Normal and mirror searches can alternate on one worker thread. Keeping two
/// entries avoids rebuilding either query's rotation matrix while bounding the
/// retained allocations and query `Arc`s. `least_recently_used` is sufficient
/// for exact LRU replacement because this cache has exactly two entries.
struct PrerotatedQueryCache {
    entries: [Option<PrerotatedQueryRowMajor>; PREROTATED_CACHE_CAPACITY],
    least_recently_used: usize,
}

impl PrerotatedQueryCache {
    const fn new() -> Self {
        Self {
            entries: [None, None],
            least_recently_used: 0,
        }
    }

    fn get_or_fill<const ROTATIONS: usize>(
        &mut self,
        query: &ArcIris,
    ) -> (&mut PrerotatedQueryRowMajor, bool) {
        let hit_index = self.entries.iter().position(|entry| {
            entry
                .as_ref()
                .is_some_and(|storage| storage.matches::<ROTATIONS>(query))
        });

        let (index, hit) = if let Some(index) = hit_index {
            (index, true)
        } else {
            let index = self
                .entries
                .iter()
                .position(Option::is_none)
                .unwrap_or(self.least_recently_used);
            (index, false)
        };

        // With two entries, the other slot is now the least recently used.
        self.least_recently_used = 1 - index;

        let storage = self.entries[index].get_or_insert_with(PrerotatedQueryRowMajor::new_buffer);
        let mut prerotated = PrerotatedQueryRowMajorView::<ROTATIONS> { storage };
        prerotated.fill_if_changed(query);
        (prerotated.storage, hit)
    }
}

// Thread-local storage for the bounded reusable prerotated-query cache.
thread_local! {
    static PREROTATED_CACHE: RefCell<PrerotatedQueryCache> = const { RefCell::new(PrerotatedQueryCache::new()) };
}

/// Row-major rotation-aware distance - processes row-by-row for L1 cache efficiency.
/// Uses a bounded two-entry thread-local cache to minimize allocations and
/// prerotation work at high thread counts.
pub fn rotation_aware_pairwise_distance_rowmajor<'a, const ROTATIONS: usize, I>(
    query: &'a ArcIris,
    targets: I,
) -> Vec<RingElement<u16>>
where
    I: Iterator<Item = Option<&'a ArcIris>> + ExactSizeIterator,
{
    let target_count = targets.len();
    let mut additive_shares = vec![RingElement(0u16); 2 * ROTATIONS * target_count];

    // Use the thread-local cache to avoid allocations and prerotation rebuilds.
    PREROTATED_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        let (storage, _) = cache.get_or_fill::<ROTATIONS>(query);
        let prerotated = PrerotatedQueryRowMajorView::<ROTATIONS> { storage };
        rotation_aware_inner(&prerotated, targets, &mut additive_shares);
    });

    additive_shares
}

/// Inner implementation that works with a borrowed prerotated buffer.
#[inline(never)]
fn rotation_aware_inner<'a, const ROTATIONS: usize, I>(
    prerotated: &PrerotatedQueryRowMajorView<ROTATIONS>,
    targets: I,
    additive_shares: &mut [RingElement<u16>],
) where
    I: Iterator<Item = Option<&'a ArcIris>> + ExactSizeIterator,
{
    #[cfg(not(target_arch = "aarch64"))]
    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;
    const CODE_ROWS: usize = PrerotatedQueryRowMajor::CODE_ROWS;
    const MASK_ROWS: usize = PrerotatedQueryRowMajor::MASK_ROWS;

    let targets: Vec<_> = targets.collect();

    #[cfg(target_arch = "aarch64")]
    {
        let code_targets = targets
            .iter()
            .map(|target| target.as_ref().map(|iris| iris.code.coefs.as_slice()))
            .collect::<Vec<_>>();
        accumulate_component_tiled_6x4::<ROTATIONS>(
            &prerotated.storage.code_data,
            &code_targets,
            CODE_ROWS,
            0,
            additive_shares,
        );

        let mask_targets = targets
            .iter()
            .map(|target| target.as_ref().map(|iris| iris.mask.coefs.as_slice()))
            .collect::<Vec<_>>();
        accumulate_component_tiled_6x4::<ROTATIONS>(
            &prerotated.storage.mask_data,
            &mask_targets,
            MASK_ROWS,
            1,
            additive_shares,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // Process row-by-row: all 31 rotations of one row stay in L1
        for row_idx in 0..CODE_ROWS {
            let query_rows = prerotated.code_row_rotations(row_idx); // 50KB, fits in L1

            for (target_idx, target_opt) in targets.iter().enumerate() {
                if let Some(target) = target_opt {
                    let target_row =
                        &target.code.coefs[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];

                    for rot_idx in 0..ROTATIONS {
                        let query_row = &query_rows[rot_idx * ROW_SIZE..(rot_idx + 1) * ROW_SIZE];
                        let partial = simple_dot_product(query_row, target_row);
                        let result_idx = target_idx * ROTATIONS * 2 + rot_idx * 2;
                        additive_shares[result_idx] =
                            RingElement(additive_shares[result_idx].0.wrapping_add(partial));
                    }
                }
            }
        }
        // Process mask rows - accumulate first, multiply by 2 later
        for row_idx in 0..MASK_ROWS {
            let query_rows = prerotated.mask_row_rotations(row_idx);

            for (target_idx, target_opt) in targets.iter().enumerate() {
                if let Some(target) = target_opt {
                    let target_row =
                        &target.mask.coefs[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];

                    for rot_idx in 0..ROTATIONS {
                        let query_row = &query_rows[rot_idx * ROW_SIZE..(rot_idx + 1) * ROW_SIZE];
                        let partial = simple_dot_product(query_row, target_row);
                        let result_idx = target_idx * ROTATIONS * 2 + rot_idx * 2 + 1;
                        additive_shares[result_idx] =
                            RingElement(additive_shares[result_idx].0.wrapping_add(partial));
                    }
                }
            }
        }
    }

    // Multiply mask results by 2 for Some targets, set max distance for None targets
    for (target_idx, target_opt) in targets.iter().enumerate() {
        let base_idx = target_idx * ROTATIONS * 2;
        if target_opt.is_some() {
            for rot_idx in 0..ROTATIONS {
                let mask_idx = base_idx + rot_idx * 2 + 1;
                additive_shares[mask_idx] = RingElement(2) * additive_shares[mask_idx];
            }
        } else {
            let (a, b) = SHARE_OF_MAX_DISTANCE;
            for rot_idx in 0..ROTATIONS {
                let code_idx = base_idx + rot_idx * 2;
                additive_shares[code_idx] = RingElement(a);
                additive_shares[code_idx + 1] = RingElement(b);
            }
        }
    }
}

/// Accumulate six rotations against four targets at a time. The tile keeps
/// twenty-four independent SIMD accumulators live, so each query vector load
/// is reused by four targets and each target vector load by six rotations.
/// The 24 accumulators, 6 query vectors, and one streamed target vector fit in
/// the 32-register AArch64 SIMD register file without hot-loop spills.
/// Arithmetic stays in `u16` lanes and is therefore exactly the same wrapping
/// modulo-2^16 arithmetic as [`simple_dot_product`].
#[cfg(target_arch = "aarch64")]
fn accumulate_component_tiled_6x4<const ROTATIONS: usize>(
    query: &[u16],
    targets: &[Option<&[u16]>],
    rows: usize,
    result_lane: usize,
    additive_shares: &mut [RingElement<u16>],
) {
    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;

    for row_idx in 0..rows {
        let query_rows_start = row_idx * ROTATIONS * ROW_SIZE;
        let query_rows = &query[query_rows_start..query_rows_start + ROTATIONS * ROW_SIZE];
        let target_row_start = row_idx * ROW_SIZE;

        let mut target_idx = 0;
        while target_idx + 4 <= targets.len() {
            let group = &targets[target_idx..target_idx + 4];
            if let (Some(target0), Some(target1), Some(target2), Some(target3)) =
                (group[0], group[1], group[2], group[3])
            {
                let target0 = &target0[target_row_start..target_row_start + ROW_SIZE];
                let target1 = &target1[target_row_start..target_row_start + ROW_SIZE];
                let target2 = &target2[target_row_start..target_row_start + ROW_SIZE];
                let target3 = &target3[target_row_start..target_row_start + ROW_SIZE];

                let mut rotation_idx = 0;
                while rotation_idx + 6 <= ROTATIONS {
                    let query0 =
                        &query_rows[rotation_idx * ROW_SIZE..(rotation_idx + 1) * ROW_SIZE];
                    let query1 =
                        &query_rows[(rotation_idx + 1) * ROW_SIZE..(rotation_idx + 2) * ROW_SIZE];
                    let query2 =
                        &query_rows[(rotation_idx + 2) * ROW_SIZE..(rotation_idx + 3) * ROW_SIZE];
                    let query3 =
                        &query_rows[(rotation_idx + 3) * ROW_SIZE..(rotation_idx + 4) * ROW_SIZE];
                    let query4 =
                        &query_rows[(rotation_idx + 4) * ROW_SIZE..(rotation_idx + 5) * ROW_SIZE];
                    let query5 =
                        &query_rows[(rotation_idx + 5) * ROW_SIZE..(rotation_idx + 6) * ROW_SIZE];
                    let partials = dot_product_6x4_u16(
                        [query0, query1, query2, query3, query4, query5],
                        [target0, target1, target2, target3],
                    );

                    for (rotation_offset, row) in partials.into_iter().enumerate() {
                        for (target_offset, partial) in row.into_iter().enumerate() {
                            let result_idx = (target_idx + target_offset) * ROTATIONS * 2
                                + (rotation_idx + rotation_offset) * 2
                                + result_lane;
                            additive_shares[result_idx].0 =
                                additive_shares[result_idx].0.wrapping_add(partial);
                        }
                    }
                    rotation_idx += 6;
                }

                // Preserve an efficient 4-wide remainder for the 11-rotation
                // HNSW path (6 + 4 + 1) instead of evaluating five rotations
                // as independent dot products.
                while rotation_idx + 4 <= ROTATIONS {
                    let query0 =
                        &query_rows[rotation_idx * ROW_SIZE..(rotation_idx + 1) * ROW_SIZE];
                    let query1 =
                        &query_rows[(rotation_idx + 1) * ROW_SIZE..(rotation_idx + 2) * ROW_SIZE];
                    let query2 =
                        &query_rows[(rotation_idx + 2) * ROW_SIZE..(rotation_idx + 3) * ROW_SIZE];
                    let query3 =
                        &query_rows[(rotation_idx + 3) * ROW_SIZE..(rotation_idx + 4) * ROW_SIZE];
                    let partials = dot_product_4x4_u16(
                        [query0, query1, query2, query3],
                        [target0, target1, target2, target3],
                    );

                    for (rotation_offset, row) in partials.into_iter().enumerate() {
                        for (target_offset, partial) in row.into_iter().enumerate() {
                            let result_idx = (target_idx + target_offset) * ROTATIONS * 2
                                + (rotation_idx + rotation_offset) * 2
                                + result_lane;
                            additive_shares[result_idx].0 =
                                additive_shares[result_idx].0.wrapping_add(partial);
                        }
                    }
                    rotation_idx += 4;
                }

                // ROTATIONS is normally 11, 13, or 31; each leaves one final
                // rotation after the 6-wide and optional 4-wide tiles.
                while rotation_idx < ROTATIONS {
                    let query_row =
                        &query_rows[rotation_idx * ROW_SIZE..(rotation_idx + 1) * ROW_SIZE];
                    for (target_offset, target_row) in
                        [target0, target1, target2, target3].into_iter().enumerate()
                    {
                        let partial = simple_dot_product(query_row, target_row);
                        let result_idx = (target_idx + target_offset) * ROTATIONS * 2
                            + rotation_idx * 2
                            + result_lane;
                        additive_shares[result_idx].0 =
                            additive_shares[result_idx].0.wrapping_add(partial);
                    }
                    rotation_idx += 1;
                }
            } else {
                // Missing vectors are uncommon in a full scan. Preserve their
                // sentinel handling below while still evaluating live members
                // of a mixed group exactly.
                for (target_offset, target) in group.iter().enumerate() {
                    let Some(target) = target else {
                        continue;
                    };
                    let target_row = &target[target_row_start..target_row_start + ROW_SIZE];
                    accumulate_scalar_target::<ROTATIONS>(
                        query_rows,
                        target_row,
                        target_idx + target_offset,
                        result_lane,
                        additive_shares,
                    );
                }
            }
            target_idx += 4;
        }

        for (target_offset, target) in targets[target_idx..].iter().enumerate() {
            let Some(target) = target else {
                continue;
            };
            let target_row = &target[target_row_start..target_row_start + ROW_SIZE];
            accumulate_scalar_target::<ROTATIONS>(
                query_rows,
                target_row,
                target_idx + target_offset,
                result_lane,
                additive_shares,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn accumulate_scalar_target<const ROTATIONS: usize>(
    query_rows: &[u16],
    target_row: &[u16],
    target_idx: usize,
    result_lane: usize,
    additive_shares: &mut [RingElement<u16>],
) {
    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;
    for rotation_idx in 0..ROTATIONS {
        let query_row = &query_rows[rotation_idx * ROW_SIZE..(rotation_idx + 1) * ROW_SIZE];
        let partial = simple_dot_product(query_row, target_row);
        let result_idx = target_idx * ROTATIONS * 2 + rotation_idx * 2 + result_lane;
        additive_shares[result_idx].0 = additive_shares[result_idx].0.wrapping_add(partial);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_product_4x4_u16(queries: [&[u16]; 4], targets: [&[u16]; 4]) -> [[u16; 4]; 4] {
    use std::arch::aarch64::{uint16x8_t, vaddvq_u16, vdupq_n_u16, vld1q_u16, vmlaq_u16};

    debug_assert!(queries.iter().all(|query| query.len() == 800));
    debug_assert!(targets.iter().all(|target| target.len() == 800));

    // SAFETY: AArch64 guarantees Advanced SIMD. Each pointer is derived from
    // an 800-element slice and the loop only issues unaligned-safe 8-lane
    // loads at offsets 0..792. All arithmetic deliberately wraps in u16.
    unsafe {
        let zero = vdupq_n_u16(0);
        let mut acc00: uint16x8_t = zero;
        let mut acc01: uint16x8_t = zero;
        let mut acc02: uint16x8_t = zero;
        let mut acc03: uint16x8_t = zero;
        let mut acc10: uint16x8_t = zero;
        let mut acc11: uint16x8_t = zero;
        let mut acc12: uint16x8_t = zero;
        let mut acc13: uint16x8_t = zero;
        let mut acc20: uint16x8_t = zero;
        let mut acc21: uint16x8_t = zero;
        let mut acc22: uint16x8_t = zero;
        let mut acc23: uint16x8_t = zero;
        let mut acc30: uint16x8_t = zero;
        let mut acc31: uint16x8_t = zero;
        let mut acc32: uint16x8_t = zero;
        let mut acc33: uint16x8_t = zero;

        let mut idx = 0;
        while idx < 800 {
            let query0 = vld1q_u16(queries[0].as_ptr().add(idx));
            let query1 = vld1q_u16(queries[1].as_ptr().add(idx));
            let query2 = vld1q_u16(queries[2].as_ptr().add(idx));
            let query3 = vld1q_u16(queries[3].as_ptr().add(idx));
            let target0 = vld1q_u16(targets[0].as_ptr().add(idx));
            let target1 = vld1q_u16(targets[1].as_ptr().add(idx));
            let target2 = vld1q_u16(targets[2].as_ptr().add(idx));
            let target3 = vld1q_u16(targets[3].as_ptr().add(idx));

            acc00 = vmlaq_u16(acc00, query0, target0);
            acc01 = vmlaq_u16(acc01, query0, target1);
            acc02 = vmlaq_u16(acc02, query0, target2);
            acc03 = vmlaq_u16(acc03, query0, target3);
            acc10 = vmlaq_u16(acc10, query1, target0);
            acc11 = vmlaq_u16(acc11, query1, target1);
            acc12 = vmlaq_u16(acc12, query1, target2);
            acc13 = vmlaq_u16(acc13, query1, target3);
            acc20 = vmlaq_u16(acc20, query2, target0);
            acc21 = vmlaq_u16(acc21, query2, target1);
            acc22 = vmlaq_u16(acc22, query2, target2);
            acc23 = vmlaq_u16(acc23, query2, target3);
            acc30 = vmlaq_u16(acc30, query3, target0);
            acc31 = vmlaq_u16(acc31, query3, target1);
            acc32 = vmlaq_u16(acc32, query3, target2);
            acc33 = vmlaq_u16(acc33, query3, target3);
            idx += 8;
        }

        [
            [
                vaddvq_u16(acc00),
                vaddvq_u16(acc01),
                vaddvq_u16(acc02),
                vaddvq_u16(acc03),
            ],
            [
                vaddvq_u16(acc10),
                vaddvq_u16(acc11),
                vaddvq_u16(acc12),
                vaddvq_u16(acc13),
            ],
            [
                vaddvq_u16(acc20),
                vaddvq_u16(acc21),
                vaddvq_u16(acc22),
                vaddvq_u16(acc23),
            ],
            [
                vaddvq_u16(acc30),
                vaddvq_u16(acc31),
                vaddvq_u16(acc32),
                vaddvq_u16(acc33),
            ],
        ]
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_product_6x4_u16(queries: [&[u16]; 6], targets: [&[u16]; 4]) -> [[u16; 4]; 6] {
    use std::arch::aarch64::{uint16x8_t, vaddvq_u16, vdupq_n_u16, vld1q_u16, vmlaq_u16};

    debug_assert!(queries.iter().all(|query| query.len() == 800));
    debug_assert!(targets.iter().all(|target| target.len() == 800));

    // SAFETY: AArch64 guarantees Advanced SIMD. Each pointer is derived from
    // an 800-element slice and the loop only issues unaligned-safe 8-lane
    // loads at offsets 0..792. All arithmetic deliberately wraps in u16.
    unsafe {
        let zero = vdupq_n_u16(0);
        let mut acc00: uint16x8_t = zero;
        let mut acc01: uint16x8_t = zero;
        let mut acc02: uint16x8_t = zero;
        let mut acc03: uint16x8_t = zero;
        let mut acc10: uint16x8_t = zero;
        let mut acc11: uint16x8_t = zero;
        let mut acc12: uint16x8_t = zero;
        let mut acc13: uint16x8_t = zero;
        let mut acc20: uint16x8_t = zero;
        let mut acc21: uint16x8_t = zero;
        let mut acc22: uint16x8_t = zero;
        let mut acc23: uint16x8_t = zero;
        let mut acc30: uint16x8_t = zero;
        let mut acc31: uint16x8_t = zero;
        let mut acc32: uint16x8_t = zero;
        let mut acc33: uint16x8_t = zero;
        let mut acc40: uint16x8_t = zero;
        let mut acc41: uint16x8_t = zero;
        let mut acc42: uint16x8_t = zero;
        let mut acc43: uint16x8_t = zero;
        let mut acc50: uint16x8_t = zero;
        let mut acc51: uint16x8_t = zero;
        let mut acc52: uint16x8_t = zero;
        let mut acc53: uint16x8_t = zero;

        let mut idx = 0;
        while idx < 800 {
            let query0 = vld1q_u16(queries[0].as_ptr().add(idx));
            let query1 = vld1q_u16(queries[1].as_ptr().add(idx));
            let query2 = vld1q_u16(queries[2].as_ptr().add(idx));
            let query3 = vld1q_u16(queries[3].as_ptr().add(idx));
            let query4 = vld1q_u16(queries[4].as_ptr().add(idx));
            let query5 = vld1q_u16(queries[5].as_ptr().add(idx));

            // Stream one target at a time. Keeping all four target vectors
            // live alongside 24 accumulators and 6 query vectors would exceed
            // the architectural SIMD register file and force stack spills.
            let target0 = vld1q_u16(targets[0].as_ptr().add(idx));
            acc00 = vmlaq_u16(acc00, query0, target0);
            acc10 = vmlaq_u16(acc10, query1, target0);
            acc20 = vmlaq_u16(acc20, query2, target0);
            acc30 = vmlaq_u16(acc30, query3, target0);
            acc40 = vmlaq_u16(acc40, query4, target0);
            acc50 = vmlaq_u16(acc50, query5, target0);

            let target1 = vld1q_u16(targets[1].as_ptr().add(idx));
            acc01 = vmlaq_u16(acc01, query0, target1);
            acc11 = vmlaq_u16(acc11, query1, target1);
            acc21 = vmlaq_u16(acc21, query2, target1);
            acc31 = vmlaq_u16(acc31, query3, target1);
            acc41 = vmlaq_u16(acc41, query4, target1);
            acc51 = vmlaq_u16(acc51, query5, target1);

            let target2 = vld1q_u16(targets[2].as_ptr().add(idx));
            acc02 = vmlaq_u16(acc02, query0, target2);
            acc12 = vmlaq_u16(acc12, query1, target2);
            acc22 = vmlaq_u16(acc22, query2, target2);
            acc32 = vmlaq_u16(acc32, query3, target2);
            acc42 = vmlaq_u16(acc42, query4, target2);
            acc52 = vmlaq_u16(acc52, query5, target2);

            let target3 = vld1q_u16(targets[3].as_ptr().add(idx));
            acc03 = vmlaq_u16(acc03, query0, target3);
            acc13 = vmlaq_u16(acc13, query1, target3);
            acc23 = vmlaq_u16(acc23, query2, target3);
            acc33 = vmlaq_u16(acc33, query3, target3);
            acc43 = vmlaq_u16(acc43, query4, target3);
            acc53 = vmlaq_u16(acc53, query5, target3);
            idx += 8;
        }

        [
            [
                vaddvq_u16(acc00),
                vaddvq_u16(acc01),
                vaddvq_u16(acc02),
                vaddvq_u16(acc03),
            ],
            [
                vaddvq_u16(acc10),
                vaddvq_u16(acc11),
                vaddvq_u16(acc12),
                vaddvq_u16(acc13),
            ],
            [
                vaddvq_u16(acc20),
                vaddvq_u16(acc21),
                vaddvq_u16(acc22),
                vaddvq_u16(acc23),
            ],
            [
                vaddvq_u16(acc30),
                vaddvq_u16(acc31),
                vaddvq_u16(acc32),
                vaddvq_u16(acc33),
            ],
            [
                vaddvq_u16(acc40),
                vaddvq_u16(acc41),
                vaddvq_u16(acc42),
                vaddvq_u16(acc43),
            ],
            [
                vaddvq_u16(acc50),
                vaddvq_u16(acc51),
                vaddvq_u16(acc52),
                vaddvq_u16(acc53),
            ],
        ]
    }
}

#[inline]
fn simple_dot_product(a: &[u16], b: &[u16]) -> u16 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0u16;
    for i in 0..a.len() {
        sum = sum.wrapping_add(a[i].wrapping_mul(b[i]));
    }
    sum
}

pub fn non_existent_distance() -> Vec<RingElement<u16>> {
    vec![
        RingElement(SHARE_OF_MAX_DISTANCE.0),
        RingElement(SHARE_OF_MAX_DISTANCE.1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::{
            local::{generate_local_identities, LocalRuntime},
            session::Session,
        },
        network::mpc::{NetworkInt, NetworkValue},
        protocol::shared_iris::GaloisRingSharedIris,
        shares::{int_ring::IntRing2k, ring_impl::RingElement},
    };
    use aes_prng::AesRng;
    use ampc_actor_utils::protocol::prf::Prf;
    use eyre::{bail, Result};
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::{RngCore, SeedableRng};
    use rstest::rstest;
    use std::sync::Arc;
    use tokio::task::JoinSet;
    use tracing::instrument;

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_t_many<T>(session: &mut Session, shares: Vec<Share<T>>) -> Result<Vec<T>>
    where
        T: IntRing2k + NetworkInt,
    {
        let network = &mut session.network_session;

        let shares_b: Vec<_> = shares.iter().map(|s| s.b).collect();
        let message = shares_b;
        network.send_next(T::new_network_vec(message)).await?;

        // receiving from previous party
        let shares_c = {
            let net_message = network.receive_prev().await?;
            T::into_vec(net_message)
        }?;

        let res = shares
            .into_iter()
            .zip(shares_c)
            .map(|(s, c)| {
                let (a, b) = s.get_ab();
                (a + b + c).convert()
            })
            .collect();
        Ok(res)
    }

    #[tokio::test]
    async fn test_async_prf_setup() {
        let num_parties = 3;
        let identities = generate_local_identities();
        let mut seeds = Vec::new();
        for i in 0..num_parties {
            let mut seed = [0_u8; 16];
            seed[0] = i;
            seeds.push(seed);
        }
        let mut runtime = LocalRuntime::new(identities.clone(), seeds.clone())
            .await
            .unwrap();

        // check whether parties have sent/received the correct seeds.
        // P0: [seed_0, seed_2]
        // P1: [seed_1, seed_0]
        // P2: [seed_2, seed_1]
        // This is done by calling next() on the PRFs and see whether they match with
        // the ones created from scratch.

        // Alice
        let prf0 = &mut runtime.sessions[0].prf;
        assert_eq!(
            prf0.get_my_prf().next_u64(),
            Prf::new(seeds[0], seeds[2]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf0.get_prev_prf().next_u64(),
            Prf::new(seeds[0], seeds[2]).get_prev_prf().next_u64()
        );

        // Bob
        let prf1 = &mut runtime.sessions[1].prf;
        assert_eq!(
            prf1.get_my_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf1.get_prev_prf().next_u64(),
            Prf::new(seeds[1], seeds[0]).get_prev_prf().next_u64()
        );

        // Charlie
        let prf2 = &mut runtime.sessions[2].prf;
        assert_eq!(
            prf2.get_my_prf().next_u64(),
            Prf::new(seeds[2], seeds[1]).get_my_prf().next_u64()
        );
        assert_eq!(
            prf2.get_prev_prf().next_u64(),
            Prf::new(seeds[2], seeds[1]).get_prev_prf().next_u64()
        );
    }

    #[instrument(level = "trace", target = "searcher::network", skip_all)]
    async fn open_additive(session: &mut Session, x: Vec<RingElement<u16>>) -> Result<Vec<u16>> {
        let network = &mut session.network_session;

        network
            .send_next(NetworkValue::VecRing16(x.clone()))
            .await?;

        let message_bytes = NetworkValue::VecRing16(x.clone());
        network.send_prev(message_bytes).await?;

        let reply_0 = network.receive_prev().await;
        let reply_1 = network.receive_next().await;

        let missing_share_0 = match reply_0 {
            Ok(NetworkValue::VecRing16(element)) => element,
            _ => bail!("Could not deserialize VecRingElement16"),
        };
        let missing_share_1 = match reply_1 {
            Ok(NetworkValue::VecRing16(element)) => element,
            _ => bail!("Could not deserialize VecRingElement16"),
        };
        let opened_value: Vec<u16> = x
            .iter()
            .enumerate()
            .map(|(i, v)| (missing_share_0[i] + missing_share_1[i] + v).convert())
            .collect();
        Ok(opened_value)
    }

    #[tokio::test]
    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    async fn test_galois_ring_to_rep3(#[case] seed: u64) {
        let sessions = LocalRuntime::mock_sessions_with_channel().await.unwrap();
        let mut rng = AesRng::seed_from_u64(seed);

        let iris_db = IrisDB::new_random_rng(2, &mut rng).db;

        let first_entry =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());
        let second_entry =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[1].clone());

        let mut jobs = JoinSet::new();
        for (index, session) in sessions.iter().enumerate() {
            let own_shares = vec![(first_entry[index].clone(), second_entry[index].clone())]
                .into_iter()
                .map(|(x, mut y)| {
                    y.code.preprocess_iris_code_query_share();
                    y.mask.preprocess_mask_code_query_share();
                    Some((Arc::new(x), Arc::new(y)))
                })
                .collect_vec();
            let session = session.clone();
            jobs.spawn(async move {
                let mut player_session = session.lock().await;
                let x = galois_ring_pairwise_distance(own_shares);
                let opened_x = open_additive(&mut player_session, x.clone()).await.unwrap();
                let x_rep = galois_ring_to_rep3(&mut player_session, x).await.unwrap();
                let opened_x_rep = open_t_many(&mut player_session, x_rep).await.unwrap();
                (opened_x, opened_x_rep)
            });
        }
        let output0 = jobs.join_next().await.unwrap().unwrap();
        let output1 = jobs.join_next().await.unwrap().unwrap();
        let output2 = jobs.join_next().await.unwrap().unwrap();
        assert_eq!(output0, output1);
        assert_eq!(output0, output2);

        let (plain_d1, plain_d2) = iris_db[0].get_dot_distance_fraction(&iris_db[1]);
        assert_eq!(output0.0[0], plain_d1 as u16);
        assert_eq!(output0.0[1], plain_d2);

        assert_eq!(output0.1[0], plain_d1 as u16);
        assert_eq!(output0.1[1], plain_d2);
    }

    #[test]
    fn test_prerotated_query_cache_keeps_two_alternating_queries() {
        let mut rng = AesRng::seed_from_u64(0xCA_C4_E5);
        let iris_db = IrisDB::new_random_rng(2, &mut rng).db;
        let queries: Vec<ArcIris> = iris_db
            .into_iter()
            .map(|iris| {
                let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, iris);
                let mut query = shares[0].clone();
                query.code.preprocess_iris_code_query_share();
                query.mask.preprocess_mask_code_query_share();
                Arc::new(query)
            })
            .collect();

        let mut cache = PrerotatedQueryCache::new();

        let (storage, hit) = cache.get_or_fill::<31>(&queries[0]);
        assert!(!hit);
        let first_buffer = storage.code_data.as_ptr();

        let (storage, hit) = cache.get_or_fill::<31>(&queries[1]);
        assert!(!hit);
        let second_buffer = storage.code_data.as_ptr();
        assert_ne!(first_buffer, second_buffer);

        // Cloning an Arc preserves the cache key, and alternating between the
        // two live query allocations must not trigger either prerotation again.
        let first_query_clone = queries[0].clone();
        let (storage, hit) = cache.get_or_fill::<31>(&first_query_clone);
        assert!(hit);
        assert_eq!(storage.code_data.as_ptr(), first_buffer);

        let (storage, hit) = cache.get_or_fill::<31>(&queries[1]);
        assert!(hit);
        assert_eq!(storage.code_data.as_ptr(), second_buffer);
        assert_eq!(cache.entries.iter().flatten().count(), 2);

        // Rotation count is part of the key even when the Arc allocation is
        // identical. The replacement remains bounded to the same two buffers.
        let (storage, hit) = cache.get_or_fill::<13>(&queries[0]);
        assert!(!hit);
        assert_eq!(storage.cached_rotations, 13);
        let (storage, hit) = cache.get_or_fill::<13>(&queries[0]);
        assert!(hit);
        assert_eq!(storage.cached_rotations, 13);
        assert_eq!(cache.entries.iter().flatten().count(), 2);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dot_product_6x4_u16_matches_scalar_wrapping_reference() {
        const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;

        for seed in [0, 1, 42, u64::MAX] {
            let mut rng = AesRng::seed_from_u64(seed);
            let queries: [Vec<u16>; 6] =
                std::array::from_fn(|_| (0..ROW_SIZE).map(|_| rng.next_u32() as u16).collect());
            let targets: [Vec<u16>; 4] =
                std::array::from_fn(|_| (0..ROW_SIZE).map(|_| rng.next_u32() as u16).collect());
            let query_refs: [&[u16]; 6] = std::array::from_fn(|idx| queries[idx].as_slice());
            let target_refs: [&[u16]; 4] = std::array::from_fn(|idx| targets[idx].as_slice());

            let tiled = dot_product_6x4_u16(query_refs, target_refs);
            let reference = std::array::from_fn(|query_idx| {
                std::array::from_fn(|target_idx| {
                    simple_dot_product(&queries[query_idx], &targets[target_idx])
                })
            });

            assert_eq!(tiled, reference, "6x4 tile mismatch for seed {seed}");
        }
    }

    #[rstest]
    #[case(0, 1)]
    #[case(1, 5)]
    #[case(42, 10)]
    fn test_rotation_aware_pairwise_distance_rowmajor_equivalence(
        #[case] seed: u64,
        #[case] num_targets: usize,
    ) {
        test_rotation_aware_pairwise_distance_rowmajor_equivalence_generic::<11>(seed, num_targets);
        test_rotation_aware_pairwise_distance_rowmajor_equivalence_generic::<13>(seed, num_targets);
        test_rotation_aware_pairwise_distance_rowmajor_equivalence_generic::<31>(seed, num_targets);
    }

    fn test_rotation_aware_pairwise_distance_rowmajor_equivalence_generic<
        const ROTATIONS: usize,
    >(
        seed: u64,
        num_targets: usize,
    ) {
        use crate::protocol::shared_iris::GaloisRingSharedIris;

        let mut rng = AesRng::seed_from_u64(seed);

        // Create enough iris codes: 1 for query + num_targets for targets
        let iris_db = IrisDB::new_random_rng(1 + num_targets, &mut rng).db;

        // Generate shares for the query
        let query_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());

        // Generate shares for targets
        let target_shares: Vec<_> = iris_db[1..]
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        // Test for each party's share
        for party_id in 0..3 {
            // Prepare query with preprocessing
            let mut query = query_shares[party_id].clone();
            query.code.preprocess_iris_code_query_share();
            query.mask.preprocess_mask_code_query_share();
            let query_arc = Arc::new(query);

            // Prepare targets (no preprocessing needed for targets)
            let targets: Vec<ArcIris> = target_shares
                .iter()
                .map(|shares| Arc::new(shares[party_id].clone()))
                .collect();

            // Call the original function
            let result_original = rotation_aware_pairwise_distance::<ROTATIONS, _>(
                &query_arc,
                targets.iter().map(Some),
            );

            // Call the row-major function
            let result_rowmajor = rotation_aware_pairwise_distance_rowmajor::<ROTATIONS, _>(
                &query_arc,
                targets.iter().map(Some),
            );

            // Verify results are identical
            assert_eq!(
                result_original.len(),
                result_rowmajor.len(),
                "Result lengths should match"
            );
            for (i, (orig, rowmaj)) in result_original
                .iter()
                .zip(result_rowmajor.iter())
                .enumerate()
            {
                assert_eq!(
                    orig, rowmaj,
                    "Mismatch at index {} for party {} with seed {}: original {:?} != rowmajor {:?}",
                    i, party_id, seed, orig, rowmaj
                );
            }
        }
    }

    fn test_rotation_aware_pairwise_distance_rowmajor_with_none_targets_generic<
        const ROTATIONS: usize,
    >() {
        use crate::protocol::shared_iris::GaloisRingSharedIris;

        let mut rng = AesRng::seed_from_u64(123);

        // Create iris codes
        let iris_db = IrisDB::new_random_rng(3, &mut rng).db;

        // Generate shares
        let query_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());
        let target_shares: Vec<_> = iris_db[1..]
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        // Test for party 0
        let party_id = 0;

        // Prepare query with preprocessing
        let mut query = query_shares[party_id].clone();
        query.code.preprocess_iris_code_query_share();
        query.mask.preprocess_mask_code_query_share();
        let query_arc = Arc::new(query);

        // Keep the Arc values alive
        let target0: ArcIris = Arc::new(target_shares[0][party_id].clone());
        let target1: ArcIris = Arc::new(target_shares[1][party_id].clone());
        let targets_with_none: Vec<Option<&ArcIris>> = vec![
            Some(&target0),
            None, // This should get SHARE_OF_MAX_DISTANCE
            Some(&target1),
        ];

        // Call both functions
        let result_original = rotation_aware_pairwise_distance::<ROTATIONS, _>(
            &query_arc,
            targets_with_none.clone().into_iter(),
        );
        let result_rowmajor = rotation_aware_pairwise_distance_rowmajor::<ROTATIONS, _>(
            &query_arc,
            targets_with_none.into_iter(),
        );

        // Verify results are identical
        assert_eq!(result_original.len(), result_rowmajor.len());
        for (i, (orig, rowmaj)) in result_original
            .iter()
            .zip(result_rowmajor.iter())
            .enumerate()
        {
            assert_eq!(orig, rowmaj, "Mismatch at index {}", i);
        }

        // Verify that None targets produce SHARE_OF_MAX_DISTANCE
        let (max_code, max_mask) = SHARE_OF_MAX_DISTANCE;
        // Second target is None, so indices 31*2..31*4 should all be max distance
        for rot in 0..ROTATIONS {
            let idx = ROTATIONS * 2 + rot * 2;
            assert_eq!(
                result_rowmajor[idx].0, max_code,
                "Code at rotation {} should be max",
                rot
            );
            assert_eq!(
                result_rowmajor[idx + 1].0,
                max_mask,
                "Mask at rotation {} should be max",
                rot
            );
        }
    }

    #[test]
    fn test_rotation_aware_pairwise_distance_rowmajor_with_none_targets() {
        test_rotation_aware_pairwise_distance_rowmajor_with_none_targets_generic::<11>();
        test_rotation_aware_pairwise_distance_rowmajor_with_none_targets_generic::<13>();
        test_rotation_aware_pairwise_distance_rowmajor_with_none_targets_generic::<31>();
    }
}
