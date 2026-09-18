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

    /// Compile-time proof that every rotation amount is a multiple of 4.
    /// The doubled-row mixed-plane kernel addresses each rotation as an
    /// 8-byte-aligned window plus a 4-element phase
    /// (`mixed_scan::query_window`), which is only sound under this property.
    /// Its sole consumer is that aarch64-only kernel, so gate it to keep
    /// non-aarch64 builds free of dead code.
    #[cfg(target_arch = "aarch64")]
    const ROTATION_AMOUNTS_ARE_MULTIPLES_OF_FOUR: () = {
        let mut i = 0;
        while i < ROTATIONS {
            assert!(
                Self::ROTATION_AMOUNTS[i] % 4 == 0,
                "rotation amounts must be multiples of 4"
            );
            i += 1;
        }
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
    // Iterate target groups in the outer loop so each target is streamed as
    // one contiguous `rows * ROW_SIZE` run. Row-outer ordering instead visits
    // every target once per row, which restarts the hardware prefetcher on a
    // short 1600-byte burst for each (row, target) pair. The query rotation
    // matrix is re-streamed per group, but it stays L2-resident and its
    // (row-major) traversal is itself perfectly sequential.
    let mut target_idx = 0;
    while target_idx + 4 <= targets.len() {
        accumulate_group_rows::<ROTATIONS>(
            query,
            &targets[target_idx..target_idx + 4],
            rows,
            target_idx,
            result_lane,
            additive_shares,
        );
        target_idx += 4;
    }

    for (target_offset, target) in targets[target_idx..].iter().enumerate() {
        let Some(target) = target else {
            continue;
        };
        accumulate_scalar_target_all_rows::<ROTATIONS>(
            query,
            target,
            rows,
            target_idx + target_offset,
            result_lane,
            additive_shares,
        );
    }
}

/// One group of four targets against every rotation tile, row by row.
#[cfg(target_arch = "aarch64")]
fn accumulate_group_rows<const ROTATIONS: usize>(
    query: &[u16],
    group: &[Option<&[u16]>],
    rows: usize,
    target_idx: usize,
    result_lane: usize,
    additive_shares: &mut [RingElement<u16>],
) {
    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;

    {
        if let (Some(full_target0), Some(full_target1), Some(full_target2), Some(full_target3)) =
            (group[0], group[1], group[2], group[3])
        {
            for row_idx in 0..rows {
                let query_rows_start = row_idx * ROTATIONS * ROW_SIZE;
                let query_rows = &query[query_rows_start..query_rows_start + ROTATIONS * ROW_SIZE];
                let target_row_start = row_idx * ROW_SIZE;
                let target0 = &full_target0[target_row_start..target_row_start + ROW_SIZE];
                let target1 = &full_target1[target_row_start..target_row_start + ROW_SIZE];
                let target2 = &full_target2[target_row_start..target_row_start + ROW_SIZE];
                let target3 = &full_target3[target_row_start..target_row_start + ROW_SIZE];

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
                    let partials = dot_product_nx4_u16(
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
                    let partials = dot_product_nx4_u16(
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
                    let [partials] =
                        dot_product_nx4_u16([query_row], [target0, target1, target2, target3]);
                    for (target_offset, partial) in partials.into_iter().enumerate() {
                        let result_idx = (target_idx + target_offset) * ROTATIONS * 2
                            + rotation_idx * 2
                            + result_lane;
                        additive_shares[result_idx].0 =
                            additive_shares[result_idx].0.wrapping_add(partial);
                    }
                    rotation_idx += 1;
                }
            }
        } else {
            // Missing vectors are uncommon in a full scan. Preserve their
            // sentinel handling below while still evaluating live members
            // of a mixed group exactly.
            for (target_offset, target) in group.iter().enumerate() {
                let Some(target) = target else {
                    continue;
                };
                accumulate_scalar_target_all_rows::<ROTATIONS>(
                    query,
                    target,
                    rows,
                    target_idx + target_offset,
                    result_lane,
                    additive_shares,
                );
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn accumulate_scalar_target_all_rows<const ROTATIONS: usize>(
    query: &[u16],
    target: &[u16],
    rows: usize,
    target_idx: usize,
    result_lane: usize,
    additive_shares: &mut [RingElement<u16>],
) {
    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;
    for row_idx in 0..rows {
        let query_rows_start = row_idx * ROTATIONS * ROW_SIZE;
        let query_rows = &query[query_rows_start..query_rows_start + ROTATIONS * ROW_SIZE];
        let target_row_start = row_idx * ROW_SIZE;
        let target_row = &target[target_row_start..target_row_start + ROW_SIZE];
        accumulate_scalar_target::<ROTATIONS>(
            query_rows,
            target_row,
            target_idx,
            result_lane,
            additive_shares,
        );
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

/// `N` query rotations against four targets in one pass.
///
/// One 8-lane block streams each target vector past all `N` query vectors, so
/// every target load is reused for `N` MLAs and the `N * 4` accumulators stay
/// in registers. The 6-wide instantiation is the main scan tile; 4 and 1 cover
/// the remainders of the 11/13/31-rotation schedules (6+4+1, 6+6+1, 6x5+1).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_product_nx4_u16<const N: usize>(
    queries: [&[u16]; N],
    targets: [&[u16]; 4],
) -> [[u16; 4]; N] {
    use std::arch::aarch64::{vaddvq_u16, vdupq_n_u16, vld1q_u16, vmlaq_u16};

    debug_assert!(queries.iter().all(|query| query.len() == 800));
    debug_assert!(targets.iter().all(|target| target.len() == 800));

    // SAFETY: AArch64 guarantees Advanced SIMD. Each pointer is derived from
    // an 800-element slice and the loop only issues unaligned-safe 8-lane
    // loads at offsets 0..792. All arithmetic deliberately wraps in u16.
    unsafe {
        let zero = vdupq_n_u16(0);
        let mut acc = [[zero; 4]; N];

        let mut idx = 0;
        while idx < 800 {
            let mut query = [zero; N];
            for (vector, source) in query.iter_mut().zip(&queries) {
                *vector = vld1q_u16(source.as_ptr().add(idx));
            }
            // Stream one target at a time. Keeping all four target vectors
            // live alongside the accumulators and query vectors would exceed
            // the architectural SIMD register file for the 6-wide tile and
            // force stack spills.
            for (lane, source) in targets.iter().enumerate() {
                let target = vld1q_u16(source.as_ptr().add(idx));
                for (row, query) in acc.iter_mut().zip(&query) {
                    row[lane] = vmlaq_u16(row[lane], *query, target);
                }
            }
            idx += 8;
        }

        let mut out = [[0u16; 4]; N];
        for (row, accs) in out.iter_mut().zip(&acc) {
            for (value, acc) in row.iter_mut().zip(accs) {
                *value = vaddvq_u16(*acc);
            }
        }
        out
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

#[cfg(target_arch = "aarch64")]
pub use mixed_scan::{
    rotation_aware_pairwise_distance_mixed, rotation_aware_pairwise_distance_mixed_pair,
};

/// UMMLA-based full-rotation scan over mixed lo/hi plane residents.
///
/// Every u16 product is decomposed as
/// `x*y mod 2^16 = xl*yl + 2^8*(xl*yh + xh*yl)` — the `xh*yh` term vanishes
/// modulo 2^16. With rows stored as 8-byte-interleaved planes
/// `[lo0..7 | hi0..7]` on both the query and target side, a single UMMLA per
/// (rotation, target) per 8 coefficients accumulates all three needed
/// partial products into separate u32 lanes (`[ll, lh, hl, discard]`), and
/// the discarded lane is exactly the vanishing high-high term. Results are
/// bit-identical to the u16 MLA kernel.
#[cfg(target_arch = "aarch64")]
mod mixed_scan {
    use super::{
        PrerotatedQueryRowMajor, PrerotatedQueryRowMajorView, RingElement, SHARE_OF_MAX_DISTANCE,
    };
    use crate::protocol::shared_iris::{ArcIris, MixedPlaneIris};
    use std::arch::asm;
    use std::cell::RefCell;

    const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;
    /// A mixed-plane row occupies the same bytes as the u16 row.
    const MIXED_ROW_BYTES: usize = 2 * ROW_SIZE;
    /// Two copies of a row, used to expose every circular 800-element window.
    const DOUBLED_MIXED_ROW_BYTES: usize = 2 * MIXED_ROW_BYTES;
    const QUERY_PHASES: usize = 2;
    const CODE_ROWS: usize = PrerotatedQueryRowMajor::CODE_ROWS;
    const MASK_ROWS: usize = PrerotatedQueryRowMajor::MASK_ROWS;
    const GROUP_TARGETS: usize = 4;
    const TILE_ROTATIONS: usize = 6;

    /// Compact mixed-plane query. Each row has two doubled copies: one starts
    /// at coefficient 0 and one at coefficient 4. Since every supported
    /// rotation is a multiple of four, selecting a phase makes its start
    /// 8-element aligned and therefore directly loadable by UMMLA.
    struct DoubledQueryMixed {
        code: Vec<u8>,
        mask: Vec<u8>,
        cached_query: Option<ArcIris>,
        cached_rotations: usize,
    }

    impl DoubledQueryMixed {
        fn new_buffer() -> Self {
            Self {
                code: vec![0u8; CODE_ROWS * QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES],
                mask: vec![0u8; MASK_ROWS * QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES],
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

        fn fill_if_changed<const ROTATIONS: usize>(&mut self, query: &ArcIris) {
            if self.matches::<ROTATIONS>(query) {
                return;
            }
            let fill_component = |component: &mut [u8], rows: usize, coefs: &[u16]| {
                for row_idx in 0..rows {
                    let src_row = &coefs[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];
                    for phase in 0..QUERY_PHASES {
                        let dst_start = (row_idx * QUERY_PHASES + phase) * DOUBLED_MIXED_ROW_BYTES;
                        let dst = &mut component[dst_start..dst_start + DOUBLED_MIXED_ROW_BYTES];
                        for group in 0..(2 * ROW_SIZE / 8) {
                            for lane in 0..8 {
                                let src_idx = (phase * 4 + group * 8 + lane) % ROW_SIZE;
                                let value = src_row[src_idx];
                                dst[group * 16 + lane] = value as u8;
                                dst[group * 16 + 8 + lane] = (value >> 8) as u8;
                            }
                        }
                    }
                }
            };
            fill_component(&mut self.code, CODE_ROWS, &query.code.coefs);
            fill_component(&mut self.mask, MASK_ROWS, &query.mask.coefs);
            self.cached_query = Some(query.clone());
            self.cached_rotations = ROTATIONS;
        }
    }

    thread_local! {
        static DOUBLED_MIXED: RefCell<[Option<DoubledQueryMixed>; 2]> =
            const { RefCell::new([None, None]) };
        static MIXED_LRU: RefCell<usize> = const { RefCell::new(0) };
        static PAIR_PACKED: RefCell<Option<PairPackedQueryMixed>> = const { RefCell::new(None) };
    }

    /// Cross-query packed doubled rows for the fused two-query scan. For each
    /// 8-coefficient group, one 16-byte block holds a single byte plane of
    /// BOTH queries: `[qa_plane(8) | qb_plane(8)]`. Loaded as the UMMLA "B"
    /// operand against a target block `[t_lo(8) | t_hi(8)]` (or a
    /// `trn1`-combined `[t1_lo | t2_lo]`), one instruction then produces
    /// partial products for both queries at once, eliminating the discarded
    /// high-high lane of the single-query scheme: three UMMLA cover what
    /// four cover in the unpacked layout.
    ///
    /// Both orientations rotate by the same amounts, so the two-phase doubled
    /// window addressing is identical to [`DoubledQueryMixed`].
    struct PairPackedQueryMixed {
        code_lo: Vec<u8>,
        code_hi: Vec<u8>,
        mask_lo: Vec<u8>,
        mask_hi: Vec<u8>,
        cached_queries: Option<(ArcIris, ArcIris)>,
        cached_rotations: usize,
    }

    impl PairPackedQueryMixed {
        fn new_buffer() -> Self {
            Self {
                code_lo: vec![0u8; CODE_ROWS * QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES],
                code_hi: vec![0u8; CODE_ROWS * QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES],
                mask_lo: vec![0u8; MASK_ROWS * QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES],
                mask_hi: vec![0u8; MASK_ROWS * QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES],
                cached_queries: None,
                cached_rotations: 0,
            }
        }

        fn matches<const ROTATIONS: usize>(&self, queries: [&ArcIris; 2]) -> bool {
            self.cached_rotations == ROTATIONS
                && self.cached_queries.as_ref().is_some_and(|(a, b)| {
                    std::sync::Arc::ptr_eq(a, queries[0]) && std::sync::Arc::ptr_eq(b, queries[1])
                })
        }

        fn fill_if_changed<const ROTATIONS: usize>(&mut self, queries: [&ArcIris; 2]) {
            if self.matches::<ROTATIONS>(queries) {
                return;
            }
            let fill_component =
                |lo: &mut [u8], hi: &mut [u8], rows: usize, coefs_a: &[u16], coefs_b: &[u16]| {
                    for row_idx in 0..rows {
                        let src_a = &coefs_a[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];
                        let src_b = &coefs_b[row_idx * ROW_SIZE..(row_idx + 1) * ROW_SIZE];
                        for phase in 0..QUERY_PHASES {
                            let dst_start =
                                (row_idx * QUERY_PHASES + phase) * DOUBLED_MIXED_ROW_BYTES;
                            let lo = &mut lo[dst_start..dst_start + DOUBLED_MIXED_ROW_BYTES];
                            let hi = &mut hi[dst_start..dst_start + DOUBLED_MIXED_ROW_BYTES];
                            for group in 0..(2 * ROW_SIZE / 8) {
                                for lane in 0..8 {
                                    let src_idx = (phase * 4 + group * 8 + lane) % ROW_SIZE;
                                    let value_a = src_a[src_idx];
                                    let value_b = src_b[src_idx];
                                    lo[group * 16 + lane] = value_a as u8;
                                    lo[group * 16 + 8 + lane] = value_b as u8;
                                    hi[group * 16 + lane] = (value_a >> 8) as u8;
                                    hi[group * 16 + 8 + lane] = (value_b >> 8) as u8;
                                }
                            }
                        }
                    }
                };
            fill_component(
                &mut self.code_lo,
                &mut self.code_hi,
                CODE_ROWS,
                &queries[0].code.coefs,
                &queries[1].code.coefs,
            );
            fill_component(
                &mut self.mask_lo,
                &mut self.mask_hi,
                MASK_ROWS,
                &queries[0].mask.coefs,
                &queries[1].mask.coefs,
            );
            self.cached_queries = Some((queries[0].clone(), queries[1].clone()));
            self.cached_rotations = ROTATIONS;
        }
    }

    #[inline(always)]
    fn reduce_ummla(lanes: [u32; 4]) -> u16 {
        lanes[0].wrapping_add(lanes[1].wrapping_add(lanes[2]) << 8) as u16
    }

    /// Six rotations against four targets. The explicit assembly keeps all
    /// 24 accumulators in registers and lets Neoverse V2 issue UMMLA on all
    /// four SIMD pipes. The stable Rust i8mm intrinsic is not available yet.
    #[target_feature(enable = "i8mm")]
    unsafe fn dot_product_6x4_ummla(
        queries: [*const u8; TILE_ROTATIONS],
        targets: [*const u8; GROUP_TARGETS],
    ) -> [[u16; GROUP_TARGETS]; TILE_ROTATIONS] {
        let mut raw = [[0u32; 4]; TILE_ROTATIONS * GROUP_TARGETS];
        let q0 = queries[0];
        let q1 = queries[1];
        let q2 = queries[2];
        let q3 = queries[3];
        let q4 = queries[4];
        let q5 = queries[5];
        let t0 = targets[0];
        let t1 = targets[1];
        let t2 = targets[2];
        let t3 = targets[3];
        asm!(
            "movi v0.4s, #0", "movi v1.4s, #0", "movi v2.4s, #0", "movi v3.4s, #0",
            "movi v4.4s, #0", "movi v5.4s, #0", "movi v6.4s, #0", "movi v7.4s, #0",
            "movi v8.4s, #0", "movi v9.4s, #0", "movi v10.4s, #0", "movi v11.4s, #0",
            "movi v12.4s, #0", "movi v13.4s, #0", "movi v14.4s, #0", "movi v15.4s, #0",
            "movi v16.4s, #0", "movi v17.4s, #0", "movi v18.4s, #0", "movi v19.4s, #0",
            "movi v20.4s, #0", "movi v21.4s, #0", "movi v22.4s, #0", "movi v23.4s, #0",
            "mov {groups}, #100",
            "2:",
            "ldr q24, [{q0}], #16", "ldr q25, [{q1}], #16", "ldr q26, [{q2}], #16",
            "ldr q27, [{q3}], #16", "ldr q28, [{q4}], #16", "ldr q29, [{q5}], #16",
            "ldr q30, [{t0}], #16",
            "ummla v0.4s, v24.16b, v30.16b", "ummla v4.4s, v25.16b, v30.16b",
            "ummla v8.4s, v26.16b, v30.16b", "ummla v12.4s, v27.16b, v30.16b",
            "ummla v16.4s, v28.16b, v30.16b", "ummla v20.4s, v29.16b, v30.16b",
            "ldr q30, [{t1}], #16",
            "ummla v1.4s, v24.16b, v30.16b", "ummla v5.4s, v25.16b, v30.16b",
            "ummla v9.4s, v26.16b, v30.16b", "ummla v13.4s, v27.16b, v30.16b",
            "ummla v17.4s, v28.16b, v30.16b", "ummla v21.4s, v29.16b, v30.16b",
            "ldr q30, [{t2}], #16",
            "ummla v2.4s, v24.16b, v30.16b", "ummla v6.4s, v25.16b, v30.16b",
            "ummla v10.4s, v26.16b, v30.16b", "ummla v14.4s, v27.16b, v30.16b",
            "ummla v18.4s, v28.16b, v30.16b", "ummla v22.4s, v29.16b, v30.16b",
            "ldr q30, [{t3}], #16",
            "ummla v3.4s, v24.16b, v30.16b", "ummla v7.4s, v25.16b, v30.16b",
            "ummla v11.4s, v26.16b, v30.16b", "ummla v15.4s, v27.16b, v30.16b",
            "ummla v19.4s, v28.16b, v30.16b", "ummla v23.4s, v29.16b, v30.16b",
            "subs {groups}, {groups}, #1", "b.ne 2b",
            "stp q0, q1, [{out}, #0]", "stp q2, q3, [{out}, #32]",
            "stp q4, q5, [{out}, #64]", "stp q6, q7, [{out}, #96]",
            "stp q8, q9, [{out}, #128]", "stp q10, q11, [{out}, #160]",
            "stp q12, q13, [{out}, #192]", "stp q14, q15, [{out}, #224]",
            "stp q16, q17, [{out}, #256]", "stp q18, q19, [{out}, #288]",
            "stp q20, q21, [{out}, #320]", "stp q22, q23, [{out}, #352]",
            q0 = inout(reg) q0 => _, q1 = inout(reg) q1 => _, q2 = inout(reg) q2 => _,
            q3 = inout(reg) q3 => _, q4 = inout(reg) q4 => _, q5 = inout(reg) q5 => _,
            t0 = inout(reg) t0 => _, t1 = inout(reg) t1 => _, t2 = inout(reg) t2 => _,
            t3 = inout(reg) t3 => _, out = in(reg) raw.as_mut_ptr(), groups = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, options(nostack),
        );

        std::array::from_fn(|rotation| {
            std::array::from_fn(|target| reduce_ummla(raw[rotation * GROUP_TARGETS + target]))
        })
    }

    #[target_feature(enable = "i8mm")]
    unsafe fn dot_product_1x4_ummla(
        query: *const u8,
        targets: [*const u8; GROUP_TARGETS],
    ) -> [u16; GROUP_TARGETS] {
        let mut raw = [[0u32; 4]; GROUP_TARGETS];
        let t0 = targets[0];
        let t1 = targets[1];
        let t2 = targets[2];
        let t3 = targets[3];
        asm!(
            "movi v0.4s, #0", "movi v1.4s, #0", "movi v2.4s, #0", "movi v3.4s, #0",
            "mov {groups}, #100",
            "2:",
            "ldr q24, [{query}], #16",
            "ldr q30, [{t0}], #16", "ummla v0.4s, v24.16b, v30.16b",
            "ldr q30, [{t1}], #16", "ummla v1.4s, v24.16b, v30.16b",
            "ldr q30, [{t2}], #16", "ummla v2.4s, v24.16b, v30.16b",
            "ldr q30, [{t3}], #16", "ummla v3.4s, v24.16b, v30.16b",
            "subs {groups}, {groups}, #1", "b.ne 2b",
            "stp q0, q1, [{out}, #0]", "stp q2, q3, [{out}, #32]",
            query = inout(reg) query => _,
            t0 = inout(reg) t0 => _, t1 = inout(reg) t1 => _,
            t2 = inout(reg) t2 => _, t3 = inout(reg) t3 => _,
            out = in(reg) raw.as_mut_ptr(), groups = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v24") _, out("v30") _, options(nostack),
        );
        std::array::from_fn(|target| reduce_ummla(raw[target]))
    }

    /// Four rotations of a packed query pair against four targets (two
    /// target pairs). Per (rotation, target pair), three UMMLA produce all
    /// twelve needed byte-plane products for both queries and both targets —
    /// none of the multiplier work lands in a discarded lane:
    /// - `acc_a = [t_lo|t_hi] x [qa_lo|qb_lo]` -> ll and lo*hi for target 2p
    /// - `acc_b = same for target 2p+1`
    /// - `acc_c = trn1(t_2p, t_2p+1) x [qa_hi|qb_hi]` -> hi*lo for both
    ///
    /// The trn1-combined lo rows are built from the already-loaded target
    /// registers, so targets are still loaded once per group step. Exactly
    /// fills the 32-register file: 24 accumulators + 4 targets + 2 trn +
    /// 2 query operands. Adds this row's products into `raw` so callers
    /// accumulate a whole component across rows before one reduction.
    #[target_feature(enable = "i8mm")]
    unsafe fn dot_product_pair_4x2p_acc(
        query_lo: [*const u8; 4],
        query_hi: [*const u8; 4],
        targets: [*const u8; GROUP_TARGETS],
        raw: &mut [[u32; 4]; 24],
    ) {
        let ql0 = query_lo[0];
        let ql1 = query_lo[1];
        let ql2 = query_lo[2];
        let ql3 = query_lo[3];
        let qh0 = query_hi[0];
        let qh1 = query_hi[1];
        let qh2 = query_hi[2];
        let qh3 = query_hi[3];
        let t0 = targets[0];
        let t1 = targets[1];
        let t2 = targets[2];
        let t3 = targets[3];
        asm!(
            // Accumulators are loaded and stored rather than zero-initialized:
            // callers accumulate a whole component's rows in u32 and reduce
            // once per tile. ldp/stp run on the load/store pipes, which have
            // headroom, instead of movi on the contended SIMD pipes.
            "ldp q0, q1, [{out}, #0]", "ldp q2, q3, [{out}, #32]",
            "ldp q4, q5, [{out}, #64]", "ldp q6, q7, [{out}, #96]",
            "ldp q8, q9, [{out}, #128]", "ldp q10, q11, [{out}, #160]",
            "ldp q12, q13, [{out}, #192]", "ldp q14, q15, [{out}, #224]",
            "ldp q16, q17, [{out}, #256]", "ldp q18, q19, [{out}, #288]",
            "ldp q20, q21, [{out}, #320]", "ldp q22, q23, [{out}, #352]",
            "mov {groups}, #100",
            "2:",
            "ldr q24, [{t0}], #16", "ldr q25, [{t1}], #16",
            "ldr q26, [{t2}], #16", "ldr q27, [{t3}], #16",
            "trn1 v28.2d, v24.2d, v25.2d",
            "trn1 v29.2d, v26.2d, v27.2d",
            "ldr q30, [{ql0}], #16", "ldr q31, [{qh0}], #16",
            "ummla v0.4s, v24.16b, v30.16b", "ummla v1.4s, v25.16b, v30.16b",
            "ummla v2.4s, v28.16b, v31.16b", "ummla v3.4s, v26.16b, v30.16b",
            "ummla v4.4s, v27.16b, v30.16b", "ummla v5.4s, v29.16b, v31.16b",
            "ldr q30, [{ql1}], #16", "ldr q31, [{qh1}], #16",
            "ummla v6.4s, v24.16b, v30.16b", "ummla v7.4s, v25.16b, v30.16b",
            "ummla v8.4s, v28.16b, v31.16b", "ummla v9.4s, v26.16b, v30.16b",
            "ummla v10.4s, v27.16b, v30.16b", "ummla v11.4s, v29.16b, v31.16b",
            "ldr q30, [{ql2}], #16", "ldr q31, [{qh2}], #16",
            "ummla v12.4s, v24.16b, v30.16b", "ummla v13.4s, v25.16b, v30.16b",
            "ummla v14.4s, v28.16b, v31.16b", "ummla v15.4s, v26.16b, v30.16b",
            "ummla v16.4s, v27.16b, v30.16b", "ummla v17.4s, v29.16b, v31.16b",
            "ldr q30, [{ql3}], #16", "ldr q31, [{qh3}], #16",
            "ummla v18.4s, v24.16b, v30.16b", "ummla v19.4s, v25.16b, v30.16b",
            "ummla v20.4s, v28.16b, v31.16b", "ummla v21.4s, v26.16b, v30.16b",
            "ummla v22.4s, v27.16b, v30.16b", "ummla v23.4s, v29.16b, v31.16b",
            "subs {groups}, {groups}, #1", "b.ne 2b",
            "stp q0, q1, [{out}, #0]", "stp q2, q3, [{out}, #32]",
            "stp q4, q5, [{out}, #64]", "stp q6, q7, [{out}, #96]",
            "stp q8, q9, [{out}, #128]", "stp q10, q11, [{out}, #160]",
            "stp q12, q13, [{out}, #192]", "stp q14, q15, [{out}, #224]",
            "stp q16, q17, [{out}, #256]", "stp q18, q19, [{out}, #288]",
            "stp q20, q21, [{out}, #320]", "stp q22, q23, [{out}, #352]",
            ql0 = inout(reg) ql0 => _, ql1 = inout(reg) ql1 => _,
            ql2 = inout(reg) ql2 => _, ql3 = inout(reg) ql3 => _,
            qh0 = inout(reg) qh0 => _, qh1 = inout(reg) qh1 => _,
            qh2 = inout(reg) qh2 => _, qh3 = inout(reg) qh3 => _,
            t0 = inout(reg) t0 => _, t1 = inout(reg) t1 => _, t2 = inout(reg) t2 => _,
            t3 = inout(reg) t3 => _, out = in(reg) raw.as_mut_ptr(), groups = out(reg) _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _, options(nostack),
        );
    }

    /// Scatter one rotation's packed-pair accumulator block (`[a, b, c]` for
    /// two targets x two queries) into both queries' share vectors. The
    /// packed path writes each result exactly once, so the mask lane's
    /// doubling (the epilogue's job on the other paths) is folded in here.
    #[inline(always)]
    fn scatter_pair_block<const ROTATIONS: usize>(
        block: &[[u32; 4]],
        base_target_idx: usize,
        pair: usize,
        rotation: usize,
        result_lane: usize,
        additive_shares: &mut [Vec<RingElement<u16>>; 2],
    ) {
        let acc_a = &block[0];
        let acc_b = &block[1];
        let acc_c = &block[2];
        let lane_scale = 1 + result_lane as u16;
        for (query, shares) in additive_shares.iter_mut().enumerate() {
            let first =
                (acc_a[query].wrapping_add(acc_a[2 + query].wrapping_add(acc_c[query]) << 8)
                    as u16)
                    .wrapping_mul(lane_scale);
            let second = (acc_b[query]
                .wrapping_add(acc_b[2 + query].wrapping_add(acc_c[2 + query]) << 8)
                as u16)
                .wrapping_mul(lane_scale);
            let first_idx =
                (base_target_idx + pair * 2) * ROTATIONS * 2 + rotation * 2 + result_lane;
            let second_idx =
                (base_target_idx + pair * 2 + 1) * ROTATIONS * 2 + rotation * 2 + result_lane;
            shares[first_idx].0 = shares[first_idx].0.wrapping_add(first);
            shares[second_idx].0 = shares[second_idx].0.wrapping_add(second);
        }
    }

    /// Packed-pair scan of one component of four present targets. Rows
    /// accumulate into persistent per-tile u32 buffers (u32 lanes cannot
    /// overflow: 16 rows x 800 coefficients x 255^2 < 2^32) and each tile is
    /// reduced and scattered once at the end, instead of once per row.
    #[target_feature(enable = "i8mm")]
    unsafe fn scan_four_targets_pair_packed<const ROTATIONS: usize>(
        query_lo: &[u8],
        query_hi: &[u8],
        rows: usize,
        targets: [&[u8]; GROUP_TARGETS],
        base_target_idx: usize,
        result_lane: usize,
        additive_shares: &mut [Vec<RingElement<u16>>; 2],
    ) {
        const PAIR_TILE_ROTATIONS: usize = 4;
        const MAX_TILES: usize = 8;
        let n_tiles = ROTATIONS.div_ceil(PAIR_TILE_ROTATIONS);
        assert!(n_tiles <= MAX_TILES, "unsupported rotation count");
        let mut raws = [[[0u32; 4]; 24]; MAX_TILES];

        // Row `r`'s window for a rotation is the row-0 window plus a constant
        // stride, so the 2 x 4 window pointers per tile are derived once.
        // The final tile is padded: rotation slots beyond `ROTATIONS` reuse
        // the last valid window and their results are simply not scattered.
        // For the 31-rotation scan this wastes 1/32 of the tile work, far
        // less than single-rotation remainder passes.
        let query_row_stride = QUERY_PHASES * DOUBLED_MIXED_ROW_BYTES;
        let window_bases: [([*const u8; 4], [*const u8; 4]); MAX_TILES] =
            std::array::from_fn(|tile| {
                let rotation = (tile * PAIR_TILE_ROTATIONS).min(ROTATIONS - 1);
                let live_rotations = PAIR_TILE_ROTATIONS.min(ROTATIONS - rotation);
                (
                    std::array::from_fn(|offset| {
                        query_window::<ROTATIONS>(
                            query_lo,
                            0,
                            rotation + offset.min(live_rotations - 1),
                        )
                    }),
                    std::array::from_fn(|offset| {
                        query_window::<ROTATIONS>(
                            query_hi,
                            0,
                            rotation + offset.min(live_rotations - 1),
                        )
                    }),
                )
            });

        for row in 0..rows {
            let target_row_start = row * MIXED_ROW_BYTES;
            let target_ptrs =
                std::array::from_fn(|target| targets[target][target_row_start..].as_ptr());
            let row_offset = row * query_row_stride;
            for (tile, raw) in raws.iter_mut().enumerate().take(n_tiles) {
                let (lo_bases, hi_bases) = &window_bases[tile];
                let lo_ptrs = std::array::from_fn(|offset| lo_bases[offset].add(row_offset));
                let hi_ptrs = std::array::from_fn(|offset| hi_bases[offset].add(row_offset));
                dot_product_pair_4x2p_acc(lo_ptrs, hi_ptrs, target_ptrs, raw);
            }
        }

        for (tile, raw) in raws.iter().enumerate().take(n_tiles) {
            let rotation = tile * PAIR_TILE_ROTATIONS;
            let live_rotations = PAIR_TILE_ROTATIONS.min(ROTATIONS - rotation);
            for rotation_offset in 0..live_rotations {
                for pair in 0..2 {
                    scatter_pair_block::<ROTATIONS>(
                        &raw[rotation_offset * 6 + pair * 3..rotation_offset * 6 + pair * 3 + 3],
                        base_target_idx,
                        pair,
                        rotation + rotation_offset,
                        result_lane,
                        additive_shares,
                    );
                }
            }
        }
    }

    /// Packed-pair scan of a whole component: all groups of four targets.
    /// Callers guarantee `targets.len()` is a multiple of four with every
    /// target present.
    #[target_feature(enable = "i8mm")]
    unsafe fn accumulate_component_pair_packed<const ROTATIONS: usize>(
        query_lo: &[u8],
        query_hi: &[u8],
        targets: &[&[u8]],
        rows: usize,
        result_lane: usize,
        additive_shares: &mut [Vec<RingElement<u16>>; 2],
    ) {
        debug_assert_eq!(targets.len() % GROUP_TARGETS, 0);
        let mut target_idx = 0;
        while target_idx + GROUP_TARGETS <= targets.len() {
            let group = std::array::from_fn(|offset| targets[target_idx + offset]);
            scan_four_targets_pair_packed::<ROTATIONS>(
                query_lo,
                query_hi,
                rows,
                group,
                target_idx,
                result_lane,
                additive_shares,
            );
            target_idx += GROUP_TARGETS;
        }
    }

    #[inline(always)]
    unsafe fn query_window<const ROTATIONS: usize>(
        query: &[u8],
        row: usize,
        rotation: usize,
    ) -> *const u8 {
        // Referencing the const forces its compile-time evaluation for this
        // ROTATIONS instantiation.
        const { PrerotatedQueryRowMajorView::<ROTATIONS>::ROTATION_AMOUNTS_ARE_MULTIPLES_OF_FOUR }
        let amount = PrerotatedQueryRowMajorView::<ROTATIONS>::ROTATION_AMOUNTS[rotation];
        let phase = (amount / 4) & 1;
        let aligned_amount = amount - phase * 4;
        query
            .as_ptr()
            .add((row * QUERY_PHASES + phase) * DOUBLED_MIXED_ROW_BYTES + aligned_amount * 2)
    }

    /// All rotation tiles of one query row against four loaded target rows.
    /// Factored out so the single-query and paired-query scans accumulate in
    /// exactly the same instruction order (bit-identical results).
    #[target_feature(enable = "i8mm")]
    unsafe fn scan_row_rotations<const ROTATIONS: usize>(
        query: &[u8],
        row: usize,
        target_ptrs: [*const u8; GROUP_TARGETS],
        base_target_idx: usize,
        result_lane: usize,
        additive_shares: &mut [RingElement<u16>],
    ) {
        let mut rotation = 0;
        while rotation + TILE_ROTATIONS <= ROTATIONS {
            let query_ptrs = std::array::from_fn(|offset| {
                query_window::<ROTATIONS>(query, row, rotation + offset)
            });
            let partials = dot_product_6x4_ummla(query_ptrs, target_ptrs);
            for (rotation_offset, values) in partials.into_iter().enumerate() {
                for (target_offset, partial) in values.into_iter().enumerate() {
                    let result_idx = (base_target_idx + target_offset) * ROTATIONS * 2
                        + (rotation + rotation_offset) * 2
                        + result_lane;
                    additive_shares[result_idx].0 =
                        additive_shares[result_idx].0.wrapping_add(partial);
                }
            }
            rotation += TILE_ROTATIONS;
        }
        while rotation < ROTATIONS {
            let partials =
                dot_product_1x4_ummla(query_window::<ROTATIONS>(query, row, rotation), target_ptrs);
            for (target_offset, partial) in partials.into_iter().enumerate() {
                let result_idx =
                    (base_target_idx + target_offset) * ROTATIONS * 2 + rotation * 2 + result_lane;
                additive_shares[result_idx].0 = additive_shares[result_idx].0.wrapping_add(partial);
            }
            rotation += 1;
        }
    }

    #[target_feature(enable = "i8mm")]
    unsafe fn scan_four_targets<const ROTATIONS: usize>(
        query: &[u8],
        rows: usize,
        targets: [&[u8]; GROUP_TARGETS],
        base_target_idx: usize,
        result_lane: usize,
        additive_shares: &mut [RingElement<u16>],
    ) {
        for row in 0..rows {
            let target_row_start = row * MIXED_ROW_BYTES;
            let target_ptrs =
                std::array::from_fn(|target| targets[target][target_row_start..].as_ptr());
            scan_row_rotations::<ROTATIONS>(
                query,
                row,
                target_ptrs,
                base_target_idx,
                result_lane,
                additive_shares,
            );
        }
    }

    /// Two queries against the same four targets in one target traversal. The
    /// four target rows (6.4 KB) stay L1-resident across both queries' tiles,
    /// so the second query's rotations cost no additional target streaming.
    #[target_feature(enable = "i8mm")]
    unsafe fn scan_four_targets_pair<const ROTATIONS: usize>(
        queries: [&[u8]; 2],
        rows: usize,
        targets: [&[u8]; GROUP_TARGETS],
        base_target_idx: usize,
        result_lane: usize,
        additive_shares: &mut [Vec<RingElement<u16>>; 2],
    ) {
        for row in 0..rows {
            let target_row_start = row * MIXED_ROW_BYTES;
            let target_ptrs =
                std::array::from_fn(|target| targets[target][target_row_start..].as_ptr());
            for (query, shares) in queries.iter().zip(additive_shares.iter_mut()) {
                scan_row_rotations::<ROTATIONS>(
                    query,
                    row,
                    target_ptrs,
                    base_target_idx,
                    result_lane,
                    shares,
                );
            }
        }
    }

    #[target_feature(enable = "i8mm")]
    unsafe fn scan_one_target<const ROTATIONS: usize>(
        query: &[u8],
        rows: usize,
        target: &[u8],
        target_idx: usize,
        result_lane: usize,
        additive_shares: &mut [RingElement<u16>],
    ) {
        for row in 0..rows {
            let target_ptr = target[row * MIXED_ROW_BYTES..].as_ptr();
            let target_ptrs = [target_ptr; GROUP_TARGETS];
            for rotation in 0..ROTATIONS {
                let partial = dot_product_1x4_ummla(
                    query_window::<ROTATIONS>(query, row, rotation),
                    target_ptrs,
                )[0];
                let result_idx = target_idx * ROTATIONS * 2 + rotation * 2 + result_lane;
                additive_shares[result_idx].0 = additive_shares[result_idx].0.wrapping_add(partial);
            }
        }
    }

    #[target_feature(enable = "i8mm")]
    unsafe fn accumulate_component_mixed<const ROTATIONS: usize>(
        query: &[u8],
        targets: &[Option<&[u8]>],
        rows: usize,
        result_lane: usize,
        additive_shares: &mut [RingElement<u16>],
    ) {
        let mut target_idx = 0;
        while target_idx < targets.len() {
            if target_idx + GROUP_TARGETS <= targets.len()
                && targets[target_idx..target_idx + GROUP_TARGETS]
                    .iter()
                    .all(Option::is_some)
            {
                let group = std::array::from_fn(|offset| {
                    targets[target_idx + offset].expect("checked present target")
                });
                scan_four_targets::<ROTATIONS>(
                    query,
                    rows,
                    group,
                    target_idx,
                    result_lane,
                    additive_shares,
                );
                target_idx += GROUP_TARGETS;
            } else {
                if let Some(target) = targets[target_idx] {
                    scan_one_target::<ROTATIONS>(
                        query,
                        rows,
                        target,
                        target_idx,
                        result_lane,
                        additive_shares,
                    );
                }
                target_idx += 1;
            }
        }
    }

    /// Paired-query counterpart of [`accumulate_component_mixed`]: every
    /// target group is streamed once and evaluated by both queries.
    #[target_feature(enable = "i8mm")]
    unsafe fn accumulate_component_mixed_pair<const ROTATIONS: usize>(
        queries: [&[u8]; 2],
        targets: &[Option<&[u8]>],
        rows: usize,
        result_lane: usize,
        additive_shares: &mut [Vec<RingElement<u16>>; 2],
    ) {
        let mut target_idx = 0;
        while target_idx < targets.len() {
            if target_idx + GROUP_TARGETS <= targets.len()
                && targets[target_idx..target_idx + GROUP_TARGETS]
                    .iter()
                    .all(Option::is_some)
            {
                let group = std::array::from_fn(|offset| {
                    targets[target_idx + offset].expect("checked present target")
                });
                scan_four_targets_pair::<ROTATIONS>(
                    queries,
                    rows,
                    group,
                    target_idx,
                    result_lane,
                    additive_shares,
                );
                target_idx += GROUP_TARGETS;
            } else {
                if let Some(target) = targets[target_idx] {
                    for (query, shares) in queries.iter().zip(additive_shares.iter_mut()) {
                        scan_one_target::<ROTATIONS>(
                            query,
                            rows,
                            target,
                            target_idx,
                            result_lane,
                            shares,
                        );
                    }
                }
                target_idx += 1;
            }
        }
    }

    /// Mixed-plane counterpart of
    /// [`super::rotation_aware_pairwise_distance_rowmajor`]: identical inputs,
    /// outputs, and sentinel semantics, operating on plane residents.
    ///
    /// # Panics
    /// The caller must only invoke this when the `i8mm` CPU feature is
    /// present (pools only adopt the mixed layout in that case).
    pub fn rotation_aware_pairwise_distance_mixed<const ROTATIONS: usize>(
        query: &ArcIris,
        targets: &[Option<&MixedPlaneIris>],
    ) -> Vec<RingElement<u16>> {
        assert!(
            std::arch::is_aarch64_feature_detected!("i8mm"),
            "mixed-plane scan kernel requires the i8mm CPU feature"
        );
        let mut additive_shares = vec![RingElement(0u16); 2 * ROTATIONS * targets.len()];

        DOUBLED_MIXED.with(|cell| {
            let mut entries = cell.borrow_mut();
            let hit = entries.iter().position(|entry| {
                entry
                    .as_ref()
                    .is_some_and(|doubled| doubled.matches::<ROTATIONS>(query))
            });
            let index = hit.unwrap_or_else(|| {
                entries
                    .iter()
                    .position(Option::is_none)
                    .unwrap_or_else(|| MIXED_LRU.with(|lru| *lru.borrow()))
            });
            MIXED_LRU.with(|lru| *lru.borrow_mut() = 1 - index);
            let doubled = entries[index].get_or_insert_with(DoubledQueryMixed::new_buffer);
            doubled.fill_if_changed::<ROTATIONS>(query);

            let code_targets: Vec<Option<&[u8]>> = targets
                .iter()
                .map(|target| target.map(MixedPlaneIris::code_planes))
                .collect();
            let mask_targets: Vec<Option<&[u8]>> = targets
                .iter()
                .map(|target| target.map(MixedPlaneIris::mask_planes))
                .collect();
            // SAFETY: i8mm presence asserted above.
            unsafe {
                accumulate_component_mixed::<ROTATIONS>(
                    &doubled.code,
                    &code_targets,
                    CODE_ROWS,
                    0,
                    &mut additive_shares,
                );
                accumulate_component_mixed::<ROTATIONS>(
                    &doubled.mask,
                    &mask_targets,
                    MASK_ROWS,
                    1,
                    &mut additive_shares,
                );
            }
        });

        apply_scan_epilogue::<ROTATIONS>(targets, &mut additive_shares);
        additive_shares
    }

    /// Same epilogue as the u16 kernel: double the mask lanes of present
    /// targets, fill sentinel distances for missing ones.
    fn apply_scan_epilogue<const ROTATIONS: usize>(
        targets: &[Option<&MixedPlaneIris>],
        additive_shares: &mut [RingElement<u16>],
    ) {
        for (target_idx, target) in targets.iter().enumerate() {
            let base_idx = target_idx * ROTATIONS * 2;
            if target.is_some() {
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

    /// Fused two-query scan: identical outputs to two independent
    /// [`rotation_aware_pairwise_distance_mixed`] calls, but each target row
    /// is streamed once and feeds both queries' rotation tiles. Used by the
    /// exact scan to evaluate the normal and mirror orientations in one pass
    /// over the resident database.
    ///
    /// Full four-target groups with every target present take the packed-pair
    /// kernel (three UMMLA per four query/target/rotation results); calls
    /// containing a missing target or a tail group fall back to the unpacked
    /// pair scan. Both produce bit-identical results.
    ///
    /// # Panics
    /// The caller must only invoke this when the `i8mm` CPU feature is
    /// present (pools only adopt the mixed layout in that case).
    pub fn rotation_aware_pairwise_distance_mixed_pair<const ROTATIONS: usize>(
        queries: [&ArcIris; 2],
        targets: &[Option<&MixedPlaneIris>],
    ) -> [Vec<RingElement<u16>>; 2] {
        assert!(
            std::arch::is_aarch64_feature_detected!("i8mm"),
            "mixed-plane scan kernel requires the i8mm CPU feature"
        );
        let packed_eligible = targets.len().is_multiple_of(GROUP_TARGETS)
            && !targets.is_empty()
            && targets.iter().all(Option::is_some);
        if packed_eligible {
            return rotation_aware_pairwise_distance_mixed_pair_packed::<ROTATIONS>(
                queries, targets,
            );
        }
        rotation_aware_pairwise_distance_mixed_pair_unpacked::<ROTATIONS>(queries, targets)
    }

    fn rotation_aware_pairwise_distance_mixed_pair_packed<const ROTATIONS: usize>(
        queries: [&ArcIris; 2],
        targets: &[Option<&MixedPlaneIris>],
    ) -> [Vec<RingElement<u16>>; 2] {
        let mut additive_shares = [
            vec![RingElement(0u16); 2 * ROTATIONS * targets.len()],
            vec![RingElement(0u16); 2 * ROTATIONS * targets.len()],
        ];

        PAIR_PACKED.with(|cell| {
            let mut entry = cell.borrow_mut();
            let packed = entry.get_or_insert_with(PairPackedQueryMixed::new_buffer);
            packed.fill_if_changed::<ROTATIONS>(queries);

            let code_targets: Vec<&[u8]> = targets
                .iter()
                .map(|target| target.expect("packed pair scan requires present targets"))
                .map(MixedPlaneIris::code_planes)
                .collect();
            let mask_targets: Vec<&[u8]> = targets
                .iter()
                .map(|target| target.expect("packed pair scan requires present targets"))
                .map(MixedPlaneIris::mask_planes)
                .collect();
            // SAFETY: i8mm presence asserted by the public entry point.
            unsafe {
                accumulate_component_pair_packed::<ROTATIONS>(
                    &packed.code_lo,
                    &packed.code_hi,
                    &code_targets,
                    CODE_ROWS,
                    0,
                    &mut additive_shares,
                );
                accumulate_component_pair_packed::<ROTATIONS>(
                    &packed.mask_lo,
                    &packed.mask_hi,
                    &mask_targets,
                    MASK_ROWS,
                    1,
                    &mut additive_shares,
                );
            }
        });

        // No epilogue: every target is present by construction and the mask
        // doubling is applied inside the single per-result scatter.
        additive_shares
    }

    fn rotation_aware_pairwise_distance_mixed_pair_unpacked<const ROTATIONS: usize>(
        queries: [&ArcIris; 2],
        targets: &[Option<&MixedPlaneIris>],
    ) -> [Vec<RingElement<u16>>; 2] {
        let mut additive_shares = [
            vec![RingElement(0u16); 2 * ROTATIONS * targets.len()],
            vec![RingElement(0u16); 2 * ROTATIONS * targets.len()],
        ];

        DOUBLED_MIXED.with(|cell| {
            let mut entries = cell.borrow_mut();
            // Materialize both queries in the two cache slots. If a query is
            // already cached, keep its slot; otherwise fill the slot that the
            // other query does not occupy.
            let slot_of = |entries: &[Option<DoubledQueryMixed>; 2], query: &ArcIris| {
                entries.iter().position(|entry| {
                    entry
                        .as_ref()
                        .is_some_and(|doubled| doubled.matches::<ROTATIONS>(query))
                })
            };
            let index_a = slot_of(&entries, queries[0]).unwrap_or_else(|| {
                let index = match slot_of(&entries, queries[1]) {
                    Some(index_b) => 1 - index_b,
                    None => 0,
                };
                let doubled = entries[index].get_or_insert_with(DoubledQueryMixed::new_buffer);
                doubled.fill_if_changed::<ROTATIONS>(queries[0]);
                index
            });
            let index_b = slot_of(&entries, queries[1]).unwrap_or_else(|| {
                let index = 1 - index_a;
                let doubled = entries[index].get_or_insert_with(DoubledQueryMixed::new_buffer);
                doubled.fill_if_changed::<ROTATIONS>(queries[1]);
                index
            });

            let doubled_a = entries[index_a].as_ref().expect("slot filled above");
            let doubled_b = entries[index_b].as_ref().expect("slot filled above");

            let code_targets: Vec<Option<&[u8]>> = targets
                .iter()
                .map(|target| target.map(MixedPlaneIris::code_planes))
                .collect();
            let mask_targets: Vec<Option<&[u8]>> = targets
                .iter()
                .map(|target| target.map(MixedPlaneIris::mask_planes))
                .collect();
            // SAFETY: i8mm presence asserted above.
            unsafe {
                accumulate_component_mixed_pair::<ROTATIONS>(
                    [&doubled_a.code, &doubled_b.code],
                    &code_targets,
                    CODE_ROWS,
                    0,
                    &mut additive_shares,
                );
                accumulate_component_mixed_pair::<ROTATIONS>(
                    [&doubled_a.mask, &doubled_b.mask],
                    &mask_targets,
                    MASK_ROWS,
                    1,
                    &mut additive_shares,
                );
            }
        });

        for shares in &mut additive_shares {
            apply_scan_epilogue::<ROTATIONS>(targets, shares);
        }
        additive_shares
    }
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
    fn dot_product_nx4_u16_matches_scalar_wrapping_reference() {
        const ROW_SIZE: usize = PrerotatedQueryRowMajor::ROW_SIZE;

        fn check_width<const N: usize>(seed: u64) {
            let mut rng = AesRng::seed_from_u64(seed);
            let queries: [Vec<u16>; N] =
                std::array::from_fn(|_| (0..ROW_SIZE).map(|_| rng.next_u32() as u16).collect());
            let targets: [Vec<u16>; 4] =
                std::array::from_fn(|_| (0..ROW_SIZE).map(|_| rng.next_u32() as u16).collect());
            let query_refs: [&[u16]; N] = std::array::from_fn(|idx| queries[idx].as_slice());
            let target_refs: [&[u16]; 4] = std::array::from_fn(|idx| targets[idx].as_slice());

            let tiled = dot_product_nx4_u16(query_refs, target_refs);
            let reference: [[u16; 4]; N] = std::array::from_fn(|query_idx| {
                std::array::from_fn(|target_idx| {
                    simple_dot_product(&queries[query_idx], &targets[target_idx])
                })
            });

            assert_eq!(tiled, reference, "{N}x4 tile mismatch for seed {seed}");
        }

        for seed in [0, 1, 42, u64::MAX] {
            check_width::<6>(seed);
            check_width::<4>(seed);
            check_width::<1>(seed);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn mixed_ummla_scan_matches_u16_scan_with_tail_and_missing_targets() {
        use crate::protocol::shared_iris::{GaloisRingSharedIris, MixedPlaneIris};

        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let mut rng = AesRng::seed_from_u64(0x1_8_8_4);
        let iris_db = IrisDB::new_random_rng(10, &mut rng).db;
        let query_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());
        let target_shares: Vec<_> = iris_db[1..]
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        for party in 0..3 {
            let mut query = query_shares[party].clone();
            query.code.preprocess_iris_code_query_share();
            query.mask.preprocess_mask_code_query_share();
            let query = Arc::new(query);
            let targets: Vec<ArcIris> = target_shares
                .iter()
                .map(|shares| Arc::new(shares[party].clone()))
                .collect();
            let mixed: Vec<MixedPlaneIris> = targets
                .iter()
                .map(|target| MixedPlaneIris::from_iris(target))
                .collect();
            let present = [true, false, true, true, true, true, false, true, true];

            let u16_targets = targets
                .iter()
                .zip(present)
                .map(|(target, present)| present.then_some(target));
            let mixed_targets: Vec<Option<&MixedPlaneIris>> = mixed
                .iter()
                .zip(present)
                .map(|(target, present)| present.then_some(target))
                .collect();

            let expected = rotation_aware_pairwise_distance_rowmajor::<31, _>(&query, u16_targets);
            let actual = rotation_aware_pairwise_distance_mixed::<31>(&query, &mixed_targets);
            assert_eq!(actual, expected, "party {party}");
        }
    }

    /// The fused two-query pass must be byte-identical to two independent
    /// single-query passes, across group/tail/missing-target shapes. The
    /// query pair mirrors production: one normal-preprocessed query and one
    /// mirrored-preprocessed query of a different iris.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn mixed_ummla_pair_scan_matches_two_single_scans() {
        use crate::protocol::shared_iris::{GaloisRingSharedIris, MixedPlaneIris};

        if !std::arch::is_aarch64_feature_detected!("i8mm") {
            return;
        }

        let mut rng = AesRng::seed_from_u64(0x1_8_8_5);
        let iris_db = IrisDB::new_random_rng(12, &mut rng).db;
        let query_shares =
            GaloisRingSharedIris::generate_shares_locally(&mut rng, iris_db[0].clone());
        let mirrored_query_shares =
            GaloisRingSharedIris::generate_mirrored_shares_locally(&mut rng, iris_db[1].clone());
        let target_shares: Vec<_> = iris_db[2..]
            .iter()
            .map(|iris| GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.clone()))
            .collect();

        for party in 0..3 {
            let mut query_a = query_shares[party].clone();
            query_a.code.preprocess_iris_code_query_share();
            query_a.mask.preprocess_mask_code_query_share();
            let query_a = Arc::new(query_a);
            let mut query_b = mirrored_query_shares[party].clone();
            query_b.code.preprocess_iris_code_query_share();
            query_b.mask.preprocess_mask_code_query_share();
            let query_b = Arc::new(query_b);

            let targets: Vec<ArcIris> = target_shares
                .iter()
                .map(|shares| Arc::new(shares[party].clone()))
                .collect();
            let mixed: Vec<MixedPlaneIris> = targets
                .iter()
                .map(|target| MixedPlaneIris::from_iris(target))
                .collect();

            // Cover: the packed fast path (all present, multiple of four —
            // one and two groups), a missing target inside a group, a tail
            // shorter than a group, and an all-missing prefix.
            let present_patterns: [&[bool]; 6] = [
                &[true; 4],
                &[true; 8],
                &[true; 10],
                &[true, false, true, true, true, true, false, true, true, true],
                &[true, true, true],
                &[false, false, true, true, true, true, true],
            ];
            for present in present_patterns {
                let mixed_targets: Vec<Option<&MixedPlaneIris>> = mixed
                    .iter()
                    .zip(present)
                    .map(|(target, &present)| present.then_some(target))
                    .collect();

                let expected_a =
                    rotation_aware_pairwise_distance_mixed::<31>(&query_a, &mixed_targets);
                let expected_b =
                    rotation_aware_pairwise_distance_mixed::<31>(&query_b, &mixed_targets);
                let [actual_a, actual_b] = rotation_aware_pairwise_distance_mixed_pair::<31>(
                    [&query_a, &query_b],
                    &mixed_targets,
                );
                assert_eq!(actual_a, expected_a, "party {party} query A");
                assert_eq!(actual_b, expected_b, "party {party} query B");

                // Same-query pairing (both slots resolve to one cache entry).
                let [same_a, same_b] = rotation_aware_pairwise_distance_mixed_pair::<31>(
                    [&query_a, &query_a],
                    &mixed_targets,
                );
                assert_eq!(same_a, expected_a, "party {party} same-query A");
                assert_eq!(same_b, expected_a, "party {party} same-query B");
            }
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
