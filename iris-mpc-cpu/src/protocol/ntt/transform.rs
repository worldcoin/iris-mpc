//! The measured CPU spectral kernel: mixed-radix NTT, fused packed products,
//! and a paired selected inverse. Output shares include degree-two Lagrange weights.
use super::FieldIris;
use std::sync::LazyLock;
const COLS: usize = 200;
const CODE_LEN: usize = 12_800;
const MASK_LEN: usize = 6_400;
const ROTATIONS: usize = 31;
const ROW_SIZE: usize = 800;
const PAIRS: usize = 99;
const P: u64 = super::MODULUS as u64;
const ROOT: u64 = 43_061;
static NTT: LazyLock<Ntt> = LazyLock::new(Ntt::new);
fn pow_mod(mut base: u64, mut exponent: u64) -> u64 {
    let mut out = 1;
    while exponent != 0 {
        if exponent & 1 == 1 {
            out = out * base % P;
        }
        base = base * base % P;
        exponent >>= 1;
    }
    out
}

struct NttNode {
    n: usize,
    radix: usize,
    weights: Vec<u16>,
    child: Option<Box<NttNode>>,
}

impl NttNode {
    fn new(n: usize, root: u64) -> Self {
        if n == 1 {
            return Self {
                n,
                radix: 1,
                weights: vec![],
                child: None,
            };
        }
        let radix = if n.is_multiple_of(2) { 2 } else { 5 };
        Self {
            n,
            radix,
            weights: (0..n)
                .flat_map(|k| (0..radix).map(move |j| pow_mod(root, (k * j) as u64) as u16))
                .collect(),
            child: Some(Box::new(Self::new(n / radix, pow_mod(root, radix as u64)))),
        }
    }

    fn transform(&self, source: &[u16], stride: usize, out: &mut [u16]) {
        if self.n == 1 {
            out[0] = source[0] % P as u16;
            return;
        }
        let width = self.n / self.radix;
        for j in 0..self.radix {
            self.child.as_ref().unwrap().transform(
                &source[j * stride..],
                stride * self.radix,
                &mut out[j * width..(j + 1) * width],
            );
        }
        let mut temp = [0u16; COLS];
        temp[..self.n].copy_from_slice(&out[..self.n]);
        for k in 0..self.n {
            let mut acc = 0u64;
            for j in 0..self.radix {
                acc += temp[j * width + k % width] as u64 * self.weights[k * self.radix + j] as u64;
            }
            out[k] = (acc % P) as u16;
        }
    }
}

struct Ntt {
    forward: NttNode,
    padded_sum_weights: Vec<u16>,
    padded_diff_weights: Vec<u16>,
}

impl Ntt {
    fn new() -> Self {
        Self {
            forward: NttNode::new(COLS, ROOT),
            padded_sum_weights: Self::pair_weights(false, 104),
            padded_diff_weights: Self::pair_weights(true, 104),
        }
    }

    fn pair_weights(difference: bool, stride: usize) -> Vec<u16> {
        let half = P.div_ceil(2);
        (1..=ROTATIONS / 2)
            .flat_map(|r| {
                (1..=stride).map(move |k| {
                    if k > PAIRS {
                        return 0;
                    }
                    let a = pow_mod(ROOT, ((COLS - r) * k % COLS) as u64);
                    let b = pow_mod(ROOT, (r * k % COLS) as u64);
                    let value = if difference { a + P - b } else { a + b };
                    centered((value * half % P) as u16)
                })
            })
            .collect()
    }

    fn prepare_query(&self, record: &FieldIris, party: usize) -> Vec<u16> {
        let mut result = self.prepare(record);
        // Linearity moves inverse normalization to the one-time query transform.
        let inv_n = pow_mod(COLS as u64, P - 2) as i64 * [3, -3, 1][party];
        for v in &mut result {
            *v = centered(((*v as i16 as i64 * inv_n).rem_euclid(P as i64)) as u16);
        }
        result
    }

    fn inverse_paired(&self, spectrum: &[u16; COLS]) -> [u16; ROTATIONS] {
        let mut sums = [0u16; 104];
        let mut diffs = [0u16; 104];
        let (stride, sum_weights, diff_weights) =
            (104, &self.padded_sum_weights, &self.padded_diff_weights);
        // Both paired inputs must be centered for the larger prime before
        // narrowing to i16. The wide dot below flushes every 48 terms.
        for k in 1..=PAIRS {
            let a = spectrum[k] as i16 as i32;
            let b = spectrum[COLS - k] as i16 as i32;
            sums[k - 1] = center_sum(a + b);
            diffs[k - 1] = center_sum(a - b);
        }
        let dc = spectrum[0] as i16 as i32;
        let nyquist = spectrum[COLS / 2] as i16 as i32;
        let mut out = [0u16; ROTATIONS];
        out[ROTATIONS / 2] = (dc + nyquist + sums.iter().map(|v| *v as i16 as i32).sum::<i32>())
            .rem_euclid(P as i32) as u16;
        for r in 1..=ROTATIONS / 2 {
            let start = (r - 1) * stride;

            // Combine exact wide sums before reducing; no per-dot modular
            // reductions are needed. Each final output has one reduction.
            let even = field_dot_unreduced(&sums[..stride], &sum_weights[start..start + stride]);
            let odd = field_dot_unreduced(&diffs[..stride], &diff_weights[start..start + stride]);
            let endpoint = (dc + if r % 2 == 0 { nyquist } else { -nyquist }) as i64;
            out[ROTATIONS / 2 + r] = (endpoint + even + odd).rem_euclid(P as i64) as u16;
            out[ROTATIONS / 2 - r] = (endpoint + even - odd).rem_euclid(P as i64) as u16;
        }
        out
    }

    fn prepare(&self, record: &FieldIris) -> Vec<u16> {
        let mut out = vec![0u16; CODE_LEN + MASK_LEN];
        let mut source = [0u16; COLS];
        let mut result = [0u16; COLS];
        for (data, offset) in [(&record.code, 0), (&record.mask, CODE_LEN)] {
            let channels = data.len() / COLS;
            for (block, row) in data.chunks_exact(ROW_SIZE).enumerate() {
                for lane in 0..4 {
                    for t in 0..COLS {
                        source[t] = row[t * 4 + lane] % P as u16;
                    }
                    self.forward.transform(&source, 1, &mut result);
                    for k in 0..COLS {
                        out[offset + k * channels + block * 4 + lane] = centered(result[k]);
                    }
                }
            }
        }
        out
    }
}

#[inline]
fn center_sum(value: i32) -> u16 {
    let value = if value > P as i32 / 2 {
        value - P as i32
    } else {
        value
    };
    let value = if value < -(P as i32 / 2) {
        value + P as i32
    } else {
        value
    };
    value as i16 as u16
}

#[inline]
fn centered(value: u16) -> u16 {
    if value > P as u16 / 2 {
        value.wrapping_sub(P as u16)
    } else {
        value
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn field_dot_unreduced(a: &[u16], b: &[u16]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x as i16 as i64 * y as i16 as i64)
        .sum::<i64>()
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn field_dot_unreduced(a: &[u16], b: &[u16]) -> i64 {
    use std::arch::aarch64::*;
    assert_eq!(a.len(), b.len());
    debug_assert!(a.len() <= 200);
    // Each i32 lane accumulates at most three products before widening:
    // 3 * 26100^2 = 2,043,630,000 < i32::MAX.
    // SAFETY: NEON is guaranteed on AArch64; loads remain inside each chunk.
    unsafe {
        let mut total = 0i64;
        for (a, b) in a.chunks(48).zip(b.chunks(48)) {
            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut acc2 = vdupq_n_s32(0);
            let mut acc3 = vdupq_n_s32(0);
            let mut i = 0;
            while i + 16 <= a.len() {
                let x0 = vld1q_s16(a.as_ptr().add(i).cast());
                let y0 = vld1q_s16(b.as_ptr().add(i).cast());
                let x1 = vld1q_s16(a.as_ptr().add(i + 8).cast());
                let y1 = vld1q_s16(b.as_ptr().add(i + 8).cast());
                acc0 = vmlal_s16(acc0, vget_low_s16(x0), vget_low_s16(y0));
                acc1 = vmlal_high_s16(acc1, x0, y0);
                acc2 = vmlal_s16(acc2, vget_low_s16(x1), vget_low_s16(y1));
                acc3 = vmlal_high_s16(acc3, x1, y1);
                i += 16;
            }
            if i + 8 <= a.len() {
                let x = vld1q_s16(a.as_ptr().add(i).cast());
                let y = vld1q_s16(b.as_ptr().add(i).cast());
                acc0 = vmlal_s16(acc0, vget_low_s16(x), vget_low_s16(y));
                acc1 = vmlal_high_s16(acc1, x, y);
                i += 8;
            }
            let mut sum =
                vaddlvq_s32(acc0) + vaddlvq_s32(acc1) + vaddlvq_s32(acc2) + vaddlvq_s32(acc3);
            while i < a.len() {
                sum += a[i] as i16 as i64 * b[i] as i16 as i64;
                i += 1;
            }
            total += sum;
        }
        total
    }
}

/// Resident preprocessed database share: two signed byte planes, 38,400 bytes.
#[derive(Clone, Debug)]
pub struct SpectralIris {
    packed: Box<[i8]>,
}
impl SpectralIris {
    pub fn prepare(iris: &FieldIris) -> Self {
        assert_eq!(iris.code.len(), CODE_LEN);
        assert_eq!(iris.mask.len(), MASK_LEN);
        Self {
            packed: pack(&NTT.prepare(iris)).into_boxed_slice(),
        }
    }
    /// Payload for versioned spectral persistence.
    pub fn packed_bytes(&self) -> &[i8] {
        &self.packed
    }
    pub fn from_packed(packed: Box<[i8]>) -> eyre::Result<Self> {
        eyre::ensure!(
            packed.len() == 2 * (CODE_LEN + MASK_LEN),
            "invalid spectral record length"
        );
        eyre::ensure!(
            packed.chunks_exact(16).all(|b| (0..8).all(|i| {
                (-26_100..=26_100).contains(&(i32::from(b[i]) + 256 * i32::from(b[i + 8])))
            })),
            "noncanonical spectral record"
        );
        Ok(Self { packed })
    }
}
const TILE: usize = 8;

#[inline]
fn split(value: i16) -> (i8, i8) {
    let low = value as i8;
    let high = ((value as i32 - low as i32) / 256) as i8;
    (low, high)
}

fn pack(values: &[u16]) -> Vec<i8> {
    assert!(values.len().is_multiple_of(8));
    let mut out = vec![0i8; values.len() * 2];
    for (source, dest) in values.chunks_exact(8).zip(out.chunks_exact_mut(16)) {
        for k in 0..8 {
            let (low, high) = split(source[k] as i16);
            dest[k] = low;
            dest[8 + k] = high;
        }
    }
    out
}

#[derive(Debug)]
pub struct SpectralQuery {
    low: Vec<i8>,
    high: Vec<i8>,
    orientations: usize,
}

impl SpectralQuery {
    /// Merge two already-prepared query orientations without repeating either NTT.
    pub fn pair(first: &Self, second: &Self) -> Self {
        let merge = |a: &[i8], b: &[i8]| {
            a.chunks_exact(16)
                .zip(b.chunks_exact(16))
                .flat_map(|(a, b)| a[..8].iter().chain(&b[..8]).copied())
                .collect()
        };
        Self {
            low: merge(&first.low, &second.low),
            high: merge(&first.high, &second.high),
            orientations: 2,
        }
    }

    pub fn prepare(queries: &[&FieldIris], party: usize) -> Self {
        assert!(party < 3);
        let ntt = &*NTT;
        assert!((1..=2).contains(&queries.len()));
        let values: Vec<_> = queries
            .iter()
            .map(|q| ntt.prepare_query(q, party))
            .collect();
        let mut out = Self {
            low: vec![0; 2 * (CODE_LEN + MASK_LEN)],
            high: vec![0; 2 * (CODE_LEN + MASK_LEN)],
            orientations: queries.len(),
        };
        for (offset, channels) in [(0, 64), (CODE_LEN, 32)] {
            for k in 0..COLS {
                let source = offset + ((COLS - k) % COLS) * channels;
                let dest = 2 * (offset + k * channels);
                for block in 0..channels / 8 {
                    for q in 0..2 {
                        for j in 0..8 {
                            let mut value = values[q.min(values.len() - 1)][source + block * 8 + j];
                            // Fold g=2*C-m into query weights before aggregation.
                            value = center_sum(
                                value as i16 as i32 * if offset == CODE_LEN { -1 } else { 2 },
                            );
                            let (low, high) = split(value as i16);
                            out.low[dest + block * 16 + q * 8 + j] = low;
                            out.high[dest + block * 16 + q * 8 + j] = high;
                        }
                    }
                }
            }
        }
        out
    }
}

pub fn score_chunk(query: &SpectralQuery, targets: &[&SpectralIris]) -> Vec<u16> {
    let ntt = &*NTT;
    let mut out = vec![0u16; targets.len() * query.orientations * ROTATIONS];
    // Frequency products for an eight-record/two-orientation tile fit in 6,400 bytes.
    let mut spectra = [[[0u16; COLS]; 2]; TILE];
    for (tile_index, tile) in targets.chunks(TILE).enumerate() {
        for (offset, channels) in [(0, 64), (CODE_LEN, 32)] {
            let component_bytes = 2 * channels * COLS;
            let low = &query.low[2 * offset..2 * offset + component_bytes];
            let high = &query.high[2 * offset..2 * offset + component_bytes];
            for (k, (low, high)) in low
                .chunks_exact(2 * channels)
                .zip(high.chunks_exact(2 * channels))
                .enumerate()
            {
                let start = 2 * (offset + k * channels);
                let pointers = std::array::from_fn(|t: usize| {
                    tile[t.min(tile.len() - 1)].packed[start..].as_ptr()
                });
                // SAFETY: callers select this method only on i8mm AArch64;
                // each pointer covers 2*channels bytes, and queries have two
                // packed lanes. Tail targets repeat a valid last record.
                let scores = unsafe {
                    dispatched_dot_8x2(low.as_ptr(), high.as_ptr(), pointers, channels / 8)
                };
                for (record, scores) in spectra.iter_mut().zip(scores).take(tile.len()) {
                    for (spectrum, score) in record.iter_mut().zip(scores).take(query.orientations)
                    {
                        spectrum[k] = if offset == 0 {
                            centered(score)
                        } else {
                            center_sum(spectrum[k] as i16 as i32 + centered(score) as i16 as i32)
                        };
                    }
                }
            }
        }
        for (t, record) in spectra.iter().take(tile.len()).enumerate() {
            for (q, spectrum) in record.iter().take(query.orientations).enumerate() {
                let scores = ntt.inverse_paired(spectrum);
                for (r, score) in scores.into_iter().enumerate() {
                    let idx = ((tile_index * TILE + t) * query.orientations + q) * ROTATIONS + r;
                    out[idx] = score;
                }
            }
        }
    }
    out
}

// Signed byte planes represent v exactly as lo + 256*hi, including negative
// low bytes and the carry into hi. |lo|<=128 and |hi|<=102, so each i32 SMMLA
// partial over 64 terms is at most 64*128^2 = 1,048,576 in magnitude. All four
// byte products are required in this prime field. Recombine in i64 then mod p.
// The SMMLA intrinsic is unstable on the supported Rust toolchain. Keep only
// this instruction in assembly; Rust manages the loop, loads, and registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn smmla(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    std::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack, preserves_flags),
    );
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn dot_8x2(
    low: *const i8,
    high: *const i8,
    targets: [*const i8; TILE],
    blocks: usize,
) -> [[u16; 2]; TILE] {
    use std::arch::aarch64::{vdupq_n_s32, vld1q_s8, vst1q_s32};
    let mut acc = [[vdupq_n_s32(0); 2]; TILE];
    for block in 0..blocks {
        let query_low = vld1q_s8(low.add(block * 16));
        let query_high = vld1q_s8(high.add(block * 16));
        for (lanes, target) in acc.iter_mut().zip(targets) {
            let value = vld1q_s8(target.add(block * 16));
            lanes[0] = smmla(lanes[0], value, query_low);
            lanes[1] = smmla(lanes[1], value, query_high);
        }
    }
    std::array::from_fn(|t| {
        let mut lo = [0i32; 4];
        let mut hi = [0i32; 4];
        vst1q_s32(lo.as_mut_ptr(), acc[t][0]);
        vst1q_s32(hi.as_mut_ptr(), acc[t][1]);
        std::array::from_fn(|q| {
            let value =
                lo[q] as i64 + 256 * (lo[2 + q] as i64 + hi[q] as i64) + 65536 * hi[2 + q] as i64;
            value.rem_euclid(P as i64) as u16
        })
    })
}

// All pointers cover blocks*16 bytes. This portable path also serves as an
// independent implementation for testing architectures without SMMLA.
unsafe fn dispatched_dot_8x2(
    low: *const i8,
    high: *const i8,
    targets: [*const i8; TILE],
    blocks: usize,
) -> [[u16; 2]; TILE] {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("i8mm") {
        return dot_8x2(low, high, targets, blocks);
    }
    std::array::from_fn(|t| {
        std::array::from_fn(|q| {
            let mut sum = 0i64;
            for block in 0..blocks {
                for j in 0..8 {
                    let d = *targets[t].add(block * 16 + j) as i64
                        + 256 * *targets[t].add(block * 16 + 8 + j) as i64;
                    let x = *low.add(block * 16 + q * 8 + j) as i64
                        + 256 * *high.add(block * 16 + q * 8 + j) as i64;
                    sum += d * x;
                }
            }
            super::reduce(sum)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn wide_inverse_accumulation_and_boundary_spectra() {
        for len in [0, 8, 16, 47, 48, 49, 99, 104, 200] {
            for (a, b) in [(26100i16, 26100i16), (-26100, 26100), (-26100, -26100)] {
                assert_eq!(
                    field_dot_unreduced(&vec![a as u16; len], &vec![b as u16; len]),
                    len as i64 * i64::from(a) * i64::from(b)
                );
            }
        }
        let mut rng = ChaCha20Rng::seed_from_u64(52201200);
        for case in 0..32 {
            let spectrum = std::array::from_fn(|k| match case {
                0 => 26100u16,
                1 => (-26100i16) as u16,
                2 => {
                    if k % 2 == 0 {
                        26100
                    } else {
                        (-26100i16) as u16
                    }
                }
                _ => centered(rng.gen_range(0..P as u16)),
            });
            let actual = NTT.inverse_paired(&spectrum);
            for (index, rotation) in (-15i64..=15).enumerate() {
                let expected = spectrum
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        i64::from(v as i16)
                            * pow_mod(ROOT, (-rotation * k as i64).rem_euclid(200) as u64) as i64
                    })
                    .sum::<i64>()
                    .rem_euclid(P as i64) as u16;
                assert_eq!(actual[index], expected, "case {case}, rotation {rotation}");
            }
        }
    }
}
