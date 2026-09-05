use rayon::prelude::*;
use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::{hint::black_box, sync::Arc, time::Instant};

const COLS: usize = 200;
const BINS: usize = COLS / 2 + 1;
const ROW_SIZE: usize = 800;
const ROTATIONS: usize = 31;
const CODE_LEN: usize = 12_800;
const MASK_LEN: usize = 6_400;
const CHUNK: usize = 256;
const PAIRS: usize = COLS / 2 - 1;
const METHODS: [&str; 8] = [
    "legacy_neon",
    "pr2348_ummla_pair",
    "fft_f64",
    "ntt_direct",
    "ntt_paired",
    "ntt_paired_lazy",
    "ntt_paired_simd",
    "ntt_smmla",
];
const P: u64 = 25_601;
const ROOT: u64 = 9_217;

#[derive(Clone, Copy)]
pub struct RingElement<T>(T);
impl std::ops::Mul for RingElement<u16> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
pub struct PrerotatedQueryRowMajor;
impl PrerotatedQueryRowMajor {
    pub const ROW_SIZE: usize = ROW_SIZE;
    pub const CODE_ROWS: usize = 16;
    pub const MASK_ROWS: usize = 8;
}
include!(concat!(env!("OUT_DIR"), "/baseline.rs"));
include!(concat!(env!("OUT_DIR"), "/rotation_constants.rs"));
include!(concat!(env!("OUT_DIR"), "/mixed_scan.rs"));
mod packed_ntt;
mod protocol;
use protocol::shared_iris::{ArcIris, GaloisRingSharedIris, MixedPlaneIris};
#[cfg(target_arch = "aarch64")]
const SHARE_OF_MAX_DISTANCE: (u16, u16) = (u16::MAX, 1);

fn production(queries: &[ArcIris], targets: &[MixedPlaneIris]) -> Vec<Vec<u16>> {
    #[cfg(target_arch = "aarch64")]
    {
        let targets: Vec<_> = targets.iter().map(Some).collect();
        let result = if queries.len() == 2 {
            mixed_scan::rotation_aware_pairwise_distance_mixed_pair::<ROTATIONS>(
                [&queries[0], &queries[1]],
                &targets,
            )
            .into_iter()
            .collect::<Vec<_>>()
        } else {
            vec![mixed_scan::rotation_aware_pairwise_distance_mixed::<
                ROTATIONS,
            >(&queries[0], &targets)]
        };
        result
            .into_iter()
            .map(|v| v.into_iter().map(|v| v.0).collect())
            .collect()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (queries, targets);
        panic!("the PR #2348 mixed-plane baseline requires AArch64 with i8mm");
    }
}

#[derive(Clone)]
struct Record {
    code: Vec<u16>,
    mask: Vec<u16>,
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u16 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (z ^ (z >> 31)) as u16
    }
    fn record(&mut self) -> Record {
        Record {
            code: (0..CODE_LEN).map(|_| self.next()).collect(),
            mask: (0..MASK_LEN).map(|_| self.next()).collect(),
        }
    }
}

fn prerotate(input: &[u16]) -> Vec<u16> {
    let mut out = vec![0; input.len() * ROTATIONS];
    for (row, source) in input.chunks_exact(ROW_SIZE).enumerate() {
        for r in 0..ROTATIONS {
            let left = ((15 - r as isize) * 4).rem_euclid(800) as usize;
            let start = (row * ROTATIONS + r) * ROW_SIZE;
            let dest = &mut out[start..start + ROW_SIZE];
            dest[..ROW_SIZE - left].copy_from_slice(&source[left..]);
            dest[ROW_SIZE - left..].copy_from_slice(&source[..left]);
        }
    }
    out
}

fn prepare_baseline(query: &Record) -> Record {
    Record {
        code: prerotate(&query.code),
        mask: prerotate(&query.mask),
    }
}

fn baseline(prepared: &Record, targets: &[Record]) -> Vec<u16> {
    let mut out = vec![RingElement(0u16); targets.len() * ROTATIONS * 2];
    #[cfg(target_arch = "aarch64")]
    {
        let codes: Vec<_> = targets.iter().map(|r| Some(r.code.as_slice())).collect();
        let masks: Vec<_> = targets.iter().map(|r| Some(r.mask.as_slice())).collect();
        accumulate_component_tiled_6x4::<ROTATIONS>(&prepared.code, &codes, 16, 0, &mut out);
        accumulate_component_tiled_6x4::<ROTATIONS>(&prepared.mask, &masks, 8, 1, &mut out);
    }
    #[cfg(not(target_arch = "aarch64"))]
    for (lane, rows) in [(0, 16), (1, 8)] {
        let query = if lane == 0 {
            &prepared.code
        } else {
            &prepared.mask
        };
        for row in 0..rows {
            for (target_idx, target) in targets.iter().enumerate() {
                let target = if lane == 0 {
                    &target.code
                } else {
                    &target.mask
                };
                let t = &target[row * ROW_SIZE..(row + 1) * ROW_SIZE];
                for r in 0..ROTATIONS {
                    let start = (row * ROTATIONS + r) * ROW_SIZE;
                    let val = simple_dot_product(&query[start..start + ROW_SIZE], t);
                    let idx = target_idx * ROTATIONS * 2 + r * 2 + lane;
                    out[idx].0 = out[idx].0.wrapping_add(val);
                }
            }
        }
    }
    out.into_iter()
        .enumerate()
        .map(|(i, v)| if i % 2 == 1 { v.0.wrapping_mul(2) } else { v.0 })
        .collect()
}

struct Fourier {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl Fourier {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        Self {
            forward: planner.plan_fft_forward(COLS),
            inverse: planner.plan_fft_inverse(COLS),
        }
    }

    fn prepare(&self, record: &Record) -> Vec<Complex64> {
        let mut out = Vec::with_capacity(96 * BINS);
        let mut buffer = vec![Complex64::default(); COLS];
        let mut scratch = vec![Complex64::default(); self.forward.get_inplace_scratch_len()];
        for data in [&record.code, &record.mask] {
            for row in data.chunks_exact(ROW_SIZE) {
                for lane in 0..4 {
                    for t in 0..COLS {
                        // A signed representative preserves the dot modulo 2^16
                        // and reduces numerical error compared with unsigned u16.
                        buffer[t] = Complex64::new(row[t * 4 + lane] as i16 as f64, 0.0);
                    }
                    self.forward.process_with_scratch(&mut buffer, &mut scratch);
                    out.extend_from_slice(&buffer[..BINS]);
                }
            }
        }
        out
    }

    fn chunk(&self, query: &[Complex64], targets: &[Vec<Complex64>]) -> Vec<u16> {
        let mut out = Vec::with_capacity(targets.len() * ROTATIONS * 2);
        let mut spectrum = vec![Complex64::default(); COLS * 2];
        let mut scratch = vec![Complex64::default(); self.inverse.get_inplace_scratch_len()];
        for target in targets {
            self.scores(query, target, &mut spectrum, &mut scratch);
            for r in 0..ROTATIONS {
                let idx = (r + COLS - 15) % COLS;
                out.push((spectrum[idx].re / COLS as f64).round() as i64 as u16);
                out.push(
                    ((spectrum[COLS + idx].re / COLS as f64).round() as i64 as u16).wrapping_mul(2),
                );
            }
        }
        out
    }

    fn scores(
        &self,
        query: &[Complex64],
        target: &[Complex64],
        spectrum: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        spectrum.fill(Complex64::default());
        for (lane, begin, end) in [(0, 0, 64), (1, 64, 96)] {
            let accum = &mut spectrum[lane * COLS..(lane + 1) * COLS];
            for channel in begin..end {
                let q = &query[channel * BINS..(channel + 1) * BINS];
                let d = &target[channel * BINS..(channel + 1) * BINS];
                for k in 0..BINS {
                    accum[k].re = q[k]
                        .re
                        .mul_add(d[k].re, q[k].im.mul_add(d[k].im, accum[k].re));
                    accum[k].im = q[k]
                        .re
                        .mul_add(d[k].im, (-q[k].im).mul_add(d[k].re, accum[k].im));
                }
            }
            for k in 1..COLS / 2 {
                accum[COLS - k] = accum[k].conj();
            }
            self.inverse.process_with_scratch(accum, scratch);
        }
    }
}

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
    selected_inverse: Vec<u16>,
    pair_sum_weights: Vec<u16>,
    pair_diff_weights: Vec<u16>,
    padded_sum_weights: Vec<u16>,
    padded_diff_weights: Vec<u16>,
}

impl Ntt {
    fn new() -> Self {
        let inv_n = pow_mod(COLS as u64, P - 2);
        Self {
            forward: NttNode::new(COLS, ROOT),
            pair_sum_weights: Self::pair_weights(false, PAIRS),
            pair_diff_weights: Self::pair_weights(true, PAIRS),
            padded_sum_weights: Self::pair_weights(false, 104),
            padded_diff_weights: Self::pair_weights(true, 104),
            selected_inverse: (0..ROTATIONS)
                .flat_map(|r| {
                    let r = (r + COLS - 15) % COLS;
                    (0..COLS).map(move |k| {
                        centered((inv_n * pow_mod(ROOT, ((COLS - r) * k % COLS) as u64) % P) as u16)
                    })
                })
                .collect(),
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

    fn prepare_query<const PAIRED: bool>(&self, record: &Record) -> Vec<u16> {
        let mut result = self.prepare(record);
        if PAIRED {
            // Linearity moves inverse normalization to the one-time query transform.
            let inv_n = pow_mod(COLS as u64, P - 2) as i64;
            for v in &mut result {
                *v = centered(((*v as i16 as i64 * inv_n).rem_euclid(P as i64)) as u16);
            }
        }
        result
    }

    // MODE: 1 reduces each dot; 2 defers reductions; 3 also pads the SIMD tail.
    fn inverse_paired<const MODE: usize>(&self, spectrum: &[u16; COLS]) -> [u16; ROTATIONS] {
        let mut sums = [0u16; 104];
        let mut diffs = [0u16; 104];
        let (stride, sum_weights, diff_weights) = if MODE == 3 {
            (104, &self.padded_sum_weights, &self.padded_diff_weights)
        } else {
            (PAIRS, &self.pair_sum_weights, &self.pair_diff_weights)
        };
        // The simple variant centers sums/differences modulo p. The lazy
        // variant retains [-25600,25600]: a 99-term dot accumulates only
        // 96 terms in SIMD (six products/lane) and the last three in i64.
        // 6 * 25600 * 12800 = 1,966,080,000 < i32::MAX, so it is exact too.
        // Mode 3 pads to 104 terms and centers just the three live tail terms.
        // Six wide products plus one centered product still fit in each i32
        // lane (2,129,920,000), and the five padding products are zero.
        for k in 1..=PAIRS {
            let a = spectrum[k] as i16 as i32;
            let b = spectrum[COLS - k] as i16 as i32;
            sums[k - 1] = if MODE >= 2 && !(MODE == 3 && k > 96) {
                (a + b) as i16 as u16
            } else {
                center_sum(a + b)
            };
            diffs[k - 1] = if MODE >= 2 && !(MODE == 3 && k > 96) {
                (a - b) as i16 as u16
            } else {
                center_sum(a - b)
            };
        }
        let dc = spectrum[0] as i16 as i32;
        let nyquist = spectrum[COLS / 2] as i16 as i32;
        let mut out = [0u16; ROTATIONS];
        out[ROTATIONS / 2] = (dc + nyquist + sums.iter().map(|v| *v as i16 as i32).sum::<i32>())
            .rem_euclid(P as i32) as u16;
        for r in 1..=ROTATIONS / 2 {
            let start = (r - 1) * stride;
            if MODE >= 2 {
                // Combine exact wide sums before reducing; no per-dot modular
                // reductions are needed. Each final output has one reduction.
                let even =
                    field_dot_unreduced(&sums[..stride], &sum_weights[start..start + stride]);
                let odd =
                    field_dot_unreduced(&diffs[..stride], &diff_weights[start..start + stride]);
                let endpoint = (dc + if r % 2 == 0 { nyquist } else { -nyquist }) as i64;
                out[ROTATIONS / 2 + r] = (endpoint + even + odd).rem_euclid(P as i64) as u16;
                out[ROTATIONS / 2 - r] = (endpoint + even - odd).rem_euclid(P as i64) as u16;
            } else {
                let even = field_dot(&sums[..stride], &sum_weights[start..start + stride]) as i32;
                let odd = field_dot(&diffs[..stride], &diff_weights[start..start + stride]) as i32;
                let endpoint = dc + if r % 2 == 0 { nyquist } else { -nyquist };
                out[ROTATIONS / 2 + r] = (endpoint + even + odd).rem_euclid(P as i32) as u16;
                out[ROTATIONS / 2 - r] = (endpoint + even - odd).rem_euclid(P as i32) as u16;
            }
        }
        out
    }

    fn prepare(&self, record: &Record) -> Vec<u16> {
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

    fn chunk<const INVERSE: usize>(&self, queries: &[Vec<u16>], targets: &[Vec<u16>]) -> Vec<u16> {
        let mut out = vec![0u16; targets.len() * queries.len() * ROTATIONS * 2];
        let mut spectrum = [0u16; COLS];
        // All inverse variants use the same target-first traversal. Normal and
        // mirror reuse the resident record within this worker, as in the PR.
        for (target_index, target) in targets.iter().enumerate() {
            for (query_index, query) in queries.iter().enumerate() {
                for (lane, offset, channels) in [(0, 0, 64), (1, CODE_LEN, 32)] {
                    for k in 0..COLS {
                        let q = &query[offset + ((COLS - k) % COLS) * channels..][..channels];
                        let d = &target[offset + k * channels..][..channels];
                        spectrum[k] = centered(field_dot(q, d));
                    }
                    let scores = if INVERSE != 0 {
                        self.inverse_paired::<INVERSE>(&spectrum)
                    } else {
                        std::array::from_fn(|r| {
                            field_dot(&spectrum, &self.selected_inverse[r * COLS..(r + 1) * COLS])
                        })
                    };
                    for (r, score) in scores.into_iter().enumerate() {
                        let idx = (target_index * queries.len() + query_index) * ROTATIONS * 2
                            + r * 2
                            + lane;
                        out[idx] = if lane == 1 {
                            (2 * score as u64 % P) as u16
                        } else {
                            score
                        };
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
    // Centered inputs lie in [-12800,12800]. Sixteen independent i32
    // accumulators receive at most ceil(200/16)=13 products each:
    // 13 * 12800^2 = 2,129,920,000 < i32::MAX. Widen the horizontal sum.
    // Paired inverse also permits |a|<=25600, |b|<=12800 at length 99:
    // the 96 vectorized terms contribute six products per accumulator lane,
    // bounded by 1,966,080,000; the remaining three use scalar i64.
    // Padded paired inverse uses 104 terms: first 96 have |a|<=25600,
    // next three |a|<=12800, last five are zero; |b|<=12800 throughout.
    // Its largest SIMD lane is bounded by 6*25600*12800 + 12800^2
    // = 2,129,920,000 < i32::MAX. No scalar tail is needed in that mode.
    // SAFETY: NEON is guaranteed on AArch64; both slices have equal length,
    // vector loads stay in bounds, and the accumulation bound is given above.
    unsafe {
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
        let mut sum = vaddlvq_s32(acc0) + vaddlvq_s32(acc1) + vaddlvq_s32(acc2) + vaddlvq_s32(acc3);
        while i < a.len() {
            sum += a[i] as i16 as i64 * b[i] as i16 as i64;
            i += 1;
        }
        sum
    }
}

#[inline]
fn field_dot(a: &[u16], b: &[u16]) -> u16 {
    field_dot_unreduced(a, b).rem_euclid(P as i64) as u16
}

fn direct_prime(query: &Record, target: &Record) -> Vec<u16> {
    let mut out = Vec::new();
    for r in 0..ROTATIONS {
        let left = ((15 - r as isize) * 4).rem_euclid(800) as usize;
        for (lane, q, d) in [
            (0, &query.code, &target.code),
            (1, &query.mask, &target.mask),
        ] {
            let mut acc = 0u64;
            for (q, d) in q.chunks_exact(ROW_SIZE).zip(d.chunks_exact(ROW_SIZE)) {
                for t in 0..ROW_SIZE {
                    acc += (q[(t + left) % ROW_SIZE] as u64 % P) * (d[t] as u64 % P);
                }
            }
            out.push((acc * (lane + 1) % P) as u16);
        }
    }
    out
}

fn direct_signed(query: &Record, target: &Record) -> Vec<i64> {
    let mut out = Vec::new();
    for r in 0..ROTATIONS {
        let left = ((15 - r as isize) * 4).rem_euclid(800) as usize;
        for (q, d) in [(&query.code, &target.code), (&query.mask, &target.mask)] {
            let mut acc = 0i64;
            for (q, d) in q.chunks_exact(ROW_SIZE).zip(d.chunks_exact(ROW_SIZE)) {
                for t in 0..ROW_SIZE {
                    acc += q[(t + left) % ROW_SIZE] as i16 as i64 * d[t] as i16 as i64;
                }
            }
            out.push(acc);
        }
    }
    out
}

fn validate(fft: &Fourier, ntt: &Ntt) {
    let mut rng = Rng(90210);
    let mut max_absolute_error = 0.0f64;
    for size in [1, 8, 16, 32, 64, 99, 199, 200] {
        for a in [-12800i16, 12800] {
            for b in [-12800i16, 12800] {
                let expected = (size as i64 * a as i64 * b as i64).rem_euclid(P as i64) as u16;
                assert_eq!(
                    field_dot(&vec![a as u16; size], &vec![b as u16; size]),
                    expected
                );
            }
        }
    }
    for a in [-25600i16, 25600] {
        for b in [-12800i16, 12800] {
            assert_eq!(
                field_dot_unreduced(&[a as u16; 99], &[b as u16; 99]),
                99 * a as i64 * b as i64
            );
        }
    }
    for a in [-25600i16, 25600] {
        for b in [-12800i16, 12800] {
            let mut lhs = [0u16; 104];
            lhs[..96].fill(a as u16);
            lhs[96..99].fill((a / 2) as u16);
            let mut rhs = [0u16; 104];
            rhs[..99].fill(b as u16);
            assert_eq!(
                field_dot_unreduced(&lhs, &rhs),
                (96 * a as i64 + 3 * (a / 2) as i64) * b as i64
            );
        }
    }
    let inverse_checks = validate_paired_inverse(ntt);
    let mut checks = 0;
    let mut cases: Vec<(Record, Record)> = (0..64).map(|_| (rng.record(), rng.record())).collect();
    for v in [0u16, 1, 32767, 32768, 65535] {
        let constant = Record {
            code: vec![v; CODE_LEN],
            mask: vec![v; MASK_LEN],
        };
        cases.push((constant.clone(), constant));
    }
    // Large DC terms stress rounding much more than zero-mean random shares.
    for jitter in [2u16, 128, 1024] {
        for negative in [false, true] {
            let mut q = rng.record();
            let mut d = rng.record();
            for record in [&mut q, &mut d] {
                for data in [&mut record.code, &mut record.mask] {
                    for value in data {
                        let signed = 32767i16 - (*value % jitter) as i16;
                        *value = if negative { -signed } else { signed } as u16;
                    }
                }
            }
            cases.push((q, d));
        }
    }
    let mut alternating = rng.record();
    for data in [&mut alternating.code, &mut alternating.mask] {
        for (i, v) in data.iter_mut().enumerate() {
            *v = if i % 8 < 4 { 32768 } else { 32767 };
        }
    }
    cases.push((alternating.clone(), alternating));
    let mut impulse = Record {
        code: vec![0; CODE_LEN],
        mask: vec![0; MASK_LEN],
    };
    impulse.code[799] = 32768;
    impulse.mask[0] = 65535;
    cases.push((impulse.clone(), impulse));
    for (query, target) in &cases {
        let reference = baseline(&prepare_baseline(query), std::slice::from_ref(target));
        let q = fft.prepare(query);
        let d = fft.prepare(target);
        assert_eq!(
            fft.chunk(&q, std::slice::from_ref(&d)),
            reference,
            "FFT differs from repository kernel"
        );
        let mut scores = vec![Complex64::default(); 2 * COLS];
        let mut scratch = vec![Complex64::default(); fft.inverse.get_inplace_scratch_len()];
        fft.scores(&q, &d, &mut scores, &mut scratch);
        let integer_reference = direct_signed(query, target);
        for r in 0..ROTATIONS {
            for lane in 0..2 {
                let v = scores[lane * COLS + (r + COLS - 15) % COLS].re / COLS as f64;
                let exact = integer_reference[r * 2 + lane];
                assert_eq!(v.round() as i64, exact, "FFT differs before ring reduction");
                max_absolute_error = max_absolute_error.max((v - exact as f64).abs());
            }
        }
        let q = ntt.prepare(query);
        let d = ntt.prepare(target);
        assert_eq!(
            ntt.chunk::<0>(&[q], std::slice::from_ref(&d)),
            direct_prime(query, target),
            "NTT differs from direct field dot"
        );
        assert_eq!(
            ntt.chunk::<1>(&[ntt.prepare_query::<true>(query)], &[d]),
            direct_prime(query, target),
            "paired NTT differs from direct field dot"
        );
        checks += ROTATIONS * 2;
    }
    // Exercise all 6x4 tiles and both tails, not only singleton fallback.
    let query = rng.record();
    let targets: Vec<_> = (0..131).map(|_| rng.record()).collect();
    let reference = baseline(&prepare_baseline(&query), &targets);
    let spectra: Vec<_> = targets.iter().map(|r| fft.prepare(r)).collect();
    assert_eq!(fft.chunk(&fft.prepare(&query), &spectra), reference);
    validate_production_and_pair(ntt);
    eprintln!("VALIDATION: {inverse_checks} paired inverse outputs; {checks} FFT/ring and NTT/prime scalar scores; 8122 additional FFT scores against tiled NEON; 40 SIMD accumulation boundary cases. Max observed absolute FFT error before rounding={max_absolute_error:.9}. This is empirical, not an all-input error proof.");
}

fn validate_paired_inverse(ntt: &Ntt) -> usize {
    for value in -25600..=25600 {
        let actual = center_sum(value) as i16 as i32;
        assert!((-12800..=12800).contains(&actual));
        assert_eq!(actual.rem_euclid(P as i32), value.rem_euclid(P as i32));
    }
    let mut rng = Rng(2348);
    let mut cases: Vec<[u16; COLS]> = (0..1000)
        .map(|_| std::array::from_fn(|_| centered(rng.next() % P as u16)))
        .collect();
    for k in 0..COLS {
        let mut impulse = [0u16; COLS];
        impulse[k] = 12800;
        cases.push(impulse);
    }
    for value in [-12800i16, 12800] {
        cases.push([value as u16; COLS]);
        cases.push(std::array::from_fn(
            |k| if k % 2 == 0 { value } else { -value } as u16,
        ));
    }
    let inv_n = pow_mod(COLS as u64, P - 2) as i64;
    for spectrum in &cases {
        let normalized =
            spectrum.map(|v| centered((v as i16 as i64 * inv_n).rem_euclid(P as i64) as u16));
        let expected: [u16; ROTATIONS] = std::array::from_fn(|r| {
            spectrum
                .iter()
                .zip(&ntt.selected_inverse[r * COLS..(r + 1) * COLS])
                .map(|(&a, &b)| a as i16 as i64 * b as i16 as i64)
                .sum::<i64>()
                .rem_euclid(P as i64) as u16
        });
        assert_eq!(
            ntt.inverse_paired::<1>(&normalized),
            expected,
            "paired inverse spectrum mismatch"
        );
        assert_eq!(
            ntt.inverse_paired::<2>(&normalized),
            expected,
            "lazy paired inverse spectrum mismatch"
        );
        assert_eq!(
            ntt.inverse_paired::<3>(&normalized),
            expected,
            "padded paired inverse mismatch"
        );
    }
    cases.len() * ROTATIONS
}

fn validate_production_and_pair(ntt: &Ntt) {
    let mut rng = Rng(23482348);
    let normal = rng.record();
    let queries = [normal.clone(), mirror(&normal)];
    let arcs: Vec<_> = queries
        .iter()
        .map(|q| Arc::new(GaloisRingSharedIris::from_record(q)))
        .collect();
    let targets: Vec<_> = (0..132).map(|_| rng.record()).collect();
    let mixed: Vec<_> = targets
        .iter()
        .map(|r| MixedPlaneIris::from_iris(&GaloisRingSharedIris::from_record(r)))
        .collect();
    for (raw, packed) in targets.iter().zip(&mixed) {
        let decoded = packed.to_iris();
        assert_eq!(raw.code, decoded.code.coefs);
        assert_eq!(raw.mask, decoded.mask.coefs);
    }
    let finite: Vec<_> = targets.iter().map(|r| ntt.prepare(r)).collect();
    let direct: Vec<_> = queries
        .iter()
        .map(|q| ntt.prepare_query::<false>(q))
        .collect();
    let paired: Vec<_> = queries
        .iter()
        .map(|q| ntt.prepare_query::<true>(q))
        .collect();
    let mut checks = 0;
    for size in [0, 1, 2, 3, 4, 5, 8, 131, 132] {
        for count in [1, 2] {
            let actual = production(&arcs[..count], &mixed[..size]);
            for (q, result) in queries[..count].iter().zip(&actual) {
                assert_eq!(
                    *result,
                    baseline(&prepare_baseline(q), &targets[..size]),
                    "PR #2348 kernel mismatch"
                );
                checks += result.len();
            }
            let expected: Vec<_> = targets[..size]
                .iter()
                .flat_map(|d| {
                    queries[..count]
                        .iter()
                        .flat_map(move |q| direct_prime(q, d))
                })
                .collect();
            assert_eq!(ntt.chunk::<0>(&direct[..count], &finite[..size]), expected);
            assert_eq!(ntt.chunk::<1>(&paired[..count], &finite[..size]), expected);
            assert_eq!(ntt.chunk::<2>(&paired[..count], &finite[..size]), expected);
            assert_eq!(ntt.chunk::<3>(&paired[..count], &finite[..size]), expected);
        }
    }
    eprintln!("PR2348_VALIDATION: {checks} mixed/legacy and four NTT variants/direct scalar scores, normal + mirror, singleton and packed paths, 4-target and 31-rotation tails; 132 resident roundtrips.");
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn checksum(scores: Vec<u16>) -> u64 {
    black_box(scores).into_iter().map(|v| v as u64).sum()
}

#[cfg(target_os = "linux")]
fn pin_cpu(cpu: usize) {
    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u8) -> i32;
    }
    let mut mask = [0u8; 128];
    assert!(cpu < mask.len() * 8);
    mask[cpu / 8] = 1 << (cpu % 8);
    // SAFETY: the initialized mask is valid for its supplied size; pid zero
    // selects this calling thread. No other process's affinity is modified.
    let rc = unsafe { sched_setaffinity(0, mask.len(), mask.as_ptr()) };
    assert_eq!(rc, 0, "pin CPU {cpu}: {}", std::io::Error::last_os_error());
}

#[cfg(not(target_os = "linux"))]
fn pin_cpu(_: usize) {
    panic!("explicit CPU affinity is implemented for Linux only");
}

fn cpu_list(value: &str) -> Vec<usize> {
    if value.is_empty() {
        return Vec::new();
    }
    value
        .split(',')
        .flat_map(|part| {
            if let Some((first, last)) = part.split_once('-') {
                (first.parse::<usize>().unwrap()..=last.parse::<usize>().unwrap())
                    .collect::<Vec<_>>()
            } else {
                vec![part.parse::<usize>().unwrap()]
            }
        })
        .collect()
}

fn make_pool(threads: usize, cpus: &[usize]) -> rayon::ThreadPool {
    assert!(cpus.is_empty() || threads <= cpus.len());
    let cpus = cpus.to_vec();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |index| {
            if !cpus.is_empty() {
                pin_cpu(cpus[index]);
            }
        })
        .build()
        .unwrap()
}

fn mirror(record: &Record) -> Record {
    let mut out = record.clone();
    for (code, data) in [(true, &mut out.code), (false, &mut out.mask)] {
        let source = if code { &record.code } else { &record.mask };
        for (block, row) in data.chunks_exact_mut(ROW_SIZE).enumerate() {
            for c in 0..COLS {
                let flipped = if c < 100 { 99 - c } else { 299 - c };
                for lane in 0..4 {
                    let v = source[block * ROW_SIZE + flipped * 4 + lane];
                    row[c * 4 + lane] = if code && block >= 8 {
                        v.wrapping_neg()
                    } else {
                        v
                    };
                }
            }
        }
    }
    out
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let option = |name: &str, default: &str| {
        args.windows(2)
            .find(|w| w[0] == name)
            .map(|w| w[1].clone())
            .unwrap_or_else(|| default.to_owned())
    };
    let sizes: Vec<usize> = option("--sizes", "1,128,1024,16384")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let thread_counts: Vec<usize> = option("--threads", "1,6")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let repetitions: usize = option("--reps", "9").parse().unwrap();
    let chunk: usize = option("--chunk", &CHUNK.to_string()).parse().unwrap();
    let orientations: usize = option("--orientations", "1").parse().unwrap();
    let pre_threads: usize = option("--preprocess-threads", "1").parse().unwrap();
    let cpus = cpu_list(&option("--cpu-list", ""));
    let main_cpu = option("--main-cpu", "");
    if !main_cpu.is_empty() {
        pin_cpu(main_cpu.parse().unwrap());
    }
    assert!(chunk > 0 && (1..=2).contains(&orientations));
    assert!(repetitions >= 3);
    eprintln!("CONFIG chunk={chunk} orientations={orientations} cpu_list={cpus:?} main_cpu={main_cpu:?} preprocess_threads={pre_threads}");
    let fft = Fourier::new();
    let ntt = Ntt::new();
    validate(&fft, &ntt);
    packed_ntt::validate(&ntt);
    if args.iter().any(|a| a == "--validate-only") {
        return;
    }
    let mut rng = Rng(20260905);
    let queries: Vec<_> = (0..4).map(|_| rng.record()).collect();
    let mirrored: Vec<_> = queries.iter().map(mirror).collect();
    let query_arcs: Vec<Vec<_>> = queries
        .iter()
        .zip(&mirrored)
        .map(|(a, b)| {
            [a, b]
                .into_iter()
                .map(|q| Arc::new(GaloisRingSharedIris::from_record(q)))
                .collect()
        })
        .collect();
    let max_size = *sizes.iter().max().unwrap();
    let start = Instant::now();
    let raw: Vec<_> = (0..max_size).map(|_| rng.record()).collect();
    eprintln!(
        "Generated {max_size} full-width synthetic share records in {:.3}s",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let pre_pool = make_pool(pre_threads, &cpus);
    let mixed: Vec<_> = pre_pool.install(|| {
        raw.par_iter()
            .map(|record| MixedPlaneIris::from_iris(&GaloisRingSharedIris::from_record(record)))
            .collect()
    });
    eprintln!(
        "OFFLINE PR2348 mixed-plane preprocessing {:.3}s, 38400 bytes/record",
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let fourier: Vec<_> =
        pre_pool.install(|| raw.par_iter().map(|record| fft.prepare(record)).collect());
    eprintln!(
        "OFFLINE FFT DB preprocessing {:.3}s, {} bytes/record",
        start.elapsed().as_secs_f64(),
        96 * BINS * 16
    );
    let start = Instant::now();
    let finite: Vec<_> =
        pre_pool.install(|| raw.par_iter().map(|record| ntt.prepare(record)).collect());
    eprintln!(
        "OFFLINE NTT DB preprocessing {:.3}s, {} bytes/record",
        start.elapsed().as_secs_f64(),
        (CODE_LEN + MASK_LEN) * 2
    );
    let start = Instant::now();
    let packed: Vec<_> = pre_pool.install(|| {
        finite
            .par_iter()
            .map(|record| packed_ntt::pack(record))
            .collect()
    });
    eprintln!(
        "OFFLINE signed-byte NTT packing {:.3}s, 38400 bytes/record",
        start.elapsed().as_secs_f64()
    );
    drop(pre_pool);

    for method in [0, 2, 3, 4, 5, 6] {
        let mut times = Vec::new();
        for repeat in 0..51 {
            let query = &queries[repeat % queries.len()];
            let start = Instant::now();
            match method {
                0 => {
                    black_box(prepare_baseline(query));
                }
                2 => {
                    black_box(fft.prepare(query));
                }
                3 => {
                    black_box(ntt.prepare_query::<false>(query));
                }
                _ => {
                    black_box(ntt.prepare_query::<true>(query));
                }
            }
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!(
            "QUERY_PREP,{},median_us={:.3}",
            METHODS[method],
            median(&mut times)
        );
    }
    eprintln!("PR2348 query packing is cached per worker and included in scan timing; each repetition changes the query.");
    println!(
        "threads,records,method,median_ms,min_ms,max_ms,ns_per_record,records_per_second,checksum,orientations,chunk,orientation_comparisons_per_second"
    );
    for threads in thread_counts {
        let pool = make_pool(threads, &cpus);
        for &size in &sizes {
            let mut times: [Vec<f64>; METHODS.len()] = std::array::from_fn(|_| Vec::new());
            let mut sums = [0u64; METHODS.len()];
            // Rotate method order, warm every method, and vary the query to
            // include fresh query preprocessing instead of pointer-cache hits.
            for repeat in 0..repetitions + 2 {
                let query_index = repeat % queries.len();
                let query_variants = [&queries[query_index], &mirrored[query_index]];
                let query_variants = &query_variants[..orientations];
                for shift in 0..METHODS.len() {
                    let method = (repeat + shift) % METHODS.len();
                    let start = Instant::now();
                    let sum: u64 = match method {
                        0 => {
                            let prepared: Vec<_> =
                                query_variants.iter().map(|q| prepare_baseline(q)).collect();
                            pool.install(|| {
                                prepared
                                    .par_iter()
                                    .map(|q| {
                                        raw[..size]
                                            .par_chunks(chunk)
                                            .map(|targets| checksum(baseline(q, targets)))
                                            .sum::<u64>()
                                    })
                                    .sum()
                            })
                        }
                        1 => pool.install(|| {
                            mixed[..size]
                                .par_chunks(chunk)
                                .map(|targets| {
                                    production(&query_arcs[query_index][..orientations], targets)
                                        .into_iter()
                                        .map(checksum)
                                        .sum::<u64>()
                                })
                                .sum()
                        }),
                        2 => {
                            let prepared: Vec<_> =
                                query_variants.iter().map(|q| fft.prepare(q)).collect();
                            pool.install(|| {
                                prepared
                                    .par_iter()
                                    .map(|q| {
                                        fourier[..size]
                                            .par_chunks(chunk)
                                            .map(|targets| checksum(fft.chunk(q, targets)))
                                            .sum::<u64>()
                                    })
                                    .sum()
                            })
                        }
                        3 => {
                            let prepared: Vec<_> = query_variants
                                .iter()
                                .map(|q| ntt.prepare_query::<false>(q))
                                .collect();
                            pool.install(|| {
                                finite[..size]
                                    .par_chunks(chunk)
                                    .map(|targets| checksum(ntt.chunk::<0>(&prepared, targets)))
                                    .sum()
                            })
                        }
                        4 => {
                            let prepared: Vec<_> = query_variants
                                .iter()
                                .map(|q| ntt.prepare_query::<true>(q))
                                .collect();
                            pool.install(|| {
                                finite[..size]
                                    .par_chunks(chunk)
                                    .map(|targets| checksum(ntt.chunk::<1>(&prepared, targets)))
                                    .sum()
                            })
                        }
                        5 => {
                            let prepared: Vec<_> = query_variants
                                .iter()
                                .map(|q| ntt.prepare_query::<true>(q))
                                .collect();
                            pool.install(|| {
                                finite[..size]
                                    .par_chunks(chunk)
                                    .map(|targets| checksum(ntt.chunk::<2>(&prepared, targets)))
                                    .sum()
                            })
                        }
                        6 => {
                            let prepared: Vec<_> = query_variants
                                .iter()
                                .map(|q| ntt.prepare_query::<true>(q))
                                .collect();
                            pool.install(|| {
                                finite[..size]
                                    .par_chunks(chunk)
                                    .map(|targets| checksum(ntt.chunk::<3>(&prepared, targets)))
                                    .sum()
                            })
                        }
                        _ => {
                            let prepared = packed_ntt::Query::prepare(&ntt, query_variants);
                            pool.install(|| {
                                packed[..size]
                                    .par_chunks(chunk)
                                    .map(|targets| {
                                        checksum(packed_ntt::chunk(&ntt, &prepared, targets))
                                    })
                                    .sum()
                            })
                        }
                    };
                    let elapsed = start.elapsed().as_secs_f64();
                    black_box(sum);
                    eprintln!(
                        "SAMPLE,{threads},{size},{},{repeat},{:.6},{sum}",
                        METHODS[method],
                        elapsed * 1e3
                    );
                    sums[method] = sum;
                    if repeat >= 2 {
                        times[method].push(elapsed);
                    }
                }
                assert_eq!(sums[0], sums[1], "PR2348/legacy checksum mismatch");
                assert_eq!(sums[0], sums[2], "FFT/legacy checksum mismatch");
                assert_eq!(sums[3], sums[4], "paired/direct NTT checksum mismatch");
                assert_eq!(sums[3], sums[5], "lazy paired/direct NTT checksum mismatch");
                assert_eq!(
                    sums[3], sums[6],
                    "padded paired/direct NTT checksum mismatch"
                );
                assert_eq!(sums[3], sums[7], "packed SMMLA/direct NTT mismatch");
            }
            for (method, samples) in times.iter_mut().enumerate() {
                let med = median(samples);
                println!(
                    "{threads},{size},{},{:.6},{:.6},{:.6},{:.3},{:.3},{},{orientations},{chunk},{:.3}",
                    METHODS[method],
                    med * 1e3,
                    samples[0] * 1e3,
                    samples[samples.len() - 1] * 1e3,
                    med * 1e9 / size as f64,
                    size as f64 / med,
                    sums[method],
                    (size * orientations) as f64 / med
                );
            }
        }
    }
}
