//! Exact signed-byte matrix products for the spectral score stage.
use super::{center_sum, centered, Ntt, Record, CODE_LEN, COLS, MASK_LEN, P, ROTATIONS};

const TILE: usize = 8;

#[inline]
fn split(value: i16) -> (i8, i8) {
    let low = value as i8;
    let high = ((value as i32 - low as i32) / 256) as i8;
    (low, high)
}

pub fn pack(values: &[u16]) -> Vec<i8> {
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

pub struct Query {
    low: Vec<i8>,
    high: Vec<i8>,
    orientations: usize,
}

impl Query {
    pub fn prepare(ntt: &Ntt, queries: &[&Record]) -> Self {
        assert!((1..=2).contains(&queries.len()));
        let values: Vec<_> = queries
            .iter()
            .map(|q| ntt.prepare_query::<true>(q))
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
                            // Absorb mask doubling into query preparation too.
                            if offset == CODE_LEN {
                                value = center_sum(2 * value as i16 as i32);
                            }
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

pub fn chunk(ntt: &Ntt, query: &Query, targets: &[Vec<i8>]) -> Vec<u16> {
    let mut out = vec![0u16; targets.len() * query.orientations * ROTATIONS * 2];
    // Frequency products for an eight-record/two-orientation tile fit in 6.4 KiB.
    let mut spectra = [[[0u16; COLS]; 2]; TILE];
    for (tile_index, tile) in targets.chunks(TILE).enumerate() {
        for (lane, offset, channels) in [(0, 0, 64), (1, CODE_LEN, 32)] {
            let component_bytes = 2 * channels * COLS;
            let low = &query.low[2 * offset..2 * offset + component_bytes];
            let high = &query.high[2 * offset..2 * offset + component_bytes];
            for (k, (low, high)) in low
                .chunks_exact(2 * channels)
                .zip(high.chunks_exact(2 * channels))
                .enumerate()
            {
                let start = 2 * (offset + k * channels);
                let pointers =
                    std::array::from_fn(|t: usize| tile[t.min(tile.len() - 1)][start..].as_ptr());
                // SAFETY: callers select this method only on i8mm AArch64;
                // each pointer covers 2*channels bytes, and queries have two
                // packed lanes. Tail targets repeat a valid last record.
                let scores =
                    unsafe { dot_8x2(low.as_ptr(), high.as_ptr(), pointers, channels / 8) };
                for (record, scores) in spectra.iter_mut().zip(scores).take(tile.len()) {
                    for (spectrum, score) in record.iter_mut().zip(scores).take(query.orientations)
                    {
                        spectrum[k] = centered(score);
                    }
                }
            }
            for (t, record) in spectra.iter().take(tile.len()).enumerate() {
                for (q, spectrum) in record.iter().take(query.orientations).enumerate() {
                    let scores = ntt.inverse_paired::<3>(spectrum);
                    for (r, score) in scores.into_iter().enumerate() {
                        let idx =
                            ((tile_index * TILE + t) * query.orientations + q) * ROTATIONS * 2
                                + r * 2
                                + lane;
                        out[idx] = score;
                    }
                }
            }
        }
    }
    out
}

// Signed byte planes represent v exactly as lo + 256*hi, including negative
// low bytes and the carry into hi. |lo|<=128 and |hi|<=50, so each i32 SMMLA
// partial over 64 terms is at most 64*128^2 = 1,048,576 in magnitude. All four
// byte products are required in this prime field. Recombine in i64 then mod p.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "i8mm")]
unsafe fn dot_8x2(
    low: *const i8,
    high: *const i8,
    targets: [*const i8; TILE],
    blocks: usize,
) -> [[u16; 2]; TILE] {
    use std::arch::asm;
    let mut raw = [[0i32; 4]; 2 * TILE];
    let t0 = targets[0];
    let t1 = targets[1];
    let t2 = targets[2];
    let t3 = targets[3];
    let t4 = targets[4];
    let t5 = targets[5];
    let t6 = targets[6];
    let t7 = targets[7];
    asm!(
        "movi v0.4s, #0",
        "movi v1.4s, #0",
        "movi v2.4s, #0",
        "movi v3.4s, #0",
        "movi v4.4s, #0",
        "movi v5.4s, #0",
        "movi v6.4s, #0",
        "movi v7.4s, #0",
        "movi v8.4s, #0",
        "movi v9.4s, #0",
        "movi v10.4s, #0",
        "movi v11.4s, #0",
        "movi v12.4s, #0",
        "movi v13.4s, #0",
        "movi v14.4s, #0",
        "movi v15.4s, #0",
        "2:",
        "ldr q16, [{low}], #16", "ldr q17, [{high}], #16",
        "ldr q18, [{t0}], #16",
        "ldr q19, [{t1}], #16",
        "ldr q20, [{t2}], #16",
        "ldr q21, [{t3}], #16",
        "ldr q22, [{t4}], #16",
        "ldr q23, [{t5}], #16",
        "ldr q24, [{t6}], #16",
        "ldr q25, [{t7}], #16",
        "smmla v0.4s, v18.16b, v16.16b", "smmla v1.4s, v18.16b, v17.16b",
        "smmla v2.4s, v19.16b, v16.16b", "smmla v3.4s, v19.16b, v17.16b",
        "smmla v4.4s, v20.16b, v16.16b", "smmla v5.4s, v20.16b, v17.16b",
        "smmla v6.4s, v21.16b, v16.16b", "smmla v7.4s, v21.16b, v17.16b",
        "smmla v8.4s, v22.16b, v16.16b", "smmla v9.4s, v22.16b, v17.16b",
        "smmla v10.4s, v23.16b, v16.16b", "smmla v11.4s, v23.16b, v17.16b",
        "smmla v12.4s, v24.16b, v16.16b", "smmla v13.4s, v24.16b, v17.16b",
        "smmla v14.4s, v25.16b, v16.16b", "smmla v15.4s, v25.16b, v17.16b",
        "subs {blocks}, {blocks}, #1", "b.ne 2b",
        "stp q0, q1, [{out}, #0]",
        "stp q2, q3, [{out}, #32]",
        "stp q4, q5, [{out}, #64]",
        "stp q6, q7, [{out}, #96]",
        "stp q8, q9, [{out}, #128]",
        "stp q10, q11, [{out}, #160]",
        "stp q12, q13, [{out}, #192]",
        "stp q14, q15, [{out}, #224]",
        low = inout(reg) low => _, high = inout(reg) high => _,
        t0 = inout(reg) t0 => _,
        t1 = inout(reg) t1 => _,
        t2 = inout(reg) t2 => _,
        t3 = inout(reg) t3 => _,
        t4 = inout(reg) t4 => _,
        t5 = inout(reg) t5 => _,
        t6 = inout(reg) t6 => _,
        t7 = inout(reg) t7 => _,
        blocks = inout(reg) blocks => _, out = in(reg) raw.as_mut_ptr(),
        out("v0") _,
        out("v1") _,
        out("v2") _,
        out("v3") _,
        out("v4") _,
        out("v5") _,
        out("v6") _,
        out("v7") _,
        out("v8") _,
        out("v9") _,
        out("v10") _,
        out("v11") _,
        out("v12") _,
        out("v13") _,
        out("v14") _,
        out("v15") _,
        out("v16") _,
        out("v17") _,
        out("v18") _,
        out("v19") _,
        out("v20") _,
        out("v21") _,
        out("v22") _,
        out("v23") _,
        out("v24") _,
        out("v25") _,
        options(nostack),
    );
    std::array::from_fn(|t| {
        std::array::from_fn(|q| {
            let lo = &raw[2 * t];
            let hi = &raw[2 * t + 1];
            let value =
                lo[q] as i64 + 256 * (lo[2 + q] as i64 + hi[q] as i64) + 65536 * hi[2 + q] as i64;
            value.rem_euclid(P as i64) as u16
        })
    })
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn dot_8x2(_: *const i8, _: *const i8, _: [*const i8; TILE], _: usize) -> [[u16; 2]; TILE] {
    panic!("packed NTT requires AArch64 i8mm");
}

pub fn validate(ntt: &Ntt) {
    #[cfg(target_arch = "aarch64")]
    assert!(std::arch::is_aarch64_feature_detected!("i8mm"));
    for v in -12800i16..=12800 {
        let (lo, hi) = split(v);
        assert_eq!(lo as i32 + 256 * hi as i32, v as i32);
        assert!((-50..=50).contains(&hi));
    }
    let mut rng = super::Rng(81002348);
    let mut checks = 0;
    // Directly exercise SMMLA extrema, all plane products, and both widths.
    for width in [32, 64] {
        for a in [-12800i16, -129, -128, -1, 0, 127, 128, 12800] {
            for b in [-12800i16, -129, -128, -1, 0, 127, 128, 12800] {
                let db = pack(&vec![a as u16; width]);
                let mut lo = vec![0; width * 2];
                let mut hi = vec![0; width * 2];
                let (ql, qh) = split(b);
                lo.fill(ql);
                hi.fill(qh);
                // SAFETY: buffers cover the complete selected width; all tile
                // pointers alias a valid read-only record. i8mm checked above.
                let actual =
                    unsafe { dot_8x2(lo.as_ptr(), hi.as_ptr(), [db.as_ptr(); TILE], width / 8) };
                let expected = (width as i64 * a as i64 * b as i64).rem_euclid(P as i64) as u16;
                assert_eq!(actual, [[expected; 2]; TILE]);
                checks += TILE * 2;
            }
        }
    }
    let normal = rng.record();
    let mirror = super::mirror(&normal);
    let queries = [&normal, &mirror];
    let raw: Vec<_> = (0..19).map(|_| rng.record()).collect();
    let db: Vec<_> = raw.iter().map(|r| pack(&ntt.prepare(r))).collect();
    for count in [1, 2] {
        let query = Query::prepare(ntt, &queries[..count]);
        for size in [0, 1, 7, 8, 9, 15, 16, 19] {
            let expected: Vec<_> = raw[..size]
                .iter()
                .flat_map(|d| {
                    queries[..count]
                        .iter()
                        .flat_map(move |q| super::direct_prime(q, d))
                })
                .collect();
            assert_eq!(
                chunk(ntt, &query, &db[..size]),
                expected,
                "SMMLA NTT differs from direct field scores"
            );
            checks += expected.len();
        }
    }
    eprintln!("SMMLA_VALIDATION: {checks} exact field outputs; signed-byte roundtrip for all 25601 representatives, channel widths 32/64, both orientations, 8-record tiles and tails.");
}
