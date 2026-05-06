//! Compact at-rest encoding for HNSW graph neighborhoods.
//!
//! A neighborhood is a sorted, strictly-increasing slice of `u32` IDs. We
//! delta-encode it (`ids[0]` kept as an absolute value, then each later id as
//! `ids[i] - ids[i-1] - 1`) and Rice-code the resulting `k` values into a
//! packed bitstream. Decoding reverses this: read the header, Rice-decode `k`
//! symbols, then prefix-sum them back into ids.
//!
//! # Wire format
//!
//! ```text
//! +-------------+-----+-----------------------------+
//! | k (u16 LE)  | b   | Rice-coded body, MSB-first  |
//! +-------------+-----+-----------------------------+
//!  bytes 0..2    2     3..
//! ```
//!
//! Each Rice-coded value `v` is written as `q = v >> b` zero bits, a `1`
//! terminator, then the low `b` bits of `v`. The Rice parameter `b` (which is
//! a power-of-two Golomb divisor) is the same for every symbol in a blob and
//! is chosen as described below. The body is zero-padded on the right to a
//! whole number of bytes.
//!
//! # Choosing `b`
//!
//! For values drawn from a geometric distribution with mean `μ`, the
//! bit-optimal Rice parameter is `b ≈ floor(log2(μ · ln 2))`. We approximate
//! this as `floor(log2(μ))` (about half a bit too high on average — at most
//! one extra bit per symbol), where `μ` is the exact mean of the `k` Rice-coded
//! values: `(ids[k-1] - (k-1)) / k`. See `compute_b` for the implementation.
//!
//! # Example
//!
//! Encoding `[0, 5, 9]`:
//!
//! - Delta stream: `0` (= `ids[0]`), `4` (= 5-0-1), `3` (= 9-5-1).
//! - `compute_b`: mean = `(9 - 2)/3 = 2`, so `b = floor(log2(2)) = 1`.
//! - Per symbol, splitting `v` into quotient (unary) | terminator | low `b` bits:
//!     - `0` → `1 | 0`              (2 bits)
//!     - `4` → `00 | 1 | 0`         (4 bits)
//!     - `3` → `0 | 1 | 1`          (3 bits)
//! - Body bitstream `100010011` packs MSB-first (with 7 bits of zero padding)
//!   into `[0x89, 0x80]`.
//! - Full blob: `[0x03, 0x00, 0x01, 0x89, 0x80]` — 5 bytes for 3 ids.
//!
//! See `module_doc_example_bytes` for the test that pins these bytes.

use thiserror::Error;

/// Maximum supported neighborhood size. Limited by the 16-bit `k` field in the header.
pub const MAX_K: usize = u16::MAX as usize;

const HEADER_LEN: usize = 3;

/// Compact at-rest form of a graph neighborhood: Rice-coded gap-encoded bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedNeighborhood {
    bytes: Box<[u8]>,
}

/// Errors returned by [`EncodedNeighborhood::encode`].
#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("input ids are empty")]
    Empty,
    #[error("input ids not sorted strictly ascending at position {0}")]
    NotSorted(usize),
    #[error("neighborhood size {0} exceeds maximum {MAX_K}")]
    TooLarge(usize),
}

/// Errors returned by [`EncodedNeighborhood::decode`].
#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("encoded blob too short: header truncated, have {0} bytes")]
    TruncatedHeader(usize),
    #[error("encoded blob too short: body needs more bits at gap {0}")]
    TruncatedBody(usize),
    #[error("invalid rice parameter b={0} (must be 0..=31)")]
    InvalidParameter(u8),
    #[error("malformed bitstream: quotient overflow at gap {0}")]
    QuotientOverflow(usize),
    #[error("decoded id overflow at gap {0}")]
    IdOverflow(usize),
}

impl EncodedNeighborhood {
    /// Encode a sorted-ascending, strictly-increasing slice of u32 IDs.
    pub fn encode(ids: &[u32]) -> Result<Self, EncodeError> {
        if ids.is_empty() {
            return Err(EncodeError::Empty);
        }
        if ids.len() > MAX_K {
            return Err(EncodeError::TooLarge(ids.len()));
        }
        for i in 1..ids.len() {
            if ids[i] <= ids[i - 1] {
                return Err(EncodeError::NotSorted(i));
            }
        }

        let k = ids.len() as u16;
        let b = compute_b(ids);

        // Per-symbol body cost is `b + 1 + q` bits with `E[q] ≈ 1` for an optimal `b`
        // (geometric tail), so budgeting `b + 4` gives ~2 bits of slack per symbol.
        // A reallocation needs `Σ q_i > 3k`, a `√(2k)`-sigma tail — negligible for
        // any realistic `k`. Empirically (criterion bench across slack ∈ {2,3,4,6,8,16}
        // and k ∈ {10, 450, 2000, 65535}), enlarging the slack does not change encode
        // timing within noise, so `+ 4` is not undersized.
        let cap = HEADER_LEN + (ids.len() * (b as usize + 4)).div_ceil(8);
        let mut writer = BitWriter::with_capacity(cap);

        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(&k.to_le_bytes());
        header.push(b);

        rice_encode(&mut writer, ids[0], b);
        for i in 1..ids.len() {
            let gap = ids[i] - ids[i - 1] - 1;
            rice_encode(&mut writer, gap, b);
        }

        let body = writer.finish();
        let mut out = header;
        out.extend_from_slice(&body);
        Ok(Self {
            bytes: out.into_boxed_slice(),
        })
    }

    /// Wrap an already-encoded byte blob (e.g. loaded from storage) without
    /// validating its contents; validation happens on `decode`.
    pub fn from_bytes(bytes: Box<[u8]>) -> Self {
        Self { bytes }
    }

    /// Return the underlying encoded bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Decode back to a sorted-ascending `Vec<u32>` of neighbor IDs.
    pub fn decode(&self) -> Result<Vec<u32>, DecodeError> {
        if self.bytes.len() < HEADER_LEN {
            return Err(DecodeError::TruncatedHeader(self.bytes.len()));
        }
        let k = u16::from_le_bytes([self.bytes[0], self.bytes[1]]) as usize;
        let b = self.bytes[2];
        if b > 31 {
            return Err(DecodeError::InvalidParameter(b));
        }
        if k == 0 {
            return Ok(vec![]);
        }

        let mut reader = BitReader::new(&self.bytes[HEADER_LEN..]);
        let mut out = Vec::with_capacity(k);

        let first = rice_decode(&mut reader, b, 0)?;
        out.push(first);
        let mut prev = first;
        for i in 1..k {
            let gap = rice_decode(&mut reader, b, i)?;
            let next = prev
                .checked_add(gap)
                .and_then(|x| x.checked_add(1))
                .ok_or(DecodeError::IdOverflow(i))?;
            out.push(next);
            prev = next;
        }
        Ok(out)
    }
}

/// MSB-first bit writer that flushes whole bytes to an internal `Vec<u8>`.
struct BitWriter {
    bytes: Vec<u8>,
    /// Buffer holding up-to-7 not-yet-flushed bits in its most-significant positions.
    buf: u8,
    /// Number of valid bits in `buf` (0..8).
    nbits: u8,
}

impl BitWriter {
    fn with_capacity(cap_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(cap_bytes),
            buf: 0,
            nbits: 0,
        }
    }

    /// Write the low `n` bits of `value` MSB-first. `n` must be 0..=32.
    fn write_bits(&mut self, value: u32, n: u8) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            self.buf = (self.buf << 1) | bit;
            self.nbits += 1;
            if self.nbits == 8 {
                self.bytes.push(self.buf);
                self.buf = 0;
                self.nbits = 0;
            }
        }
    }

    fn write_one(&mut self) {
        self.write_bits(1, 1);
    }

    fn write_zeros(&mut self, n: u32) {
        // Write up to 32 bits at a time so very large quotients don't
        // recursively explode through `write_bits`.
        let mut remaining = n;
        while remaining > 0 {
            let chunk = remaining.min(32) as u8;
            self.write_bits(0, chunk);
            remaining -= chunk as u32;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.buf <<= pad;
            self.bytes.push(self.buf);
        }
        self.bytes
    }
}

/// MSB-first bit reader over a byte slice. Tracks position in bits.
struct BitReader<'a> {
    bytes: &'a [u8],
    /// Bit offset from the start of `bytes` (MSB of byte 0 is bit 0).
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn total_bits(&self) -> usize {
        self.bytes.len() * 8
    }

    fn read_bits(&mut self, n: u8) -> Option<u32> {
        debug_assert!(n <= 32);
        if self.pos + n as usize > self.total_bits() {
            return None;
        }
        let mut out: u32 = 0;
        for _ in 0..n {
            let byte = self.bytes[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            out = (out << 1) | bit as u32;
            self.pos += 1;
        }
        Some(out)
    }

    fn read_unary(&mut self) -> Option<u32> {
        let mut zeros: u32 = 0;
        loop {
            let bit = self.read_bits(1)?;
            if bit == 1 {
                return Some(zeros);
            }
            zeros = zeros.checked_add(1)?;
        }
    }
}

/// Pick the Rice parameter `b` used by [`rice_encode`] / [`rice_decode`].
///
/// Rice coding is the power-of-two case of Golomb coding: a value `v` is split
/// into `q = v >> b` (written in unary) and the low `b` bits (written raw), so
/// each symbol costs `q + 1 + b` bits. For values drawn from a geometric
/// distribution with mean `μ`, the bit-optimal choice is
/// `b ≈ floor(log2(μ · ln 2))`, which we approximate here as `floor(log2(μ))`
/// (about half a bit too high on average — at most one extra bit per symbol).
///
/// `μ` is the mean of the `k` values actually Rice-coded by [`encode`]: the
/// absolute first id plus the `(k-1)` inter-element gaps `ids[i] - ids[i-1] - 1`,
/// which sum to `ids[k-1] - (k-1)`.
fn compute_b(ids: &[u32]) -> u8 {
    debug_assert!(!ids.is_empty());
    let k = ids.len() as u32;
    // `ids[k-1] >= k-1` for any strictly-increasing u32 slice, so this never underflows.
    let mean_gap = ((ids[ids.len() - 1] - (k - 1)) / k).max(1);
    // floor(log2(mean_gap)) for mean_gap >= 1; result is always in [0, 31].
    31u8.saturating_sub(mean_gap.leading_zeros() as u8)
}

fn rice_encode(writer: &mut BitWriter, value: u32, b: u8) {
    let q = if b == 0 { value } else { value >> b };
    writer.write_zeros(q);
    writer.write_one();
    if b > 0 {
        let mask = (1u32 << b) - 1;
        writer.write_bits(value & mask, b);
    }
}

fn rice_decode(reader: &mut BitReader<'_>, b: u8, gap_idx: usize) -> Result<u32, DecodeError> {
    debug_assert!(b <= 31, "rice parameter b must be 0..=31");
    let q = reader
        .read_unary()
        .ok_or(DecodeError::TruncatedBody(gap_idx))?;
    // `q << b` must fit in u32: the bit-width of `q` plus `b` cannot exceed 32.
    let bits_in_q = 32u8.saturating_sub(q.leading_zeros() as u8);
    if bits_in_q + b > 32 {
        return Err(DecodeError::QuotientOverflow(gap_idx));
    }
    let r = if b == 0 {
        0
    } else {
        reader
            .read_bits(b)
            .ok_or(DecodeError::TruncatedBody(gap_idx))?
    };
    Ok((q << b) | r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_doc_example_bytes() {
        let encoded = EncodedNeighborhood::encode(&[0u32, 5, 9]).unwrap();
        let actual: Vec<u8> = encoded.as_bytes().to_vec();
        eprintln!("BYTES = {actual:02x?}");
        assert_eq!(actual, vec![0x03, 0x00, 0x01, 0x89, 0x80]);
    }

    #[test]
    fn rejects_empty() {
        let err = EncodedNeighborhood::encode(&[]).unwrap_err();
        assert!(matches!(err, EncodeError::Empty), "got {:?}", err);
    }

    #[test]
    fn rejects_unsorted() {
        let err = EncodedNeighborhood::encode(&[5, 3]).unwrap_err();
        assert!(matches!(err, EncodeError::NotSorted(1)), "got {:?}", err);
    }

    #[test]
    fn rejects_duplicates() {
        let err = EncodedNeighborhood::encode(&[5, 5]).unwrap_err();
        assert!(matches!(err, EncodeError::NotSorted(1)), "got {:?}", err);
    }

    #[test]
    fn rejects_too_large() {
        let oversized: Vec<u32> = (0..(MAX_K as u32 + 1)).collect();
        let err = EncodedNeighborhood::encode(&oversized).unwrap_err();
        match err {
            EncodeError::TooLarge(n) => assert_eq!(n, MAX_K + 1),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn single_element_round_trip_zero() {
        let encoded = EncodedNeighborhood::encode(&[0]).expect("encode");
        let decoded = encoded.decode().expect("decode");
        assert_eq!(decoded, vec![0u32]);
    }

    #[test]
    fn single_element_round_trip_small() {
        let encoded = EncodedNeighborhood::encode(&[42]).expect("encode");
        assert_eq!(decoded_or_panic(&encoded), vec![42u32]);
    }

    #[test]
    fn single_element_round_trip_max() {
        let encoded = EncodedNeighborhood::encode(&[u32::MAX]).expect("encode");
        assert_eq!(decoded_or_panic(&encoded), vec![u32::MAX]);
    }

    #[test]
    fn round_trip_consecutive() {
        let ids: Vec<u32> = (0..10).collect();
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
        assert_eq!(encoded.decode().expect("decode"), ids);
    }

    #[test]
    fn round_trip_handcrafted_small() {
        let ids = vec![0u32, 1, 5, 100, 1000];
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
        assert_eq!(encoded.decode().expect("decode"), ids);
    }

    #[test]
    fn round_trip_widely_spaced() {
        let ids = vec![0u32, 1_000, 1_000_000, 100_000_000, u32::MAX - 1];
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
        assert_eq!(encoded.decode().expect("decode"), ids);
    }

    #[test]
    fn round_trip_max_k() {
        // Sample a strictly-increasing sequence at MAX_K length.
        // Evenly spaced across u32 to exercise a typical mean_gap.
        let k = MAX_K;
        let step = (u32::MAX as u64 / k as u64) as u32;
        let mut ids: Vec<u32> = (0..k).map(|i| (i as u32).saturating_mul(step)).collect();
        // Ensure strict monotonicity (saturating multiply can collide near u32::MAX).
        for i in 1..ids.len() {
            if ids[i] <= ids[i - 1] {
                ids[i] = ids[i - 1] + 1;
            }
        }
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode max_k");
        assert_eq!(encoded.decode().expect("decode max_k"), ids);
    }

    #[test]
    fn round_trip_high_universe() {
        // IDs clustered near u32::MAX.
        let ids: Vec<u32> = (0..450u32).map(|i| u32::MAX - 449 + i).collect();
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode high");
        assert_eq!(encoded.decode().expect("decode high"), ids);
    }

    #[test]
    fn decode_truncated_header() {
        for len in 0..HEADER_LEN {
            let bytes: Box<[u8]> = vec![0u8; len].into_boxed_slice();
            let e = EncodedNeighborhood::from_bytes(bytes);
            match e.decode() {
                Err(DecodeError::TruncatedHeader(n)) => assert_eq!(n, len),
                other => panic!("expected TruncatedHeader({len}), got {other:?}"),
            }
        }
    }

    #[test]
    fn decode_invalid_b() {
        // k=0 so body is empty; b=32 is illegal.
        let bytes: Box<[u8]> = vec![0u8, 0u8, 32u8].into_boxed_slice();
        let e = EncodedNeighborhood::from_bytes(bytes);
        match e.decode() {
            Err(DecodeError::InvalidParameter(32)) => {}
            other => panic!("expected InvalidParameter(32), got {other:?}"),
        }
    }

    #[test]
    fn decode_truncated_body() {
        let ids = vec![0u32, 100, 200, 300];
        let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
        let full = encoded.as_bytes();
        // Truncate every prefix that is longer than the header but shorter than the full body.
        for cut in HEADER_LEN..full.len() {
            let bytes: Box<[u8]> = full[..cut].to_vec().into_boxed_slice();
            let e = EncodedNeighborhood::from_bytes(bytes);
            match e.decode() {
                Err(DecodeError::TruncatedBody(_))
                | Err(DecodeError::QuotientOverflow(_))
                | Err(DecodeError::IdOverflow(_))
                | Err(DecodeError::TruncatedHeader(_)) => {}
                Ok(decoded) => panic!("decoded a truncated blob ok: cut={cut}, got {decoded:?}"),
                Err(other) => panic!("unexpected error at cut={cut}: {other:?}"),
            }
        }
    }

    #[test]
    fn decode_garbage_does_not_panic() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
        for _ in 0..200 {
            let len = rng.gen_range(0..64);
            let bytes: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let e = EncodedNeighborhood::from_bytes(bytes.into_boxed_slice());
            // We only require that decode does not panic. Either Ok or any DecodeError is acceptable.
            let _ = e.decode();
        }
    }

    #[test]
    fn random_round_trip_many_seeds() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        for seed in 0..32u64 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            // Vary k across a useful range, including small and near-MAX_K cases.
            let k = match seed % 4 {
                0 => rng.gen_range(1..16),
                1 => rng.gen_range(16..1024),
                2 => rng.gen_range(1024..16_384),
                _ => 65_535,
            };
            // Vary the universe to exercise different `b` values.
            let universe: u32 = match seed % 3 {
                0 => 1_000_000,
                1 => 100_000_000,
                _ => u32::MAX,
            };

            // Sample k unique values, sort, deduplicate.
            let mut set = std::collections::BTreeSet::new();
            while set.len() < k {
                set.insert(rng.gen_range(0..universe));
            }
            let ids: Vec<u32> = set.into_iter().collect();

            let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
            let decoded = encoded.decode().expect("decode");
            assert_eq!(decoded, ids, "seed={seed} k={k} universe={universe}");
        }
    }

    #[test]
    fn encoded_size_matches_model() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let k: usize = 450;
        let cases: &[(u32, u8)] = &[
            (1_000_000, 11),   // log2(1M/450) ~= 11
            (10_000_000, 14),  // log2(10M/450) ~= 14
            (100_000_000, 17), // log2(100M/450) ~= 17
            (u32::MAX, 23),    // log2(2^32/450) ~= 23
        ];

        for &(universe, expected_b) in cases {
            let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE_F00D ^ universe as u64);
            let mut set = std::collections::BTreeSet::new();
            while set.len() < k {
                set.insert(rng.gen_range(0..universe));
            }
            let ids: Vec<u32> = set.into_iter().collect();

            let encoded = EncodedNeighborhood::encode(&ids).expect("encode");
            let actual_b = encoded.as_bytes()[2];
            assert!(
                actual_b.abs_diff(expected_b) <= 1,
                "universe={universe}: expected_b={expected_b}, got {actual_b}",
            );

            let model_bytes = (k * (actual_b as usize + 2)).div_ceil(8) + HEADER_LEN;
            let actual_bytes = encoded.as_bytes().len();
            let lower = (model_bytes as f64 * 0.90) as usize;
            let upper = (model_bytes as f64 * 1.10) as usize;
            assert!(
                (lower..=upper).contains(&actual_bytes),
                "universe={universe}: actual={actual_bytes}, model={model_bytes}, b={actual_b}",
            );

            // Round-trip sanity check.
            assert_eq!(encoded.decode().expect("decode"), ids);
        }
    }

    fn decoded_or_panic(e: &EncodedNeighborhood) -> Vec<u32> {
        e.decode().expect("decode succeeded")
    }
}
