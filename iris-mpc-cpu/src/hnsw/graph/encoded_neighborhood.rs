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
    pub fn encode(ids: &[u32]) -> Result<Self, EncodeError> {
        if ids.len() > MAX_K {
            return Err(EncodeError::TooLarge(ids.len()));
        }
        for i in 1..ids.len() {
            if ids[i] <= ids[i - 1] {
                return Err(EncodeError::NotSorted(i));
            }
        }

        let k = ids.len() as u16;
        if ids.is_empty() {
            let mut out = Vec::with_capacity(HEADER_LEN);
            out.extend_from_slice(&k.to_le_bytes());
            out.push(0);
            return Ok(Self {
                bytes: out.into_boxed_slice(),
            });
        }

        let b = compute_b(ids);

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

    pub fn from_bytes(bytes: Box<[u8]>) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

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
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            buf: 0,
            nbits: 0,
        }
    }

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

fn compute_b(ids: &[u32]) -> u8 {
    debug_assert!(!ids.is_empty());
    let mean_gap: u32 = if ids.len() == 1 {
        ids[0].max(1)
    } else {
        // Strict-ascending guarantees `last >= first + (k-1)`, so the division yields >= 1.
        let span = ids[ids.len() - 1] - ids[0];
        let denom = (ids.len() as u32) - 1;
        (span / denom).max(1)
    };
    // floor(log2(mean_gap)) for mean_gap >= 1.
    let b = 31u8.saturating_sub(mean_gap.leading_zeros() as u8);
    b.min(31)
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
    if b == 32 {
        // Defense-in-depth: `compute_b` clamps to 31, but if a future change
        // ever loosens that, keep the shift well-defined.
        return Ok(r);
    }
    Ok((q << b) | r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_round_trip() {
        let encoded = EncodedNeighborhood::encode(&[]).expect("encode empty");
        let decoded = encoded.decode().expect("decode empty");
        assert!(decoded.is_empty());
        assert_eq!(encoded.as_bytes().len(), HEADER_LEN);
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

    fn decoded_or_panic(e: &EncodedNeighborhood) -> Vec<u32> {
        e.decode().expect("decode succeeded")
    }
}
