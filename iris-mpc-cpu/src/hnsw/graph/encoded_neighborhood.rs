use thiserror::Error;

/// Maximum supported neighborhood size. Limited by the 16-bit `k` field in the header.
pub const MAX_K: usize = u16::MAX as usize;

const HEADER_LEN: usize = 3;

/// Compact at-rest form of a graph neighborhood: Rice-coded gap-encoded bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedNeighborhood {
    bytes: Box<[u8]>,
}

#[derive(Debug, Error)]
pub enum EncodeError {
    #[error("input ids not sorted strictly ascending at position {0}")]
    NotSorted(usize),
    #[error("neighborhood size {0} exceeds maximum {MAX_K}")]
    TooLarge(usize),
}

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
        let k = ids.len() as u16;
        let b: u8 = 0;
        let mut out = Vec::with_capacity(HEADER_LEN);
        out.extend_from_slice(&k.to_le_bytes());
        out.push(b);
        // Body is empty for k=0; later tasks will append Rice-coded gaps here.
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
            return Ok(Vec::new());
        }
        // Body decoding lands in a later task.
        unimplemented!("body decode not implemented yet for k > 0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_round_trip() {
        let encoded = EncodedNeighborhood::encode(&[]).expect("encode empty");
        let decoded = encoded.decode().expect("decode empty");
        assert_eq!(decoded, Vec::<u32>::new());
        assert_eq!(encoded.as_bytes().len(), HEADER_LEN);
    }
}
