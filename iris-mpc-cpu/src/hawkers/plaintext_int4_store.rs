//! Plaintext `VectorStore` over packed int4 vectors with inner-product distance.
//!
//! Benchmarking/experimentation harness — not a production path and not an MPC
//! mirror. Each vector is 512 signed nibbles in the range `{-7..=7}` packed two
//! per byte. Distance between two vectors is their integer inner product
//! (similarity, not Hamming distance). The store fires a match when the inner
//! product exceeds a configurable threshold.

#[allow(unused_imports)]
use crate::{
    hawkers::shared_irises::{SharedIrises, SharedIrisesRef},
    hnsw::{vector_store::VectorStoreMut, GraphMem, HnswSearcher, SortedNeighborhood, VectorStore},
};
#[allow(unused_imports)]
use aes_prng::AesRng;
#[allow(unused_imports)]
use eyre::{bail, Result};
#[allow(unused_imports)]
use iris_mpc_common::vector_id::VectorId;
#[allow(unused_imports)]
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::sync::Arc;

/// Number of int4 elements in each vector.
pub const INT4_DIM: usize = 512;

/// Bytes per packed vector (two int4 elements per byte).
pub const INT4_PACKED_BYTES: usize = INT4_DIM / 2;

/// 512-element vector of signed 4-bit values in `{-7..=7}` packed two per byte
/// using two's-complement nibbles.
///
/// Byte `i` carries element `2*i` in its low nibble and element `2*i+1` in its
/// high nibble. Encoded nibble values are `0x0..=0x7` (positive 0..7) and
/// `0x9..=0xF` (negative -7..-1). `0x8` (-8) is never produced by `random`; it
/// decodes correctly to -8 if encountered, but no test path generates it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Int4Vector {
    #[serde(with = "BigArray")]
    pub packed: [u8; INT4_PACKED_BYTES],
}

impl Default for Int4Vector {
    fn default() -> Self {
        Self {
            packed: [0u8; INT4_PACKED_BYTES],
        }
    }
}

pub type Int4StoredVector = Arc<Int4Vector>;
pub type Int4SharedVectors = SharedIrises<Int4StoredVector>;
pub type Int4SharedVectorsRef = SharedIrisesRef<Int4StoredVector>;

impl Int4Vector {
    /// Generate a random `Int4Vector` with each element drawn i.i.d. uniformly
    /// from `{-7..=7}`.
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        let mut packed = [0u8; INT4_PACKED_BYTES];
        for byte in packed.iter_mut() {
            let lo = Self::encode_nibble(rng.gen_range(-7..=7));
            let hi = Self::encode_nibble(rng.gen_range(-7..=7));
            *byte = lo | (hi << 4);
        }
        Self { packed }
    }

    /// Decode element at index `i` (0-based).
    ///
    /// Returns a value in `{-8..=7}`. Vectors produced by [`Int4Vector::random`]
    /// only ever yield values in `{-7..=7}`.
    pub fn get(&self, i: usize) -> i8 {
        let byte = self.packed[i / 2];
        let nibble = if i.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        Self::decode_nibble(nibble)
    }

    /// Integer inner product of two int4 vectors.
    ///
    /// Decodes nibbles on the fly and accumulates element-wise products in
    /// `i16`. The tightest bound is `512 * 7 * 7 = 25_088 < i16::MAX = 32_767`,
    /// so the sum never overflows for vectors produced by [`Int4Vector::random`].
    pub fn dot(&self, other: &Self) -> i16 {
        let mut acc: i16 = 0;
        for (a, b) in self.packed.iter().zip(other.packed.iter()) {
            let a_lo = i16::from(Self::decode_nibble(*a & 0x0F));
            let a_hi = i16::from(Self::decode_nibble(*a >> 4));
            let b_lo = i16::from(Self::decode_nibble(*b & 0x0F));
            let b_hi = i16::from(Self::decode_nibble(*b >> 4));
            acc += a_lo * b_lo + a_hi * b_hi;
        }
        acc
    }

    /// Encode a value in `{-7..=7}` (also accepts `-8`) as a 4-bit
    /// two's-complement nibble.
    #[inline]
    fn encode_nibble(value: i8) -> u8 {
        (value as u8) & 0x0F
    }

    /// Decode a 4-bit two's-complement nibble to a signed `i8`.
    #[inline]
    fn decode_nibble(nibble: u8) -> i8 {
        let n = nibble & 0x0F;
        if n & 0x08 != 0 {
            (n as i8) | !0x0F_i8 // sign-extend
        } else {
            n as i8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_prng::AesRng;
    use rand::SeedableRng;

    #[test]
    fn test_int4_pack_roundtrip() {
        let mut rng = AesRng::seed_from_u64(0x1234_5678_9abc_def0);
        let v = Int4Vector::random(&mut rng);
        for i in 0..INT4_DIM {
            let x = v.get(i);
            assert!(
                (-7..=7).contains(&x),
                "element {i} = {x} is outside {{-7..=7}}",
            );
        }

        // Hand-constructed vector: alternating +7 and -7
        let mut packed = [0u8; INT4_PACKED_BYTES];
        for byte in packed.iter_mut() {
            // low nibble = +7 (0x07), high nibble = -7 (0x09)
            *byte = 0x07 | (0x09 << 4);
        }
        let v = Int4Vector { packed };
        for i in 0..INT4_DIM {
            let expected = if i.is_multiple_of(2) { 7 } else { -7 };
            assert_eq!(v.get(i), expected, "mismatch at element {i}");
        }
    }

    fn make_vec_with(value: i8) -> Int4Vector {
        let mut packed = [0u8; INT4_PACKED_BYTES];
        let n = Int4Vector::encode_nibble(value);
        let byte = n | (n << 4);
        for b in packed.iter_mut() {
            *b = byte;
        }
        Int4Vector { packed }
    }

    #[test]
    fn test_dot_known_values() {
        let zero = Int4Vector::default();
        assert_eq!(zero.dot(&zero), 0);

        let all_plus_7 = make_vec_with(7);
        let all_minus_7 = make_vec_with(-7);

        // 512 * 7 * 7 = 25_088
        assert_eq!(all_plus_7.dot(&all_plus_7), 25_088);
        assert_eq!(all_minus_7.dot(&all_minus_7), 25_088);
        assert_eq!(all_plus_7.dot(&all_minus_7), -25_088);

        // Half +7, half -7 in one vector; dotted against all +7 → 0
        let mut split = Int4Vector::default();
        for i in 0..INT4_PACKED_BYTES {
            // first 128 bytes: both nibbles +7; next 128 bytes: both nibbles -7
            let v = if i < INT4_PACKED_BYTES / 2 {
                7_i8
            } else {
                -7_i8
            };
            let n = Int4Vector::encode_nibble(v);
            split.packed[i] = n | (n << 4);
        }
        assert_eq!(split.dot(&all_plus_7), 0);
    }
}
