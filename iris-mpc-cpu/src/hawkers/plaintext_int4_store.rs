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
    hnsw::{
        vector_store::VectorStoreMut, GraphMem, HnswSearcher, SortedNeighborhood, VectorStore,
    },
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
