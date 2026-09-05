//! Exact spectral matching over F_52201 and private conversion of existing iris shares.
//!
//! The input conversion uses the iris encoding's ternary code and binary mask
//! domains. It must receive original Galois-ring shares, before query preprocessing.

mod conversion;
pub mod persistence;
mod threshold;
mod transform;

pub use conversion::{convert_irises, FieldIris};
pub use threshold::anon_stats_greater_than;
pub use transform::{score_chunk, SpectralIris, SpectralQuery};

/// The smallest prime supporting a 200-point transform and the exact combined anonymous-statistics score.
pub const MODULUS: u16 = 52_201;

#[inline]
pub(crate) fn reduce(value: i64) -> u16 {
    value.rem_euclid(i64::from(MODULUS)) as u16
}
