#![allow(clippy::needless_range_loop)]
pub mod config;
pub mod error;
pub mod galois;
pub mod galois_engine;
pub mod helpers;
pub mod id;
pub mod iris_db;
pub mod shamir;

pub const IRIS_CODE_LENGTH: usize = 12_800;
pub const MASK_CODE_LENGTH: usize = 6_400;

/// Iris code database type; .0 = iris code, .1 = mask
pub type IrisCodeDb = (Vec<u16>, Vec<u16>);
/// Borrowed version of iris database; .0 = iris code, .1 = mask
pub type IrisCodeDbSlice<'a> = (&'a [u16], &'a [u16]);
