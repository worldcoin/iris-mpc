#![allow(clippy::needless_range_loop)]
pub mod anon_stats;
pub mod config;
pub mod error;
pub mod fast_metrics;
pub mod galois;
pub mod galois_engine;
pub mod helpers;
pub mod id;
pub mod iris_db;
pub mod job;
pub mod postgres;
pub mod server_coordination;
pub mod shamir;
#[cfg(feature = "helpers")]
pub mod test;
pub mod tracing;
pub mod vector_id;

pub const IRIS_CODE_LENGTH: usize = 12_800;
pub const MASK_CODE_LENGTH: usize = 6_400;
pub const ROTATIONS: usize = 31;

pub const PRE_PROC_ROW_PADDING: usize = 120;
pub const IRIS_CODE_ROWS: usize = 16;
// 16 = 12800 / 800 = (IRIS_CODE_LENGTH) / (CODE_COLS * 4)
pub const PRE_PROC_IRIS_CODE_LENGTH: usize =
    IRIS_CODE_LENGTH + (IRIS_CODE_ROWS * PRE_PROC_ROW_PADDING);
pub const PRE_PROC_MASK_CODE_LENGTH: usize = MASK_CODE_LENGTH + (8 * PRE_PROC_ROW_PADDING);

/// Iris code database type; .0 = iris code, .1 = mask
pub type IrisCodeDb = (Vec<u16>, Vec<u16>);
/// Borrowed version of iris database; .0 = iris code, .1 = mask
pub type IrisCodeDbSlice<'a> = (&'a [u16], &'a [u16]);

pub use vector_id::SerialId as IrisSerialId;
pub use vector_id::VectorId as IrisVectorId;
pub use vector_id::VersionId as IrisVersionId;
