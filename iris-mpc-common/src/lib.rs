#![allow(clippy::needless_range_loop)]
pub mod config;
pub mod error;
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

// Count of MPC protocol parties.
pub const PARTY_COUNT: usize = 3;

pub const IRIS_CODE_LENGTH: usize = 12_800;
pub const MASK_CODE_LENGTH: usize = 6_400;
pub const ROTATIONS: usize = 31;

/// Type alias: Ordinal identifier of an MPC participant.
pub type PartyIdx = usize;

/// MPC party ordinal identifiers.
pub const PARTY_IDX_0: PartyIdx = 0;
pub const PARTY_IDX_1: PartyIdx = 1;
pub const PARTY_IDX_2: PartyIdx = 2;

/// MPC party ordinal identifier set.
pub const PARTY_IDX_SET: [PartyIdx; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];

/// Iris code database type; .0 = iris code, .1 = mask
pub type IrisCodeDb = (Vec<u16>, Vec<u16>);
/// Borrowed version of iris database; .0 = iris code, .1 = mask
pub type IrisCodeDbSlice<'a> = (&'a [u16], &'a [u16]);

pub use vector_id::SerialId as IrisSerialId;
pub use vector_id::VectorId as IrisVectorId;
pub use vector_id::VersionId as IrisVersionId;
