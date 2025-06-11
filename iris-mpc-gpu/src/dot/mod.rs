pub mod distance_comparator;
pub mod share_db;

use std::collections::HashMap;

pub const IRIS_CODE_LENGTH: usize = iris_mpc_common::IRIS_CODE_LENGTH;
pub const MASK_CODE_LENGTH: usize = iris_mpc_common::MASK_CODE_LENGTH;
pub const ROTATIONS: usize = iris_mpc_common::ROTATIONS;

/// Type alias for partial results with rotations: query_id -> db_id -> list of matching rotations
pub type PartialResultsWithRotations = HashMap<u32, HashMap<u32, Vec<i8>>>;
