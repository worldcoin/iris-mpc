pub mod distance_comparator;
pub mod share_db;

use std::collections::HashMap;

pub const IRIS_CODE_LENGTH: usize = iris_mpc_common::IRIS_CODE_LENGTH;
pub const MASK_CODE_LENGTH: usize = iris_mpc_common::MASK_CODE_LENGTH;
pub const ROTATIONS: usize = iris_mpc_common::ROTATIONS;

pub const THRESHOLD_B: u32 = 1u32 << 16;
pub const MATCHING_THRESHOLD: f64 = iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
pub const THRESHOLD_A: u32 = ((1.0 - 2.0 * MATCHING_THRESHOLD) * (THRESHOLD_B as f64)) as u32;
pub const MATCHING_THRESHOLD_ANON_STATS: f64 =
    iris_mpc_common::iris_db::iris::MATCH_THRESHOLD_RATIO;
pub const THRESHOLD_ANON_STATS_A: u32 =
    ((1.0 - 2.0 * MATCHING_THRESHOLD_ANON_STATS) * (THRESHOLD_B as f64)) as u32;

/// Type alias for partial results with rotations: query_id -> db_id -> list of matching rotations
pub type PartialResultsWithRotations = HashMap<u32, HashMap<u32, Vec<i8>>>;
