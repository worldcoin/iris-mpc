mod batch_generator;
mod hawk_handle;
mod hawk_job;
pub mod plaintext;
pub mod state_accessor;
pub mod state_sync;
pub mod utils;

use serde::{Deserialize, Serialize};
use std::fmt;

pub use batch_generator::{Batch, BatchGenerator, BatchIterator, BatchSize};
pub use hawk_handle::Handle;
pub use hawk_job::{Job, JobRequest, JobResult};
pub use state_accessor::{
    get_iris_deletions, get_last_indexed_iris_id, get_last_indexed_modification_id,
    set_last_indexed_iris_id, set_last_indexed_modification_id,
};
pub use utils::logger;
pub use utils::{
    errors::IndexationError,
    logger::{log_error, log_info, log_warn},
};

/// Configuration for batch sizing during genesis indexation.
///
/// Supports two modes:
/// - `Static`: Fixed batch size for all batches
/// - `Dynamic`: Batch size grows with graph size, capped at a maximum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchSizeConfig {
    /// Fixed batch size.
    Static { size: usize },
    /// Dynamic batch size with formula: `min(N/(M*r-1)+1, cap)`
    /// where N = graph size, M = HNSW M parameter, r = error_rate.
    Dynamic { cap: usize, error_rate: usize },
}

impl fmt::Display for BatchSizeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BatchSizeConfig::Static { size } => write!(f, "static:{}", size),
            BatchSizeConfig::Dynamic { cap, error_rate } => {
                write!(f, "dynamic:cap={},error_rate={}", cap, error_rate)
            }
        }
    }
}
