mod batch_generator;
mod hawk_handle;
mod hawk_job;
pub mod state_accessor;
pub mod sync;
pub mod utils;

pub use batch_generator::{BatchGenerator, BatchIterator, Batch};
pub use hawk_handle::Handle;
pub use hawk_job::{Job, JobRequest, JobResult};
pub use state_accessor::{fetch_iris_deletions, get_last_indexed, set_last_indexed};
pub use utils::logger;
pub use utils::{
    errors::{handle_error, IndexationError},
    logger::{log_error, log_info, log_warn},
};
