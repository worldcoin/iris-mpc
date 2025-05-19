mod batch_generator;
mod hawk_handle;
mod hawk_job;
mod state_accessor;
mod utils;

pub use batch_generator::{BatchGenerator, BatchIterator};
pub use hawk_handle::Handle;
pub use hawk_job::{Job, JobRequest, JobResult};
pub use state_accessor::{fetch_height_of_indexed, fetch_iris_deletions, set_height_of_indexed};
use utils::logger;
pub use utils::{
    errors::{handle_error, IndexationError},
    logger::{log_error, log_info, log_warn},
};
