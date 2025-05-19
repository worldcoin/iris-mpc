mod batch_generator;
mod hawk_handle;
mod hawk_job;
mod utils;

pub use batch_generator::{BatchGenerator, BatchIterator};
pub use hawk_handle::Handle;
pub use hawk_job::{Job, JobRequest, JobResult};
use utils::logger;
pub use utils::{
    errors::{handle_error, IndexationError},
    fetcher::{fetch_height_of_indexed, fetch_iris_deletions, set_height_of_indexed},
    logger::{log_error, log_info, log_warn},
};
