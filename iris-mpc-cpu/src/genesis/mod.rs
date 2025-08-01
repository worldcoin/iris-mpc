mod batch_generator;
mod hawk_handle;
mod hawk_job;
pub mod plaintext;
pub mod state_accessor;
pub mod state_sync;
pub mod utils;

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
