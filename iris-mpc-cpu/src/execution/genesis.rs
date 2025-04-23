use super::hawk_main::{BothEyes, VecRequests};
use crate::hawkers::aby3::aby3_store::QueryRef;
use eyre::Result;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Handle managing concurrent interactions with Hawk actor.
#[derive(Clone, Debug)]
pub struct Handle {
    job_queue: mpsc::Sender<Job>,
}

/// Encapsulates an indexation job that materialises an in-mem graph.
pub struct Job {
    // A request encapsulating data for indexation.
    request: JobRequest,

    // Tokio channel through which job result will be signalled.
    return_channel: oneshot::Sender<Result<JobResult>>,
}

/// Encapsulates indexation job request information.
#[derive(Clone, Debug)]
pub struct JobRequest {
    // Indexation queries over both eyes.
    queries: Arc<BothEyes<VecRequests<QueryRef>>>,
}

/// Encapsulates indexation job result information.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {}
