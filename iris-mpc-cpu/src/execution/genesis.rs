use super::hawk_main::{BothEyes, HawkActor, HawkSessionRef, VecRequests};
use crate::hawkers::aby3::aby3_store::QueryRef;
use eyre::Result;
use std::{future::Future, sync::Arc};
use tokio::sync::{mpsc, oneshot};

/// Handle to manage concurrent interactions with a Hawk actor.
#[derive(Clone, Debug)]
pub struct Handle {
    job_queue: mpsc::Sender<Job>,
}

/// An indexation job that materialises an in-mem graph.
pub struct Job {
    // A request encapsulating data for indexation.
    request: JobRequest,

    // Tokio channel through which job result will be signalled.
    return_channel: oneshot::Sender<Result<JobResult>>,
}

/// An indexation job request.
#[derive(Clone, Debug)]
pub struct JobRequest {
    // Indexation queries over both eyes.
    queries: Arc<BothEyes<VecRequests<QueryRef>>>,
}

/// An indexation job result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {}

// Constructor.
impl Handle {
    pub async fn new(mut actor: HawkActor) -> Result<Self> {
        unimplemented!();
    }
}

// Methods.
impl Handle {
    pub async fn handle_job(
        _actor: &mut HawkActor,
        _sessions: &BothEyes<Vec<HawkSessionRef>>,
        _request: &JobRequest,
    ) -> Result<JobRequest> {
        unimplemented!()
    }

    async fn health_check(
        _actor: &mut HawkActor,
        _sessions: &mut BothEyes<Vec<HawkSessionRef>>,
        _job_failed: bool,
    ) -> Result<()> {
        unimplemented!()
    }

    async fn submit_batch(&mut self, _batch: Vec<u64>) -> impl Future<Output = Result<u64>> {
        async move { unimplemented!() }
    }
}
