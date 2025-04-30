use crate::execution::hawk_main::{
    BothEyes, HawkActor, HawkSession, HawkSessionRef, VecRequests, LEFT, RIGHT,
};
use crate::hawkers::aby3::aby3_store::QueryRef;
use eyre::Result;
use futures::try_join;
use iris_mpc_store::DbStoredIris;
use std::{future::Future, sync::Arc};
use tokio::sync::{mpsc, oneshot};

/// Handle to manage concurrent interactions with a Hawk actor.
#[derive(Clone, Debug)]
#[allow(dead_code)]
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
#[allow(dead_code)]
pub struct JobRequest {
    // Indexation queries over both eyes.
    queries: Arc<BothEyes<VecRequests<QueryRef>>>,
}

/// An indexation job result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobResult {}

/// ---------------------------------------------
/// Constructors.
/// ---------------------------------------------

impl Handle {
    pub async fn new(mut actor: HawkActor) -> Result<Self> {
        // Initiate sessions with other MPC nodes & perform state consistency check.
        let mut sessions = actor.new_sessions().await?;
        try_join!(
            HawkSession::state_check(&sessions[LEFT][0]),
            HawkSession::state_check(&sessions[RIGHT][0]),
        )?;

        // Process jobs over mpsc channel until MPC network health check fails or channel closes.
        let (tx, mut rx) = mpsc::channel::<Job>(1);
        tokio::spawn(async move {
            // Job processing loop.
            while let Some(job) = rx.recv().await {
                let job_result = Self::handle_job(&mut actor, &sessions, &job.request).await;
                let health =
                    Self::health_check(&mut actor, &mut sessions, job_result.is_err()).await;
                let stop = health.is_err();
                let _ = job.return_channel.send(health.and(job_result));
                if stop {
                    tracing::error!("HawkActor is in an inconsistent state, therefore stopping.");
                    break;
                }
            }

            // Clean up.
            rx.close();
            while let Some(job) = rx.recv().await {
                let _ = job.return_channel.send(Err(eyre::eyre!("stopping")));
            }
        });

        Ok(Self { job_queue: tx })
    }
}

/// ---------------------------------------------
/// Convertors.
/// ---------------------------------------------

impl From<&Vec<DbStoredIris>> for JobRequest {
    fn from(_batch: &Vec<DbStoredIris>) -> Self {
        unimplemented!()
    }
}

/// ---------------------------------------------
/// Methods.
/// ---------------------------------------------
#[allow(dead_code)]
impl Handle {
    pub async fn handle_job(
        _actor: &mut HawkActor,
        _sessions: &BothEyes<Vec<HawkSessionRef>>,
        _request: &JobRequest,
    ) -> Result<JobResult> {
        unimplemented!()
    }

    async fn health_check(
        _actor: &mut HawkActor,
        _sessions: &mut BothEyes<Vec<HawkSessionRef>>,
        _job_failed: bool,
    ) -> Result<()> {
        unimplemented!()
    }

    /// Enqueues a job to process a batch of Iris records pulled from a remote store. It returns
    /// a future that resolves to the processed results.
    ///
    /// # Arguments
    ///
    /// * `batch` - A vector of `DbStoredIris` records to be processed.
    ///
    /// # Returns
    ///
    /// A future that resolves to the processed results.
    ///
    /// # Errors
    ///
    /// This method may return an error if the job queue channel is closed or if the job fails.
    pub async fn submit_batch(
        &mut self,
        batch: Vec<DbStoredIris>,
    ) -> impl Future<Output = Result<Vec<u64>>> {
        // Set job queue channel.
        let (tx, rx) = oneshot::channel();

        // Set job.
        let job = Job {
            request: JobRequest::from(&batch),
            return_channel: tx,
        };

        // Enqueue job.
        let sent = self.job_queue.send(job).await;

        // Execute job & await result.
        async move {
            // In a second Future, wait for the result.
            sent?;
            let _result = rx.await??;

            // TODO: Implement job result processing.
            Ok(Vec::new())
        }
    }
}
