use super::hawk_job::{Job, JobRequest, JobResult};
use crate::execution::hawk_main::{BothEyes, HawkActor, HawkSession, HawkSessionRef, LEFT, RIGHT};
use eyre::Result;
use futures::try_join;
use iris_mpc_store::DbStoredIris;
use std::{future::Future, time::Instant};
use tokio::sync::{mpsc, oneshot};

/// Handle to manage concurrent interactions with a Hawk actor.
#[derive(Clone, Debug)]
pub struct Handle {
    // Identifier of party participating in MPC protocol.
    party_id: usize,

    // Queue of indexation jobs for processing.
    job_queue: mpsc::Sender<Job>,
}

/// Constructors.
impl Handle {
    pub async fn new(party_id: usize, mut actor: HawkActor) -> Result<Self> {
        /// Performs post job processing health checks:
        /// - resets Hawk sessions upon job failure
        /// - ensures system state is in sync.
        async fn do_health_check(
            actor: &mut HawkActor,
            sessions: &mut BothEyes<Vec<HawkSessionRef>>,
            job_failed: bool,
        ) -> Result<()> {
            // Reset sessions upon an error.
            if job_failed {
                *sessions = actor.new_sessions().await?;
            }

            // Validate the common state after processing the requests.
            try_join!(
                HawkSession::state_check(&sessions[LEFT][0]),
                HawkSession::state_check(&sessions[RIGHT][0]),
            )?;

            Ok(())
        }

        // Initiate sessions with other MPC nodes & perform state consistency check.
        let mut sessions = actor.new_sessions().await?;
        try_join!(
            HawkSession::state_check(&sessions[LEFT][0]),
            HawkSession::state_check(&sessions[RIGHT][0]),
        )?;

        // Process jobs until health check fails or channel closes.
        let (tx, mut rx) = mpsc::channel::<Job>(1);
        tokio::spawn(async move {
            // Processing loop.
            while let Some(job) = rx.recv().await {
                let job_result = Self::handle_job(&mut actor, &sessions, &job.request).await;
                let health = do_health_check(&mut actor, &mut sessions, job_result.is_err()).await;
                let stop = health.is_err();
                let _ = job.return_channel.send(health.and(job_result));
                if stop {
                    tracing::error!("HNSW GENESIS :: Hawk Handle :: HawkActor is in an inconsistent state, therefore stopping.");
                    break;
                }
            }

            // Clean up.
            rx.close();
            while let Some(job) = rx.recv().await {
                let _ = job.return_channel.send(Err(eyre::eyre!("stopping")));
            }
        });

        Ok(Self {
            party_id,
            job_queue: tx,
        })
    }
}

/// Methods.
impl Handle {
    /// Processes a single genesis indexation job by sending it to the Hawk actor.
    ///
    /// # Arguments
    ///
    /// * `actor` - A mutable instance of a Hawk actor.
    /// * `sessions` - Hawk sessions to other MPC nodes.
    /// * `request` - Indexation job to be processed.
    ///
    /// # Returns
    ///
    /// Indexation processing results.
    ///
    pub async fn handle_job(
        _actor: &mut HawkActor,
        _sessions: &BothEyes<Vec<HawkSessionRef>>,
        request: &JobRequest,
    ) -> Result<JobResult> {
        tracing::info!(
            "HNSW GENESIS :: Hawk Handle :: Genesis Hawk job processing ::{} elements within batch",
            request.identifiers.len()
        );
        let _ = Instant::now();

        // TODO implement business logic.

        Ok(JobResult {
            results: Vec::new(),
        })
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
        batch: &[DbStoredIris],
    ) -> impl Future<Output = Result<()>> {
        // Set job queue channel.
        let (tx, rx) = oneshot::channel();

        // Set job.
        let job = Job {
            request: JobRequest::new(self.party_id, batch),
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
            Ok(())
        }
    }
}
