use super::hawk_job::{Job, JobRequest, JobResult};
use crate::{
    execution::hawk_main::{
        insert, join_plans, scheduler::parallelize, search::search_single_query_no_match_count, BothEyes, HawkActor, HawkSession, HawkSessionRef, InsertPlan, LEFT, RIGHT
    },
    genesis::utils::types::IrisIdentifier,
    hawkers::aby3::aby3_store::QueryRef as Aby3QueryRef,
    hnsw::HnswSearcher,
};
use eyre::{ContextCompat, OptionExt, Result};
use futures::try_join;
use iris_mpc_store::DbStoredIris;
use itertools::{izip, Itertools};
use std::{future::Future, time::Instant};
use tokio::{sync::{mpsc, oneshot}, task::JoinSet};

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
        actor: &mut HawkActor,
        sessions: &BothEyes<Vec<HawkSessionRef>>,
        request: &JobRequest,
    ) -> Result<JobResult> {
        tracing::info!(
            "Genesis Hawk job processing ::{} elements within batch",
            request.identifiers.len()
        );
        let _ = Instant::now();

        // Use all sessions per iris side to search for insertion indices per
        // batch, number configured by `args.request_parallelism`.

        // TODO implement automatic parallelism scaling

        // ----------

        // 1. Iterate through chunks of requests

        let mut join: JoinSet<Result<()>> = JoinSet::new();
        for (queries_side, sessions_side) in izip!(request.queries.iter(), sessions.iter()) {
            let searcher = actor.searcher();
            let queries = queries_side.clone();
            let sessions = sessions_side.clone();

            // Per side do searches and insertions
            join.spawn(async move {
                let n_sessions = sessions.len();
                let insert_session = sessions.first().ok_or_eyre("Sessions for side are empty")?;

                for queries_batch in queries.chunks(n_sessions) {
                    let search_jobs = izip!(queries_batch.iter(), sessions.iter())
                        .map(|(query, session)| {
                            let query = query.clone();
                            let searcher = searcher.clone();
                            let session = session.clone();
                            async move {
                                search_single_query_no_match_count(session, query, &searcher).await
                            }
                        });
                    
                    let plans: Vec<Option<_>> = parallelize(search_jobs)
                        .await?
                        .into_iter()
                        .map(Some)
                        .collect();

                    insert(plans, &searcher, insert_session).await?;
                }

                Ok(())
            });
        }

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
