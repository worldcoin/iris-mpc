use super::{
    batch_generator::Batch,
    hawk_job::{Job, JobRequest, JobResult},
    utils::{self, PartyId},
};
use crate::execution::hawk_main::{
    insert::insert, scheduler::parallelize, search::search_single_query_no_match_count, BothEyes,
    HawkActor, HawkMutation, HawkSession, HawkSessionRef, SingleHawkMutation, LEFT, RIGHT,
};
use eyre::{OptionExt, Result};
use itertools::{izip, Itertools};
use std::{future::Future, time::Instant};
use tokio::sync::{mpsc, oneshot};

// Component name for logging purposes.
const COMPONENT: &str = "Hawk-Handle";

/// Handle to manage concurrent interactions with a Hawk actor.
#[derive(Clone, Debug)]
pub struct Handle {
    // Identifier of party participating in MPC protocol.
    party_id: PartyId,

    // Queue of indexation jobs for processing.
    job_queue: mpsc::Sender<Job>,
}

/// Constructors.
impl Handle {
    pub async fn new(party_id: PartyId, mut actor: HawkActor) -> Result<Self> {
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
            HawkSession::state_check([&sessions[LEFT][0], &sessions[RIGHT][0]]).await?;

            Ok(())
        }

        // Initiate sessions with other MPC nodes & perform state consistency check.
        let mut sessions = actor.new_sessions().await?;
        Self::log_info(String::from("Starting State check"));
        HawkSession::state_check([&sessions[LEFT][0], &sessions[RIGHT][0]]).await?;

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
                    Self::log_error(String::from(
                        "HawkActor is in an inconsistent state, therefore stopping.",
                    ));
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
        Self::log_info(format!(
            "Hawk Job :: processing batch-id={}; batch-size={}",
            request.batch_id,
            request.batch_size()
        ));
        let _ = Instant::now();

        // Use all sessions per iris side to search for insertion indices per
        // batch, number configured by `args.request_parallelism`.

        // TODO implement automatic parallelism scaling

        // Iterate per side
        let jobs_per_side = izip!(request.queries.iter(), sessions.iter())
            .map(|(queries_side, sessions_side)| {
                let searcher = actor.searcher();
                let queries_with_ids =
                    izip!(queries_side.clone(), request.vector_ids.clone()).collect_vec();
                let sessions = sessions_side.clone();

                // Per side do searches and insertions
                async move {
                    let n_sessions = sessions.len();
                    let insert_session =
                        sessions.first().ok_or_eyre("Sessions for side are empty")?;
                    let mut connect_plans = Vec::new();

                    // Process queries in a logical insertion batch for this side
                    for queries_batch in queries_with_ids.chunks(n_sessions) {
                        let search_jobs = izip!(queries_batch.iter(), sessions.iter()).map(
                            |((query, _id), session)| {
                                let query = query.clone();
                                let searcher = searcher.clone();
                                let session = session.clone();
                                async move {
                                    search_single_query_no_match_count(session, query, &searcher)
                                        .await
                                }
                            },
                        );

                        let plans = parallelize(search_jobs)
                            .await?
                            .into_iter()
                            .map(Some)
                            .collect_vec();

                        let batch_ids = queries_batch
                            .iter()
                            .map(|(_query, id)| Some(*id))
                            .collect_vec();

                        // Insert into in-memory store, and return insertion plans for use by DB
                        let plans = insert(insert_session, &searcher, plans, &batch_ids).await?;
                        connect_plans.extend(plans);
                    }

                    Ok(connect_plans)
                }
            })
            .collect_vec();

        let results_ = parallelize(jobs_per_side.into_iter()).await?;
        let results: [_; 2] = results_.try_into().unwrap();

        // Convert the results into SingleHawkMutation format
        let [left_plans, right_plans] = results;
        let max_len = left_plans.len().max(right_plans.len());
        let mut mutations = Vec::new();

        for i in 0..max_len {
            let left_plan = left_plans.get(i).cloned().flatten();
            let right_plan = right_plans.get(i).cloned().flatten();

            // Only create mutation if at least one side has a plan
            if left_plan.is_some() || right_plan.is_some() {
                mutations.push(SingleHawkMutation {
                    plans: [left_plan, right_plan],
                    modification_key: None, // Genesis doesn't use modification keys
                });
            }
        }

        Ok(JobResult::new(request, HawkMutation(mutations)))
    }

    // Helper: component error logging.
    fn log_error(msg: String) {
        utils::log_error(COMPONENT, msg);
    }

    // Helper: component logging.
    fn log_info(msg: String) {
        utils::log_info(COMPONENT, msg);
    }

    /// Enqueues a job to process a batch of Iris records pulled from a remote store. It returns
    /// a future that resolves to the processed results.
    ///
    /// # Arguments
    ///
    /// * `batch` - A set of `DbStoredIris` records to be processed.
    ///
    /// # Returns
    ///
    /// A future that resolves to the processed results.
    ///
    /// # Errors
    ///
    /// This method may return an error if the job queue channel is closed or if the job fails.
    pub async fn submit_batch(&mut self, batch: Batch) -> impl Future<Output = Result<JobResult>> {
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
            let result = rx.await??;

            Ok(result)
        }
    }
}
