use super::{
    hawk_job::{Job, JobRequest, JobResult, SYNC_DONE, SYNC_ERROR, SYNC_RUNNING},
    utils,
};
use crate::{
    execution::hawk_main::{
        insert::insert,
        iris_worker::{IrisWorkerPool, QueryId},
        scheduler::parallelize,
        search::search_single_query_no_match_count,
        BothEyes, HawkActor, HawkSession, LEFT, RIGHT, STORE_IDS,
    },
    hawkers::aby3::aby3_store::Aby3Query,
};
use eyre::{bail, eyre, OptionExt, Result};
use iris_mpc_common::helpers::smpc_request;
use itertools::{izip, Itertools};
use std::{
    future::Future,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{self, mpsc, oneshot};

// Component name for logging purposes.
const COMPONENT: &str = "Hawk-Handle";

/// Maximum time to wait for all parties to complete the sync_peers exchange.
/// This bounds the retry loop inside `HawkSession::sync_peers` so that
/// persistent failures (crashed peer, broken connection) surface as errors
/// rather than hanging indefinitely.
const SYNC_PEERS_TIMEOUT: Duration = Duration::from_secs(5 * 60);

/// Handle to manage concurrent interactions with a Hawk actor.
#[derive(Clone, Debug)]
pub struct Handle {
    // Queue of indexation jobs for processing.
    job_queue: mpsc::Sender<Job>,
}

/// Constructors.
impl Handle {
    pub async fn new(mut actor: HawkActor) -> Result<Self> {
        /// Performs post job processing health checks:
        /// - resets Hawk sessions upon job failure
        /// - ensures system state is in sync.
        async fn do_health_check(
            actor: &mut HawkActor,
            sessions: &mut BothEyes<Vec<HawkSession>>,
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
            while let Some(Job {
                request,
                return_channel,
            }) = rx.recv().await
            {
                let job_result = Self::handle_job(&mut actor, &sessions, request).await;
                let health = do_health_check(&mut actor, &mut sessions, job_result.is_err()).await;
                let stop = health.is_err();
                let _ = return_channel.send(health.and(job_result));
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

        Ok(Self { job_queue: tx })
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
        sessions: &BothEyes<Vec<HawkSession>>,
        request: JobRequest,
    ) -> Result<(sync::oneshot::Receiver<()>, JobResult)> {
        let now = Instant::now();
        let (done_tx, done_rx) = sync::oneshot::channel();

        match request {
            JobRequest::BatchIndexation {
                batch_id,
                vector_ids,
                queries,
                vector_ids_to_persist,
                irises_to_cache,
            } => {
                Self::log_info(format!(
                    "Hawk Job :: processing batch-id={}; batch-size={}",
                    batch_id,
                    vector_ids.len(),
                ));

                // Cache all batch irises in both worker pools before search.
                // Both pools need all irises (mirror queries cross eyes).
                let all_to_cache: Vec<_> = irises_to_cache[LEFT]
                    .iter()
                    .chain(irises_to_cache[RIGHT].iter())
                    .cloned()
                    .collect();
                futures::try_join!(
                    actor.worker_pools[LEFT].cache_queries(all_to_cache.clone()),
                    actor.worker_pools[RIGHT].cache_queries(all_to_cache),
                )?;

                // Use all sessions per iris side to search for insertion indices per
                // batch, number configured by `args.request_parallelism`.

                // Iterate per side
                let jobs_per_side = izip!(STORE_IDS, queries.iter(), sessions.iter())
                    .map(|(side, queries_side, sessions_side)| {
                        let searcher = actor.searcher();
                        let queries_with_ids =
                            izip!(queries_side.clone(), vector_ids.clone()).collect_vec();
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
                                    |((query, id), session)| {
                                        let query = *query;
                                        let searcher = searcher.clone();
                                        let session = session.clone();
                                        let identifier = (*id, side);
                                        async move {
                                            search_single_query_no_match_count(
                                                session,
                                                query,
                                                &searcher,
                                                &identifier,
                                            )
                                            .await
                                        }
                                    },
                                );
                                let start = Instant::now();
                                let plans = parallelize(search_jobs)
                                    .await?
                                    .into_iter()
                                    .map(Some)
                                    .collect_vec();
                                metrics::histogram!("genesis_all_searches_duration")
                                    .record(start.elapsed().as_secs_f64());

                                let batch_ids = queries_batch
                                    .iter()
                                    .map(|(_query, id)| Some(*id))
                                    .collect_vec();

                                // Insert into in-memory store, and return insertion plans for use by DB
                                {
                                    let start = Instant::now();
                                    let mut store = insert_session.aby3_store.write().await;
                                    let mut graph = insert_session.graph_store.write().await;

                                    let plans = insert(
                                        &mut *store,
                                        &mut *graph,
                                        &searcher,
                                        plans,
                                        &batch_ids,
                                    )
                                    .await?;
                                    metrics::histogram!("genesis_insert_duration")
                                        .record(start.elapsed().as_secs_f64());
                                    connect_plans.extend(plans);
                                }
                            }

                            Ok(connect_plans)
                        }
                    })
                    .collect_vec();

                let results_ = parallelize(jobs_per_side.into_iter()).await?;
                let results: [_; 2] = results_.try_into().unwrap();

                // Convert the results into SingleHawkMutation format
                let [left_plans, right_plans] = results;
                assert_eq!(left_plans.len(), right_plans.len());

                // Evict cached queries now that the batch is fully processed.
                let all_query_ids: Vec<QueryId> = irises_to_cache[LEFT]
                    .iter()
                    .chain(irises_to_cache[RIGHT].iter())
                    .map(|(qid, _)| *qid)
                    .collect();
                futures::try_join!(
                    actor.worker_pools[LEFT].evict_queries(all_query_ids.clone()),
                    actor.worker_pools[RIGHT].evict_queries(all_query_ids),
                )?;

                metrics::histogram!("genesis_batch_duration").record(now.elapsed().as_secs_f64());
                metrics::gauge!("genesis_batch_size").set(vector_ids.len() as f64);

                Ok((
                    done_rx,
                    JobResult::new_batch_result(
                        batch_id,
                        vector_ids,
                        vector_ids_to_persist,
                        done_tx,
                    ),
                ))
            }
            JobRequest::Modification { modification } => {
                let serial_id = modification.serial_id.ok_or(eyre!(
                    "Genesis received modification with empty serial_id field"
                ))? as u32;

                let jobs_per_side =
                    izip!(STORE_IDS, sessions.iter()).map(|(side, sessions_side)| {
                        let sessions = sessions_side.clone();
                        let registry = actor.registry(side);
                        let modification = modification.clone();
                        let searcher = actor.searcher();

                        async move {
                            let vector_id_ = registry.get_vector_id(serial_id).await;

                            let session =
                                sessions.first().ok_or_eyre("Sessions for side are empty")?;

                            match modification.request_type.as_str() {
                                smpc_request::RESET_UPDATE_MESSAGE_TYPE
                                | smpc_request::RECOVERY_UPDATE_MESSAGE_TYPE
                                | smpc_request::REAUTH_MESSAGE_TYPE => {
                                    let vector_id = vector_id_.ok_or_eyre(
                                        "Expected vector serial id of update is missing from store",
                                    )?;
                                    let identifier = (vector_id, side);

                                    // TODO remove any prior versions of this vector id from graph

                                    // Fetch the iris from the worker pool and cache it before search.
                                    let query_id = QueryId::new();
                                    let irises = {
                                        let store = session.aby3_store.read().await;
                                        store.workers.fetch_irises(vec![vector_id]).await?
                                    };
                                    {
                                        let store = session.aby3_store.read().await;
                                        store
                                            .workers
                                            .cache_queries(vec![(query_id, irises.into_iter().next().unwrap())])
                                            .await?;
                                    }
                                    let query = Aby3Query::new(query_id);
                                    let insert_plan = search_single_query_no_match_count(
                                        session.clone(),
                                        query,
                                        &searcher,
                                        &identifier,
                                    )
                                    .await?;
                                    let plans = vec![Some(insert_plan)];
                                    let ids = vec![Some(vector_id)];

                                    let mut store = session.aby3_store.write().await;
                                    let mut graph = session.graph_store.write().await;
                                    let connect_plan =
                                        insert(&mut *store, &mut *graph, &searcher, plans, &ids).await?;

                                    // Evict the cached query now that search + insert are done.
                                    {
                                        let store = session.aby3_store.read().await;
                                        store.workers.evict_queries(vec![query_id]).await?;
                                    }

                                    Ok((connect_plan, vector_id))
                                }
                                smpc_request::IDENTITY_DELETION_MESSAGE_TYPE => {
                                    let msg = Self::log_error(format!(
                                        "HawkActor does not support deletion of identities: modification: {:?}",
                                        modification
                                    ));
                                    Err(eyre!(msg))
                                }
                                _ => {
                                    let msg = Self::log_error(format!(
                                        "Invalid modification type received: {:?}",
                                        modification,
                                    ));
                                    Err(eyre!(msg))
                                }
                            }
                        }
                    });

                let results_ = parallelize(jobs_per_side.into_iter()).await?;
                let results: [_; 2] = results_.try_into().unwrap();

                // Convert the results into SingleHawkMutation format
                let [left_plans_and_vector, right_plans_and_vector] = results;
                let left_vector = left_plans_and_vector.1;
                let right_vector = right_plans_and_vector.1;

                assert_eq!(left_vector.version_id(), right_vector.version_id());
                assert_eq!(left_vector.serial_id(), right_vector.serial_id());

                metrics::histogram!("genesis_modification_duration")
                    .record(now.elapsed().as_secs_f64());

                Ok((
                    done_rx,
                    JobResult::new_modification_result(modification.id, left_vector, done_tx),
                ))
            }
            JobRequest::Sync {
                shutdown,
                sync_status,
            } => {
                let _ = done_tx;
                let mismatched = tokio::time::timeout(
                    SYNC_PEERS_TIMEOUT,
                    HawkSession::sync_peers(shutdown, sync_status, sessions),
                )
                .await
                .map_err(|_| eyre!("sync_peers timed out after {SYNC_PEERS_TIMEOUT:?}"))??;
                Ok((done_rx, JobResult::Sync { mismatched }))
            }
        }
    }

    // Helper: component error logging.
    fn log_error(msg: String) -> String {
        utils::log_error(COMPONENT, msg)
    }
    // Helper: component logging.
    fn log_info(msg: String) -> String {
        utils::log_info(COMPONENT, msg)
    }

    /// Enqueues a job request for the genesis indexer HNSW processing thread. It returns
    /// a future that resolves to the processed results.
    ///
    /// # Arguments
    ///
    /// * `request` - A request to be processed.
    ///
    /// # Returns
    ///
    /// A future that resolves to the processed results.
    ///
    /// # Errors
    ///
    /// This method may return an error if the job queue channel is closed or if the job fails.
    pub async fn submit_request(
        &mut self,
        request: JobRequest,
    ) -> impl Future<Output = Result<(sync::oneshot::Receiver<()>, JobResult)>> {
        // Set job queue channel.
        let (tx, rx) = oneshot::channel();

        // Set job.
        let job = Job {
            request,
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

    /// Synchronizes MPC nodes, checking for mismatch in the values of nodes'
    /// shutdown states.
    ///
    /// Used to periodically synchronize db persistence threads of MPC nodes in
    /// genesis protocol.
    pub async fn sync_peers(
        &mut self,
        shutdown: bool,
        sync_done: Option<oneshot::Receiver<()>>,
    ) -> Result<bool> {
        let status = Arc::new(AtomicU8::new(SYNC_RUNNING));
        if let Some(ch) = sync_done {
            let status = status.clone();
            tokio::spawn(async move {
                match ch.await {
                    Ok(()) => status.store(SYNC_DONE, Ordering::Relaxed),
                    Err(_) => status.store(SYNC_ERROR, Ordering::Relaxed),
                }
            });
        } else {
            status.store(SYNC_DONE, Ordering::Relaxed);
        }

        let r = self
            .submit_request(JobRequest::Sync {
                shutdown,
                sync_status: status,
            })
            .await;
        let (_, r) = r.await?;
        match r {
            JobResult::Sync { mismatched } => Ok(mismatched),
            _ => bail!("invalid job result"),
        }
    }
}
