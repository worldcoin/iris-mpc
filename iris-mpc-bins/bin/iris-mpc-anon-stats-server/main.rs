use ampc_anon_stats::anon_stats::{DistanceBundle1D, LiftedDistanceBundle1D};
use ampc_anon_stats::store::postgres::AccessMode as AnonStatsAccessMode;
use ampc_anon_stats::store::postgres::PostgresClient as AnonStatsPgClient;
use ampc_anon_stats::{
    process_1d_anon_stats_job, process_1d_lifted_anon_stats_job, process_2d_anon_stats_job,
    start_coordination_server, sync_on_id_hash, sync_on_job_sizes, AnonStatsContext,
    AnonStatsMapping, AnonStatsOrientation, AnonStatsOrigin, AnonStatsServerConfig, AnonStatsStore,
    Opt,
};
use ampc_server_utils::{
    init_heartbeat_task, shutdown_handler::ShutdownHandler, wait_for_others_ready,
    wait_for_others_unready, BucketStatistics, BucketStatistics2D, Eye, TaskMonitor,
};
use aws_sdk_sns::{config::Region, types::MessageAttributeValue, Client as SNSClient};
use clap_builder::Parser;
use eyre::{bail, eyre, Context, Result};
use iris_mpc_common::{
    helpers::{
        smpc_request::{ANONYMIZED_STATISTICS_2D_MESSAGE_TYPE, ANONYMIZED_STATISTICS_MESSAGE_TYPE},
        smpc_response::create_message_type_attribute_map,
    },
    tracing::initialize_tracing,
};
use iris_mpc_cpu::{
    execution::session::Session,
    network::tcp::{build_network_handle, NetworkHandleArgs},
};
use std::collections::HashSet;
use std::{
    collections::HashMap,
    convert::TryFrom,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info, warn};
use uuid::Uuid;

const GPU_1D_ORIGINS: [AnonStatsOrigin; 2] = [
    AnonStatsOrigin {
        side: Some(Eye::Left),
        orientation: AnonStatsOrientation::Normal,
        context: AnonStatsContext::GPU,
    },
    AnonStatsOrigin {
        side: Some(Eye::Right),
        orientation: AnonStatsOrientation::Normal,
        context: AnonStatsContext::GPU,
    },
];

const HNSW_1D_ORIGINS: [AnonStatsOrigin; 2] = [
    AnonStatsOrigin {
        side: Some(Eye::Left),
        orientation: AnonStatsOrientation::Normal,
        context: AnonStatsContext::HNSW,
    },
    AnonStatsOrigin {
        side: Some(Eye::Right),
        orientation: AnonStatsOrientation::Normal,
        context: AnonStatsContext::HNSW,
    },
];

const GPU_2D_ORIGINS: [AnonStatsOrigin; 1] = [AnonStatsOrigin {
    side: None,
    orientation: AnonStatsOrientation::Normal,
    context: AnonStatsContext::GPU,
}];

#[derive(Clone, Copy, Debug)]
enum JobKind {
    Gpu1D,
    Hnsw1D,
    Gpu2D,
}

struct PublishTargets {
    topic_arn: String,
    message_group_id: String,
    attributes_1d: HashMap<String, MessageAttributeValue>,
    attributes_2d: HashMap<String, MessageAttributeValue>,
}

struct AnonStatsProcessor {
    config: Arc<AnonStatsServerConfig>,
    store: AnonStatsStore,
    sns_client: SNSClient,
    publish: PublishTargets,
    sync_failures: HashMap<AnonStatsOrigin, usize>,
}

impl AnonStatsProcessor {
    fn new(
        config: Arc<AnonStatsServerConfig>,
        store: AnonStatsStore,
        sns_client: SNSClient,
    ) -> Self {
        let publish = PublishTargets {
            topic_arn: config.results_topic_arn.clone(),
            message_group_id: format!("party-id-{}", config.party_id),
            attributes_1d: create_message_type_attribute_map(ANONYMIZED_STATISTICS_MESSAGE_TYPE),
            attributes_2d: create_message_type_attribute_map(ANONYMIZED_STATISTICS_2D_MESSAGE_TYPE),
        };

        Self {
            config,
            store,
            sns_client,
            publish,
            sync_failures: HashMap::new(),
        }
    }

    async fn run_iteration(&mut self, session: &mut Session) -> Result<()> {
        for origin in GPU_1D_ORIGINS {
            self.run_1d_job(session, origin, JobKind::Gpu1D).await?;
        }
        for origin in HNSW_1D_ORIGINS {
            self.run_1d_job(session, origin, JobKind::Hnsw1D).await?;
        }
        for origin in GPU_2D_ORIGINS {
            self.run_2d_job(session, origin).await?;
        }
        Ok(())
    }

    async fn run_1d_job(
        &mut self,
        session: &mut Session,
        origin: AnonStatsOrigin,
        kind: JobKind,
    ) -> Result<()> {
        let available = match kind {
            JobKind::Gpu1D => self.store.num_available_anon_stats_1d(origin).await?,
            JobKind::Hnsw1D => {
                self.store
                    .num_available_anon_stats_1d_lifted(origin)
                    .await?
            }
            JobKind::Gpu2D => 0,
        };
        let available = usize::try_from(available).unwrap_or(0);
        info!(
            "Available anon stats entries for {:?}: {}",
            origin, available
        );
        if available == 0 {
            info!("No anon stats entries for {:?}", origin);
            return Ok(());
        }

        let min_job_size = sync_on_job_sizes(session, available).await?;
        if min_job_size < self.config.min_1d_job_size {
            debug!(
                ?origin,
                available,
                min_job_size,
                required = self.config.min_1d_job_size,
                "Not enough entries yet for 1D anon stats job"
            );
            return Ok(());
        }

        match kind {
            JobKind::Gpu1D => {
                let (ids, bundles) = self
                    .store
                    .get_available_anon_stats_1d(origin, min_job_size)
                    .await?;
                if bundles.is_empty() {
                    return Ok(());
                }

                let mut job: AnonStatsMapping<DistanceBundle1D> = AnonStatsMapping::new(bundles);
                if job.len() > min_job_size {
                    job.truncate(min_job_size);
                }
                let job_size = job.len();
                let job_hash = job.get_id_hash();

                if !sync_on_id_hash(session, job_hash).await? {
                    warn!(
                        ?origin,
                        job_size, "Mismatched 1D anon stats job hash detected; scheduling recovery"
                    );
                    self.handle_sync_failure(origin, kind).await?;
                    return Ok(());
                }

                let start = Instant::now();
                let stats =
                    process_1d_anon_stats_job(session, job, &origin, self.config.as_ref()).await?;
                info!(
                    ?origin,
                    job_size,
                    elapsed_ms = start.elapsed().as_millis(),
                    "Completed 1D anon stats job"
                );

                self.publish_1d_stats(&stats).await?;
                self.store.mark_anon_stats_processed_1d(&ids).await?;
                self.sync_failures.remove(&origin);
                Ok(())
            }
            JobKind::Hnsw1D => {
                let (ids, bundles) = self
                    .store
                    .get_available_anon_stats_1d_lifted(origin, min_job_size)
                    .await?;
                if bundles.is_empty() {
                    return Ok(());
                }

                let mut job: AnonStatsMapping<LiftedDistanceBundle1D> =
                    AnonStatsMapping::new(bundles);
                if job.len() > min_job_size {
                    job.truncate(min_job_size);
                }
                let job_size = job.len();
                let job_hash = job.get_id_hash();

                if !sync_on_id_hash(session, job_hash).await? {
                    warn!(
                        ?origin,
                        job_size, "Mismatched 1D anon stats job hash detected; scheduling recovery"
                    );
                    self.handle_sync_failure(origin, kind).await?;
                    return Ok(());
                }

                let start = Instant::now();
                let stats =
                    process_1d_lifted_anon_stats_job(session, job, &origin, self.config.as_ref())
                        .await?;
                info!(
                    ?origin,
                    job_size,
                    elapsed_ms = start.elapsed().as_millis(),
                    "Completed 1D anon stats job"
                );

                self.publish_1d_stats(&stats).await?;
                self.store.mark_anon_stats_processed_1d_lifted(&ids).await?;
                self.sync_failures.remove(&origin);
                Ok(())
            }
            JobKind::Gpu2D => unreachable!(),
        }
    }

    async fn run_2d_job(&mut self, session: &mut Session, origin: AnonStatsOrigin) -> Result<()> {
        let available =
            usize::try_from(self.store.num_available_anon_stats_2d(origin).await?).unwrap_or(0);
        if available == 0 {
            return Ok(());
        }

        let min_job_size = sync_on_job_sizes(session, available).await?;
        if min_job_size < self.config.min_1d_job_size {
            debug!(
                ?origin,
                available,
                min_job_size,
                required = self.config.min_1d_job_size,
                "Not enough entries yet for 2D anon stats job"
            );
            return Ok(());
        }

        let (ids, bundles) = self
            .store
            .get_available_anon_stats_2d(origin, min_job_size)
            .await?;

        info!(
            ?origin,
            "Fetched {} anon stats 2D bundles for processing",
            bundles.len()
        );

        if bundles.is_empty() {
            info!("No anon stats entries for {:?}", origin);
            return Ok(());
        }

        let mut job = AnonStatsMapping::new(bundles);
        if job.len() > min_job_size {
            job.truncate(min_job_size);
        }
        let job_size = job.len();
        let job_hash = job.get_id_hash();

        if !sync_on_id_hash(session, job_hash).await? {
            warn!(
                ?origin,
                job_size, "Mismatched 2D anon stats job hash detected; scheduling recovery"
            );
            self.handle_sync_failure(origin, JobKind::Gpu2D).await?;
            return Ok(());
        }

        let start = Instant::now();
        let stats = process_2d_anon_stats_job(session, job, self.config.as_ref()).await?;
        info!(
            ?origin,
            job_size,
            elapsed_ms = start.elapsed().as_millis(),
            "Completed 2D anon stats job"
        );

        self.publish_2d_stats(&stats).await?;
        self.store.mark_anon_stats_processed_2d(&ids).await?;
        self.sync_failures.remove(&origin);
        Ok(())
    }

    async fn publish_1d_stats(&self, stats: &BucketStatistics) -> Result<()> {
        let payload =
            serde_json::to_string(stats).wrap_err("failed to serialize 1D anon stats payload")?;
        self.publish_message(payload, &self.publish.attributes_1d)
            .await
    }

    async fn publish_2d_stats(&self, stats: &BucketStatistics2D) -> Result<()> {
        let payload =
            serde_json::to_string(stats).wrap_err("failed to serialize 2D anon stats payload")?;
        self.publish_message(payload, &self.publish.attributes_2d)
            .await
    }

    async fn publish_message(
        &self,
        payload: String,
        attributes: &HashMap<String, MessageAttributeValue>,
    ) -> Result<()> {
        self.sns_client
            .publish()
            .topic_arn(&self.publish.topic_arn)
            .message(payload)
            .message_group_id(&self.publish.message_group_id)
            .set_message_attributes(Some(attributes.clone()))
            .send()
            .await
            .wrap_err("failed to publish anon stats result to SNS")?;
        Ok(())
    }

    async fn handle_sync_failure(&mut self, origin: AnonStatsOrigin, kind: JobKind) -> Result<()> {
        let failures = self.sync_failures.entry(origin).or_insert(0);
        *failures += 1;

        if *failures < self.config.max_sync_failures_before_reset {
            debug!(
                ?origin,
                failure_count = *failures,
                "Anon stats sync mismatch recorded"
            );
            return Ok(());
        }

        warn!(
            ?origin,
            "Exceeded sync mismatches threshold; clearing local anon stats queue"
        );

        let cleared = match kind {
            JobKind::Gpu1D => self.store.clear_unprocessed_anon_stats_1d(origin).await?,
            JobKind::Hnsw1D => {
                self.store
                    .clear_unprocessed_anon_stats_1d_lifted(origin)
                    .await?
            }
            JobKind::Gpu2D => self.store.clear_unprocessed_anon_stats_2d(origin).await?,
        };
        info!(?origin, cleared, "Cleared unprocessed anon stats entries");
        self.sync_failures.insert(origin, 0);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    info!("Init config");
    let mut config = AnonStatsServerConfig::load_config("SMPC")?;
    config.overwrite_defaults_with_cli_args(Opt::parse());

    if config.results_topic_arn.is_empty() {
        bail!("SMPC__RESULTS_TOPIC_ARN must be provided");
    }

    let config = Arc::new(config);

    let server_coord_config = config.server_coordination.as_ref().ok_or_else(|| {
        eyre!("Server coordination configuration must be provided for anon stats server")
    })?;

    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.register_signal_handler().await;

    let node_addresses: Vec<String> = server_coord_config
        .node_hostnames
        .iter()
        .zip(config.service_ports.iter())
        .map(|(host, port)| format!("{}:{}", host, port))
        .collect();
    if node_addresses.is_empty() {
        bail!("SMPC__NODE_HOSTNAMES and SMPC__SERVICE_PORTS must be provided");
    }

    let _tracing_shutdown_handle =
        initialize_tracing(config.service.clone()).wrap_err("Failed to initialize tracing")?;

    info!("Connecting to database");
    let postgres_client = AnonStatsPgClient::new(
        &config.db_url,
        &config.db_schema_name,
        AnonStatsAccessMode::ReadWrite,
    )
    .await?;
    let anon_stats_store = AnonStatsStore::new(&postgres_client).await?;

    let sns_client = build_sns_client(config.as_ref()).await?;

    info!("Starting anon stats server.");
    let mut background_tasks = TaskMonitor::new();
    let coordination_handles =
        start_coordination_server(&server_coord_config, &mut background_tasks);

    background_tasks.check_tasks();
    let health_port = server_coord_config
        .healthcheck_ports
        .get(config.party_id)
        .cloned()
        .unwrap_or_else(|| "8080".to_string());
    info!("Healthcheck server running on port {}", health_port);

    let verified_peers = Arc::new(Mutex::new(HashSet::new()));
    let uuid = Uuid::new_v4().to_string();

    wait_for_others_unready(&server_coord_config, &verified_peers, &uuid).await?;

    init_heartbeat_task(
        &server_coord_config,
        &mut background_tasks,
        &shutdown_handler,
    )
    .await
    .wrap_err("failed to start heartbeat task")?;
    background_tasks.check_tasks();

    let args = NetworkHandleArgs {
        party_index: server_coord_config.party_id,
        addresses: node_addresses.clone(),
        outbound_addresses: node_addresses,
        connection_parallelism: 8,
        request_parallelism: 8,
        sessions_per_request: 1,
        tls: None,
    };
    let ct = shutdown_handler.get_cancellation_token();

    let mut networking = build_network_handle(args, ct.child_token()).await?;
    let mut sessions = networking.as_mut().make_sessions().await?;
    let session = sessions
        .0
        .get_mut(0)
        .ok_or_else(|| eyre!("expected at least one network session"))?;

    let mut processor = AnonStatsProcessor::new(config.clone(), anon_stats_store, sns_client);

    coordination_handles.set_ready();
    wait_for_others_ready(&server_coord_config)
        .await
        .wrap_err("waiting for other anon stats servers to become ready")?;

    let mut poll_interval = interval(Duration::from_secs(config.poll_interval_secs));
    poll_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

    if let Err(err) = processor.run_iteration(session).await {
        warn!(error = ?err, "Anon stats iteration failed");
        shutdown_handler.trigger_manual_shutdown();
    }

    if !shutdown_handler.is_shutting_down() {
        let shutdown_wait = shutdown_handler.wait_for_shutdown();
        tokio::pin!(shutdown_wait);

        loop {
            tokio::select! {
                _ = poll_interval.tick() => {
                    if let Err(err) = processor.run_iteration(session).await {
                        warn!(error = ?err, "Anon stats iteration failed");
                        shutdown_handler.trigger_manual_shutdown();
                break;
                    }
                }
                _ = shutdown_wait.as_mut() => {
                    info!("Shutdown triggered, stopping anon stats processor.");
                    break;
                }
            }
        }
    }

    shutdown_handler.wait_for_shutdown().await;
    shutdown_handler.wait_for_pending_batches_completion().await;
    ct.cancel();
    background_tasks.abort_and_wait_for_finish().await;

    Ok(())
}

async fn build_sns_client(config: &AnonStatsServerConfig) -> Result<SNSClient> {
    let mut loader = aws_config::from_env();
    if let Some(aws) = &config.aws {
        if let Some(region) = &aws.region {
            loader = loader.region(Region::new(region.clone()));
        }
    }
    let shared_config = loader.load().await;
    let mut sns_config_builder = aws_sdk_sns::config::Builder::from(&shared_config);
    if let Some(aws) = &config.aws {
        if let Some(endpoint) = &aws.endpoint {
            sns_config_builder = sns_config_builder.endpoint_url(endpoint);
        }
    }
    Ok(SNSClient::from_conf(sns_config_builder.build()))
}
