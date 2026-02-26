use ampc_anon_stats::store::postgres::AccessMode as AnonStatsAccessMode;
use ampc_anon_stats::store::postgres::PostgresClient as AnonStatsPgClient;
use ampc_anon_stats::types::Eye;
use ampc_anon_stats::{
    process_1d_anon_stats_job, process_1d_lifted_anon_stats_job, process_2d_anon_stats_job,
    start_coordination_server, sync_on_id_hash, sync_on_job_sizes, AnonStatsContext,
    AnonStatsMapping, AnonStatsOperation, AnonStatsOrientation, AnonStatsOrigin,
    AnonStatsServerConfig, AnonStatsStore, BucketStatistics, BucketStatistics2D, DistanceBundle1D,
    LiftedDistanceBundle1D, Opt,
};
use ampc_server_utils::{
    init_heartbeat_task, shutdown_handler::ShutdownHandler, wait_for_others_ready,
    wait_for_others_unready, TaskMonitor,
};
use aws_sdk_s3::{config::Builder as S3ConfigBuilder, Client as S3Client};
use aws_sdk_sns::{config::Region, types::MessageAttributeValue, Client as SNSClient};
use aws_smithy_types::retry::RetryConfig;
use chrono::{DateTime, Utc};
use clap_builder::Parser;
use eyre::{bail, eyre, Context, Result};
use iris_mpc_common::config::{ENV_PROD, ENV_STAGE};
use iris_mpc_common::helpers::sqs_s3_helper::upload_file_to_s3;
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
use sodiumoxide::hex;
use std::collections::HashSet;
use std::{
    collections::HashMap,
    convert::TryFrom,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::Mutex;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, event, info, warn};
use uuid::Uuid;

const GPU_1D_ORIGINS: [AnonStatsOrigin; 4] = [
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
    AnonStatsOrigin {
        side: Some(Eye::Left),
        orientation: AnonStatsOrientation::Mirror,
        context: AnonStatsContext::GPU,
    },
    AnonStatsOrigin {
        side: Some(Eye::Right),
        orientation: AnonStatsOrientation::Mirror,
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

const GPU_2D_ORIGINS: [AnonStatsOrigin; 2] = [
    AnonStatsOrigin {
        side: None,
        orientation: AnonStatsOrientation::Normal,
        context: AnonStatsContext::GPU,
    },
    AnonStatsOrigin {
        side: None,
        orientation: AnonStatsOrientation::Mirror,
        context: AnonStatsContext::GPU,
    },
];

#[derive(Clone, Copy, Debug)]
enum JobKind {
    Gpu1D,
    Hnsw1D,
    Gpu2D,
    Gpu1DReauth,
    Gpu2DReauth,
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
    s3_client: S3Client,
    publish: PublishTargets,
    sync_failures: HashMap<(AnonStatsOrigin, AnonStatsOperation), usize>,
    last_report_times: HashMap<(AnonStatsOrigin, AnonStatsOperation), DateTime<Utc>>,
}

impl AnonStatsProcessor {
    fn new(
        config: Arc<AnonStatsServerConfig>,
        store: AnonStatsStore,
        sns_client: SNSClient,
        s3_client: S3Client,
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
            s3_client,
            publish,
            sync_failures: HashMap::new(),
            last_report_times: HashMap::new(),
        }
    }

    async fn run_iteration(&mut self, session: &mut Session) -> Result<()> {
        for origin in GPU_1D_ORIGINS {
            self.run_1d_job(
                session,
                origin,
                JobKind::Gpu1D,
                AnonStatsOperation::Uniqueness,
            )
            .await?;
        }
        for origin in GPU_1D_ORIGINS {
            self.run_1d_job(
                session,
                origin,
                JobKind::Gpu1DReauth,
                AnonStatsOperation::Reauth,
            )
            .await?;
        }
        for origin in HNSW_1D_ORIGINS {
            self.run_1d_job(
                session,
                origin,
                JobKind::Hnsw1D,
                AnonStatsOperation::Uniqueness,
            )
            .await?;
        }
        for origin in GPU_2D_ORIGINS {
            self.run_2d_job(
                session,
                origin,
                JobKind::Gpu2D,
                AnonStatsOperation::Uniqueness,
            )
            .await?;
        }
        for origin in GPU_2D_ORIGINS {
            self.run_2d_job(
                session,
                origin,
                JobKind::Gpu2DReauth,
                AnonStatsOperation::Reauth,
            )
            .await?;
        }
        Ok(())
    }

    async fn run_1d_job(
        &mut self,
        session: &mut Session,
        origin: AnonStatsOrigin,
        kind: JobKind,
        operation: AnonStatsOperation,
    ) -> Result<()> {
        let available = match kind {
            JobKind::Gpu1D | JobKind::Gpu1DReauth => {
                self.store
                    .num_available_anon_stats_1d(origin, Some(operation))
                    .await?
            }
            JobKind::Hnsw1D => {
                self.store
                    .num_available_anon_stats_1d_lifted(origin, Some(operation))
                    .await?
            }
            JobKind::Gpu2D | JobKind::Gpu2DReauth => 0,
        };
        let available = usize::try_from(available).unwrap_or(0);

        if available == 0 {
            info!("No anon stats entries for {:?}", origin);
            return Ok(());
        }

        // Cap the number of rows we consider for a single job to avoid fetching an
        // unbounded amount of data when the backlog is large.
        let available_capped = available.min(self.config.max_rows_per_job_1d);
        if available_capped < available {
            info!(
                "Capping 1D anon stats job fetch size: available = {}, capped = {}, cap = {}",
                available, available_capped, self.config.max_rows_per_job_1d
            );
        }

        let min_job_size = sync_on_job_sizes(session, available_capped).await?;
        let required_min = match kind {
            JobKind::Gpu1DReauth => self.config.min_1d_job_size_reauth,
            JobKind::Gpu1D | JobKind::Hnsw1D => self.config.min_1d_job_size,
            _ => panic!("Invalid job kind for 1D job"),
        };

        self.log_available_entries(min_job_size, required_min, origin, kind)
            .await;

        if min_job_size < required_min {
            return Ok(());
        }

        match kind {
            JobKind::Gpu1D | JobKind::Gpu1DReauth => {
                let (ids, bundles) = self
                    .store
                    .get_available_anon_stats_1d(origin, Some(operation), min_job_size)
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
                    self.handle_sync_failure(origin, operation, kind).await?;
                    return Ok(());
                }

                let last_report_time = self.last_report_times.get(&(origin, operation)).copied();
                let start = Instant::now();
                let mut stats = process_1d_anon_stats_job(
                    session,
                    job,
                    &origin,
                    self.config.as_ref(),
                    Some(operation),
                    last_report_time,
                )
                .await?;

                self.log_job_metrics("1d", origin, kind, job_size, start.elapsed())
                    .await;

                let report_time = Utc::now();
                let window_start = last_report_time.unwrap_or(report_time);
                stats.start_time_utc_timestamp = window_start;
                stats.end_time_utc_timestamp = Some(report_time);
                stats.next_start_time_utc_timestamp = Some(report_time);

                self.publish_1d_stats(&stats).await?;
                self.last_report_times
                    .insert((origin, operation), report_time);
                self.store.mark_anon_stats_processed_1d(&ids).await?;
                self.sync_failures.remove(&(origin, operation));
                Ok(())
            }
            JobKind::Hnsw1D => {
                let (ids, bundles) = self
                    .store
                    .get_available_anon_stats_1d_lifted(origin, Some(operation), min_job_size)
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
                    self.handle_sync_failure(origin, operation, kind).await?;
                    return Ok(());
                }

                let last_report_time = self.last_report_times.get(&(origin, operation)).copied();
                let start = Instant::now();
                let mut stats = process_1d_lifted_anon_stats_job(
                    session,
                    job,
                    &origin,
                    self.config.as_ref(),
                    Some(operation),
                    last_report_time,
                )
                .await?;

                self.log_job_metrics("1d", origin, kind, job_size, start.elapsed())
                    .await;

                let report_time = Utc::now();
                let window_start = last_report_time.unwrap_or(report_time);
                stats.start_time_utc_timestamp = window_start;
                stats.end_time_utc_timestamp = Some(report_time);
                stats.next_start_time_utc_timestamp = Some(report_time);

                self.publish_1d_stats(&stats).await?;
                self.last_report_times
                    .insert((origin, operation), report_time);
                self.store.mark_anon_stats_processed_1d_lifted(&ids).await?;
                self.sync_failures.remove(&(origin, operation));
                Ok(())
            }
            JobKind::Gpu2D | JobKind::Gpu2DReauth => unreachable!(),
        }
    }

    async fn run_2d_job(
        &mut self,
        session: &mut Session,
        origin: AnonStatsOrigin,
        kind: JobKind,
        operation: AnonStatsOperation,
    ) -> Result<()> {
        let available = usize::try_from(
            self.store
                .num_available_anon_stats_2d(origin, Some(operation))
                .await?,
        )
        .unwrap_or(0);
        if available == 0 {
            return Ok(());
        }

        // Cap the number of rows we consider for a single job to avoid fetching an
        // unbounded amount of data when the backlog is large.
        let available_capped = available.min(self.config.max_rows_per_job_2d);
        if available_capped < available {
            info!(
                "Capping 1D anon stats job fetch size: available = {}, capped = {}, cap = {}",
                available, available_capped, self.config.max_rows_per_job_1d
            );
        }

        let min_job_size = sync_on_job_sizes(session, available_capped).await?;
        let required_min = match kind {
            JobKind::Gpu2DReauth => self.config.min_2d_job_size_reauth,
            JobKind::Gpu2D => self.config.min_2d_job_size,
            _ => panic!("Invalid job kind for 2D job"),
        };

        self.log_available_entries(min_job_size, required_min, origin, kind)
            .await;

        if min_job_size < required_min {
            return Ok(());
        }

        let (ids, bundles) = self
            .store
            .get_available_anon_stats_2d(origin, Some(operation), min_job_size)
            .await?;

        info!(
            ?origin,
            ?operation,
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
            self.handle_sync_failure(origin, operation, kind).await?;
            return Ok(());
        }

        let last_report_time = self.last_report_times.get(&(origin, operation)).copied();
        let start = Instant::now();
        let mut stats = process_2d_anon_stats_job(
            session,
            job,
            &origin,
            self.config.as_ref(),
            Some(operation),
            last_report_time,
        )
        .await?;

        self.log_job_metrics("2d", origin, kind, job_size, start.elapsed())
            .await;

        let report_time = Utc::now();
        let window_start = last_report_time.unwrap_or(report_time);
        stats.start_time_utc_timestamp = window_start;
        stats.end_time_utc_timestamp = Some(report_time);
        stats.next_start_time_utc_timestamp = Some(report_time);

        self.publish_2d_stats(&stats).await?;
        self.last_report_times
            .insert((origin, operation), report_time);
        self.store.mark_anon_stats_processed_2d(&ids).await?;
        self.sync_failures.remove(&(origin, operation));
        Ok(())
    }

    async fn publish_1d_stats(&self, stats: &BucketStatistics) -> Result<()> {
        let payload =
            serde_json::to_string(stats).wrap_err("failed to serialize 1D anon stats payload")?;
        self.publish_message(payload, &self.publish.attributes_1d)
            .await
    }

    async fn publish_2d_stats(&self, stats: &BucketStatistics2D) -> Result<()> {
        info!("Sending 2D anonymized stats results");
        let serialized = serde_json::to_string(&stats)
            .wrap_err("failed to serialize 2D anonymized statistics result")?;

        // offloading 2D anon stats file to s3 to avoid sending large messages to SNS
        // with 2D stats we were exceeding the SNS message size limit
        let now_ms = Utc::now().timestamp_millis();
        let sha = iris_mpc_common::helpers::sha256::sha256_bytes(&serialized);
        let content_hash = hex::encode(sha);
        let s3_key = format!("stats2d/{}_{}.json", now_ms, content_hash);

        upload_file_to_s3(
            &self.config.sns_buffer_bucket_name,
            &s3_key,
            self.s3_client.clone(),
            serialized.as_bytes(),
        )
        .await
        .wrap_err("failed to upload 2D anonymized statistics to s3")?;

        // Publish only the S3 key to SNS
        let payload = serde_json::to_string(&serde_json::json!({
            "s3_key": s3_key,
        }))?;
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

    async fn handle_sync_failure(
        &mut self,
        origin: AnonStatsOrigin,
        operation: AnonStatsOperation,
        kind: JobKind,
    ) -> Result<()> {
        let failures = self.sync_failures.entry((origin, operation)).or_insert(0);
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
        event!(
            tracing::Level::ERROR,
            ?origin,
            "Exceeded sync mismatches threshold; clearing local anon stats queue"
        );

        let cleared = match kind {
            JobKind::Gpu1D | JobKind::Gpu1DReauth => {
                self.store
                    .clear_unprocessed_anon_stats_1d(origin, Some(operation))
                    .await?
            }
            JobKind::Hnsw1D => {
                self.store
                    .clear_unprocessed_anon_stats_1d_lifted(origin, Some(operation))
                    .await?
            }
            JobKind::Gpu2D | JobKind::Gpu2DReauth => {
                self.store
                    .clear_unprocessed_anon_stats_2d(origin, Some(operation))
                    .await?
            }
        };
        info!(?origin, cleared, "Cleared unprocessed anon stats entries");
        self.sync_failures.insert((origin, operation), 0);
        Ok(())
    }

    async fn log_job_metrics(
        &self,
        metric_name_suffix: &str,
        origin: AnonStatsOrigin,
        kind: JobKind,
        job_size: usize,
        duration: Duration,
    ) {
        let duration_metric_name = format!("job_{}.duration", metric_name_suffix);
        let job_size_metric_name = format!("job_{}.size", metric_name_suffix);

        let side = match origin.side {
            Some(eye) => format!("{:?}", eye),
            None => "both".to_string(),
        };

        metrics::histogram!(
            duration_metric_name,
            "orientation" => format!("{:?}", origin.orientation),
            "kind" => format!("{:?}", kind),
            "side" => side.clone(),
        )
        .record(duration.as_millis() as f64);

        metrics::histogram!(
            job_size_metric_name,
            "orientation" => format!("{:?}", origin.orientation),
            "kind" => format!("{:?}", kind),
            "side" => side,
        )
        .record(job_size as f64);

        info!("Completed anon stats job of kind: {:?}", kind);
    }

    async fn log_available_entries(
        &self,
        min_job_size: usize,
        required_min: usize,
        origin: AnonStatsOrigin,
        kind: JobKind,
    ) {
        let side = match origin.side {
            Some(eye) => format!("{:?}", eye),
            None => "both".to_string(),
        };

        metrics::gauge!(
            "available_entries",
            "orientation" => format!("{:?}", origin.orientation),
            "kind" => format!("{:?}", kind),
            "side" => side
        )
        .set(min_job_size as f64);

        metrics::gauge!(
            "required_min_entries",
            "orientation" => format!("{:?}", origin.orientation),
            "kind" => format!("{:?}", kind),
        )
        .set(required_min as f64);

        info!(
            "Available syncable entries for side {:?}, orientation {:?}, kind {:?}: {}/{}",
            origin.side, origin.orientation, kind, min_job_size, required_min
        );
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
        bail!("SMPC__SERVER_COORDINATION__NODE_HOSTNAMES and SMPC__SERVICE_PORTS must be provided");
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

    let sns_client = build_sns_client(&config).await?;
    let s3_client = build_s3_client(&config).await?;

    info!("Starting anon stats server.");
    let mut background_tasks = TaskMonitor::new();
    let verified_peers = Arc::new(Mutex::new(HashSet::new()));
    let uuid = Uuid::new_v4().to_string();
    let coordination_handles = start_coordination_server(
        server_coord_config,
        &mut background_tasks,
        verified_peers.clone(),
        uuid.clone(),
    );

    background_tasks.check_tasks();
    let health_port = server_coord_config
        .healthcheck_ports
        .get(config.party_id)
        .cloned()
        .unwrap_or_else(|| "8080".to_string());
    info!("Healthcheck server running on port {}", health_port);

    wait_for_others_unready(server_coord_config, &verified_peers, &uuid).await?;

    init_heartbeat_task(
        server_coord_config,
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
        tls: config.tls.clone(),
    };
    let ct = shutdown_handler.get_network_cancellation_token();

    let mut networking = build_network_handle(args, ct.child_token()).await?;
    let mut sessions = networking.as_mut().make_sessions().await?;
    let session = sessions
        .0
        .get_mut(0)
        .ok_or_else(|| eyre!("expected at least one network session"))?;

    let mut processor =
        AnonStatsProcessor::new(config.clone(), anon_stats_store, sns_client, s3_client);

    coordination_handles.set_ready();
    wait_for_others_ready(server_coord_config)
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
                    tokio::select! {
                        res = processor.run_iteration(session) => {
                             if let Err(err) = res {
                                warn!(error = ?err, "Anon stats iteration failed");
                                shutdown_handler.trigger_manual_shutdown();
                                break;
                            }
                        }
                        _ = ct.cancelled() => {
                             info!("Shutdown triggered during iteration, stopping.");
                             break;
                        }
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

async fn build_s3_client(config: &AnonStatsServerConfig) -> Result<S3Client> {
    let force_path_style = config.environment != ENV_PROD && config.environment != ENV_STAGE;
    let retry_config = RetryConfig::standard().with_max_attempts(5);

    let mut loader = aws_config::from_env();
    if let Some(aws) = &config.aws {
        if let Some(region) = &aws.region {
            loader = loader.region(Region::new(region.clone()));
        }
    }

    let shared_config = loader.load().await;
    let mut s3_config = S3ConfigBuilder::from(&shared_config).retry_config(retry_config.clone());
    if let Some(aws) = &config.aws {
        if let Some(endpoint) = &aws.endpoint {
            s3_config = s3_config.endpoint_url(endpoint);
        }
        if force_path_style {
            s3_config = s3_config.force_path_style(force_path_style);
        }
    }
    Ok(S3Client::from_conf(s3_config.build()))
}
