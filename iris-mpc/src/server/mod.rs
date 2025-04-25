pub mod utils;

use crate::server::utils::get_check_addresses;
use crate::services::aws::clients::AwsClients;
use crate::services::processors::batch::receive_batch;
use crate::services::processors::job::process_job_result;
use crate::services::processors::process_identity_deletions;
use crate::services::processors::result_message::send_results_to_sns;
use crate::services::store::{load_db, S3LoaderParams};
use aws_sdk_sns::types::MessageAttributeValue;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{bail, eyre, Error, Report, Result, WrapErr};
use iris_mpc_common::config::{CommonConfig, Config, ModeOfCompute, ModeOfDeployment};
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{
    ANONYMIZED_STATISTICS_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::create_message_type_attribute_map;
use iris_mpc_common::helpers::sqs::{delete_messages_until_sequence_num, get_next_sns_seq_num};
use iris_mpc_common::helpers::sync::{SyncResult, SyncState};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::iris_db::get_dummy_shares_for_deletion;
use iris_mpc_common::job::JobSubmissionHandle;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::execution::hawk_main::{
    GraphStore, HawkActor, HawkArgs, HawkHandle, ServerJobResult,
};
use iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store;
use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
use iris_mpc_store::{S3Store, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::Sender;
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

const RNG_SEED_INIT_DB: u64 = 42;
pub const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
pub const MAX_CONCURRENT_REQUESTS: usize = 32;
pub static CURRENT_BATCH_SIZE: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

/// Main logic for initialization and execution of AMPC iris uniqueness server
/// nodes.
pub async fn server_main(config: Config) -> Result<()> {
    let shutdown_handler = init_shutdown_handler(&config).await;

    process_config(&config);

    let (iris_store, graph_store) = prepare_stores(&config).await?;

    let aws_clients = init_aws_services(&config).await?;
    let shares_encryption_key_pair = get_shares_encryption_key_pair(&config, &aws_clients).await?;
    let sns_attributes_maps = init_sns(&config, &aws_clients, &iris_store).await?;

    maybe_seed_random_shares(&config, &iris_store).await?;
    check_store_consistency(&config, &iris_store).await?;
    let my_state = build_sync_state(&config, &aws_clients, &iris_store).await?;

    let mut background_tasks = init_task_monitor();

    let is_ready_flag =
        start_coordination_server(&config, &mut background_tasks, &shutdown_handler, &my_state)
            .await;

    background_tasks.check_tasks();

    wait_for_others_unready(&config).await?;
    init_heartbeat_task(&config, &mut background_tasks, &shutdown_handler).await?;

    background_tasks.check_tasks();

    let sync_result = get_others_sync_state(&config, &my_state).await?;
    sync_result.check_common_config()?;

    maybe_sync_sqs_queues(&config, &sync_result, &aws_clients).await?;
    sync_dbs_rollback(&config, &sync_result, &iris_store).await?;

    if shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    let mut hawk_actor = init_hawk_actor(&config).await?;

    load_database(
        &config,
        &iris_store,
        &graph_store,
        &aws_clients,
        &shutdown_handler,
        &mut hawk_actor,
    )
    .await?;

    background_tasks.check_tasks();

    let tx_results = start_results_thread(
        &config,
        &iris_store,
        graph_store,
        &aws_clients,
        &mut background_tasks,
        &shutdown_handler,
        sns_attributes_maps,
    )
    .await?;

    background_tasks.check_tasks();

    set_node_ready(is_ready_flag);
    wait_for_others_ready(&config).await?;

    background_tasks.check_tasks();

    run_main_server_loop(
        &config,
        &iris_store,
        &aws_clients,
        shares_encryption_key_pair,
        &sync_result,
        background_tasks,
        &shutdown_handler,
        hawk_actor,
        tx_results,
    )
    .await?;

    Ok(())
}

/// Initializes shutdown handler, which waits for shutdown signals or function
/// calls and provides a light mechanism for gracefully finishing ongoing query
/// batches before exiting.
async fn init_shutdown_handler(config: &Config) -> Arc<ShutdownHandler> {
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.wait_for_shutdown_signal().await;

    shutdown_handler
}

/// Validates server config and initializes associated static state.
fn process_config(config: &Config) {
    // Validate modes of compute/deployment.
    if config.mode_of_compute != ModeOfCompute::Cpu {
        panic!(
            "Invalid config setting: compute_mode: actual: {:?} :: expected: ModeOfCompute::CPU",
            config.mode_of_compute
        );
    } else {
        tracing::info!("Mode of compute: {:?}", config.mode_of_compute);
        tracing::info!("Mode of deployment: {:?}", config.mode_of_deployment);
    }

    // Make sure the configuration is in correct state, to avoid complex handling of ReadOnly
    // during ShadowReadOnly deployment we panic if the base store persistence is enabled.
    if config.mode_of_deployment == ModeOfDeployment::ShadowReadOnly && !config.disable_persistence
    {
        panic!(
            "The system cannot start securely in ShadowReadOnly mode with enabled base persistence flag!"
        )
    }

    // Load batch_size config
    *CURRENT_BATCH_SIZE.lock().unwrap() = config.max_batch_size;
    tracing::info!("Set batch size to {}", config.max_batch_size);
}

/// Returns computed maximum sync lookback size.
fn max_sync_lookback(config: &Config) -> usize {
    config.max_batch_size * 2
}

/// Returns computed maximum rollback size.
fn max_rollback(config: &Config) -> usize {
    config.max_batch_size * 2
}

/// Returnes initialized PostgreSQL clients for interacting
/// with iris share and HNSW graph stores.
async fn prepare_stores(config: &Config) -> Result<(Store, GraphPg<Aby3Store>), Report> {
    let schema_name = format!(
        "{}_{}_{}",
        config.app_name, config.environment, config.party_id
    );

    match config.mode_of_deployment {
        // use the hawk db for both stores
        ModeOfDeployment::ShadowIsolation => {
            // This mode uses only Hawk DB
            let hawk_db_config = config
                .cpu_database
                .as_ref()
                .ok_or(eyre!("Missing CPU database config in ShadowIsolation"))?;
            let hawk_postgres_client =
                PostgresClient::new(&hawk_db_config.url, &schema_name, AccessMode::ReadWrite)
                    .await?;

            // Store -> CPU
            tracing::info!(
                "Creating new iris store from: {:?} in mode {:?}",
                hawk_db_config,
                config.mode_of_deployment
            );
            let iris_store = Store::new(&hawk_postgres_client).await?;

            // Graph -> CPU
            tracing::info!(
                "Creating new graph store from: {:?} in mode {:?}",
                hawk_db_config,
                config.mode_of_deployment
            );
            let graph_store = GraphStore::new(&hawk_postgres_client).await?;

            Ok((iris_store, graph_store))
        }

        // use base db for iris store and hawk db for graph store
        ModeOfDeployment::ShadowReadOnly => {
            let db_config = config
                .database
                .as_ref()
                .ok_or(eyre!("Missing database config"))?;

            let postgres_client =
                PostgresClient::new(&db_config.url, &schema_name, AccessMode::ReadOnly).await?;

            tracing::info!(
                "Creating new iris store from: {:?} in mode {:?}",
                db_config,
                config.mode_of_deployment
            );

            let iris_store = Store::new(&postgres_client).await?;

            let hawk_db_config = config
                .cpu_database
                .as_ref()
                .ok_or(eyre!("Missing CPU database config in ShadowReadOnly"))?;
            let hawk_postgres_client =
                PostgresClient::new(&hawk_db_config.url, &schema_name, AccessMode::ReadWrite)
                    .await?;

            tracing::info!(
                "Creating new graph store from: {:?} in mode {:?}",
                hawk_db_config,
                config.mode_of_deployment
            );
            let graph_store = GraphStore::new(&hawk_postgres_client).await?;

            Ok((iris_store, graph_store))
        }

        // use the base db for both stores
        _ => {
            let db_config = config
                .database
                .as_ref()
                .ok_or(eyre!("Missing database config"))?;

            let postgres_client =
                PostgresClient::new(&db_config.url, &schema_name, AccessMode::ReadWrite).await?;

            tracing::info!(
                "Creating new iris store from: {:?} in mode {:?}",
                db_config,
                config.mode_of_deployment
            );
            let iris_store = Store::new(&postgres_client).await?;

            tracing::info!(
                "Creating new graph store from: {:?} in mode {:?}",
                db_config,
                config.mode_of_deployment
            );
            let graph_store = GraphStore::new(&postgres_client).await?;

            Ok((iris_store, graph_store))
        }
    }
}

/// Returns AWS service clients for SQS, SNS, S3, and Secrets Manager.
async fn init_aws_services(config: &Config) -> Result<AwsClients> {
    tracing::info!("Initialising AWS services");
    AwsClients::new(config).await
}

/// Returns a party's keypair used to decrypt iris code secret shares from SQS input queue.
async fn get_shares_encryption_key_pair(
    config: &Config,
    aws_clients: &AwsClients,
) -> Result<SharesEncryptionKeyPairs> {
    let key_pair_result = SharesEncryptionKeyPairs::from_storage(
        aws_clients.secrets_manager_client.clone(),
        &config.environment,
        &config.party_id,
    )
    .await;

    if let Err(e) = &key_pair_result {
        tracing::error!("Failed to initialize shares encryption key pairs: {:?}", e);
    }

    Ok(key_pair_result?)
}

struct SnsAttributesMaps {
    uniqueness_result_attributes: HashMap<String, MessageAttributeValue>,
    reauth_result_attributes: HashMap<String, MessageAttributeValue>,
    anonymized_statistics_attributes: HashMap<String, MessageAttributeValue>,
    identity_deletion_result_attributes: HashMap<String, MessageAttributeValue>,
}

/// Returns a set of attribute maps used to interact with AWS SNS.
///
/// Also replays recent SNS results to ensure delivery occurred in case of previous server failure.
async fn init_sns(
    config: &Config,
    aws_clients: &AwsClients,
    store: &Store,
) -> Result<SnsAttributesMaps> {
    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_result_attributes = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let anonymized_statistics_attributes =
        create_message_type_attribute_map(ANONYMIZED_STATISTICS_MESSAGE_TYPE);
    let identity_deletion_result_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);

    tracing::info!("Replaying results");
    send_results_to_sns(
        store.last_results(max_sync_lookback(config)).await?,
        &Vec::new(),
        &aws_clients.sns_client,
        config,
        &uniqueness_result_attributes,
        UNIQUENESS_MESSAGE_TYPE,
    )
    .await?;

    Ok(SnsAttributesMaps {
        uniqueness_result_attributes,
        reauth_result_attributes,
        anonymized_statistics_attributes,
        identity_deletion_result_attributes,
    })
}

/// Seeds iris dB with random shares.
///
/// Note: seeds only if store length is 0 and `config.init_db_size` > 0.
async fn maybe_seed_random_shares(config: &Config, iris_store: &Store) -> Result<()> {
    let store_len = iris_store.count_irises().await?;
    if store_len == 0 && config.init_db_size > 0 {
        tracing::info!(
            "Initialize persistent iris DB with {} randomly generated shares",
            config.init_db_size
        );
        tracing::info!("Resetting the db: {}", config.clear_db_before_init);
        iris_store
            .init_db_with_random_shares(
                RNG_SEED_INIT_DB,
                config.party_id,
                config.init_db_size,
                config.clear_db_before_init,
            )
            .await?
    }

    Ok(())
}

/// Conducts consistency checks on iris shares store.
async fn check_store_consistency(config: &Config, iris_store: &Store) -> Result<()> {
    let store_len = iris_store.count_irises().await?;
    tracing::info!("Size of the database after init: {}", store_len);

    // Check if the sequence id is consistent with the number of irises
    let max_serial_id = iris_store.get_max_serial_id().await?;
    if max_serial_id != store_len {
        tracing::error!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );

        bail!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );
    }

    if store_len > config.max_db_size {
        tracing::error!("Database size exceeds maximum allowed size: {}", store_len);
        bail!("Database size exceeds maximum allowed size: {}", store_len);
    }

    Ok(())
}

// Returns a new task monitor.
fn init_task_monitor() -> TaskMonitor {
    tracing::info!("Preparing task monitor");
    TaskMonitor::new()
}

/// Build this node's synchronization state, which is compared against the
/// states provided by the other MPC nodes to reconstruct a consistent initial
/// state for MPC operation.
async fn build_sync_state(
    config: &Config,
    aws_clients: &AwsClients,
    store: &Store,
) -> Result<SyncState> {
    let db_len = store.count_irises().await? as u64;
    let deleted_request_ids = store
        .last_deleted_requests(max_sync_lookback(config))
        .await?;
    let modifications = store.last_modifications(max_sync_lookback(config)).await?;
    let next_sns_sequence_num = get_next_sns_seq_num(config, &aws_clients.sqs_client).await?;
    let common_config = CommonConfig::from(config.clone());

    Ok(SyncState {
        db_len,
        deleted_request_ids,
        modifications,
        next_sns_sequence_num,
        common_config,
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ReadyProbeResponse {
    image_name: String,
    uuid: String,
    shutting_down: bool,
}

/// Initializes and starts HTTP server for coordinating healthcheck, readiness,
/// and synchronization between MPC nodes.
///
/// Note: returns a reference to a readiness flag, an `AtomicBool`, which can later
/// be set to indicate to other MPC nodes that this server is ready for operation.
async fn start_coordination_server(
    config: &Config,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    my_state: &SyncState,
) -> Arc<AtomicBool> {
    tracing::info!("⚓️ ANCHOR: Starting Healthcheck, Readiness and Sync server");

    let is_ready_flag = Arc::new(AtomicBool::new(false));

    let health_shutdown_handler = Arc::clone(shutdown_handler);
    let health_check_port = config.hawk_server_healthcheck_port;

    let _health_check_abort = task_monitor.spawn({
        let uuid = uuid::Uuid::new_v4().to_string();
        let is_ready_flag = Arc::clone(&is_ready_flag);
        let ready_probe_response = ReadyProbeResponse {
            image_name: config.image_name.clone(),
            shutting_down: false,
            uuid: uuid.clone(),
        };
        let ready_probe_response_shutdown = ReadyProbeResponse {
            image_name: config.image_name.clone(),
            shutting_down: true,
            uuid: uuid.clone(),
        };
        let serialized_response = serde_json::to_string(&ready_probe_response)
            .expect("Serialization to JSON to probe response failed");
        let serialized_response_shutdown = serde_json::to_string(&ready_probe_response_shutdown)
            .expect("Serialization to JSON to probe response failed");
        tracing::info!("Healthcheck probe response: {}", serialized_response);
        let my_state = my_state.clone();
        async move {
            // Generate a random UUID for each run.
            let app = Router::new()
                .route(
                    "/health",
                    get(move || {
                        let shutdown_handler_clone = Arc::clone(&health_shutdown_handler);
                        async move {
                            if shutdown_handler_clone.is_shutting_down() {
                                serialized_response_shutdown.clone()
                            } else {
                                serialized_response.clone()
                            }
                        }
                    }),
                )
                .route(
                    "/ready",
                    get({
                        // We are only ready once this flag is set to true.
                        let is_ready_flag = Arc::clone(&is_ready_flag);
                        move || async move {
                            if is_ready_flag.load(Ordering::SeqCst) {
                                "ready".into_response()
                            } else {
                                StatusCode::SERVICE_UNAVAILABLE.into_response()
                            }
                        }
                    }),
                )
                .route(
                    "/startup-sync",
                    get(move || async move { serde_json::to_string(&my_state).unwrap() }),
                );
            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", health_check_port))
                .await
                .wrap_err("healthcheck listener bind error")?;
            axum::serve(listener, app)
                .await
                .wrap_err("healthcheck listener server launch error")?;

            Ok::<(), Error>(())
        }
    });

    tracing::info!(
        "Healthcheck and Readiness server running on port {}.",
        health_check_port.clone()
    );

    is_ready_flag
}

/// Awaits until other MPC nodes respond to "ready" queries
/// indicating that their coordination servers are running.
///
/// Note: The response to this query is expected initially to be `503 Service Unavailable`.
async fn wait_for_others_unready(config: &Config) -> Result<()> {
    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be un-ready (syncing on startup)");
    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "ready",
    );

    let party_id = config.party_id;

    let unready_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(party_id + 2) % 3];
        let mut connected_but_unready = [false, false];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;

                if res.is_ok() && res.unwrap().status() == StatusCode::SERVICE_UNAVAILABLE {
                    connected_but_unready[i] = true;
                    // If all nodes are connected, notify the main thread.
                    if connected_but_unready.iter().all(|&c| c) {
                        return;
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to be unready...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        unready_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to be unready.");
            return Err(eyre!("Timeout waiting for all nodes to be unready."));
        }
    };
    tracing::info!("All nodes are starting up.");

    Ok(())
}

/// Starts a heartbeat task which periodically polls the "health" endpoints of
/// all other MPC nodes to ensure that the other nodes are still running and
/// responding to network requests.
async fn init_heartbeat_task(
    config: &Config,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);

    let all_health_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "health",
    );

    let party_id = config.party_id;
    let image_name = config.image_name.clone();
    let heartbeat_initial_retries = config.heartbeat_initial_retries;
    let heartbeat_interval_secs = config.heartbeat_interval_secs;

    let heartbeat_shutdown_handler = Arc::clone(shutdown_handler);
    let _heartbeat = task_monitor.spawn(async move {
        let next_node = &all_health_addresses[(party_id + 1) % 3];
        let prev_node = &all_health_addresses[(party_id + 2) % 3];
        let mut last_response = [String::default(), String::default()];
        let mut connected = [false, false];
        let mut retries = [0, 0];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;
                if res.is_err() || !res.as_ref().unwrap().status().is_success() {
                    // If it's the first time after startup, we allow a few retries to let the other
                    // nodes start up as well.
                    if last_response[i] == String::default()
                        && retries[i] < heartbeat_initial_retries
                    {
                        retries[i] += 1;
                        tracing::warn!("Node {} did not respond with success, retrying...", host);
                        continue;
                    }
                    tracing::info!(
                        "Node {} did not respond with success, starting graceful shutdown",
                        host
                    );
                    // if the nodes are still starting up and they get a failure - we can panic and
                    // not start graceful shutdown
                    if last_response[i] == String::default() {
                        panic!(
                            "Node {} did not respond with success during heartbeat init phase, \
                             killing server...",
                            host
                        );
                    }

                    if !heartbeat_shutdown_handler.is_shutting_down() {
                        heartbeat_shutdown_handler.trigger_manual_shutdown();
                        tracing::error!(
                            "Node {} has not completed health check, therefore graceful shutdown \
                             has been triggered",
                            host
                        );
                    } else {
                        tracing::info!("Node {} has already started graceful shutdown.", host);
                    }
                    continue;
                }

                let probe_response = res
                    .unwrap()
                    .json::<ReadyProbeResponse>()
                    .await
                    .expect("Deserialization of probe response failed");
                if probe_response.image_name != image_name {
                    // Do not create a panic as we still can continue to process before its
                    // updated
                    tracing::error!(
                        "Host {} is using image {} which differs from current node image: {}",
                        host,
                        probe_response.image_name.clone(),
                        image_name
                    );
                }
                if last_response[i] == String::default() {
                    last_response[i] = probe_response.uuid;
                    connected[i] = true;

                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        if let Some(tx) = heartbeat_tx.take() {
                            tx.send(()).unwrap();
                        }
                    }
                } else if probe_response.uuid != last_response[i] {
                    // If the UUID response is different, the node has restarted without us
                    // noticing. Our main NCCL connections cannot recover from
                    // this, so we panic.
                    panic!("Node {} seems to have restarted, killing server...", host);
                } else if probe_response.shutting_down {
                    tracing::info!("Node {} has starting graceful shutdown", host);

                    if !heartbeat_shutdown_handler.is_shutting_down() {
                        heartbeat_shutdown_handler.trigger_manual_shutdown();
                        tracing::error!(
                            "Node {} has starting graceful shutdown, therefore triggering \
                             graceful shutdown",
                            host
                        );
                    }
                } else {
                    tracing::info!("Heartbeat: Node {} is healthy", host);
                }
            }

            tokio::time::sleep(Duration::from_secs(heartbeat_interval_secs)).await;
        }
    });

    tracing::info!("Heartbeat starting...");
    heartbeat_rx.await?;
    tracing::info!("Heartbeat on all nodes started.");

    Ok(())
}

/// Retrieves synchronization state of other MPC nodes.  This data is
/// used to ensure that all nodes are in a consistent state prior
/// to starting MPC operations.
async fn get_others_sync_state(config: &Config, my_state: &SyncState) -> Result<SyncResult> {
    tracing::info!("⚓️ ANCHOR: Syncing latest node state");

    let all_startup_sync_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "startup-sync",
    );

    let next_node = &all_startup_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_startup_sync_addresses[(config.party_id + 2) % 3];

    tracing::info!("Database store length is: {}", my_state.db_len);
    let mut states = vec![my_state.clone()];
    for host in [next_node, prev_node].iter() {
        let res = reqwest::get(host.as_str()).await;
        match res {
            Ok(res) => {
                let state: SyncState = match res.json().await {
                    Ok(state) => state,
                    Err(e) => {
                        tracing::error!("Failed to parse sync state from party {}: {:?}", host, e);
                        panic!(
                            "could not get sync state from party {}, trying to restart",
                            host
                        );
                    }
                };
                states.push(state);
            }
            Err(e) => {
                tracing::error!("Failed to fetch sync state from party {}: {:?}", host, e);
                panic!(
                    "could not get sync state from party {}, trying to restart",
                    host
                );
            }
        }
    }
    Ok(SyncResult::new(my_state.clone(), states))
}

/// If enabled in `config.enable_sync_queues_on_sns_sequence_number`, delete stale
/// SQS messages in requests queue with sequence number older than the most
/// recent sequence number seen by any MPC party.
async fn maybe_sync_sqs_queues(
    config: &Config,
    sync_result: &SyncResult,
    aws_clients: &AwsClients,
) -> Result<()> {
    if config.enable_sync_queues_on_sns_sequence_number {
        let max_sqs_sequence_num = sync_result.max_sns_sequence_num();
        delete_messages_until_sequence_num(
            config,
            &aws_clients.sqs_client,
            sync_result.my_state.next_sns_sequence_num,
            max_sqs_sequence_num,
        )
        .await?;
    }

    Ok(())
}

/// Synchronize iris databases if needed by rolling back to smallest height
/// among the MPC parties.  Rollback fails if number of rolled back entries
/// is greater than a fixed maximum rollback amount determined by the
/// configuration parameters.
async fn sync_dbs_rollback(
    config: &Config,
    sync_result: &SyncResult,
    iris_store: &Store,
) -> Result<()> {
    let my_db_len = iris_store.count_irises().await?;

    if let Some(min_db_len) = sync_result.must_rollback_storage() {
        tracing::error!("Databases are out-of-sync: {:?}", sync_result);
        if min_db_len + max_rollback(config) < my_db_len {
            return Err(eyre!(
                "Refusing to rollback so much (from {} to {})",
                my_db_len,
                min_db_len,
            ));
        }
        tracing::warn!(
            "Rolling back from database length {} to other nodes length {}",
            my_db_len,
            min_db_len
        );
        iris_store.rollback(min_db_len).await?;
        metrics::counter!("db.sync.rollback").increment(1);
    }

    // refetch store_len in case we rolled back
    let store_len = iris_store.count_irises().await?;
    tracing::info!("Size of the database after sync: {}", store_len);

    Ok(())
}

/// Initialize main Hawk actor process for handling query batches using HNSW
/// approximate k-nearest neighbors graph search.
async fn init_hawk_actor(config: &Config) -> Result<HawkActor> {
    let node_addresses: Vec<String> = config
        .node_hostnames
        .iter()
        .zip(config.service_ports.iter())
        .map(|(host, port)| format!("{}:{}", host, port))
        .collect();

    let hawk_args = HawkArgs {
        party_index: config.party_id,
        addresses: node_addresses.clone(),
        request_parallelism: config.hawk_request_parallelism,
        connection_parallelism: config.hawk_connection_parallelism,
        hnsw_prng_seed: config.hawk_prng_seed,
        disable_persistence: config.cpu_disable_persistence,
        match_distances_buffer_size: config.match_distances_buffer_size,
        n_buckets: config.n_buckets,
    };

    tracing::info!(
        "Initializing HawkActor with args: party_index: {}, addresses: {:?}",
        hawk_args.party_index,
        node_addresses
    );

    HawkActor::from_cli(&hawk_args).await
}

/// Loads iris code shares & HNSW graph from Postgres and/or S3.
async fn load_database(
    config: &Config,
    iris_store: &Store,
    graph_store: &GraphPg<Aby3Store>,
    aws_clients: &AwsClients,
    shutdown_handler: &Arc<ShutdownHandler>,
    hawk_actor: &mut HawkActor,
) -> Result<()> {
    // ANCHOR: Load the database
    tracing::info!("⚓️ ANCHOR: Load the database");
    let (mut iris_loader, graph_loader) = hawk_actor.as_iris_loader().await;

    // TODO: not needed?
    if config.fake_db_size > 0 {
        iris_loader.fake_db(config.fake_db_size);
        return Ok(());
    }

    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    tracing::info!(
        "Initialize iris db: Loading from DB (parallelism: {})",
        parallelism
    );
    let download_shutdown_handler = Arc::clone(shutdown_handler);

    let store_len = iris_store.count_irises().await?;

    let s3_loader_params = S3LoaderParams {
        db_chunks_s3_store: S3Store::new(
            aws_clients.db_chunks_s3_client.clone(),
            config.db_chunks_bucket_name.clone(),
        ),
        db_chunks_s3_client: aws_clients.db_chunks_s3_client.clone(),
        s3_chunks_folder_name: config.db_chunks_folder_name.clone(),
        s3_chunks_bucket_name: config.db_chunks_bucket_name.clone(),
        s3_load_parallelism: config.load_chunks_parallelism,
        s3_load_max_retries: config.load_chunks_max_retries,
        s3_load_initial_backoff_ms: config.load_chunks_initial_backoff_ms,
    };

    load_db(
        &mut iris_loader,
        iris_store,
        store_len,
        parallelism,
        config,
        Some(s3_loader_params),
        download_shutdown_handler,
    )
    .await
    .expect("Failed to load DB");

    graph_loader.load_graph_store(graph_store).await?;

    Ok(())
}

/// Spawns thread responsible for communicating back results from batch query processing.
async fn start_results_thread(
    config: &Config,
    iris_store: &Store,
    graph_store: GraphPg<Aby3Store>,
    aws_clients: &AwsClients,
    task_monitor: &mut TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    sns_attributes_maps: SnsAttributesMaps,
) -> Result<Sender<ServerJobResult>> {
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let sns_client_bg = aws_clients.sns_client.clone();
    let config_bg = config.clone();
    let store_bg = iris_store.clone();
    let shutdown_handler_bg = Arc::clone(shutdown_handler);
    let party_id = config.party_id;
    let _result_sender_abort = task_monitor.spawn(async move {
        while let Some(job_result) = rx.recv().await {
            if let Err(e) = process_job_result(
                job_result,
                party_id,
                &store_bg,
                &graph_store,
                &sns_client_bg,
                &config_bg,
                &sns_attributes_maps.uniqueness_result_attributes,
                &sns_attributes_maps.reauth_result_attributes,
                &sns_attributes_maps.identity_deletion_result_attributes,
                &sns_attributes_maps.anonymized_statistics_attributes,
                &shutdown_handler_bg,
            )
            .await
            {
                tracing::error!("Error processing job result: {:?}", e);
            }
        }

        Ok(())
    });

    Ok(tx)
}

/// Toggle `is_ready_flag` to `true` to signal to other nodes that this node
/// is ready to execute the main server loop.
fn set_node_ready(is_ready_flag: Arc<AtomicBool>) {
    tracing::info!("⚓️ ANCHOR: Enable readiness and check all nodes");

    // Set readiness flag to true, i.e. ensure readiness server returns a 200 status code.
    is_ready_flag.store(true, Ordering::SeqCst);
}

/// Awaits until other MPC nodes respond to "ready" queries
/// indicating readiness to execute the main server loop.
async fn wait_for_others_ready(config: &Config) -> Result<()> {
    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "ready",
    );

    let party_id = config.party_id;
    let ready_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(party_id + 2) % 3];
        let mut connected = [false, false];

        loop {
            for (i, host) in [next_node, prev_node].iter().enumerate() {
                let res = reqwest::get(host.as_str()).await;

                if res.is_ok() && res.as_ref().unwrap().status().is_success() {
                    connected[i] = true;
                    // If all nodes are connected, notify the main thread.
                    if connected.iter().all(|&c| c) {
                        return;
                    }
                }
            }

            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    tracing::info!("Waiting for all nodes to be ready...");
    match tokio::time::timeout(
        Duration::from_secs(config.startup_sync_timeout_secs),
        ready_check,
    )
    .await
    {
        Ok(res) => {
            res?;
        }
        Err(_) => {
            tracing::error!("Timeout waiting for all nodes to be ready.");
            return Err(eyre!("Timeout waiting for all nodes to be ready."));
        }
    }
    tracing::info!("All nodes are ready.");

    Ok(())
}

/// Runs main processing loop in this thread.  Batches of requests are read
/// from the SQS input queue, and are passed to a `HawkHandle` processer task,
/// which distributes tasks among different threads and gRPC network sessions
/// to execute appropriate computations via MPC.  Once a batch is processed,
/// the results are passed to the results processing thread to be finalized
/// and communicated out.
#[allow(clippy::too_many_arguments)]
async fn run_main_server_loop(
    config: &Config,
    iris_store: &Store,
    aws_clients: &AwsClients,
    shares_encryption_key_pair: SharesEncryptionKeyPairs,
    sync_result: &SyncResult,
    mut task_monitor: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    hawk_actor: HawkActor,
    tx_results: Sender<ServerJobResult>,
) -> Result<()> {
    // --------------------------------------------------------------------------
    // ANCHOR: Start the main loop
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Start the main loop");

    let mut hawk_handle = HawkHandle::new(hawk_actor).await?;

    let mut skip_request_ids = sync_result.deleted_request_ids();

    let party_id = config.party_id;

    let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
    let uniqueness_error_result_attribute =
        create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_error_result_attribute = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let res: Result<()> = async {
        tracing::info!("Entering main loop");

        // Skip requests based on the startup sync, only in the first iteration.
        let skip_request_ids = mem::take(&mut skip_request_ids);

        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above

        let mut next_batch = receive_batch(
            party_id,
            &aws_clients.sqs_client,
            &aws_clients.sns_client,
            &aws_clients.s3_client,
            config,
            iris_store,
            &skip_request_ids,
            shares_encryption_key_pair.clone(),
            shutdown_handler,
            &uniqueness_error_result_attribute,
            &reauth_error_result_attribute,
        );

        let dummy_shares_for_deletions = get_dummy_shares_for_deletion(party_id);

        loop {
            let now = Instant::now();

            let _batch = next_batch.await?;
            if _batch.is_none() {
                tracing::info!("No more batches to process, exiting main loop");
                return Ok(());
            }
            let batch = _batch.unwrap();

            // start trace span - with single TraceId and single ParentTraceID
            tracing::info!("Received batch in {:?}", now.elapsed());

            metrics::histogram!("receive_batch_duration").record(now.elapsed().as_secs_f64());

            process_identity_deletions(
                &batch,
                iris_store,
                &dummy_shares_for_deletions.0,
                &dummy_shares_for_deletions.1,
            )
            .await?;

            // Iterate over a list of tracing payloads, and create logs with mappings to
            // payloads Log at least a "start" event using a log with trace.id and
            // parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Started processing share",
                );
            }

            task_monitor.check_tasks();

            let result_future = hawk_handle.submit_batch_query(batch.clone());

            next_batch = receive_batch(
                party_id,
                &aws_clients.sqs_client,
                &aws_clients.sns_client,
                &aws_clients.s3_client,
                config,
                iris_store,
                &skip_request_ids,
                shares_encryption_key_pair.clone(),
                shutdown_handler,
                &uniqueness_error_result_attribute,
                &reauth_error_result_attribute,
            );

            // await the result
            let result = timeout(processing_timeout, result_future.await)
                .await
                .map_err(|e| eyre!("HawkActor processing timeout: {:?}", e))??;

            tx_results.send(result).await?;

            shutdown_handler.increment_batches_pending_completion()
            // wrap up tracing span context
        }
    }
    .await;

    match res {
        Ok(_) => {
            tracing::info!(
                "Main loop exited normally. Waiting for last batch results to be processed before \
                 shutting down..."
            );

            shutdown_handler.wait_for_pending_batches_completion().await;
        }
        Err(e) => {
            tracing::error!("HawkActor processing error: {:?}", e);
            // drop actor handle to initiate shutdown
            drop(hawk_handle);

            // Clean up server tasks, then wait for them to finish
            task_monitor.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;

            // Check for background task hangs and shutdown panics
            task_monitor.check_tasks_finished();
        }
    }
    Ok(())
}

/// Main logic for initialization and execution of server nodes for genesis
/// indexing.  This setup builds a new HNSW graph via MPC insertion of secret
/// shared iris codes in a database snapshot.  In particular, this indexer
/// mode does not make use of AWS services, instead processing entries from
/// an isolated database snapshot of previously validated unique iris shares.
pub async fn server_main_genesis(config: Config) -> Result<()> {
    let shutdown_handler = init_shutdown_handler(&config).await;

    process_config(&config);

    let (iris_store, graph_store) = prepare_stores(&config).await?;

    // skip: init_aws_services
    // skip: get_shares_encryption_key_pair
    // skip: init_sns

    maybe_seed_random_shares(&config, &iris_store).await?;
    check_store_consistency(&config, &iris_store).await?;
    let my_state = build_genesis_sync_state(&config, &iris_store).await?;

    let mut background_tasks = init_task_monitor();

    let is_ready_flag =
        start_coordination_server(&config, &mut background_tasks, &shutdown_handler, &my_state)
            .await;

    background_tasks.check_tasks();

    wait_for_others_unready(&config).await?;
    init_heartbeat_task(&config, &mut background_tasks, &shutdown_handler).await?;

    background_tasks.check_tasks();

    let sync_result = get_others_sync_state(&config, &my_state).await?;
    sync_result.check_common_config()?;

    // skip: maybe_sync_sqs_queues
    sync_dbs_genesis(&config, &sync_result, &iris_store).await?;

    if shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    let mut hawk_actor = init_hawk_actor(&config).await?;

    load_database_genesis(
        &config,
        &iris_store,
        &graph_store,
        &shutdown_handler,
        &mut hawk_actor,
    )
    .await?;

    background_tasks.check_tasks();

    // skip: start_results_thread

    set_node_ready(is_ready_flag);
    wait_for_others_ready(&config).await?;

    background_tasks.check_tasks();

    run_genesis_main_server_loop(
        &config,
        &iris_store,
        &sync_result,
        background_tasks,
        &shutdown_handler,
        hawk_actor,
    )
    .await?;

    Ok(())
}

/// Build this node's synchronization state, which is compared against the
/// states provided by the other MPC nodes to reconstruct a consistent initial
/// state for MPC operation.
async fn build_genesis_sync_state(config: &Config, store: &Store) -> Result<SyncState> {
    let db_len = store.count_irises().await? as u64;
    let common_config = CommonConfig::from(config.clone());

    // TODO are any of these meaningful, or should they be given empty values?
    let deleted_request_ids = store
        .last_deleted_requests(max_sync_lookback(config))
        .await?;
    let modifications = store.last_modifications(max_sync_lookback(config)).await?;

    let next_sns_sequence_num = None;

    Ok(SyncState {
        db_len,
        deleted_request_ids,
        modifications,
        next_sns_sequence_num,
        common_config,
    })
}

async fn sync_dbs_genesis(
    _config: &Config,
    _sync_result: &SyncResult,
    _iris_store: &Store,
) -> Result<()> {
    todo!();
}

async fn load_database_genesis(
    config: &Config,
    iris_store: &Store,
    graph_store: &GraphPg<Aby3Store>,
    shutdown_handler: &Arc<ShutdownHandler>,
    hawk_actor: &mut HawkActor,
) -> Result<()> {
    // ANCHOR: Load the database
    tracing::info!("⚓️ ANCHOR: Load the database");
    let (mut iris_loader, graph_loader) = hawk_actor.as_iris_loader().await;

    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    tracing::info!(
        "Initialize iris db: Loading from DB (parallelism: {})",
        parallelism
    );
    let download_shutdown_handler = Arc::clone(shutdown_handler);

    // -------------------------------------------------------------------
    // TODO: use the number of currently processed entries for the amount
    //       to read into memory
    // -------------------------------------------------------------------
    let store_len = iris_store.count_irises().await?;

    load_db::<S3Store>(
        &mut iris_loader,
        iris_store,
        store_len,
        parallelism,
        config,
        None,
        download_shutdown_handler,
    )
    .await
    .expect("Failed to load DB");

    graph_loader.load_graph_store(graph_store).await?;

    Ok(())
}

async fn run_genesis_main_server_loop(
    _config: &Config,
    _iris_store: &Store,
    _sync_result: &SyncResult,
    mut _task_monitor: TaskMonitor,
    _shutdown_handler: &Arc<ShutdownHandler>,
    _hawk_actor: HawkActor,
) -> Result<()> {
    todo!()
}

// TODO genesis "num_processed" state flag

// TODO genesis results produced in large batches, update written to temporary
// table, then update applied to graph

// DB sync possibly should support limited rollback?

// DB loading should use num_processed value to choose number of entries to load
