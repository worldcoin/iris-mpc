use crate::services::aws::clients::AwsClients;
use crate::services::processors::batch::receive_batch;
use crate::services::processors::job::process_job_result;
use crate::services::processors::process_identity_deletions;
use crate::services::processors::result_message::send_results_to_sns;
use crate::services::store::load_db;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{eyre, Report, WrapErr};
use iris_mpc_common::config::{Config, ModeOfCompute, ModeOfDeployment};
use iris_mpc_common::helpers::batch_sync::get_own_batch_sync_state;
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
use iris_mpc_common::helpers::utils::get_check_addresses;
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
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

const RNG_SEED_INIT_DB: u64 = 42;
pub const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
pub const MAX_CONCURRENT_REQUESTS: usize = 32;

pub async fn server_main(config: Config) -> eyre::Result<()> {
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.wait_for_shutdown_signal().await;

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
    let max_sync_lookback: usize = config.max_batch_size * 2;
    let max_rollback: usize = config.max_batch_size * 2;
    tracing::info!("Set batch size to {}", config.max_batch_size);

    let (store, graph_store) = prepare_stores(&config).await?;

    tracing::info!("Initialising AWS services");
    let aws_clients = AwsClients::new(&config.clone()).await?;
    let next_sns_seq_number_future = get_next_sns_seq_num(&config, &aws_clients.sqs_client);

    let shares_encryption_key_pair = match SharesEncryptionKeyPairs::from_storage(
        aws_clients.secrets_manager_client,
        &config.environment,
        &config.party_id,
    )
    .await
    {
        Ok(key_pair) => key_pair,
        Err(e) => {
            tracing::error!("Failed to initialize shares encryption key pairs: {:?}", e);
            return Ok(());
        }
    };

    let party_id = config.party_id;
    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_result_attributes = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let anonymized_statistics_attributes =
        create_message_type_attribute_map(ANONYMIZED_STATISTICS_MESSAGE_TYPE);
    let identity_deletion_result_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);
    tracing::info!("Replaying results");
    send_results_to_sns(
        store.last_results(max_sync_lookback).await?,
        &Vec::new(),
        &aws_clients.sns_client,
        &config,
        &uniqueness_result_attributes,
        UNIQUENESS_MESSAGE_TYPE,
    )
    .await?;

    let store_len = store.count_irises().await?;

    tracing::info!("Size of the database before init: {}", store_len);

    // Seed the persistent storage with random shares if configured and db is still
    // empty.
    if store_len == 0 && config.init_db_size > 0 {
        tracing::info!(
            "Initialize persistent iris DB with {} randomly generated shares",
            config.init_db_size
        );
        tracing::info!("Resetting the db: {}", config.clear_db_before_init);
        store
            .init_db_with_random_shares(
                RNG_SEED_INIT_DB,
                party_id,
                config.init_db_size,
                config.clear_db_before_init,
            )
            .await?;
    }

    // Fetch again in case we've just initialized the DB
    let store_len = store.count_irises().await?;

    tracing::info!("Size of the database after init: {}", store_len);

    // Check if the sequence id is consistent with the number of irises
    let max_serial_id = store.get_max_serial_id().await?;
    if max_serial_id != store_len {
        tracing::error!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );

        eyre::bail!(
            "Detected inconsistency between max serial id {} and db size {}.",
            max_serial_id,
            store_len
        );
    }

    if store_len > config.max_db_size {
        tracing::error!("Database size exceeds maximum allowed size: {}", store_len);
        eyre::bail!("Database size exceeds maximum allowed size: {}", store_len);
    }

    tracing::info!("Preparing task monitor");
    let mut background_tasks = TaskMonitor::new();

    // --------------------------------------------------------------------------
    // ANCHOR: Starting Healthcheck, Readiness and Sync server
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Starting Healthcheck, Readiness and Sync server");

    let is_ready_flag = Arc::new(AtomicBool::new(false));
    let is_ready_flag_cloned = Arc::clone(&is_ready_flag);

    let my_state = SyncState {
        db_len: store_len as u64,
        deleted_request_ids: store.last_deleted_requests(max_sync_lookback).await?,
        modifications: store.last_modifications(max_sync_lookback).await?,
        next_sns_sequence_num: next_sns_seq_number_future.await?,
    };

    #[derive(Debug, Serialize, Deserialize, Clone)]
    struct ReadyProbeResponse {
        image_name: String,
        uuid: String,
        shutting_down: bool,
    }

    let health_shutdown_handler = Arc::clone(&shutdown_handler);
    let health_check_port = config.hawk_server_healthcheck_port;

    let _health_check_abort = background_tasks.spawn({
        let uuid = uuid::Uuid::new_v4().to_string();
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
        let sqs_client = aws_clients.sqs_client.clone();
        let config = config.clone();
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
                )
                .route(
                    "/batch-sync-state",
                    get(move || async move {
                        let batch_sync_state = get_own_batch_sync_state(&config, &sqs_client)
                            .await
                            .unwrap();
                        serde_json::to_string(&batch_sync_state).unwrap()
                    }),
                );

            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", health_check_port))
                .await
                .wrap_err("healthcheck listener bind error")?;
            axum::serve(listener, app)
                .await
                .wrap_err("healthcheck listener server launch error")?;

            Ok::<(), eyre::Error>(())
        }
    });

    background_tasks.check_tasks();
    tracing::info!(
        "Healthcheck and Readiness server running on port {}.",
        health_check_port.clone()
    );

    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be un-ready (syncing on startup)");
    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "ready",
    );

    let unready_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(config.party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(config.party_id + 2) % 3];
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

    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);

    let all_health_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "health",
    );

    let image_name = config.image_name.clone();
    let heartbeat_shutdown_handler = Arc::clone(&shutdown_handler);
    let _heartbeat = background_tasks.spawn(async move {
        let next_node = &all_health_addresses[(config.party_id + 1) % 3];
        let prev_node = &all_health_addresses[(config.party_id + 2) % 3];
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
                        && retries[i] < config.heartbeat_initial_retries
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

            tokio::time::sleep(Duration::from_secs(config.heartbeat_interval_secs)).await;
        }
    });

    tracing::info!("Heartbeat starting...");
    heartbeat_rx.await?;
    tracing::info!("Heartbeat on all nodes started.");
    let download_shutdown_handler = Arc::clone(&shutdown_handler);

    background_tasks.check_tasks();

    // Start the actor in separate task.
    // A bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
    let parallelism = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    let s3_load_parallelism = config.load_chunks_parallelism;
    let s3_chunks_bucket_name = config.db_chunks_bucket_name.clone();
    let s3_chunks_folder_name = config.db_chunks_folder_name.clone();
    let s3_load_max_retries = config.load_chunks_max_retries;
    let s3_load_initial_backoff_ms = config.load_chunks_initial_backoff_ms;

    // --------------------------------------------------------------------------
    // ANCHOR: Syncing latest node state
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Syncing latest node state");

    let all_startup_sync_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "startup-sync",
    );

    let next_node = &all_startup_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_startup_sync_addresses[(config.party_id + 2) % 3];

    tracing::info!("Database store length is: {}", store_len);
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
    let sync_result = SyncResult::new(my_state.clone(), states);

    // sync the queues
    if config.enable_sync_queues_on_sns_sequence_number {
        let max_sqs_sequence_num = sync_result.max_sns_sequence_num();
        delete_messages_until_sequence_num(
            &config,
            &aws_clients.sqs_client,
            my_state.next_sns_sequence_num,
            max_sqs_sequence_num,
        )
        .await?;
    }

    if let Some(db_len) = sync_result.must_rollback_storage() {
        tracing::error!("Databases are out-of-sync: {:?}", sync_result);
        if db_len + max_rollback < store_len {
            return Err(eyre!(
                "Refusing to rollback so much (from {} to {})",
                store_len,
                db_len,
            ));
        }
        tracing::warn!(
            "Rolling back from database length {} to other nodes length {}",
            store_len,
            db_len
        );
        store.rollback(db_len).await?;
        metrics::counter!("db.sync.rollback").increment(1);
    }

    if download_shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    // refetch store_len in case we rolled back
    let store_len = store.count_irises().await?;
    tracing::info!("Database store length after sync: {}", store_len);

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
        disable_persistence: config.cpu_disable_persistence,
        match_distances_buffer_size: config.match_distances_buffer_size,
        n_buckets: config.n_buckets,
    };

    tracing::info!(
        "Initializing HawkActor with args: party_index: {}, addresses: {:?}",
        hawk_args.party_index,
        node_addresses
    );

    let mut hawk_actor = HawkActor::from_cli(&hawk_args).await?;

    {
        // ANCHOR: Load the database
        tracing::info!("⚓️ ANCHOR: Load the database");
        let (mut iris_loader, graph_loader) = hawk_actor.as_iris_loader().await;

        if config.fake_db_size > 0 {
            // TODO: not needed?
            iris_loader.fake_db(config.fake_db_size);
        } else {
            tracing::info!(
                "Initialize iris db: Loading from DB (parallelism: {})",
                parallelism
            );
            let download_shutdown_handler = Arc::clone(&download_shutdown_handler);
            let db_chunks_s3_store = S3Store::new(
                aws_clients.db_chunks_s3_client.clone(),
                s3_chunks_bucket_name.clone(),
            );

            load_db(
                &mut iris_loader,
                &store,
                store_len,
                parallelism,
                &config,
                db_chunks_s3_store,
                aws_clients.db_chunks_s3_client,
                s3_chunks_folder_name,
                s3_chunks_bucket_name,
                s3_load_parallelism,
                s3_load_max_retries,
                s3_load_initial_backoff_ms,
                download_shutdown_handler,
            )
            .await
            .expect("Failed to load DB");

            graph_loader.load_graph_store(&graph_store).await?;
        }
    }

    let mut hawk_handle = HawkHandle::new(hawk_actor, 10).await?;

    let mut skip_request_ids = sync_result.deleted_request_ids();

    background_tasks.check_tasks();

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let sns_client_bg = aws_clients.sns_client.clone();
    let config_bg = config.clone();
    let store_bg = store.clone();
    let shutdown_handler_bg = Arc::clone(&shutdown_handler);
    let _result_sender_abort = background_tasks.spawn(async move {
        while let Some(job_result) = rx.recv().await {
            if let Err(e) = process_job_result(
                job_result,
                party_id,
                &store_bg,
                &graph_store,
                &sns_client_bg,
                &config_bg,
                &uniqueness_result_attributes,
                &reauth_result_attributes,
                &identity_deletion_result_attributes,
                &anonymized_statistics_attributes,
                &shutdown_handler_bg,
            )
            .await
            {
                tracing::error!("Error processing job result: {:?}", e);
            }
        }

        Ok(())
    });
    background_tasks.check_tasks();

    // --------------------------------------------------------------------------
    // ANCHOR: Enable readiness and check all nodes
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Enable readiness and check all nodes");

    // Set the readiness flag to true, which will make the readiness server return a
    // 200 status code.
    is_ready_flag_cloned.store(true, Ordering::SeqCst);

    // Check other nodes and wait until all nodes are ready.
    let all_readiness_addresses = get_check_addresses(
        config.node_hostnames.clone(),
        config.healthcheck_ports.clone(),
        "ready",
    );

    let ready_check = tokio::spawn(async move {
        let next_node = &all_readiness_addresses[(config.party_id + 1) % 3];
        let prev_node = &all_readiness_addresses[(config.party_id + 2) % 3];
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
    background_tasks.check_tasks();

    // --------------------------------------------------------------------------
    // ANCHOR: Start the main loop
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Start the main loop");

    let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
    let uniqueness_error_result_attribute =
        create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_error_result_attribute = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let res: eyre::Result<()> = async {
        tracing::info!("Entering main loop");
        // **Tensor format of queries**
        //
        // The functions `receive_batch` and `prepare_query_shares` will prepare the
        // _query_ variables as `Vec<Vec<u8>>` formatted as follows:
        //
        // - The inner Vec is a flattening of these dimensions (inner to outer):
        //   - One u8 limb of one iris bit.
        //   - One code: 12800 coefficients.
        //   - One query: all rotated variants of a code.
        //   - One batch: many queries.
        // - The outer Vec is the dimension of the Galois Ring (2):
        //   - A decomposition of each iris bit into two u8 limbs.

        // Skip requests based on the startup sync, only in the first iteration.
        let skip_request_ids = mem::take(&mut skip_request_ids);
        let shares_encryption_key_pair = shares_encryption_key_pair.clone();
        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above

        let mut next_batch = receive_batch(
            party_id,
            &aws_clients.sqs_client,
            &aws_clients.sns_client,
            &aws_clients.s3_client,
            &config,
            &store,
            &skip_request_ids,
            shares_encryption_key_pair.clone(),
            &shutdown_handler,
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
                &store,
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

            background_tasks.check_tasks();

            let result_future = hawk_handle.submit_batch_query(batch.clone());

            next_batch = receive_batch(
                party_id,
                &aws_clients.sqs_client,
                &aws_clients.sns_client,
                &aws_clients.s3_client,
                &config,
                &store,
                &skip_request_ids,
                shares_encryption_key_pair.clone(),
                &shutdown_handler,
                &uniqueness_error_result_attribute,
                &reauth_error_result_attribute,
            );

            // await the result
            let result = timeout(processing_timeout, result_future.await)
                .await
                .map_err(|e| eyre!("ServerActor processing timeout: {:?}", e))?;

            tx.send(result).await?;

            shutdown_handler.increment_batches_pending_completion()
            // wrap up span context
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
            tracing::error!("ServerActor processing error: {:?}", e);
            // drop actor handle to initiate shutdown
            drop(hawk_handle);

            // Clean up server tasks, then wait for them to finish
            background_tasks.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;

            // Check for background task hangs and shutdown panics
            background_tasks.check_tasks_finished();
        }
    }
    Ok(())
}

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
            let store = Store::new(&hawk_postgres_client).await?;

            // Graph -> CPU
            tracing::info!(
                "Creating new graph store from: {:?} in mode {:?}",
                hawk_db_config,
                config.mode_of_deployment
            );
            let graph_store = GraphStore::new(&hawk_postgres_client).await?;

            Ok((store, graph_store))
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

            let store = Store::new(&postgres_client).await?;

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

            Ok((store, graph_store))
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
            let store = Store::new(&postgres_client).await?;

            tracing::info!(
                "Creating new graph store from: {:?} in mode {:?}",
                db_config,
                config.mode_of_deployment
            );
            let graph_store = GraphStore::new(&postgres_client).await?;

            Ok((store, graph_store))
        }
    }
}
