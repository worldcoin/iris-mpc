mod utils;

use crate::server::utils::get_check_addresses;
use crate::services::aws::clients::AwsClients;
use crate::services::aws::sns::send_results_to_sns;
use crate::services::init::initialize_chacha_seeds;
use crate::services::processors::batch::receive_batch;
use crate::services::processors::process_identity_deletions;
use crate::services::store::load_db;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use eyre::{eyre, WrapErr};
use iris_mpc_common::config::{Config, ModeOfCompute};
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{
    ANONYMIZED_STATISTICS_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::{
    create_message_type_attribute_map, IdentityDeletionResult, ReAuthResult, UniquenessResult,
};
use iris_mpc_common::helpers::sync::{SyncResult, SyncState};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::iris_db::get_dummy_shares_for_deletion;
use iris_mpc_common::job::JobSubmissionHandle;
use iris_mpc_cpu::execution::hawk_main::{
    GraphStore, HawkActor, HawkArgs, HawkHandle, ServerJobResult,
};
use iris_mpc_store::{S3Store, Store, StoredIrisRef};
use serde::{Deserialize, Serialize};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

const RNG_SEED_INIT_DB: u64 = 42;

pub const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
pub const MAX_CONCURRENT_REQUESTS: usize = 32;
pub static CURRENT_BATCH_SIZE: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

pub async fn server_main(config: Config) -> eyre::Result<()> {
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.wait_for_shutdown_signal().await;

    // Validate modes of compute/deployment.
    if config.mode_of_compute != ModeOfCompute::CPU {
        panic!(
            "Invalid config setting: compute_mode: actual: {:?} :: expected: ModeOfCompute::CPU",
            config.mode_of_compute
        );
    } else {
        tracing::info!("Mode of compute: {:?}", config.mode_of_compute);
        tracing::info!("Mode of deployment: {:?}", config.mode_of_deployment);
    }

    // Load batch_size config
    *CURRENT_BATCH_SIZE.lock().unwrap() = config.max_batch_size;
    let max_sync_lookback: usize = config.max_batch_size * 2;
    let max_rollback: usize = config.max_batch_size * 2;
    tracing::info!("Set batch size to {}", config.max_batch_size);

    tracing::info!("Creating new storage from: {:?}", config);
    let store = Store::new_from_config(&config).await?;
    let graph_store = GraphStore::from_iris_store(&store);

    tracing::info!("Initialising AWS services");
    let aws_clients = AwsClients::new(&config.clone()).await?;

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
    tracing::info!("Deriving shared secrets");
    let _chacha_seeds = initialize_chacha_seeds(config.clone()).await?;

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
    let sync_result = SyncResult::new(my_state, states);

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
        disable_persistence: config.disable_persistence,
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
        while let Some(ServerJobResult {
            merged_results,
            request_ids,
            request_types,
            metadata,
            matches,
            matches_with_skip_persistence,
            match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            partial_match_counters_left,
            partial_match_counters_right,
            left_iris_requests,
            right_iris_requests,
            deleted_ids,
            matched_batch_request_ids,
            anonymized_bucket_statistics_left,
            anonymized_bucket_statistics_right,
            successful_reauths,
            reauth_target_indices,
            reauth_or_rule_used,
            modifications,
            actor_data: hawk_mutation,
        }) = rx.recv().await
        {
            let _modifications = modifications;

            // returned serial_ids are 0 indexed, but we want them to be 1 indexed
            let uniqueness_results = merged_results
                .iter()
                .enumerate()
                .filter(|(i, _)| request_types[*i] == UNIQUENESS_MESSAGE_TYPE)
                .map(|(i, &idx_result)| {
                    let result_event = UniquenessResult::new(
                        party_id,
                        match matches[i] {
                            true => None,
                            false => Some(idx_result + 1),
                        },
                        matches_with_skip_persistence[i],
                        request_ids[i].clone(),
                        match matches[i] {
                            true => Some(match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>()),
                            false => None,
                        },
                        match partial_match_ids_left[i].is_empty() {
                            false => Some(
                                partial_match_ids_left[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        match partial_match_ids_right[i].is_empty() {
                            false => Some(
                                partial_match_ids_right[i]
                                    .iter()
                                    .map(|x| x + 1)
                                    .collect::<Vec<_>>(),
                            ),
                            true => None,
                        },
                        Some(matched_batch_request_ids[i].clone()),
                        match partial_match_counters_left.is_empty() {
                            false => Some(partial_match_counters_left[i]),
                            true => None,
                        },
                        match partial_match_counters_right.is_empty() {
                            false => Some(partial_match_counters_right[i]),
                            true => None,
                        },
                    );

                    serde_json::to_string(&result_event).wrap_err("failed to serialize result")
                })
                .collect::<eyre::Result<Vec<_>>>()?;

            // Insert non-matching uniqueness queries into the persistent store.
            let (memory_serial_ids, codes_and_masks): (Vec<i64>, Vec<StoredIrisRef>) = matches
                .iter()
                .enumerate()
                .filter_map(
                    // Find the indices of non-matching queries in the batch.
                    |(query_idx, is_match)| if !is_match { Some(query_idx) } else { None },
                )
                .map(|query_idx| {
                    let serial_id = (merged_results[query_idx] + 1) as i64;
                    // Get the original vectors from `receive_batch`.
                    (
                        serial_id,
                        StoredIrisRef {
                            id: serial_id,
                            left_code: &left_iris_requests.code[query_idx].coefs[..],
                            left_mask: &left_iris_requests.mask[query_idx].coefs[..],
                            right_code: &right_iris_requests.code[query_idx].coefs[..],
                            right_mask: &right_iris_requests.mask[query_idx].coefs[..],
                        },
                    )
                })
                .unzip();

            let reauth_results = request_types
                .iter()
                .enumerate()
                .filter(|(_, request_type)| *request_type == REAUTH_MESSAGE_TYPE)
                .map(|(i, _)| {
                    let reauth_id = request_ids[i].clone();
                    let or_rule_used = reauth_or_rule_used.get(&reauth_id).unwrap();
                    let or_rule_matched = if *or_rule_used {
                        // if or rule was used and reauth was successful, then or rule was matched
                        Some(successful_reauths[i])
                    } else {
                        None
                    };
                    let result_event = ReAuthResult::new(
                        reauth_id.clone(),
                        party_id,
                        reauth_target_indices.get(&reauth_id).unwrap() + 1,
                        successful_reauths[i],
                        match_ids[i].iter().map(|x| x + 1).collect::<Vec<_>>(),
                        *reauth_or_rule_used.get(&reauth_id).unwrap(),
                        or_rule_matched,
                    );
                    serde_json::to_string(&result_event)
                        .wrap_err("failed to serialize reauth result")
                })
                .collect::<eyre::Result<Vec<_>>>()?;

            let mut tx = store_bg.tx().await?;

            store_bg
                .insert_results(&mut tx, &uniqueness_results)
                .await?;

            // TODO: update modifications table to store reauth and deletion results

            if !codes_and_masks.is_empty() && !config_bg.disable_persistence {
                let db_serial_ids = store_bg.insert_irises(&mut tx, &codes_and_masks).await?;

                // Check if the serial_ids match between memory and db.
                if memory_serial_ids != db_serial_ids {
                    tracing::error!(
                        "Serial IDs do not match between memory and db: {:?} != {:?}",
                        memory_serial_ids,
                        db_serial_ids
                    );
                    return Err(eyre!(
                        "Serial IDs do not match between memory and db: {:?} != {:?}",
                        memory_serial_ids,
                        db_serial_ids
                    ));
                }

                for (i, success) in successful_reauths.iter().enumerate() {
                    if !success {
                        continue;
                    }
                    let reauth_id = request_ids[i].clone();
                    // convert from memory index (0-based) to db index (1-based)
                    let serial_id = *reauth_target_indices.get(&reauth_id).unwrap() + 1;
                    tracing::info!(
                        "Persisting successful reauth update {} into postgres on serial id {} ",
                        reauth_id,
                        serial_id
                    );
                    store_bg
                        .update_iris(
                            Some(&mut tx),
                            serial_id as i64,
                            &left_iris_requests.code[i],
                            &left_iris_requests.mask[i],
                            &right_iris_requests.code[i],
                            &right_iris_requests.mask[i],
                        )
                        .await?;
                }
            }

            // Graph mutation.
            let mut graph_tx = graph_store.tx_wrap(tx);
            if !config_bg.disable_persistence {
                hawk_mutation.persist(&mut graph_tx).await?;
            }
            let tx = graph_tx.tx;

            tx.commit().await?;

            for memory_serial_id in memory_serial_ids {
                tracing::info!("Inserted serial_id: {}", memory_serial_id);
                metrics::gauge!("results_inserted.latest_serial_id").set(memory_serial_id as f64);
            }

            tracing::info!("Sending {} uniqueness results", uniqueness_results.len());
            send_results_to_sns(
                uniqueness_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &uniqueness_result_attributes,
                UNIQUENESS_MESSAGE_TYPE,
            )
            .await?;

            tracing::info!("Sending {} reauth results", reauth_results.len());
            send_results_to_sns(
                reauth_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &reauth_result_attributes,
                REAUTH_MESSAGE_TYPE,
            )
            .await?;

            // handling identity deletion results
            let identity_deletion_results = deleted_ids
                .iter()
                .map(|&serial_id| {
                    let result_event = IdentityDeletionResult::new(party_id, serial_id + 1, true);
                    serde_json::to_string(&result_event)
                        .wrap_err("failed to serialize identity deletion result")
                })
                .collect::<eyre::Result<Vec<_>>>()?;

            tracing::info!(
                "Sending {} identity deletion results",
                identity_deletion_results.len()
            );
            send_results_to_sns(
                identity_deletion_results,
                &metadata,
                &sns_client_bg,
                &config_bg,
                &identity_deletion_result_attributes,
                IDENTITY_DELETION_MESSAGE_TYPE,
            )
            .await?;

            if (config_bg.enable_sending_anonymized_stats_message)
                && (!anonymized_bucket_statistics_left.buckets.is_empty()
                    || !anonymized_bucket_statistics_right.buckets.is_empty())
            {
                tracing::info!("Sending anonymized stats results");
                let anonymized_statistics_results = [
                    anonymized_bucket_statistics_left,
                    anonymized_bucket_statistics_right,
                ];
                // transform to vector of string ands remove None values
                let anonymized_statistics_results = anonymized_statistics_results
                    .iter()
                    .map(|anonymized_bucket_statistics| {
                        serde_json::to_string(anonymized_bucket_statistics)
                            .wrap_err("failed to serialize anonymized statistics result")
                    })
                    .collect::<eyre::Result<Vec<_>>>()?;

                send_results_to_sns(
                    anonymized_statistics_results,
                    &metadata,
                    &sns_client_bg,
                    &config_bg,
                    &anonymized_statistics_attributes,
                    ANONYMIZED_STATISTICS_MESSAGE_TYPE,
                )
                .await?;
            }

            shutdown_handler_bg.decrement_batches_pending_completion();
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
