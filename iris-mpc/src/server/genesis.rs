use crate::server::utils;
use crate::services::aws::clients::AwsClients;
use crate::services::processors::batch::receive_batch;
use crate::services::processors::process_identity_deletions;
use crate::services::store::load_db;
use eyre::eyre;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE};
use iris_mpc_common::helpers::smpc_response::create_message_type_attribute_map;
use iris_mpc_common::helpers::sqs::delete_messages_until_sequence_num;
use iris_mpc_common::helpers::sync::{SyncResult, SyncState};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::iris_db::get_dummy_shares_for_deletion;
use iris_mpc_common::job::JobSubmissionHandle;
use iris_mpc_cpu::execution::hawk_main::{
    GraphStore, HawkActor, HawkArgs, HawkHandle, ServerJobResult,
};
use iris_mpc_store::{S3Store, Store};
use std::mem;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use tokio::time::timeout;

static CURRENT_BATCH_SIZE: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

pub async fn server_main(config: Config) -> eyre::Result<()> {
    // Shutdown if last results synchornization timed out.
    let shutdown_handler = Arc::new(ShutdownHandler::new(
        config.shutdown_last_results_sync_timeout_secs,
    ));
    shutdown_handler.wait_for_shutdown_signal().await;

    // Validate config - escape if invalid.
    tracing::info!("Validating config");
    utils::validate_config(&config);

    // Set batch size config settings.
    tracing::info!("Initialising processing batch size");
    *CURRENT_BATCH_SIZE.lock().unwrap() = config.max_batch_size;
    let max_sync_lookback: usize = config.max_batch_size * 2;
    let max_rollback: usize = config.max_batch_size * 2;
    tracing::info!("   batch size -> {}", config.max_batch_size);

    // Set backend service pointers.
    tracing::info!("Setting store pointers");
    let iris_pg_store = Store::new_from_config(&config).await?;
    let graph_pg_store = GraphStore::from_iris_store(&iris_pg_store);
    tracing::info!("Initialising AWS services");
    let aws_clients = AwsClients::new(&config.clone()).await?;

    // Validate length of iris store - escape if invalid.
    tracing::info!("Validating Iris store consistency");
    let store_len = utils::validate_iris_store_consistency(&config, &iris_pg_store).await?;

    // Set shares.
    tracing::info!("Setting shares encryption key");
    let shares_encryption_key_pair =
        utils::fetch_shares_encryption_key_pair(&config, aws_clients.secrets_manager_client)
            .await
            .unwrap();

    // Set background task monitor.
    tracing::info!("Setting task monitor");
    let mut background_tasks = TaskMonitor::new();

    // --------------------------------------------------------------------------
    // ANCHOR: Starting Healthcheck, Readiness and Sync server
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Starting Healthcheck, Readiness and Sync server");

    let is_ready_flag = Arc::new(AtomicBool::new(false));
    let is_ready_flag_cloned = Arc::clone(&is_ready_flag);

    let sync_state = utils::fetch_sync_state(
        &config,
        &iris_pg_store,
        &aws_clients.sqs_client,
        store_len,
        max_sync_lookback,
    )
    .await
    .unwrap();

    // On a background thread spawn the node's ops web service.
    background_tasks.spawn(
        utils::get_spinup_web_service_future(
            config.clone(),
            sync_state.clone(),
            Arc::clone(&shutdown_handler),
            Arc::clone(&is_ready_flag),
        )
        .await,
    );
    background_tasks.check_tasks();
    tracing::info!(
        "Healthcheck and Readiness server running on port {}.",
        config.hawk_server_healthcheck_port
    );

    tracing::info!("⚓️ ANCHOR: Waiting for other servers to be un-ready (syncing on startup)");
    utils::do_unreadiness_check(&config).await?;
    tracing::info!("All nodes are starting up.");

    let (heartbeat_tx, heartbeat_rx) = oneshot::channel();
    let mut heartbeat_tx = Some(heartbeat_tx);

    let all_health_addresses = utils::get_check_addresses(&config, "health");
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
                    .json::<utils::ReadyProbeResponse>()
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

    let all_startup_sync_addresses = utils::get_check_addresses(&config, "startup-sync");
    let next_node = &all_startup_sync_addresses[(config.party_id + 1) % 3];
    let prev_node = &all_startup_sync_addresses[(config.party_id + 2) % 3];

    tracing::info!("Database store length is: {}", store_len);
    let mut states = vec![sync_state.clone()];
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
    let sync_result = SyncResult::new(sync_state.clone(), states);

    // sync the queues
    if config.enable_sync_queues_on_sns_sequence_number {
        let max_sqs_sequence_num = sync_result.max_sns_sequence_num();
        delete_messages_until_sequence_num(
            &config,
            &aws_clients.sqs_client,
            sync_state.next_sns_sequence_num,
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
        iris_pg_store.rollback(db_len).await?;
        metrics::counter!("db.sync.rollback").increment(1);
    }

    if download_shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    // refetch store_len in case we rolled back
    let store_len = iris_pg_store.count_irises().await?;
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
        disable_persistence: config.disable_persistence,
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
                &iris_pg_store,
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

            graph_loader.load_graph_store(&graph_pg_store).await?;
        }
    }

    let mut hawk_handle = HawkHandle::new(hawk_actor, 10).await?;

    let mut skip_request_ids = sync_result.deleted_request_ids();

    background_tasks.check_tasks();

    // THREAD: Responsible for communicating back indexation results.
    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<ServerJobResult>(32); // TODO: pick some buffer value
    let _ = background_tasks.spawn(async move {
        while let Some(result) = rx.recv().await {
            println!("TODO: process job result: {:?}", result);
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
    let all_readiness_addresses = utils::get_check_addresses(&config, "ready");
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
            config.party_id,
            &aws_clients.sqs_client,
            &aws_clients.sns_client,
            &aws_clients.s3_client,
            &config,
            &iris_pg_store,
            &skip_request_ids,
            shares_encryption_key_pair.clone(),
            &shutdown_handler,
            &uniqueness_error_result_attribute,
            &reauth_error_result_attribute,
        );

        let dummy_shares_for_deletions = get_dummy_shares_for_deletion(config.party_id);

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
                &iris_pg_store,
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
                config.party_id,
                &aws_clients.sqs_client,
                &aws_clients.sns_client,
                &aws_clients.s3_client,
                &config,
                &iris_pg_store,
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
