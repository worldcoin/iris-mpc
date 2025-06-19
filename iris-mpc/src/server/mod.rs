use crate::services::aws::clients::AwsClients;
use crate::services::processors::batch::receive_batch_stream;
use crate::services::processors::job::process_job_result;
use aws_sdk_sns::types::MessageAttributeValue;

use crate::services::processors::modifications_sync::{
    send_last_modifications_to_sns, sync_modifications,
};
use eyre::{bail, eyre, Report, Result};
use iris_mpc_common::config::{CommonConfig, Config};
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::helpers::key_pair::SharesEncryptionKeyPairs;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::smpc_request::{
    ANONYMIZED_STATISTICS_MESSAGE_TYPE, IDENTITY_DELETION_MESSAGE_TYPE, REAUTH_MESSAGE_TYPE,
    RESET_CHECK_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE,
};
use iris_mpc_common::helpers::smpc_response::create_message_type_attribute_map;
use iris_mpc_common::helpers::sqs::{delete_messages_until_sequence_num, get_next_sns_seq_num};
use iris_mpc_common::helpers::sync::{SyncResult, SyncState};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::job::JobSubmissionHandle;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::server_coordination::{
    get_others_sync_state, init_heartbeat_task, init_task_monitor, set_node_ready,
    start_coordination_server, wait_for_others_ready, wait_for_others_unready,
};
use iris_mpc_cpu::execution::hawk_main::{
    GraphStore, HawkActor, HawkArgs, HawkHandle, ServerJobResult,
};
use iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store;
use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
use iris_mpc_store::loader::load_iris_db;
use iris_mpc_store::Store;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::sync::mpsc::Sender;
use tokio::time::timeout;

const RNG_SEED_INIT_DB: u64 = 42;
pub const SQS_POLLING_INTERVAL: Duration = Duration::from_secs(1);
pub const MAX_CONCURRENT_REQUESTS: usize = 32;

/// Main logic for initialization and execution of AMPC iris uniqueness server
/// nodes.
pub async fn server_main(config: Config) -> Result<()> {
    let shutdown_handler = init_shutdown_handler(&config).await;

    process_config(&config);

    let (iris_store, graph_store) = prepare_stores(&config).await?;

    let aws_clients = init_aws_services(&config).await?;
    let shares_encryption_key_pair = get_shares_encryption_key_pair(&config, &aws_clients).await?;
    let sns_attributes_maps = init_sns_attributes_maps()?;

    maybe_seed_random_shares(&config, &iris_store).await?;
    check_store_consistency(&config, &iris_store).await?;
    let my_state = build_sync_state(&config, &aws_clients, &iris_store).await?;

    let mut background_tasks = init_task_monitor();

    // Initialize shared current_batch_id
    let current_batch_id_atomic = Arc::new(AtomicU64::new(0));

    let is_ready_flag = start_coordination_server(
        &config,
        &aws_clients.sqs_client,
        &mut background_tasks,
        &shutdown_handler,
        &my_state,
        current_batch_id_atomic.clone(),
    )
    .await;

    background_tasks.check_tasks();

    wait_for_others_unready(&config).await?;
    init_heartbeat_task(&config, &mut background_tasks, &shutdown_handler).await?;

    background_tasks.check_tasks();

    let sync_result = get_sync_result(&config, &my_state).await?;
    sync_result.check_common_config()?;

    // Handle modifications sync
    if config.enable_modifications_sync {
        sync_modifications(
            &config,
            &iris_store,
            Some(&graph_store),
            &aws_clients,
            &shares_encryption_key_pair,
            sync_result.clone(),
        )
        .await?;
    }

    if config.enable_modifications_replay {
        // replay last `max_modification_lookback` modifications to SNS
        if let Err(e) = send_last_modifications_to_sns(
            &iris_store,
            &aws_clients.sns_client,
            &config,
            max_sync_lookback(&config),
        )
        .await
        {
            tracing::error!("Failed to replay last modifications: {:?}", e);
        }
    }

    sync_sqs_queues(&config, &sync_result, &aws_clients).await?;

    if shutdown_handler.is_shutting_down() {
        tracing::warn!("Shutting down has been triggered");
        return Ok(());
    }

    let mut hawk_actor = init_hawk_actor(&config).await?;

    load_database(
        &config,
        &iris_store,
        &graph_store,
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
        background_tasks,
        &shutdown_handler,
        hawk_actor,
        tx_results,
        current_batch_id_atomic.clone(),
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
    if config.cpu_database.is_none() {
        panic!("Missing CPU dB config settings",);
    }
    // Load batch_size config
    tracing::info!("Set max batch size to {}", config.max_batch_size);
}

/// Returns computed maximum sync lookback size.
fn max_sync_lookback(config: &Config) -> usize {
    (config.max_deletions_per_batch + config.max_batch_size) * 2
}

/// Returns initialized PostgreSQL clients for interacting
/// with iris share and HNSW graph stores.
async fn prepare_stores(config: &Config) -> Result<(Store, GraphPg<Aby3Store>), Report> {
    let hawk_schema_name = format!(
        "{}{}_{}_{}",
        config.schema_name, config.hnsw_schema_name_suffix, config.environment, config.party_id
    );

    // HNSW will always use the CPU__DATABASE_* both for irises and graph
    let hawk_db_config = config
        .cpu_database
        .as_ref()
        .ok_or(eyre!("Missing CPU database config"))?;
    let hawk_postgres_client = PostgresClient::new(
        &hawk_db_config.url,
        &hawk_schema_name,
        AccessMode::ReadWrite,
    )
    .await?;

    let iris_store = Store::new(&hawk_postgres_client).await?;
    let graph_store = GraphStore::new(&hawk_postgres_client).await?;

    Ok((iris_store, graph_store))
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
    reset_check_result_attributes: HashMap<String, MessageAttributeValue>,
    reset_update_result_attributes: HashMap<String, MessageAttributeValue>,
    anonymized_statistics_attributes: HashMap<String, MessageAttributeValue>,
    identity_deletion_result_attributes: HashMap<String, MessageAttributeValue>,
}

/// Returns a set of attribute maps used to interact with AWS SNS.
fn init_sns_attributes_maps() -> Result<SnsAttributesMaps> {
    let uniqueness_result_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_result_attributes = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_check_result_attributes = create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let reset_update_result_attributes = create_message_type_attribute_map(
        iris_mpc_common::helpers::smpc_request::RESET_UPDATE_MESSAGE_TYPE,
    );
    let anonymized_statistics_attributes =
        create_message_type_attribute_map(ANONYMIZED_STATISTICS_MESSAGE_TYPE);
    let identity_deletion_result_attributes =
        create_message_type_attribute_map(IDENTITY_DELETION_MESSAGE_TYPE);
    Ok(SnsAttributesMaps {
        uniqueness_result_attributes,
        reauth_result_attributes,
        reset_check_result_attributes,
        reset_update_result_attributes,
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

/// Build this node's synchronization state, which is compared against the
/// states provided by the other MPC nodes to reconstruct a consistent initial
/// state for MPC operation.
async fn build_sync_state(
    config: &Config,
    aws_clients: &AwsClients,
    store: &Store,
) -> Result<SyncState> {
    let db_len = store.count_irises().await? as u64;
    let modifications = store.last_modifications(max_sync_lookback(config)).await?;
    let next_sns_sequence_num = get_next_sns_seq_num(config, &aws_clients.sqs_client).await?;
    let common_config = CommonConfig::from(config.clone());

    tracing::info!("Database store length is: {}", db_len);

    Ok(SyncState {
        db_len,
        modifications,
        next_sns_sequence_num,
        common_config,
    })
}

async fn get_sync_result(config: &Config, my_state: &SyncState) -> Result<SyncResult> {
    let mut all_states = vec![my_state.clone()];
    all_states.extend(get_others_sync_state(config).await?);
    let sync_result = SyncResult::new(my_state.clone(), all_states);
    Ok(sync_result)
}

/// Delete stale SQS messages in requests queue with sequence number older than the most recent
/// sequence number seen by any MPC party.
async fn sync_sqs_queues(
    config: &Config,
    sync_result: &SyncResult,
    aws_clients: &AwsClients,
) -> Result<()> {
    let max_sqs_sequence_num = sync_result.max_sns_sequence_num();
    delete_messages_until_sequence_num(
        config,
        &aws_clients.sqs_client,
        sync_result.my_state.next_sns_sequence_num,
        max_sqs_sequence_num,
    )
    .await?;

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
        stream_parallelism: config.hawk_stream_parallelism,
        connection_parallelism: config.hawk_connection_parallelism,
        hnsw_param_ef_constr: config.hnsw_param_ef_constr,
        hnsw_param_M: config.hnsw_param_M,
        hnsw_param_ef_search: config.hnsw_param_ef_search,
        hnsw_prf_key: config.hawk_prf_key,
        disable_persistence: config.disable_persistence,
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
        .cpu_database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?
        .load_parallelism;

    tracing::info!(
        "Initialize iris db: Loading from DB (parallelism: {})",
        parallelism
    );
    let download_shutdown_handler = Arc::clone(shutdown_handler);

    let store_len = iris_store.count_irises().await?;

    let now = Instant::now();
    let iris_load_future = load_iris_db(
        &mut iris_loader,
        iris_store,
        store_len,
        parallelism,
        config,
        download_shutdown_handler,
    );

    let graph_load_future = graph_loader.load_graph_store(
        graph_store,
        config.cpu_database.as_ref().unwrap().load_parallelism,
    );

    let (iris_result, graph_result) = tokio::join!(iris_load_future, graph_load_future);
    iris_result.expect("Failed to load iris DB");
    graph_result.expect("Failed to load graph DB");
    tracing::info!(
        "Loaded both iris and graph DBs into memory in {:?}",
        now.elapsed()
    );

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
                &sns_attributes_maps.reset_check_result_attributes,
                &sns_attributes_maps.reset_update_result_attributes,
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
    mut task_monitor: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    hawk_actor: HawkActor,
    tx_results: Sender<ServerJobResult>,
    current_batch_id_atomic: Arc<AtomicU64>,
) -> Result<()> {
    // --------------------------------------------------------------------------
    // ANCHOR: Start the main loop
    // --------------------------------------------------------------------------
    tracing::info!("⚓️ ANCHOR: Start the main loop");

    let mut hawk_handle = HawkHandle::new(hawk_actor).await?;

    let party_id = config.party_id;

    let processing_timeout = Duration::from_secs(config.processing_timeout_secs);
    let uniqueness_error_result_attribute =
        create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
    let reauth_error_result_attribute = create_message_type_attribute_map(REAUTH_MESSAGE_TYPE);
    let reset_error_result_attributes = create_message_type_attribute_map(RESET_CHECK_MESSAGE_TYPE);
    let res: Result<()> = async {
        tracing::info!("Entering main loop");

        // This batch can consist of N sets of iris_share + mask
        // It also includes a vector of request ids, mapping to the sets above

        let mut batch_stream = receive_batch_stream(
            party_id,
            aws_clients.sqs_client.clone(),
            aws_clients.sns_client.clone(),
            aws_clients.s3_client.clone(),
            config.clone(),
            shares_encryption_key_pair.clone(),
            shutdown_handler.clone(),
            uniqueness_error_result_attribute.clone(),
            reauth_error_result_attribute.clone(),
            reset_error_result_attributes.clone(),
            current_batch_id_atomic,
            iris_store.clone(),
        );

        loop {
            let now = Instant::now();

            let batch = match batch_stream.recv().await {
                Some(Ok(None)) | None => {
                    tracing::info!("No more batches to process, exiting main loop");
                    return Ok(());
                }
                Some(Err(e)) => {
                    return Err(e.into());
                }
                Some(Ok(Some(batch))) => batch,
            };

            // start trace span - with single TraceId and single ParentTraceID
            tracing::info!("Received batch in {:?}", now.elapsed());

            metrics::histogram!("receive_batch_duration").record(now.elapsed().as_secs_f64());
            metrics::gauge!("batch_size").set(batch.request_types.len() as f64);

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
