use aws_config::retry::RetryConfig;
use aws_sdk_s3::config::Builder as S3ConfigBuilder;
use aws_sdk_s3::{config::Region, Client as S3Client};
use eyre::{bail, eyre, Report, Result};
use futures::stream::BoxStream;
use futures::StreamExt;
use iris_mpc_common::config::{CommonConfig, Config};
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use iris_mpc_common::helpers::shutdown_handler::ShutdownHandler;
use iris_mpc_common::helpers::sync::{SyncResult, SyncState};
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::execution::hawk_main::{GraphStore, HawkActor};
use iris_mpc_cpu::genesis::{
    BatchGenerator as GenesisBatchGenerator, BatchIterator as GenesisBatchIterator,
    Handle as GenesisHawkHandle,
};
use iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store;
use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
use iris_mpc_store::{DbStoredIris, Store};

use iris_mpc_common::server_coordination::{
    get_others_sync_state, init_heartbeat_task, init_task_monitor, set_node_ready,
    start_coordination_server, wait_for_others_ready, wait_for_others_unready,
};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;

use iris_mpc_cpu::execution::hawk_main::HawkArgs;

const DEFAULT_REGION: &str = "eu-north-1";

/// Main logic for initialization and execution of server nodes for genesis
/// indexing.  This setup builds a new HNSW graph via MPC insertion of secret
/// shared iris codes in a database snapshot.  In particular, this indexer
/// mode does not make use of AWS services, instead processing entries from
/// an isolated database snapshot of previously validated unique iris shares.
pub async fn server_main_genesis(config: Config) -> Result<()> {
    let shutdown_handler = init_shutdown_handler(&config).await;

    let (iris_store, graph_store) = prepare_stores(&config).await?;

    let aws_s3_client = create_s3_client(&config).await;

    // skip: get_shares_encryption_key_pair
    // skip: init_sns
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

    load_database_genesis(&config, &iris_store, &graph_store, &mut hawk_actor).await?;

    background_tasks.check_tasks();

    // skip: start_results_thread

    set_node_ready(is_ready_flag);
    wait_for_others_ready(&config).await?;

    background_tasks.check_tasks();

    run_genesis_main_server_loop(
        &config,
        &iris_store,
        &graph_store,
        &aws_s3_client,
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
/// We leave deleted_request_ids and modifications empty for now, as the genesis protocol can have iris-mpc running at the same time
async fn build_genesis_sync_state(config: &Config, store: &Store) -> Result<SyncState> {
    let db_len = store.count_irises().await? as u64;
    let common_config = CommonConfig::from(config.clone());

    let deleted_request_ids = Vec::new();
    let modifications = Vec::new();

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

    // -------------------------------------------------------------------
    // TODO: use the number of currently processed entries for the amount
    //       to read into memory
    // -------------------------------------------------------------------
    let store_len = iris_store.count_irises().await?;

    load_db(&mut iris_loader, iris_store, store_len, parallelism)
        .await
        .expect("Failed to load DB");

    graph_loader.load_graph_store(graph_store).await?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_genesis_main_server_loop(
    config: &Config,
    iris_store: &Store,
    graph_store: &GraphPg<Aby3Store>,
    s3_client: &S3Client,
    _sync_result: &SyncResult,
    mut task_monitor: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
    hawk_actor: HawkActor,
) -> Result<()> {
    // Initialise Hawk handle.
    let mut hawk_handle = GenesisHawkHandle::new(hawk_actor).await?;

    // Initialise batch generator.
    let mut batch_generator = GenesisBatchGenerator::new(config.max_batch_size);
    batch_generator
        .init(iris_store, graph_store, s3_client)
        .await?;

    let res: Result<()> = async {
        tracing::info!("Entering main loop");
        let now = Instant::now();
        let processing_timeout = Duration::from_secs(config.processing_timeout_secs);

        // Index until batch generator is exhausted.
        while let Some(batch) = batch_generator.next_batch(iris_store).await? {
            tracing::info!(
                "HNSW GENESIS: Indexing new batch: idx={} :: irises={} :: time {:?}",
                batch_generator.batch_count(),
                batch.len(),
                now.elapsed(),
            );
            metrics::histogram!("receive_batch_duration").record(now.elapsed().as_secs_f64());

            task_monitor.check_tasks();

            let result_future = hawk_handle.submit_batch(batch);
            timeout(processing_timeout, result_future.await)
                .await
                .map_err(|e| eyre!("HawkActor processing timeout: {:?}", e))??;

            shutdown_handler.increment_batches_pending_completion()
        }

        Ok(())
    }
    .await;

    match res {
        Ok(_) => {
            tracing::info!(
                "Main loop exited normally. Waiting for last batch results to be processed before shutting down..."
            );
            shutdown_handler.wait_for_pending_batches_completion().await;
        }
        Err(e) => {
            tracing::error!("HawkActor processing error: {:?}", e);

            // Ensure hawk handle is dropped so as to initiate shutdown.
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

/// Returns initialized PostgreSQL clients for interacting
/// with iris share and HNSW graph stores.
async fn prepare_stores(config: &Config) -> Result<(Store, GraphPg<Aby3Store>), Report> {
    let schema_name = format!(
        "{}_{}_{}",
        config.app_name, config.environment, config.party_id
    );
    let db_config = config
        .database
        .as_ref()
        .ok_or(eyre!("Missing database config"))?;

    let postgres_client =
        PostgresClient::new(&db_config.url, &schema_name, AccessMode::ReadOnly).await?;

    tracing::info!("Creating new iris store from: {:?}", db_config,);

    let iris_store = Store::new(&postgres_client).await?;

    let hawk_db_config = config
        .cpu_database
        .as_ref()
        .ok_or(eyre!("Missing CPU database config for Hawk Genesis"))?;
    let hawk_postgres_client =
        PostgresClient::new(&hawk_db_config.url, &schema_name, AccessMode::ReadWrite).await?;

    tracing::info!("Creating new graph store from: {:?}", hawk_db_config);
    let graph_store = GraphStore::new(&hawk_postgres_client).await?;

    Ok((iris_store, graph_store))
}

/// Creates an S3 client with retry configuration
pub async fn create_s3_client(config: &Config) -> S3Client {
    // Get region from config or use default
    let region = config
        .clone()
        .aws
        .and_then(|aws| aws.region)
        .unwrap_or_else(|| DEFAULT_REGION.to_owned());

    let region_provider = Region::new(region);
    let shared_config = aws_config::from_env().region(region_provider).load().await;

    let force_path_style = config.environment != "prod" && config.environment != "stage";

    let retry_config = RetryConfig::standard().with_max_attempts(5);

    let s3_config = S3ConfigBuilder::from(&shared_config)
        .force_path_style(force_path_style)
        .retry_config(retry_config.clone())
        .build();

    S3Client::from_conf(s3_config)
}

/// Conducts consistency checks on iris shares store.
async fn check_store_consistency(config: &Config, iris_store: &Store) -> Result<()> {
    // TODO - check the database size matches where genesis should run too
    // We would only want to run genesis from a certain serial id to the other serial id
    let store_len = iris_store.count_irises().await?;
    tracing::info!("Size of the database after init: {}", store_len);

    if store_len > config.max_db_size {
        tracing::error!("Database size exceeds maximum allowed size: {}", store_len);
        bail!("Database size exceeds maximum allowed size: {}", store_len);
    }

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

pub async fn load_db(
    actor: &mut impl InMemoryStore,
    store: &Store,
    store_len: usize,
    store_load_parallelism: usize,
) -> eyre::Result<()> {
    let total_load_time = Instant::now();

    let mut all_serial_ids: HashSet<i64> = HashSet::from_iter(1..=(store_len as i64));
    actor.reserve(store_len);
    tracing::info!("S3 importer disabled. Fetching only from db");
    let stream_db = store
        .stream_irises_par(None, store_load_parallelism)
        .await
        .boxed();
    load_db_records(actor, &mut all_serial_ids, stream_db).await;

    if !all_serial_ids.is_empty() {
        tracing::error!("Not all serial_ids were loaded: {:?}", all_serial_ids);
        return Err(eyre!(
            "Not all serial_ids were loaded: {:?}",
            all_serial_ids
        ));
    }

    tracing::info!("Preprocessing db");
    actor.preprocess_db();

    tracing::info!(
        "Loaded set records from db into memory in {:?} [DB sizes: {:?}]",
        total_load_time.elapsed(),
        actor.current_db_sizes()
    );

    eyre::Ok(())
}

// Helper function to load Aurora db records from the stream into memory
#[allow(clippy::needless_lifetimes)]
async fn load_db_records<'a>(
    actor: &mut impl InMemoryStore,
    all_serial_ids: &mut HashSet<i64>,
    mut stream_db: BoxStream<'a, eyre::Result<DbStoredIris>>,
) {
    let mut load_summary_ts = Instant::now();
    let mut time_waiting_for_stream = Duration::from_secs(0);
    let mut time_loading_into_memory = Duration::from_secs(0);
    let mut record_counter = 0;
    while let Some(iris) = stream_db.next().await {
        // Update time waiting for the stream
        time_waiting_for_stream += load_summary_ts.elapsed();
        load_summary_ts = Instant::now();

        let iris = iris.unwrap();

        actor.load_single_record_from_db(
            iris.serial_id() - 1,
            iris.vector_id(),
            iris.left_code(),
            iris.left_mask(),
            iris.right_code(),
            iris.right_mask(),
        );

        // Only increment db size if record has not been loaded via s3 before
        if all_serial_ids.contains(&(iris.serial_id() as i64)) {
            actor.increment_db_size(iris.serial_id() - 1);
            all_serial_ids.remove(&(iris.serial_id() as i64));
            record_counter += 1;
        }

        // Update time spent loading into memory
        time_loading_into_memory += load_summary_ts.elapsed();
        load_summary_ts = Instant::now();
    }

    tracing::info!(
        "Aurora Loading summary => Loaded {:?} items. Waited for stream: {:?}, Loaded into \
         memory: {:?}",
        record_counter,
        time_waiting_for_stream,
        time_loading_into_memory,
    );
}

// TODO genesis "num_processed" state flag

// TODO genesis results produced in large batches, update written to temporary
// table, then update applied to graph

// DB sync possibly should support limited rollback?

// DB loading should use num_processed value to choose number of entries to load
