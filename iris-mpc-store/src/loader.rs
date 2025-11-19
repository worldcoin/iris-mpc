use crate::s3_importer::create_db_chunks_s3_client;
use crate::{
    fetch_and_parse_chunks, last_snapshot_timestamp, DbStoredIris, S3Store, S3StoredIris, Store,
};
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use aws_config::Region;
use eyre::{bail, Result};
use futures::stream::BoxStream;
use futures::StreamExt;
use iris_mpc_common::config::Config;
use iris_mpc_common::helpers::inmemory_store::InMemoryStore;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Helper function to load Aurora db records from the stream into memory
#[allow(clippy::needless_lifetimes)]
async fn load_db_records_from_aurora<'a>(
    actor: &mut impl InMemoryStore,
    mut record_counter: i32,
    all_serial_ids: &mut HashSet<i64>,
    mut stream_db: BoxStream<'a, Result<DbStoredIris>>,
) {
    let mut load_summary_ts = Instant::now();
    let mut time_waiting_for_stream = Duration::from_secs(0);
    let mut time_loading_into_memory = Duration::from_secs(0);
    let n_loaded_via_s3 = record_counter;
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
        record_counter - n_loaded_via_s3,
        time_waiting_for_stream,
        time_loading_into_memory,
    );
}

/// Main iris loader method into memory. Load from either S3 + Aurora or only Aurora based on the config.
pub async fn load_iris_db(
    actor: &mut impl InMemoryStore,
    store: &Store,
    max_serial_id_to_load: usize,
    store_load_parallelism: usize,
    config: &Config,
    download_shutdown_handler: Arc<ShutdownHandler>,
) -> Result<()> {
    let total_load_time = Instant::now();
    let now = Instant::now();

    let mut record_counter = 0;
    let mut all_serial_ids: HashSet<i64> = HashSet::from_iter(1..=(max_serial_id_to_load as i64));
    actor.reserve(max_serial_id_to_load);

    if config.enable_s3_importer {
        tracing::info!("S3 importer enabled. Fetching from s3 + db");
        let region = config.clone().db_chunks_bucket_region;

        // Get s3 loading parameters from config
        let s3_load_parallelism = config.load_chunks_parallelism;
        let s3_load_max_retries = config.load_chunks_max_retries;
        let s3_load_initial_backoff_ms = config.load_chunks_initial_backoff_ms;
        let s3_chunks_folder_name = config.db_chunks_folder_name.clone();
        let s3_chunks_bucket_name = config.db_chunks_bucket_name.clone();
        let s3_load_safety_overlap_seconds = config.db_load_safety_overlap_seconds;

        // Construct s3 client and store
        let region_provider = Region::new(region);
        let shared_config = aws_config::from_env().region(region_provider).load().await;
        let s3_client = create_db_chunks_s3_client(&shared_config, true);
        let s3_store = S3Store::new(s3_client, s3_chunks_bucket_name.clone());
        let s3_arc = Arc::new(s3_store);

        // First fetch last snapshot from S3
        let last_snapshot_details =
            last_snapshot_timestamp(s3_arc.as_ref(), s3_chunks_folder_name.clone()).await?;

        let min_last_modified_at = last_snapshot_details.timestamp - s3_load_safety_overlap_seconds;
        tracing::info!(
            "Last snapshot timestamp: {}, min_last_modified_at: {}",
            last_snapshot_details.timestamp,
            min_last_modified_at
        );

        let (tx, mut rx) = mpsc::channel::<S3StoredIris>(config.load_chunks_buffer_size);
        tokio::spawn(async move {
            fetch_and_parse_chunks(
                s3_arc,
                s3_load_parallelism,
                s3_chunks_folder_name.clone(),
                last_snapshot_details,
                tx.clone(),
                s3_load_max_retries,
                s3_load_initial_backoff_ms,
            )
            .await
            .expect("Couldn't fetch and parse chunks from s3");
        });

        let mut time_waiting_for_stream = Duration::from_secs(0);
        let mut time_loading_into_memory = Duration::from_secs(0);
        let mut load_summary_ts = Instant::now();
        while let Some(iris) = rx.recv().await {
            time_waiting_for_stream += load_summary_ts.elapsed();
            load_summary_ts = Instant::now();
            let serial_id = iris.serial_id();

            if serial_id == 0 {
                tracing::error!("Invalid iris serial_id {}", serial_id);
                bail!("Invalid iris serial_id {}", serial_id);
            } else if serial_id > max_serial_id_to_load {
                tracing::warn!(
                    "Skip loading item: serial_id {} > max_serial_id_to_load {}. This can happen if the max in the case of roll-backs or where the max_serial_id_to_load is specified",
                    serial_id,
                    max_serial_id_to_load,
                );
                continue;
            } else if !all_serial_ids.contains(&(serial_id as i64)) {
                tracing::warn!("Skip loading s3 retried item: serial_id {}", serial_id);
                continue;
            }

            actor.load_single_record_from_s3(
                iris.serial_id() - 1,
                iris.vector_id(),
                iris.left_code_odd(),
                iris.left_code_even(),
                iris.right_code_odd(),
                iris.right_code_even(),
                iris.left_mask_odd(),
                iris.left_mask_even(),
                iris.right_mask_odd(),
                iris.right_mask_even(),
            );
            actor.increment_db_size(serial_id - 1);

            if record_counter % 100_000 == 0 {
                let elapsed = now.elapsed();
                tracing::info!(
                    "Loaded {} records into memory in {:?} ({:.2} entries/s)",
                    record_counter,
                    elapsed,
                    record_counter as f64 / elapsed.as_secs_f64()
                );
                if download_shutdown_handler.is_shutting_down() {
                    tracing::warn!("Shutdown requested by shutdown_handler.");
                    return Err(eyre::eyre!("Shutdown requested"));
                }
            }

            time_loading_into_memory += load_summary_ts.elapsed();
            load_summary_ts = Instant::now();

            all_serial_ids.remove(&(serial_id as i64));
            record_counter += 1;
        }
        tracing::info!(
            "S3 Loading summary => Loaded {:?} items. Waited for stream: {:?}, Loaded into \
             memory: {:?}.",
            record_counter,
            time_waiting_for_stream,
            time_loading_into_memory,
        );

        let stream_db = store
            .stream_irises_par(Some(min_last_modified_at), store_load_parallelism)
            .await
            .boxed();
        load_db_records_from_aurora(actor, record_counter, &mut all_serial_ids, stream_db).await;
    } else {
        tracing::info!("S3 importer disabled. Fetching only from Aurora db");
        let stream_db = store
            .stream_irises_par(None, store_load_parallelism)
            .await
            .boxed();
        load_db_records_from_aurora(actor, record_counter, &mut all_serial_ids, stream_db).await;
    }

    if !all_serial_ids.is_empty() {
        tracing::error!("Not all serial_ids were loaded: {:?}", all_serial_ids);
        bail!("Not all serial_ids were loaded: {:?}", all_serial_ids);
    }

    tracing::info!("Preprocessing db");
    actor.preprocess_db();

    tracing::info!(
        "Loaded {} records from db into memory in {:?} [DB sizes: {:?}]",
        record_counter,
        total_load_time.elapsed(),
        actor.current_db_sizes()
    );

    eyre::Ok(())
}
