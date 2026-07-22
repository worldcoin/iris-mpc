//! Indexation phase for genesis.
//!
//! Consumes batches of not-yet-indexed iris serial ids, submits them to the
//! Hawk handle for HNSW insertion, and drives the periodic and final S3 graph
//! checkpoints. The entry point is [`exec_indexation`].

use ampc_server_utils::{shutdown_handler::ShutdownHandler, TaskMonitor};
use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use iris_mpc_cpu::{
    execution::hawk_main::{iris_worker::IrisWorkerPool, BothEyes, GraphRef},
    genesis::{BatchGenerator, BatchIterator, Handle as GenesisHawkHandle, JobRequest, JobResult},
    hawkers::aby3::aby3_store::VectorIdRegistryRef,
};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::{
    sync::{mpsc::Sender, oneshot},
    time::timeout,
};

use super::graph_checkpoint::upload_and_sync_genesis_checkpoint;
use super::{ExecutionContextInfo, PERSIST_DELAY};

/// Index Iris's from last indexation id.
///
/// # Arguments
///
/// * `ctx` - Execution context information.
/// * `s3_client` - AWS S3 client for checkpoint uploads.
/// * `registries` - Per-eye VectorId registries used by the batch generator.
/// * `worker_pools` - Per-eye worker pools that own iris data and cache queries.
/// * `imem_graph_stores` - In-memory graph stores for checkpoints.
/// * `hawk_handle` - Hawk handle managing indexation & search over an HNSW graph.
/// * `tx_results` - Channel to send job results to DB persistence thread.
/// * `task_monitor_bg` - Tokio task monitor to coordinate with process background threads.
/// * `shutdown_handler` - Handler coordinating function termination/process shutdown.
///
#[allow(clippy::too_many_arguments)]
pub(super) async fn exec_indexation(
    ctx: &ExecutionContextInfo,
    s3_client: &S3Client,
    registries: &BothEyes<VectorIdRegistryRef>,
    worker_pools: &BothEyes<Arc<dyn IrisWorkerPool>>,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    mut hawk_handle: GenesisHawkHandle,
    tx_results: &Sender<JobResult>,
    mut task_monitor_bg: TaskMonitor,
    shutdown_handler: &Arc<ShutdownHandler>,
) -> Result<()> {
    tracing::info!(
        "Starting indexation: last_indexed_id={}, max_indexation_id={}",
        ctx.last_indexed_id,
        ctx.args.max_indexation_id
    );

    // Set batch size from config.
    let batch_size = ctx
        .args
        .batch_size_config
        .compute_batch_size(ctx.config.hnsw_param_m);

    if ctx.last_indexed_id + 1 > ctx.args.max_indexation_id {
        tracing::warn!(
            "Last indexed id {} is greater than max indexation id {}. \
                 No indexation will be performed.",
            ctx.last_indexed_id,
            ctx.args.max_indexation_id
        );
    }
    // Set batch generator.
    let mut batch_generator = BatchGenerator::new(
        ctx.last_indexed_id + 1,
        ctx.args.max_indexation_id,
        batch_size,
        ctx.excluded_serial_ids.clone(),
    );
    tracing::info!("Batch generator instantiated: {}", batch_generator);

    // Set indexation result.
    let mut persist_ch: Option<oneshot::Receiver<()>> = None;

    // Checkpoint tracking
    let checkpoint_frequency = ctx.args.checkpoint_frequency;
    // Maximum height at which an intermediate checkpoint would be run, to avoid redundancy with final checkpoint
    let max_intermediate_checkpoint_height = ctx
        .args
        .max_indexation_id
        .saturating_sub(checkpoint_frequency as u32 / 10);
    let mut irises_since_checkpoint: usize = 0;
    let mut last_indexed_id = ctx.last_indexed_id;

    let res: Result<()> = async {
        tracing::info!("Entering main indexation loop");
        tracing::info!(
            "Checkpoint frequency: {} irises per checkpoint",
            checkpoint_frequency
        );

        // Housekeeping.
        let mut now = Instant::now();
        let processing_timeout = Duration::from_secs(ctx.config.processing_timeout_secs);

        // Index until generator is exhausted.
        // N.B. assumes that generator yields non-empty batches containing serial ids > last_indexed_id.
        while let Some(batch) = batch_generator
            .next_batch(last_indexed_id, registries, worker_pools)
            .await?
        {
            // Coordinator: escape on shutdown.
            let shutdown = shutdown_handler.is_shutting_down();
            let mismatch = hawk_handle.sync_state(shutdown, None).await?;
            if shutdown || mismatch {
                tracing::warn!("Shutting down has been triggered");
                break;
            }

            // Coordinator: check background task processing.
            task_monitor_bg.check_tasks();
            last_indexed_id = batch.id_end();
            irises_since_checkpoint += batch.vector_ids.len();

            // Submit batch to Hawk handle for indexation.
            let request = JobRequest::new_batch_indexation(&batch);
            let result_future = hawk_handle.submit_request(request).await;
            let result = timeout(processing_timeout, result_future)
                .await
                .map_err(|err| {
                    tracing::error!("HawkActor processing timeout: {:?}", err);
                    eyre!("HawkActor processing timeout: {:?}", err)
                })??;

            // Send results to processing thread responsible for persisting to database.
            let (done_rx, result) = result;
            tx_results.send(result).await?;

            // Periodically synchronize batch persistence between nodes.
            let is_sync_batch = (batch.batch_id % PERSIST_DELAY) == PERSIST_DELAY - 1;
            if is_sync_batch {
                if let Some(prev_done_rx) = persist_ch.take() {
                    let wait_start = Instant::now();
                    // Wait for other nodes to finish equivalent persistence.
                    hawk_handle.sync_state(false, Some(prev_done_rx)).await?;
                    metrics::histogram!("genesis_persist_wait_duration")
                        .record(wait_start.elapsed().as_secs_f64());
                }
            }

            // Store current results thread "done" signal channel for future synchronization.
            persist_ch.replace(done_rx);

            metrics::histogram!("genesis_batch_total_duration",
                "synced" => if is_sync_batch { "true" } else { "false" },
            )
            .record(now.elapsed().as_secs_f64());
            tracing::info!(
                "Indexing new batch: {} :: time {:?}s",
                batch,
                now.elapsed().as_secs_f64(),
            );

            // Periodic checkpoint based on snapshot_frequency.  Skipped if close to end of indexation.
            if irises_since_checkpoint >= checkpoint_frequency
                && last_indexed_id <= max_intermediate_checkpoint_height
            {
                upload_and_sync_genesis_checkpoint(
                    &ctx.config.graph_checkpoint_bucket_name,
                    ctx.config.party_id,
                    imem_graph_stores,
                    s3_client,
                    last_indexed_id,
                    ctx.max_modification_persist_id, // preserve current modification state
                    false, // is_archival: periodic checkpoints are not archival
                    tx_results,
                    &mut hawk_handle,
                )
                .await?;
                irises_since_checkpoint = 0;
            };

            now = Instant::now();
        }
        Ok(())
    }
    .await;

    // Process main loop result:
    match res {
        // Success.
        Ok(_) => {
            let wait_start = Instant::now();

            // Create final archival checkpoint if any irises were indexed this run.
            // This runs unconditionally (regardless of periodic checkpoints) to ensure
            // the last checkpoint recorded for a run is always archival.
            if last_indexed_id > ctx.last_indexed_id {
                tracing::info!(
                    "Creating final archival checkpoint: last_indexed_id={}",
                    last_indexed_id
                );
                upload_and_sync_genesis_checkpoint(
                    &ctx.config.graph_checkpoint_bucket_name,
                    ctx.config.party_id,
                    imem_graph_stores,
                    s3_client,
                    last_indexed_id,
                    ctx.max_modification_persist_id, // preserve current modification state
                    true, // is_archival: final checkpoint is always archival
                    tx_results,
                    &mut hawk_handle,
                )
                .await?;
                tracing::info!(
                    "Final archival checkpoint created at iris_id={}",
                    last_indexed_id
                );
            } else if let Some(rx) = persist_ch.take() {
                hawk_handle.sync_state(false, Some(rx)).await?;
            }
            metrics::histogram!("genesis_persist_wait_duration")
                .record(wait_start.elapsed().as_secs_f64());

            hawk_handle.sync_peers().await?;
            tracing::info!("All batches have been processed, shutting down...");

            Ok(())
        }
        Err(err) => {
            tracing::error!("HawkActor processing error: {:?}", err);

            // Clean up & shutdown.
            tracing::info!("Initiating shutdown");
            drop(hawk_handle);
            task_monitor_bg.abort_all();
            tokio::time::sleep(Duration::from_secs(5)).await;
            task_monitor_bg.check_tasks_finished();

            Err(err)
        }
    }
}
