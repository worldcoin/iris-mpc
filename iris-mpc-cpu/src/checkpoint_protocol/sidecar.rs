//! Sidecar daemon: periodically run the checkpoint protocol and upload
//! a fresh checkpoint when enough WAL mutations have accumulated.
//!
//! Lifecycle per cycle:
//!
//! 1. Wait for `cycle_interval` (or shorter `retry_interval` on a previous
//!    transient).
//! 2. Open a fresh [`ControlChannel`] via the [`NetworkHandle`] — each
//!    cycle gets a new TCP connection set; nothing is reused across
//!    cycles. This isolates per-cycle failure modes from each other.
//! 3. Run [`crate::checkpoint_protocol::run_cycle`] with
//!    [`RebuildFromCheckpoint`] + [`UploadAndRecord`].
//! 4. Outcome:
//!    - `Finalized` / `Skipped` → next iteration sleeps `cycle_interval`.
//!    - `Transient` error → next iteration sleeps `retry_interval`.
//!    - `Fatal` error → daemon exits with an error (the crash signal is
//!      the alert; an external supervisor decides the restart policy).
//!
//! Shutdown via the supplied [`CancellationToken`]: the daemon honours
//! cancellation between cycles. An in-flight cycle is allowed to complete
//! (or hit its peer-round timeout); we do not abort run_cycle mid-flight.

use std::time::Duration;

use ampc_actor_utils::network::mpc::NetworkHandle;
use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use tokio_util::sync::CancellationToken;

use crate::checkpoint_protocol::{
    run_cycle, Blake3GraphHasher, CycleConfig, CycleError, MutationStore, Outcome,
    RebuildFromCheckpoint, RingConsensusTransport, UploadAndRecord,
};
use crate::hnsw::{graph::graph_store::GraphPg, VectorStore};

/// Sidecar daemon configuration. Construct from the binary's CLI args /
/// environment.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Bucket containing the checkpoint S3 objects.
    pub bucket: String,
    /// Party index (0, 1, 2). Drives S3 key prefix and metric labels.
    pub party_id: usize,
    /// Sleep between successful (or skipped) cycles.
    pub cycle_interval: Duration,
    /// Sleep after a transient cycle error before retrying.
    pub retry_interval: Duration,
    /// Per-peer-round timeout passed into the protocol's `CycleConfig`.
    pub peer_round_timeout: Duration,
    /// Lower-bound for cycle work; below this the cycle is skipped (no
    /// upload / no DB write) to avoid hammering S3 with near-empty deltas.
    pub min_mutations_per_cycle: u64,
    /// Marks the produced row as archival (skipped by pruning).
    pub is_archival: bool,
}

/// Daemon loop. Returns `Ok(())` only on graceful shutdown via
/// `shutdown_ct`; any `CycleError::Fatal` is propagated as an `Err`.
pub async fn sidecar_main<V: VectorStore + Send + Sync>(
    cfg: SidecarConfig,
    graph_store: &GraphPg<V>,
    s3_client: &S3Client,
    networking: &mut Box<dyn NetworkHandle>,
    shutdown_ct: CancellationToken,
) -> Result<()> {
    tracing::info!(
        bucket = %cfg.bucket,
        party_id = cfg.party_id,
        cycle_interval = ?cfg.cycle_interval,
        "sidecar daemon starting"
    );

    loop {
        if shutdown_ct.is_cancelled() {
            tracing::info!("sidecar daemon shutting down (cancelled)");
            return Ok(());
        }

        let sleep_after = match try_one_cycle(&cfg, graph_store, s3_client, networking).await {
            Ok(Outcome::Finalized { height, .. }) => {
                tracing::info!(height, "sidecar cycle finalized");
                cfg.cycle_interval
            }
            Ok(Outcome::Skipped(reason)) => {
                tracing::debug!(?reason, "sidecar cycle skipped");
                cfg.cycle_interval
            }
            Err(CycleError::Transient(msg)) => {
                tracing::warn!(error = %msg, "sidecar cycle transient error");
                cfg.retry_interval
            }
            Err(CycleError::Fatal(msg)) => {
                tracing::error!(error = %msg, "sidecar cycle fatal error; exiting");
                return Err(eyre!("sidecar fatal: {msg}"));
            }
        };

        tokio::select! {
            _ = shutdown_ct.cancelled() => {
                tracing::info!("sidecar daemon shutting down (cancelled during sleep)");
                return Ok(());
            }
            _ = tokio::time::sleep(sleep_after) => {}
        }
    }
}

/// One cycle. Builds a fresh `ControlChannel`, wires the materializer +
/// finalizer + transport, and dispatches to `run_cycle`. The nonce is
/// `base.checkpoint_id as u128` — deterministic across parties because all
/// three resolve `latest_checkpoint()` to the same row before exchange.
async fn try_one_cycle<V: VectorStore + Send + Sync>(
    cfg: &SidecarConfig,
    graph_store: &GraphPg<V>,
    s3_client: &S3Client,
    networking: &mut Box<dyn NetworkHandle>,
) -> Result<Outcome, CycleError> {
    // Pre-fetch the base to compute the nonce. The same row is fetched
    // again inside run_cycle's phase 1; the double-fetch is cheap and
    // keeps the function signature unchanged.
    let base = MutationStore::latest_checkpoint(graph_store).await?;
    let cycle_nonce = base.checkpoint_id as u128;

    let channel = networking
        .control_channel()
        .await
        .map_err(|e| CycleError::Transient(format!("open control_channel: {e}")))?;
    let transport = RingConsensusTransport::new(channel);

    let mut materializer = RebuildFromCheckpoint::new(graph_store, s3_client, cfg.bucket.clone());
    let mut finalizer = UploadAndRecord::new(
        graph_store,
        s3_client,
        cfg.bucket.clone(),
        cfg.party_id,
        cfg.is_archival,
    );
    let hasher = Blake3GraphHasher::new();

    let cycle_cfg = CycleConfig {
        min_mutations_to_apply: cfg.min_mutations_per_cycle,
        peer_round_timeout: cfg.peer_round_timeout,
        cycle_nonce,
    };

    run_cycle(
        &mut materializer,
        &mut finalizer,
        &transport,
        graph_store,
        &hasher,
        &cycle_cfg,
    )
    .await
}
