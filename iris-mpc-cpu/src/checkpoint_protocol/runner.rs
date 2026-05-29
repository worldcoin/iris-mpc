//! Cycle drivers: [`sidecar_main`] (daemon loop) and
//! [`restart_from_checkpoint`] (one-shot Hawk startup).
//!
//! Each cycle opens a fresh `ControlChannel`; no transport state crosses
//! cycle boundaries, so per-cycle failures stay isolated.

use std::sync::Arc;
use std::time::Duration;

use ampc_actor_utils::network::mpc::NetworkHandle;
use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::checkpoint_protocol::{
    run_cycle, Blake3GraphHasher, CycleConfig, CycleError, GraphMutationId, InstallAsServing,
    MostRecentCommon, Outcome, RebuildFromCheckpoint, RingConsensusTransport, SkipReason,
    UploadAndRecord,
};
use crate::execution::hawk_main::BothEyes;
use crate::graph_checkpoint::PruningMode;
use crate::hnsw::{
    graph::{graph_store::GraphPg, layered_graph::GraphMem},
    VectorStore,
};
use iris_mpc_common::vector_id::VectorId;

// ── Sidecar daemon ───────────────────────────────────────────────────────

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
    /// Newest-first window of recent checkpoints advertised in Phase 1.
    pub checkpoint_window: usize,
    /// Marks the produced row as archival (skipped by pruning).
    pub is_archival: bool,
    /// Optionally remove old checkoints from S3
    pub pruning_mode: PruningMode,
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

        let sleep_after = match sidecar_cycle(&cfg, graph_store, s3_client, networking).await {
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

async fn sidecar_cycle<V: VectorStore + Send + Sync>(
    cfg: &SidecarConfig,
    graph_store: &GraphPg<V>,
    s3_client: &S3Client,
    networking: &mut Box<dyn NetworkHandle>,
) -> Result<Outcome, CycleError> {
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
        cfg.pruning_mode,
    );
    let hasher = Blake3GraphHasher::new();

    let cycle_cfg = CycleConfig {
        min_mutations_to_apply: cfg.min_mutations_per_cycle,
        peer_round_timeout: cfg.peer_round_timeout,
        checkpoint_window: cfg.checkpoint_window,
    };

    run_cycle(
        &mut materializer,
        &mut finalizer,
        &transport,
        graph_store,
        &hasher,
        &MostRecentCommon,
        &cycle_cfg,
    )
    .await
}

// ── Restart ──────────────────────────────────────────────────────────────

/// Result of a restart attempt. The caller decides how to act on each.
#[derive(Debug)]
pub enum RestartOutcome {
    /// A graph was installed at the reported height.
    Installed { height: GraphMutationId },
    /// No `genesis_graph_checkpoint` row exists; caller falls back to its
    /// own bootstrap path (empty graph, plaintext checkpoint migration).
    NoCheckpoint,
    /// A peer's `current_max_mutation_id` came in below the agreed base.
    /// Restart can be retried once peers catch up; not a fatal error.
    PeerBehindBase {
        freeze: GraphMutationId,
        base: GraphMutationId,
    },
    /// The selector found no checkpoint shared across all parties (even
    /// within the recent window). Operator intervention is likely needed.
    NoCommonBase,
}

/// Rebuild the live graph from the latest checkpoint + WAL replay, then
/// install it into `target`. `min_mutations_to_apply=0` so the cycle
/// proceeds even when there are no new mutations beyond the base.
pub async fn restart_from_checkpoint<V: VectorStore + Send + Sync>(
    graph_store: &GraphPg<V>,
    s3_client: &S3Client,
    bucket: String,
    networking: &mut Box<dyn NetworkHandle>,
    target: BothEyes<Arc<RwLock<GraphMem<VectorId>>>>,
    peer_round_timeout: Duration,
    checkpoint_window: usize,
) -> Result<RestartOutcome> {
    if graph_store
        .get_latest_genesis_graph_checkpoint()
        .await
        .map_err(|e| eyre!("get_latest_genesis_graph_checkpoint: {e}"))?
        .is_none()
    {
        return Ok(RestartOutcome::NoCheckpoint);
    }

    let channel = networking
        .control_channel()
        .await
        .map_err(|e| eyre!("open control_channel: {e}"))?;
    let transport = RingConsensusTransport::new(channel);

    let mut materializer = RebuildFromCheckpoint::new(graph_store, s3_client, bucket);
    let mut finalizer = InstallAsServing::new(target);
    let hasher = Blake3GraphHasher::new();

    let cfg = CycleConfig {
        min_mutations_to_apply: 0,
        peer_round_timeout,
        checkpoint_window,
    };

    let outcome = run_cycle(
        &mut materializer,
        &mut finalizer,
        &transport,
        graph_store,
        &hasher,
        &MostRecentCommon,
        &cfg,
    )
    .await
    .map_err(|e| eyre!("restart run_cycle: {e}"))?;

    match outcome {
        Outcome::Finalized { height, .. } => {
            tracing::info!(height, "restart installed graph from checkpoint");
            Ok(RestartOutcome::Installed { height })
        }
        Outcome::Skipped(SkipReason::PeerBehindBase { freeze, base }) => {
            tracing::warn!(freeze, base, "restart skipped: peer behind base");
            Ok(RestartOutcome::PeerBehindBase { freeze, base })
        }
        Outcome::Skipped(SkipReason::NoCommonBase) => {
            tracing::warn!("restart skipped: no common base across parties");
            Ok(RestartOutcome::NoCommonBase)
        }
        Outcome::Skipped(SkipReason::NotEnoughMutations { .. }) => Err(eyre!(
            "restart unexpectedly skipped on NotEnoughMutations despite min=0"
        )),
    }
}
