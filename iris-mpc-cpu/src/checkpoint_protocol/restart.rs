//! One-shot Hawk restart: rebuild the in-memory graph from the latest
//! checkpoint + WAL replay, hash-agree with peers, and install the result
//! into the live graph reference before serving begins.
//!
//! The function is intentionally a library helper, not wired into
//! `hawk_main`'s startup flow here — callers thread it into their own
//! orchestration when ready.

use std::sync::Arc;
use std::time::Duration;

use ampc_actor_utils::network::mpc::NetworkHandle;
use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use tokio::sync::RwLock;

use crate::checkpoint_protocol::{
    run_cycle, Blake3GraphHasher, CycleConfig, InstallAsServing, Outcome, RebuildFromCheckpoint,
    RingConsensusTransport,
};
use crate::execution::hawk_main::BothEyes;
use crate::hnsw::{
    graph::{graph_store::GraphPg, layered_graph::GraphMem},
    VectorStore,
};
use iris_mpc_common::vector_id::VectorId;

/// Rebuild the live graph from the latest checkpoint + WAL replay, then
/// install it into `target`. `min_mutations_to_apply=0` so the cycle
/// proceeds even when there are no new mutations beyond the base.
///
/// Returns `Ok(true)` if a graph was installed; `Ok(false)` if no
/// `genesis_graph_checkpoint` row exists yet — the caller should fall
/// back to its own bootstrap path (empty graph, plaintext checkpoint
/// migration, etc).
pub async fn restart_from_checkpoint<V: VectorStore + Send + Sync>(
    graph_store: &GraphPg<V>,
    s3_client: &S3Client,
    bucket: String,
    networking: &mut Box<dyn NetworkHandle>,
    target: BothEyes<Arc<RwLock<GraphMem<VectorId>>>>,
    peer_round_timeout: Duration,
) -> Result<bool> {
    // Probe: if there's no base checkpoint, the caller decides what to do.
    let checkpoint_row = graph_store
        .get_latest_genesis_graph_checkpoint()
        .await
        .map_err(|e| eyre!("get_latest_genesis_graph_checkpoint: {e}"))?;
    let Some(row) = checkpoint_row else {
        return Ok(false);
    };
    let cycle_nonce = row.id as u128;

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
        cycle_nonce,
    };

    let outcome = run_cycle(
        &mut materializer,
        &mut finalizer,
        &transport,
        graph_store,
        &hasher,
        &cfg,
    )
    .await
    .map_err(|e| eyre!("restart run_cycle: {e}"))?;

    match outcome {
        Outcome::Finalized { height, .. } => {
            tracing::info!(height, "restart installed graph from checkpoint");
            Ok(true)
        }
        Outcome::Skipped(reason) => {
            // min=0 means a Skipped outcome is structurally impossible; treat
            // as a bug rather than a recoverable state.
            Err(eyre!("restart unexpectedly skipped: {reason:?}"))
        }
    }
}
