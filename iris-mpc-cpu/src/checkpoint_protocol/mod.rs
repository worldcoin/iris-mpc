//! Unified checkpoint protocol.
//!
//! Shared by the sidecar binary, an in-Hawk background checkpointing task,
//! and the Hawk startup/restart path. The protocol orchestrates five
//! phases (base agreement → height agreement → materialize → hash → hash
//! consensus) over pluggable traits. Daemon loops, sleep cadence, and
//! process lifecycle live outside this module in each caller's binary.
//!
//! See `Checkpoint Protocol - Unified Design.md` for the design rationale.

pub mod hasher;
pub mod materializer;
pub mod restart;
pub mod sidecar;
pub mod store;
pub mod terminal;
pub mod transport;

#[cfg(test)]
mod tests;

pub use hasher::Blake3GraphHasher;
pub use materializer::RebuildFromCheckpoint;
pub use restart::{restart_from_checkpoint, RestartOutcome};
pub use sidecar::{sidecar_main, SidecarConfig};
pub use terminal::{InstallAsServing, UploadAndRecord};
pub use transport::RingConsensusTransport;

use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use iris_mpc_common::vector_id::VectorId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    execution::hawk_main::BothEyes,
    hnsw::{graph::mutation::GraphMutation, GraphMem},
};

/// 32-byte BLAKE3 digest.
pub type Blake3Hash = [u8; 32];

/// Row id in `hawk_graph_mutations`. Monotonic per party.
pub type GraphMutationId = i64;

/// The graph each materializer produces and each terminal action consumes.
/// Both eyes together — left then right — matching the rest of Hawk.
pub type Graph = BothEyes<GraphMem<VectorId>>;

/// Metadata for the base checkpoint a cycle starts from.
///
/// All fields participate in base agreement (strict equality across parties).
/// `graph_version` lives here so the materializer doesn't need a separate
/// lookup to drive S3 download, and so version-skewed parties fail the base
/// round rather than the hash round.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub checkpoint_id: i64,
    pub s3_key: String,
    pub last_indexed_iris_id: i64,
    pub last_indexed_modification_id: i64,
    pub graph_mutation_id: Option<GraphMutationId>,
    pub blake3_hash: String,
    pub graph_version: i32,
}

/// Inclusive upper bound on `graph_mutation_id` to apply during materialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreezeHeight(pub GraphMutationId);

/// Cycle outcome reported back to the caller's daemon loop.
#[derive(Debug)]
pub enum Outcome {
    Finalized {
        hash: Blake3Hash,
        height: GraphMutationId,
    },
    Skipped(SkipReason),
}

#[derive(Debug)]
pub enum SkipReason {
    NotEnoughMutations {
        available: u64,
        required: u64,
    },
    /// `min(peer heights)` came in below the agreed base — a peer is behind
    /// the base checkpoint. Materializing under this condition would write
    /// a regressing WAL anchor (UploadAndRecord) or install the base graph
    /// while claiming a lower height (InstallAsServing). Caller decides
    /// whether to wait + retry.
    PeerBehindBase {
        freeze: GraphMutationId,
        base: GraphMutationId,
    },
}

#[derive(Debug, Error)]
pub enum CycleError {
    /// Caller decides whether to retry-and-loop or exit-and-fail.
    #[error("transient: {0}")]
    Transient(String),
    /// Caller exits. The crash signal is the alert.
    #[error("fatal: {0}")]
    Fatal(String),
}

#[derive(Clone, Debug)]
pub struct CycleConfig {
    /// Minimum new mutations beyond the base required to run a cycle.
    /// `0` for the restart path; e.g. `10_000` for sidecar/in-Hawk loops.
    pub min_mutations_to_apply: u64,
    pub peer_round_timeout: Duration,
    /// Per-cycle nonce all three parties must agree on. Stamps the wire
    /// frame on every `ConsensusTransport::exchange` round, so a stale
    /// message from a prior cycle (e.g. on a reused channel) fails the
    /// nonce check and aborts the cycle. The caller's daemon loop is
    /// responsible for picking a value that's deterministic across
    /// parties — e.g. derived from the latest checkpoint id plus an
    /// attempt counter. For TCP-backed transports that open a fresh
    /// stream per cycle (see ampc-common PR #103), the nonce is largely a
    /// debug breadcrumb; for reused channels (in-memory tests) it's the
    /// only guard against cross-wires.
    pub cycle_nonce: u128,
}

#[async_trait]
pub trait Materializer {
    /// Produce the graph as of `freeze`. Cross-party determinism is
    /// enforced by the subsequent hash-consensus round on the returned
    /// graph, not by any height the materializer reports.
    async fn snapshot(
        &mut self,
        base: CheckpointMeta,
        freeze: FreezeHeight,
    ) -> Result<Graph, CycleError>;
}

#[async_trait]
pub trait TerminalAction {
    /// `base` is the agreed-upon checkpoint the cycle started from;
    /// `freeze` is the agreed freeze height; `graph` is the materialized
    /// snapshot at that height; `hash` is the consensus-agreed BLAKE3 over
    /// the canonical bytes of `graph`.
    ///
    /// Implementations may carry fields forward from `base` that aren't
    /// tracked elsewhere (e.g. `last_indexed_iris_id` for Hawk Main).
    async fn finalize(
        &mut self,
        base: CheckpointMeta,
        freeze: FreezeHeight,
        graph: Graph,
        hash: Blake3Hash,
    ) -> Result<(), CycleError>;
}

/// Wire-level message types exchanged with peers during consensus rounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConsensusMessage {
    BaseProposal { checkpoint: CheckpointMeta },
    HeightProposal { height: GraphMutationId },
    HashProposal { hash: Blake3Hash },
}

#[derive(Clone, Debug)]
pub struct PeerResponses<T> {
    pub responses: Vec<T>,
}

#[async_trait]
pub trait ConsensusTransport {
    /// Send `msg` to all peers, collect their responses, project each through
    /// `expect`. A peer reply whose variant doesn't match (returns `None`) is
    /// a fatal protocol error. `cycle_nonce` lets the transport reject
    /// crossed-wires from a stale cycle.
    async fn exchange<T: Send + 'static>(
        &self,
        msg: ConsensusMessage,
        expect: fn(ConsensusMessage) -> Option<T>,
        cycle_nonce: u128,
        timeout: Duration,
    ) -> Result<PeerResponses<T>, CycleError>;
}

#[async_trait]
pub trait MutationStore {
    async fn latest_checkpoint(&self) -> Result<CheckpointMeta, CycleError>;

    /// Stream WAL rows in `(lo_exclusive, hi_inclusive]` in ascending
    /// `modification_id` order. Each item is one DB row's deserialized
    /// payload — both eyes together — so a row's left and right mutations
    /// stay atomic across replay.
    async fn mutations_in_range(
        &self,
        lo_exclusive: GraphMutationId,
        hi_inclusive: GraphMutationId,
    ) -> Result<BoxStream<'_, Result<BothEyes<Vec<GraphMutation<VectorId>>>, CycleError>>, CycleError>;

    /// Largest `graph_mutation_id` currently visible to this party.
    /// Used for the height-agreement round.
    async fn current_max_mutation_id(&self) -> Result<GraphMutationId, CycleError>;
}

pub trait GraphHasher: Send + Sync {
    fn hash_canonical(&self, graph: &Graph) -> Blake3Hash;
}

/// Runs one cycle of the protocol. No looping, no sleeping; the caller's
/// daemon loop owns cadence and retry policy.
pub async fn run_cycle<M, F, T, S, H>(
    materializer: &mut M,
    finalizer: &mut F,
    transport: &T,
    store: &S,
    hasher: &H,
    cfg: &CycleConfig,
) -> Result<Outcome, CycleError>
where
    M: Materializer + Send,
    F: TerminalAction + Send,
    T: ConsensusTransport + Send + Sync,
    S: MutationStore + Send + Sync,
    H: GraphHasher,
{
    let cycle_nonce = cfg.cycle_nonce;

    // Phase 1 — base agreement. All parties must point at the same checkpoint
    // row; any disagreement is fatal (a party that diverged on which base to
    // start from cannot reach a matching hash later).
    let local_base = store.latest_checkpoint().await?;
    let peer_bases = transport
        .exchange(
            ConsensusMessage::BaseProposal {
                checkpoint: local_base.clone(),
            },
            |m| match m {
                ConsensusMessage::BaseProposal { checkpoint } => Some(checkpoint),
                _ => None,
            },
            cycle_nonce,
            cfg.peer_round_timeout,
        )
        .await?;
    for peer in &peer_bases.responses {
        if peer != &local_base {
            return Err(CycleError::Fatal(format!(
                "base mismatch: local={local_base:?} peer={peer:?}"
            )));
        }
    }
    let agreed_base = local_base;

    // Phase 2 — height agreement. Pick the min across parties so every party
    // is guaranteed to have all WAL rows up to and including the freeze.
    let local_height = store.current_max_mutation_id().await?;
    let peer_heights = transport
        .exchange(
            ConsensusMessage::HeightProposal {
                height: local_height,
            },
            |m| match m {
                ConsensusMessage::HeightProposal { height } => Some(height),
                _ => None,
            },
            cycle_nonce,
            cfg.peer_round_timeout,
        )
        .await?;
    let freeze = FreezeHeight(
        peer_heights
            .responses
            .iter()
            .copied()
            .fold(local_height, GraphMutationId::min),
    );

    // A peer reporting a height below our agreed base means they haven't
    // ingested up to the base WAL position yet. Replaying would either
    // (a) regress the WAL anchor stored at the next checkpoint, or
    // (b) install the base graph while reporting a freeze height that's
    // lower than the base's. Bail with Skipped so callers can wait.
    let lo = agreed_base.graph_mutation_id.unwrap_or(0);
    if freeze.0 < lo {
        return Ok(Outcome::Skipped(SkipReason::PeerBehindBase {
            freeze: freeze.0,
            base: lo,
        }));
    }

    // Gate on minimum mutations to apply; restart callers pass `0`.
    let available = (freeze.0 - lo) as u64;
    if available < cfg.min_mutations_to_apply {
        return Ok(Outcome::Skipped(SkipReason::NotEnoughMutations {
            available,
            required: cfg.min_mutations_to_apply,
        }));
    }

    // Phase 3 — materialize. Pluggable: rebuild-from-checkpoint, live-clone,
    // or future live-fork.
    let graph = materializer.snapshot(agreed_base.clone(), freeze).await?;

    // Phase 4 — hash the materialized graph.
    let local_hash = hasher.hash_canonical(&graph);

    // Phase 5 — hash consensus. All parties must produce the same canonical
    // bytes; any mismatch is fatal (graph divergence; cycle cannot be trusted).
    let peer_hashes = transport
        .exchange(
            ConsensusMessage::HashProposal { hash: local_hash },
            |m| match m {
                ConsensusMessage::HashProposal { hash } => Some(hash),
                _ => None,
            },
            cycle_nonce,
            cfg.peer_round_timeout,
        )
        .await?;
    for peer_hash in &peer_hashes.responses {
        if peer_hash != &local_hash {
            return Err(CycleError::Fatal(format!(
                "hash mismatch: local={} peer={}",
                hex::encode(local_hash),
                hex::encode(peer_hash),
            )));
        }
    }

    finalizer
        .finalize(agreed_base, freeze, graph, local_hash)
        .await?;
    Ok(Outcome::Finalized {
        hash: local_hash,
        height: freeze.0,
    })
}
