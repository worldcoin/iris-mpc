//! Unified checkpoint protocol shared by the sidecar binary, an in-Hawk
//! background task, and the Hawk restart path. Five phases over pluggable
//! traits: base agreement → height → materialize → hash → hash consensus.

pub mod hasher;
pub mod materializer;
pub mod runner;
pub mod selector;
pub mod store;
pub mod terminal;
pub mod transport;

#[cfg(test)]
mod tests;

pub use hasher::Blake3GraphHasher;
pub use materializer::RebuildFromCheckpoint;
pub use runner::{restart_from_checkpoint, sidecar_main, RestartOutcome, SidecarConfig};
pub use selector::{BaseSelector, MostRecentCommon, StrictLatest};
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
/// `graph_version` is included so version-skewed parties fail the base
/// round rather than slipping through to a hash mismatch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub checkpoint_id: i64,
    pub s3_key: String,
    pub last_indexed_iris_id: i64,
    pub last_indexed_modification_id: i64,
    pub graph_mutation_id: Option<GraphMutationId>,
    pub blake3_hash: String,
    pub graph_version: i32,
}

/// `checkpoint_id` is a DB-local auto-increment primary key; each party's
/// database independently assigns its own value for the same checkpoint.
/// Equality is therefore defined over all content fields except `checkpoint_id`.
impl PartialEq for CheckpointMeta {
    fn eq(&self, other: &Self) -> bool {
        self.last_indexed_iris_id == other.last_indexed_iris_id
            && self.last_indexed_modification_id == other.last_indexed_modification_id
            && self.graph_mutation_id == other.graph_mutation_id
            && self.blake3_hash == other.blake3_hash
            && self.graph_version == other.graph_version
    }
}

impl Eq for CheckpointMeta {}

impl CheckpointMeta {
    /// Cycle nonce for Phases 2-5, derived from the same content fields as
    /// `PartialEq` so every party computes the same value for the same
    /// logical checkpoint. `checkpoint_id` must not feed this: it is
    /// DB-local, and a nonce mismatch is fatal in the transport.
    pub fn content_nonce(&self) -> u128 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.last_indexed_iris_id.to_le_bytes());
        hasher.update(&self.last_indexed_modification_id.to_le_bytes());
        match self.graph_mutation_id {
            Some(id) => {
                hasher.update(&[1]);
                hasher.update(&id.to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(self.blake3_hash.as_bytes());
        hasher.update(&self.graph_version.to_le_bytes());
        u128::from_le_bytes(
            hasher.finalize().as_bytes()[..16]
                .try_into()
                .expect("16-byte slice"),
        )
    }
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
    /// `min(peer heights)` came in below the agreed base. Materializing here
    /// would either regress the WAL anchor on the next checkpoint or install
    /// the base graph while reporting a lower freeze height.
    PeerBehindBase {
        freeze: GraphMutationId,
        base: GraphMutationId,
    },
    /// [`BaseSelector::pick`] returned `None` — no checkpoint shared across
    /// all parties.
    NoCommonBase,
}

#[derive(Debug, Error)]
pub enum CycleError {
    #[error("transient: {0}")]
    Transient(String),
    #[error("fatal: {0}")]
    Fatal(String),
}

#[derive(Clone, Debug)]
pub struct CycleConfig {
    /// Minimum new mutations beyond the base required to run a cycle.
    /// `0` for the restart path; e.g. `10_000` for sidecar/in-Hawk loops.
    pub min_mutations_to_apply: u64,
    pub peer_round_timeout: Duration,
    /// Newest-first window of recent checkpoints advertised in Phase 1.
    pub checkpoint_window: usize,
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
    /// Recent checkpoints (newest first, up to `CycleConfig.checkpoint_window`).
    /// The [`BaseSelector`] decides which row across all parties' lists is
    /// the agreed base — strict-newest or most-recent-common.
    BaseProposal {
        recent: Vec<CheckpointMeta>,
    },
    HeightProposal {
        height: GraphMutationId,
    },
    HashProposal {
        hash: Blake3Hash,
    },
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
    ) -> Result<Vec<T>, CycleError>;
}

#[async_trait]
pub trait MutationStore {
    /// Returns up to `window` most recent `genesis_graph_checkpoint` rows,
    /// **newest first**. Empty vec means no checkpoints exist yet — callers
    /// (e.g. the restart path) treat this as a bootstrap signal.
    async fn recent_checkpoints(&self, window: usize) -> Result<Vec<CheckpointMeta>, CycleError>;

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

/// Sentinel nonce for Phase 1's base-list exchange. The nonce-derived-from-
/// agreed-base trick can't be used yet (we're trying to agree on the base),
/// so we use a fixed value. Cross-cycle crossed-wire detection is weaker for
/// Phase 1 only — Phases 2-5 use the agreed base's content nonce and remain
/// fully protected.
const BASE_PHASE_NONCE: u128 = 0xC4EC_B45E_C4EC_B45E_C4EC_B45E_C4EC_B45E_u128;

/// Runs one cycle of the protocol. No looping, no sleeping; the caller's
/// daemon loop owns cadence and retry policy.
pub async fn run_cycle<M, F, T, S, H, Sel>(
    materializer: &mut M,
    finalizer: &mut F,
    transport: &T,
    store: &S,
    hasher: &H,
    selector: &Sel,
    cfg: &CycleConfig,
) -> Result<Outcome, CycleError>
where
    M: Materializer + Send,
    F: TerminalAction + Send,
    T: ConsensusTransport + Send + Sync,
    S: MutationStore + Send + Sync,
    H: GraphHasher,
    Sel: BaseSelector,
{
    // Phase 1 — base agreement.
    let my_recent = store.recent_checkpoints(cfg.checkpoint_window).await?;
    let peer_lists = transport
        .exchange(
            ConsensusMessage::BaseProposal {
                recent: my_recent.clone(),
            },
            |m| match m {
                ConsensusMessage::BaseProposal { recent } => Some(recent),
                _ => None,
            },
            BASE_PHASE_NONCE,
            cfg.peer_round_timeout,
        )
        .await?;
    let agreed_base = match selector.pick(&my_recent, &peer_lists) {
        Some(b) => b,
        None => return Ok(Outcome::Skipped(SkipReason::NoCommonBase)),
    };
    // Phases 2-5 share a nonce derived from the agreed base's content —
    // preserves the crossed-wire detection the original strict-equality
    // design had, and additionally catches parties that picked different
    // bases at Phase 2 instead of at the Phase 5 hash mismatch.
    let cycle_nonce = agreed_base.content_nonce();

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
            .iter()
            .copied()
            .fold(local_height, GraphMutationId::min),
    );

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
    for peer_hash in &peer_hashes {
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
