//! Startup phase tracking and the data-derived *epoch* that keys it.
//!
//! # Why this exists
//!
//! Peer coordination during startup is currently keyed on a random per-boot
//! UUID: the heartbeat's first-contact check and [`wait_for_others_ready`]
//! reject any peer whose UUID is outside the startup-verified set. That is the
//! right policy once a node is serving — MPC session state shared with a
//! restarted peer is unrecoverable — but during startup it is stricter than
//! necessary. A peer that bounces while the fleet is still loading forces every
//! party to restart and re-run the multi-minute DB load, even when nothing about
//! the data changed.
//!
//! The fix is to key startup coordination on the *data* rather than on identity.
//! An [`Epoch`] is a digest of the whole starting configuration: every party's
//! DB length, ingest/queue frontier, modifications tail and common config. It is
//! a deterministic function of the exchanged [`SyncState`]s, so all three
//! parties derive the identical value with no extra round trip and no leader.
//! Two consequences follow directly:
//!
//! - a peer that restarts and recomputes the same epoch is provably resuming the
//!   same startup, so it may rejoin and its peers need never restart;
//! - a peer that comes back with a different `max_persisted_sequence_number`
//!   (or db length, or modifications tail) yields a different epoch, which is
//!   exactly the signal that the initial sync is void and the whole fleet must
//!   restart.
//!
//! # Attribution-free canonicalization
//!
//! [`SyncState`] carries no party id, and the order in which peers' states are
//! collected is not something a party can attribute to a slot. So the epoch is
//! built from the *sorted multiset* of per-party fact digests rather than from a
//! party-indexed vector. Sorting makes the result independent of collection
//! order on every party, and a party can still check its own participation by
//! testing its own digest for membership ([`AgreedPlan::contains`]) — which is
//! all the rejoin decision needs. Duplicate digests (the common case, where all
//! three parties are perfectly in sync) are preserved by the multiset, so a
//! party's facts changing from "same as peers" to "different" still moves the
//! epoch.
//!
//! # The surviving peers are the durable record
//!
//! There is deliberately no persisted epoch table. The authority on "which
//! startup is in progress" is the set of peers still running it, held in their
//! live [`StartupStateHandle`]s and served on [`STARTUP_STATE_ROUTE`]. A
//! restarting party does not read a local record and try to prove it still
//! matches; it simply rebuilds its [`SyncState`] from its DB and re-derives the
//! epoch from its peers' boot snapshots. It reproduces the in-flight epoch
//! **exactly when its own facts are unchanged**, so the rejoin condition checks
//! itself — no durable state, no schema migration, and no generation counter to
//! guard against stale records, because in-process state cannot go stale.
//!
//! If every party restarts at once there is no epoch to rejoin, and all three
//! derive a fresh one from current data. That is the correct outcome, not a
//! degradation.
//!
//! # What rejoin covers, and what it does not
//!
//! Rejoin works whenever the restarting party's facts are unchanged. That covers
//! the whole [`Phase::Load`] window — the long one, and the reason any of this
//! exists — provided [`Phase::Converge`] had no work to do, which is the normal
//! case for a healthy fleet: nothing to roll forward, no rows to skip ahead.
//!
//! It does not cover a restart *after* a converge that actually mutated local
//! state. Such a party recomputes different facts, derives a different epoch, and
//! the fleet restarts. Two consequences worth knowing:
//!
//! - the crash-recovery boot — the one where converge has real work — is also
//!   the one that cannot rejoin, so it behaves exactly as it does today;
//! - with `db_backed_ingest` disabled, `next_sns_sequence_num` is a live SQS
//!   queue head that the converge-phase queue trim moves, so rejoin after
//!   converge is unavailable on that path for the same reason.
//!
//! Both fall back to a full-fleet restart, which is what happens today
//! unconditionally. Nothing gets worse; the common case gets much better.
//!
//! Extending rejoin to a mutated converge means predicting each party's
//! post-converge facts and pinning them in the plan — real work, and only worth
//! doing if the crash-during-converge window turns out to matter in practice.
//!
//! # Why an unchanged epoch also means converge is safe to replay
//!
//! A rejoining party re-runs its own converge, so replay has to be harmless. It
//! is, and not by luck: the facts projection was chosen to *see* every
//! converge-phase mutation, so "facts unchanged" already implies "converge had
//! no effect".
//!
//! - `sync_modifications` moves modification statuses and iris rows → visible in
//!   `db_len` and the modifications digest;
//! - `sync_graph_mutations` inserts graph WAL rows → visible in the modifications
//!   digest, which folds in `graph_mutation_bytes`;
//! - the ingest frontier skip-ahead moves the persisted frontier → visible in
//!   `max_persisted_sequence_number`;
//! - the SQS queue trim moves the queue head → visible in
//!   `next_sns_sequence_num`.
//!
//! The one exception is releasing unpersisted ingest claims, which touches only
//! rows above the frontier and so is invisible to the projection. That step is
//! idempotent by construction — releasing an already-released claim is a no-op,
//! and re-formation is deterministic — so replaying it is safe regardless.
//!
//! Anything added to converge later must either be fact-visible or idempotent.
//! A mutation that is neither would let a party rejoin having silently applied it
//! twice.
//!
//! [`wait_for_others_ready`]: ampc_server_utils::wait_for_others_ready

use ampc_server_utils::shutdown_handler::ShutdownHandler;
use ampc_server_utils::{try_get_endpoint_other_nodes, ServerCoordinationConfig};
use axum::routing::get;
use axum::Router;
use chrono::{DateTime, Utc};
use eyre::{bail, eyre, Result, WrapErr};
use iris_mpc_common::config::CommonConfig;
use iris_mpc_common::helpers::sha256::sha256_bytes;
use iris_mpc_common::helpers::sync::{Modification, SyncState};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sodiumoxide::hex;
use std::collections::HashSet;
use std::fmt;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

/// Axum path serving the live startup-state document.
pub const STARTUP_STATE_ROUTE: &str = "/startup-state";

/// The same route as an endpoint name, for
/// [`try_get_endpoint_other_nodes`]-style peer polling.
pub const STARTUP_STATE_ENDPOINT: &str = "startup-state";

/// Domain separators. Every digest in this module is tagged so that a byte
/// string which is valid input to one level of the construction cannot be
/// reinterpreted at another level.
const FACTS_DOMAIN: &[u8] = b"iris-mpc/startup-facts/v1";
const MODIFICATIONS_DOMAIN: &[u8] = b"iris-mpc/startup-modifications/v1";
const EPOCH_DOMAIN: &[u8] = b"iris-mpc/startup-epoch/v1";

/// Where a node is in its startup sequence.
///
/// Published to peers so they can tell "still loading" from "dead" — a
/// distinction the coordination server cannot express today, since `/health`
/// answers 200 unconditionally and `/ready` only flips after the load
/// completes. Variants are ordered, and a node only ever moves forward within
/// one epoch; going backwards requires a new epoch (and, once enforcement
/// lands, a bumped generation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Coordination server up, peers not yet mutually visible.
    Discover,
    /// Mutual visibility established; exchanging startup facts.
    Propose,
    /// Facts exchanged and an epoch derived from them.
    Commit,
    /// Applying the agreed plan to local storage (modifications roll-forward,
    /// graph WAL, ingest frontier skip-ahead, queue trim).
    Converge,
    /// Local-only heavy work: the iris/graph DB load.
    Load,
    /// Ready, heartbeat armed, main loop running.
    Serving,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::Discover => "discover",
            Phase::Propose => "propose",
            Phase::Commit => "commit",
            Phase::Converge => "converge",
            Phase::Load => "load",
            Phase::Serving => "serving",
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A 32-byte SHA-256 digest, carried as a lowercase hex string in JSON so the
/// document stays greppable in logs and readable with `curl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Digest([u8; 32]);

impl Digest {
    fn of(bytes: &[u8]) -> Self {
        Digest(sha256_bytes(bytes))
    }

    /// Short form for log lines, where the full 64 hex chars are noise.
    pub fn short(&self) -> String {
        hex::encode(&self.0[..6])
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl Serialize for Digest {
    // Spelled out because `eyre::Result` is in scope for the rest of the module.
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&hex::encode(self.0))
    }
}

impl<'de> Deserialize<'de> for Digest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s)
            .map_err(|_| D::Error::custom(format!("digest is not valid hex: {s}")))?;
        let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
            D::Error::custom(format!(
                "digest must be 32 bytes, got {} hex chars",
                s.len()
            ))
        })?;
        Ok(Digest(bytes))
    }
}

/// The epoch id: a digest over the sorted multiset of all parties' fact
/// digests. Distinct from [`Digest`] only in the type system, to keep a facts
/// digest from being passed where an epoch is expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Epoch(Digest);

impl Epoch {
    pub fn short(&self) -> String {
        self.0.short()
    }
}

impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One party's startup facts: everything about its local state that the startup
/// sync reads and that must therefore be identical across a restart for a
/// rejoin to be safe.
///
/// This is a projection of [`SyncState`] and deliberately *not* the whole
/// struct: `graph_mutation_bytes` is folded into `modifications` (it is
/// parallel-by-index to it and can be megabytes), and nothing is kept that is
/// not part of the startup decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartyFacts {
    pub db_len: u64,
    pub next_sns_sequence_num: Option<u128>,
    pub max_persisted_sequence_number: Option<String>,
    pub common_config: Digest,
    /// Digest over the modifications tail zipped with its graph WAL entries.
    pub modifications: Digest,
}

impl PartyFacts {
    pub fn from_sync_state(state: &SyncState) -> Self {
        PartyFacts {
            db_len: state.db_len,
            next_sns_sequence_num: state.next_sns_sequence_num,
            max_persisted_sequence_number: state.max_persisted_sequence_number.clone(),
            common_config: digest_common_config(&state.common_config),
            modifications: digest_modifications(&state.modifications, &state.graph_mutation_bytes),
        }
    }

    /// Canonical digest of these facts.
    ///
    /// Encoded by hand rather than via `serde_json` so the wire format of
    /// [`SyncState`] can evolve (field renames, `serde(default)` additions)
    /// without silently changing every epoch. Every variable-length field is
    /// length-prefixed, so no two distinct fact sets can encode to the same
    /// bytes by shifting a boundary.
    pub fn digest(&self) -> Digest {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(FACTS_DOMAIN);
        buf.extend_from_slice(&self.db_len.to_be_bytes());
        push_opt_u128(&mut buf, self.next_sns_sequence_num);
        push_opt_str(&mut buf, self.max_persisted_sequence_number.as_deref());
        buf.extend_from_slice(&self.common_config.0);
        buf.extend_from_slice(&self.modifications.0);
        Digest::of(&buf)
    }
}

/// The startup plan every party derives independently from the exchanged
/// [`SyncState`]s, together with the epoch that identifies it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgreedPlan {
    /// Per-party fact digests, sorted — a multiset, not a party-indexed vector
    /// (see the module docs on attribution-free canonicalization).
    facts: Vec<Digest>,
    epoch: Epoch,
}

impl AgreedPlan {
    /// Derive the plan from all parties' states, including this party's own.
    ///
    /// `states` is expected to be [`SyncResult::all_states`], which is
    /// `[my_state] ++ peers`.
    ///
    /// [`SyncResult::all_states`]: iris_mpc_common::helpers::sync::SyncResult
    pub fn from_states(states: &[SyncState]) -> Result<Self> {
        if states.len() < 2 {
            bail!(
                "cannot derive a startup epoch from {} state(s): the plan is only meaningful over \
                 the whole fleet",
                states.len()
            );
        }
        let facts = states
            .iter()
            .map(|state| PartyFacts::from_sync_state(state).digest())
            .collect();
        Ok(Self::from_fact_digests(facts))
    }

    fn from_fact_digests(mut facts: Vec<Digest>) -> Self {
        // Sort so the epoch is independent of the order peers were polled in.
        facts.sort_unstable();

        let mut buf = Vec::with_capacity(EPOCH_DOMAIN.len() + 8 + facts.len() * 32);
        buf.extend_from_slice(EPOCH_DOMAIN);
        // Party count is part of the preimage: a two-party and a three-party
        // fleet that happen to share a digest prefix must not collide.
        buf.extend_from_slice(&(facts.len() as u64).to_be_bytes());
        for digest in &facts {
            buf.extend_from_slice(&digest.0);
        }

        AgreedPlan {
            epoch: Epoch(Digest::of(&buf)),
            facts,
        }
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    pub fn party_count(&self) -> usize {
        self.facts.len()
    }

    /// Whether `facts` is one of the fact sets this plan was built from.
    ///
    /// The basis of the future rejoin check: a restarting party recomputes its
    /// facts from its DB and asks the persisted plan whether they still belong
    /// to it.
    pub fn contains(&self, facts: &Digest) -> bool {
        self.facts.contains(facts)
    }

    pub fn fact_digests(&self) -> &[Digest] {
        &self.facts
    }
}

/// The single document a node publishes about its own startup.
///
/// Consolidates what is spread across `/health`, `/ready` and `/startup-sync`
/// today, and unlike `/startup-sync` — which serves a `my_state` clone captured
/// when the coordination server was constructed — it is read live from
/// [`StartupStateHandle`], so it reflects the node's current phase rather than
/// its state at boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupState {
    pub party_id: usize,
    /// This incarnation's coordination UUID. Kept for correlation with
    /// `/health` and the heartbeat; it is deliberately *not* what peers key
    /// their startup decisions on.
    ///
    /// `None` for the brief window before the coordination server is
    /// constructed, since the UUID is minted inside it while this route has to
    /// be handed to it — correlation-only data, so it is not worth inverting
    /// that ordering upstream.
    pub uuid: Option<String>,
    /// Reserved, always 0.
    ///
    /// Intended as an ABA guard for a durable epoch record; the peer-memory
    /// design made that record unnecessary (see the module docs), so nothing
    /// sets it. Retained as a `serde(default)` field so documents stay
    /// compatible in both directions across a rolling deploy.
    #[serde(default)]
    pub generation: u64,
    pub phase: Phase,
    pub phase_started_at: DateTime<Utc>,
    /// Digest of this party's own facts, known from the moment its
    /// [`SyncState`] is built — i.e. before any peer has been contacted.
    pub facts: Digest,
    /// `None` until the facts exchange completes and the epoch is derived.
    pub epoch: Option<Epoch>,
}

/// Live, shared handle to this node's [`StartupState`]; the write side of
/// [`STARTUP_STATE_ROUTE`].
#[derive(Debug, Clone)]
pub struct StartupStateHandle {
    state: Arc<RwLock<StartupState>>,
}

impl StartupStateHandle {
    /// Create the handle in [`Phase::Discover`].
    ///
    /// Must be called before the coordination server starts, since its router
    /// is passed in as an extra route; the UUID it mints arrives afterwards via
    /// [`StartupStateHandle::set_uuid`].
    pub fn new(party_id: usize, facts: Digest) -> Self {
        StartupStateHandle {
            state: Arc::new(RwLock::new(StartupState {
                party_id,
                uuid: None,
                generation: 0,
                phase: Phase::Discover,
                phase_started_at: Utc::now(),
                facts,
                epoch: None,
            })),
        }
    }

    /// Router to hand to `start_coordination_server_with_extra_routes`.
    ///
    /// Always answers 200: the document is meaningful in every phase, and a
    /// pre-agreement node is described by `epoch: null` rather than by an error
    /// status. Peers distinguish the cases by reading the fields.
    pub fn router(&self) -> Router {
        let state = Arc::clone(&self.state);
        Router::new().route(
            STARTUP_STATE_ROUTE,
            get(move || {
                let state = Arc::clone(&state);
                async move {
                    let snapshot = state.read().await.clone();
                    serde_json::to_string(&snapshot)
                        .expect("StartupState serialization to JSON failed")
                }
            }),
        )
    }

    pub async fn snapshot(&self) -> StartupState {
        self.state.read().await.clone()
    }

    /// Record the coordination UUID once the coordination server has minted it.
    pub async fn set_uuid(&self, uuid: String) {
        self.state.write().await.uuid = Some(uuid);
    }

    /// Advance to `phase`, logging the transition and the time the previous
    /// phase took.
    ///
    /// A backwards transition is logged as an error rather than rejected: while
    /// this module is observation-only, a wrong phase label must not be able to
    /// fail a startup that would otherwise have succeeded.
    pub async fn enter(&self, phase: Phase) {
        let now = Utc::now();
        let mut state = self.state.write().await;
        let previous = state.phase;
        let elapsed = now.signed_duration_since(state.phase_started_at);

        if phase < previous {
            tracing::error!(
                "Startup phase moved backwards: {} -> {} (epoch {:?})",
                previous,
                phase,
                state.epoch.map(|e| e.short())
            );
        }

        state.phase = phase;
        state.phase_started_at = now;
        drop(state);

        tracing::info!(
            "Startup phase {} -> {} (previous phase took {}s)",
            previous,
            phase,
            elapsed.num_seconds()
        );
        metrics::counter!("startup_phase_entered", "phase" => phase.as_str()).increment(1);
    }

    /// Record the derived epoch. Called once, on entering [`Phase::Commit`].
    pub async fn set_epoch(&self, epoch: Epoch) {
        let mut state = self.state.write().await;
        if let Some(previous) = state.epoch {
            if previous != epoch {
                tracing::error!(
                    "Startup epoch changed within one boot: {} -> {}",
                    previous,
                    epoch
                );
            }
        }
        state.epoch = Some(epoch);
    }
}

/// Fetch and parse every peer's startup-state document.
///
/// A peer whose document cannot be parsed is an error rather than a skipped
/// entry: a caller counting agreeing peers must not be able to reach quorum
/// because a disagreeing peer was quietly dropped.
async fn poll_peer_states(
    config: &ServerCoordinationConfig,
    budget: Duration,
) -> Result<Vec<StartupState>> {
    // `try_get_endpoint_other_nodes` retries internally against its own
    // `startup_sync_timeout_secs` budget, which is unrelated to the caller's,
    // so cap it here rather than inheriting it.
    let responses = tokio::time::timeout(
        budget,
        try_get_endpoint_other_nodes(config, STARTUP_STATE_ENDPOINT),
    )
    .await
    .map_err(|_| eyre!("timed out polling peer startup state after {:?}", budget))??;

    responses
        .into_iter()
        .map(|(status, body)| {
            serde_json::from_slice::<StartupState>(&body).wrap_err_with(|| {
                format!("failed to deserialize peer startup state (status {status})")
            })
        })
        .collect()
}

/// Record a peer incarnation as seen, so the coordination server advertises it
/// on `/health` and the `ampc-server-utils` checks downstream accept it.
///
/// This is bookkeeping, not authorization: it says "this incarnation exists and
/// we have observed it", which is exactly what
/// `wait_until_startup_visibility_is_complete` needs from us before a restarted
/// peer can get far enough to publish an epoch at all. Withholding it until the
/// epoch matched would deadlock — the peer cannot pass its visibility barrier
/// until we list its UUID, and it cannot publish an epoch until it passes that
/// barrier. Authorization lives entirely in the epoch comparison below.
async fn record_peer_uuids(verified_peers: &Arc<Mutex<HashSet<String>>>, peers: &[StartupState]) {
    let mut verified = verified_peers.lock().await;
    for peer in peers {
        if let Some(uuid) = &peer.uuid {
            if verified.insert(uuid.clone()) {
                tracing::info!(
                    "Recorded peer party {} incarnation {} (phase {})",
                    peer.party_id,
                    uuid,
                    peer.phase
                );
            }
        }
    }
}

/// What a round of peer observations says about the epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EpochVerdict {
    /// Every peer published this epoch.
    Committed,
    /// One or more peers have not published an epoch yet: either they have not
    /// reached [`Phase::Commit`], or they restarted and are on their way back.
    Pending(Vec<(usize, Phase)>),
    /// A peer published a different epoch, so the agreed plan is void.
    Void(Vec<(usize, Option<Epoch>)>),
}

/// Classify one round of peer documents against our own epoch.
///
/// Shared by the commit barrier and the startup watch so the two cannot drift
/// apart on what counts as disagreement. Fails closed in both directions:
/// [`EpochVerdict::Void`] takes precedence over [`EpochVerdict::Pending`], and an
/// empty peer list is `Pending` rather than a vacuous `Committed`.
fn classify_peers(my_epoch: Epoch, peers: &[StartupState]) -> EpochVerdict {
    let void: Vec<_> = peers
        .iter()
        .filter(|peer| peer.epoch.is_some_and(|epoch| epoch != my_epoch))
        .map(|peer| (peer.party_id, peer.epoch))
        .collect();

    if !void.is_empty() {
        return EpochVerdict::Void(void);
    }

    if peers.is_empty() {
        return EpochVerdict::Pending(Vec::new());
    }

    let pending: Vec<_> = peers
        .iter()
        .filter(|peer| peer.epoch.is_none())
        .map(|peer| (peer.party_id, peer.phase))
        .collect();

    if pending.is_empty() {
        EpochVerdict::Committed
    } else {
        EpochVerdict::Pending(pending)
    }
}

/// Commit barrier: hold until every peer has published *this* epoch.
///
/// Nothing may mutate local storage before this returns. Converging under a plan
/// a peer never agreed to is the failure mode the barrier exists to prevent.
///
/// # What a mismatch does and does not mean
///
/// A mismatch is *not* "the parties' data diverged" — divergence is normal and
/// is precisely what the modifications roll-forward exists to repair. The epoch
/// is a digest of the whole starting configuration, differences included, so
/// three parties that legitimately disagree about their persisted frontier still
/// derive the same epoch from the same three fact sets.
///
/// A mismatch means the parties saw *different inputs*: some party's facts
/// changed between the exchange and now, which in practice means it restarted
/// and came back with different data. The agreed plan is then void, and every
/// party must abandon it and re-derive from current state — the full-fleet
/// restart this returns an error to trigger.
pub async fn wait_for_epoch_commit(
    config: &ServerCoordinationConfig,
    verified_peers: &Arc<Mutex<HashSet<String>>>,
    my_epoch: Epoch,
) -> Result<()> {
    tracing::info!("Waiting for peers to commit startup epoch {}", my_epoch);

    let budget = Duration::from_secs(config.startup_sync_timeout_secs);
    let retry_delay = Duration::from_millis(config.http_query_retry_delay_ms);
    let deadline = Instant::now() + budget;

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            metrics::counter!("startup_epoch_commit_timeout").increment(1);
            bail!(
                "peers did not commit startup epoch {} within {:?}",
                my_epoch,
                budget
            );
        }

        let peers = match poll_peer_states(config, remaining).await {
            Ok(peers) => peers,
            Err(err) => {
                tracing::warn!("Failed to poll peer startup state: {:?}", err);
                tokio::time::sleep(retry_delay.min(remaining)).await;
                continue;
            }
        };

        record_peer_uuids(verified_peers, &peers).await;

        match classify_peers(my_epoch, &peers) {
            EpochVerdict::Void(disagreeing) => {
                metrics::counter!("startup_epoch_mismatch").increment(1);
                bail!(
                    "startup epoch mismatch: mine is {}, peers report {:?}. The agreed plan is \
                     void; every party must restart and re-derive from current state.",
                    my_epoch,
                    format_epochs(&disagreeing)
                );
            }
            EpochVerdict::Committed => {
                tracing::info!(
                    "Startup epoch {} committed by all {} parties",
                    my_epoch,
                    peers.len() + 1
                );
                metrics::counter!("startup_epoch_agreement").increment(1);
                return Ok(());
            }
            EpochVerdict::Pending(pending) => {
                tracing::debug!("Peers have not committed an epoch yet: {:?}", pending);
                let remaining = deadline.saturating_duration_since(Instant::now());
                tokio::time::sleep(retry_delay.min(remaining)).await;
            }
        }
    }
}

fn format_epochs(peers: &[(usize, Option<Epoch>)]) -> Vec<(usize, String)> {
    peers
        .iter()
        .map(|(party_id, epoch)| {
            (
                *party_id,
                epoch
                    .map(|epoch| epoch.to_string())
                    .unwrap_or_else(|| "none".to_string()),
            )
        })
        .collect()
}

/// Guard for the background task that watches peers across the converge and
/// load phases. Aborts the task when dropped.
pub struct StartupWatch {
    task: JoinHandle<()>,
    /// Set once, to the reason the epoch was voided.
    verdict: Arc<StdMutex<Option<String>>>,
}

impl StartupWatch {
    /// Fail if the watch observed a peer on a different epoch.
    ///
    /// Call at every point where startup can still be abandoned cheaply. In
    /// particular, call it after the DB load: `trigger_manual_shutdown` is
    /// cooperative, and the load does not poll it, so a verdict reached while
    /// loading is only acted on once the load returns.
    pub fn check(&self) -> Result<()> {
        let verdict = self
            .verdict
            .lock()
            .expect("startup watch verdict mutex poisoned")
            .clone();
        match verdict {
            Some(reason) => bail!("{reason}"),
            None => Ok(()),
        }
    }

    /// Stop watching. Called on reaching [`Phase::Serving`], after which the
    /// heartbeat owns peer supervision and its UUID-mismatch teardown is the
    /// correct policy — a peer that restarts once sessions exist cannot rejoin.
    pub fn stop(self) {
        self.task.abort();
    }
}

impl Drop for StartupWatch {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Watch peers for the duration of converge and load — the window in which
/// nothing else is looking.
///
/// This closes the gap that forces today's full-fleet restarts. The heartbeat is
/// only armed after the load, so a peer that dies during the load is invisible
/// until the ready barrier, and a peer that *restarts* during it is rejected
/// outright: its new UUID is absent from the frozen `verified_peers` set, so
/// `wait_for_others_ready` and the heartbeat's first-contact check both kill an
/// otherwise healthy node.
///
/// The watch replaces that identity test with the epoch test:
///
/// - peer republishes this epoch (whatever its UUID) — it is provably resuming
///   the same startup, so it is recorded as verified and both parties carry on.
///   A restart is invisible to us, which is the entire point;
/// - peer publishes a different epoch — its data changed, the plan is void, and
///   we abandon startup;
/// - peer publishes no epoch, or is unreachable — it is mid-restart. Tolerated:
///   it will either republish this epoch or a different one, and the ready
///   barrier still bounds how long a peer may fail to arrive.
///
/// Note what makes the first case work without any durable record: a restarting
/// peer rebuilds its [`SyncState`] from its DB and re-derives the epoch from its
/// peers' boot snapshots, so it reproduces this epoch **exactly when its own
/// facts are unchanged** — the rejoin condition is self-checking. It therefore
/// holds for the common case of a converge that had no work to do, and correctly
/// fails when converge actually mutated local state (see the module docs).
pub fn spawn_startup_watch(
    config: ServerCoordinationConfig,
    verified_peers: Arc<Mutex<HashSet<String>>>,
    shutdown_handler: Arc<ShutdownHandler>,
    epoch: Epoch,
) -> StartupWatch {
    let verdict = Arc::new(StdMutex::new(None));

    let task = tokio::spawn({
        let verdict = Arc::clone(&verdict);
        // Reuse the heartbeat cadence: this is the same job, over the phase of
        // startup where the heartbeat itself cannot yet run.
        let poll_interval = Duration::from_secs(config.heartbeat_interval_secs);

        async move {
            loop {
                tokio::time::sleep(poll_interval).await;

                if shutdown_handler.is_shutting_down() {
                    tracing::info!("Startup watch stopping: shutdown already in progress");
                    return;
                }

                let peers = match poll_peer_states(&config, poll_interval).await {
                    Ok(peers) => peers,
                    Err(err) => {
                        // Expected while a peer restarts; the ready barrier and
                        // heartbeat bound the terminal cases.
                        tracing::warn!("Startup watch could not poll peers: {:?}", err);
                        continue;
                    }
                };

                record_peer_uuids(&verified_peers, &peers).await;

                if let EpochVerdict::Void(disagreeing) = classify_peers(epoch, &peers) {
                    let disagreeing = format_epochs(&disagreeing);
                    let reason = format!(
                        "startup watch found a peer on a different epoch: mine is {epoch}, peers \
                         report {disagreeing:?}. The agreed plan is void; abandoning startup so \
                         every party re-derives from current state."
                    );
                    tracing::error!("{}", reason);
                    metrics::counter!("startup_epoch_mismatch").increment(1);

                    *verdict
                        .lock()
                        .expect("startup watch verdict mutex poisoned") = Some(reason);

                    if !shutdown_handler.is_shutting_down() {
                        shutdown_handler.trigger_manual_shutdown();
                    }
                    return;
                }

                for peer in &peers {
                    tracing::debug!(
                        "Startup watch: party {} in phase {} (epoch {})",
                        peer.party_id,
                        peer.phase,
                        peer.epoch
                            .map(|e| e.short())
                            .unwrap_or_else(|| "pending".to_string())
                    );
                }
            }
        }
    });

    StartupWatch { task, verdict }
}

fn digest_common_config(config: &CommonConfig) -> Digest {
    // `CommonConfig` is all scalars, `Vec`s and `Option`s — no maps — so
    // `serde_json` field order is the declaration order and the encoding is
    // deterministic. It is also already compared field-by-field across parties
    // by `SyncResult::check_common_config`, so any difference the digest sees
    // is a difference that check would reject anyway.
    let bytes = serde_json::to_vec(config).expect("CommonConfig serialization to JSON failed");
    Digest::of(&bytes)
}

/// Digest the modifications tail together with its graph WAL entries.
///
/// `graph_mutation_bytes` is parallel-by-index to `modifications`, so the two
/// are zipped before being sorted by modification id — canonicalizing the order
/// without breaking the pairing. Mutation blobs are folded in by digest rather
/// than by value: they are large, and only their identity matters here.
fn digest_modifications(
    modifications: &[Modification],
    graph_mutation_bytes: &[Option<Vec<u8>>],
) -> Digest {
    let mut rows: Vec<(&Modification, Option<&Vec<u8>>)> = modifications
        .iter()
        .enumerate()
        .map(|(i, m)| (m, graph_mutation_bytes.get(i).and_then(Option::as_ref)))
        .collect();
    rows.sort_unstable_by_key(|(m, _)| m.id);

    let mut buf = Vec::with_capacity(MODIFICATIONS_DOMAIN.len() + 8 + rows.len() * 96);
    buf.extend_from_slice(MODIFICATIONS_DOMAIN);
    buf.extend_from_slice(&(rows.len() as u64).to_be_bytes());
    for (modification, mutation) in rows {
        buf.extend_from_slice(&modification.id.to_be_bytes());
        push_opt_i64(&mut buf, modification.serial_id);
        push_str(&mut buf, &modification.request_type);
        push_opt_str(&mut buf, modification.s3_url.as_deref());
        push_str(&mut buf, &modification.status);
        buf.push(u8::from(modification.persisted));
        push_opt_str(&mut buf, modification.result_message_body.as_deref());
        match mutation {
            Some(bytes) => {
                buf.push(1);
                buf.extend_from_slice(&sha256_bytes(bytes));
            }
            None => buf.push(0),
        }
    }
    Digest::of(&buf)
}

// Canonical encoding helpers. Every variable-length value is length-prefixed
// and every optional value is tagged, so distinct inputs cannot share an
// encoding by shifting a field boundary.

fn push_str(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_be_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_opt_str(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            push_str(buf, value);
        }
        None => buf.push(0),
    }
}

fn push_opt_i64(buf: &mut Vec<u8>, value: Option<i64>) {
    match value {
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_be_bytes());
        }
        None => buf.push(0),
    }
}

fn push_opt_u128(buf: &mut Vec<u8>, value: Option<u128>) {
    match value {
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_be_bytes());
        }
        None => buf.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modification(id: i64, status: &str) -> Modification {
        Modification {
            id,
            serial_id: Some(id * 10),
            request_type: "uniqueness".to_string(),
            s3_url: Some(format!("s3://bucket/{id}")),
            status: status.to_string(),
            persisted: true,
            result_message_body: None,
        }
    }

    fn state(db_len: u64, max_persisted: Option<&str>) -> SyncState {
        SyncState {
            db_len,
            modifications: vec![modification(1, "COMPLETED"), modification(2, "IN_PROGRESS")],
            next_sns_sequence_num: Some(42),
            common_config: CommonConfig::default(),
            graph_mutation_bytes: vec![Some(vec![0xab; 16]), None],
            max_persisted_sequence_number: max_persisted.map(str::to_string),
        }
    }

    fn fleet() -> Vec<SyncState> {
        vec![
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000009")),
        ]
    }

    #[test]
    fn epoch_is_independent_of_collection_order() {
        // The whole design rests on this: each party polls its peers in its own
        // order, yet all must derive the same epoch with no extra round trip.
        let baseline = AgreedPlan::from_states(&fleet()).unwrap();

        let mut rotated = fleet();
        rotated.rotate_left(1);
        assert_eq!(
            baseline.epoch(),
            AgreedPlan::from_states(&rotated).unwrap().epoch()
        );

        let mut reversed = fleet();
        reversed.reverse();
        assert_eq!(
            baseline.epoch(),
            AgreedPlan::from_states(&reversed).unwrap().epoch()
        );
    }

    #[test]
    fn epoch_changes_when_any_party_fact_changes() {
        let baseline = AgreedPlan::from_states(&fleet()).unwrap().epoch();

        // The motivating case: a peer restarts and comes back with a different
        // persisted frontier. That must void the epoch.
        let mut changed = fleet();
        changed[2].max_persisted_sequence_number = Some("0000000000000011".to_string());
        assert_ne!(
            baseline,
            AgreedPlan::from_states(&changed).unwrap().epoch(),
            "a changed persisted frontier must change the epoch"
        );

        let mut changed = fleet();
        changed[0].db_len += 1;
        assert_ne!(baseline, AgreedPlan::from_states(&changed).unwrap().epoch());

        let mut changed = fleet();
        changed[1].modifications[0].status = "COMPLETED_WITH_ERROR".to_string();
        assert_ne!(baseline, AgreedPlan::from_states(&changed).unwrap().epoch());

        let mut changed = fleet();
        changed[1].modifications[0].persisted = false;
        assert_ne!(baseline, AgreedPlan::from_states(&changed).unwrap().epoch());

        let mut changed = fleet();
        changed[1].next_sns_sequence_num = None;
        assert_ne!(baseline, AgreedPlan::from_states(&changed).unwrap().epoch());

        let mut changed = fleet();
        changed[0].graph_mutation_bytes = vec![Some(vec![0xcd; 16]), None];
        assert_ne!(baseline, AgreedPlan::from_states(&changed).unwrap().epoch());
    }

    #[test]
    fn duplicate_facts_are_kept_as_a_multiset() {
        // Two parties in sync and one behind must not hash the same as three
        // parties in sync: collapsing duplicates would erase the difference.
        let two_agree = AgreedPlan::from_states(&fleet()).unwrap();
        assert_eq!(two_agree.party_count(), 3);

        let all_agree = vec![
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
        ];
        assert_ne!(
            two_agree.epoch(),
            AgreedPlan::from_states(&all_agree).unwrap().epoch()
        );
    }

    #[test]
    fn every_party_finds_its_own_facts_in_the_plan() {
        // The rejoin check: recompute my facts, ask the plan if I belong.
        let states = fleet();
        let plan = AgreedPlan::from_states(&states).unwrap();
        for state in &states {
            assert!(plan.contains(&PartyFacts::from_sync_state(state).digest()));
        }

        let mut stranger = state(100, Some("0000000000000010"));
        stranger.db_len = 999;
        assert!(!plan.contains(&PartyFacts::from_sync_state(&stranger).digest()));
    }

    #[test]
    fn party_count_is_part_of_the_epoch() {
        let states = fleet();
        let three = AgreedPlan::from_states(&states).unwrap();
        let two = AgreedPlan::from_states(&states[..2]).unwrap();
        assert_ne!(three.epoch(), two.epoch());
    }

    #[test]
    fn a_single_state_cannot_form_a_plan() {
        assert!(AgreedPlan::from_states(&fleet()[..1]).is_err());
    }

    #[test]
    fn field_boundaries_are_unambiguous() {
        // Length prefixes must stop adjacent string fields from being shifted
        // into one another without changing the digest.
        let mut a = state(1, None);
        a.modifications[0].request_type = "ab".to_string();
        a.modifications[0].status = "cd".to_string();

        let mut b = state(1, None);
        b.modifications[0].request_type = "abc".to_string();
        b.modifications[0].status = "d".to_string();

        assert_ne!(
            PartyFacts::from_sync_state(&a).digest(),
            PartyFacts::from_sync_state(&b).digest()
        );
    }

    #[test]
    fn modification_order_does_not_affect_the_digest() {
        // `last_modifications` order is not a contract; the pairing with
        // `graph_mutation_bytes` is.
        let forward = state(100, None);
        let mut reversed = forward.clone();
        reversed.modifications.reverse();
        reversed.graph_mutation_bytes.reverse();

        assert_eq!(
            PartyFacts::from_sync_state(&forward).digest(),
            PartyFacts::from_sync_state(&reversed).digest()
        );
    }

    #[test]
    fn mispaired_graph_mutations_change_the_digest() {
        // Reversing the modifications without reversing their WAL entries
        // re-pairs them, which is a real divergence and must be visible.
        let forward = state(100, None);
        let mut mispaired = forward.clone();
        mispaired.modifications.reverse();

        assert_ne!(
            PartyFacts::from_sync_state(&forward).digest(),
            PartyFacts::from_sync_state(&mispaired).digest()
        );
    }

    #[test]
    fn digest_survives_a_json_round_trip() {
        let plan = AgreedPlan::from_states(&fleet()).unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        let parsed: AgreedPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, parsed);
        assert_eq!(plan.epoch(), parsed.epoch());
    }

    #[tokio::test]
    async fn published_document_tracks_phase_and_epoch() {
        let facts = PartyFacts::from_sync_state(&state(100, None)).digest();
        let handle = StartupStateHandle::new(1, facts);

        let initial = handle.snapshot().await;
        assert_eq!(initial.phase, Phase::Discover);
        assert!(initial.epoch.is_none());
        assert!(initial.uuid.is_none());
        assert_eq!(initial.facts, facts);

        handle.set_uuid("uuid-1".to_string()).await;

        let epoch = AgreedPlan::from_states(&fleet()).unwrap().epoch();
        handle.enter(Phase::Propose).await;
        handle.enter(Phase::Commit).await;
        handle.set_epoch(epoch).await;

        let committed = handle.snapshot().await;
        assert_eq!(committed.phase, Phase::Commit);
        assert_eq!(committed.epoch, Some(epoch));

        // The document must survive the wire: peers parse exactly this.
        let json = serde_json::to_string(&committed).unwrap();
        let parsed: StartupState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.epoch, Some(epoch));
        assert_eq!(parsed.phase, Phase::Commit);
        assert_eq!(parsed.party_id, 1);
        assert_eq!(parsed.generation, 0);
        assert_eq!(parsed.uuid.as_deref(), Some("uuid-1"));
    }

    fn peer(party_id: usize, phase: Phase, epoch: Option<Epoch>) -> StartupState {
        StartupState {
            party_id,
            uuid: Some(format!("uuid-{party_id}")),
            generation: 0,
            phase,
            phase_started_at: Utc::now(),
            facts: PartyFacts::from_sync_state(&state(100, None)).digest(),
            epoch,
        }
    }

    fn other_epoch() -> Epoch {
        let mut changed = fleet();
        changed[0].db_len = 12_345;
        AgreedPlan::from_states(&changed).unwrap().epoch()
    }

    #[test]
    fn all_peers_on_my_epoch_is_committed() {
        let mine = AgreedPlan::from_states(&fleet()).unwrap().epoch();
        let peers = [
            peer(1, Phase::Commit, Some(mine)),
            peer(2, Phase::Load, Some(mine)),
        ];
        assert_eq!(classify_peers(mine, &peers), EpochVerdict::Committed);
    }

    #[test]
    fn a_peer_without_an_epoch_is_pending_not_void() {
        // Either it has not reached Commit, or it restarted and is on its way
        // back. Neither is a disagreement.
        let mine = AgreedPlan::from_states(&fleet()).unwrap().epoch();
        let peers = [
            peer(1, Phase::Commit, Some(mine)),
            peer(2, Phase::Discover, None),
        ];
        assert_eq!(
            classify_peers(mine, &peers),
            EpochVerdict::Pending(vec![(2, Phase::Discover)])
        );
    }

    #[test]
    fn a_peer_on_a_different_epoch_voids_the_plan() {
        // The motivating failure: a peer restarted with changed data.
        let mine = AgreedPlan::from_states(&fleet()).unwrap().epoch();
        let theirs = other_epoch();
        let peers = [
            peer(1, Phase::Commit, Some(mine)),
            peer(2, Phase::Commit, Some(theirs)),
        ];
        assert_eq!(
            classify_peers(mine, &peers),
            EpochVerdict::Void(vec![(2, Some(theirs))])
        );
    }

    #[test]
    fn void_takes_precedence_over_pending() {
        // Fail closed: one peer still arriving must not mask another peer that
        // has already proven the plan void.
        let mine = AgreedPlan::from_states(&fleet()).unwrap().epoch();
        let theirs = other_epoch();
        let peers = [
            peer(1, Phase::Discover, None),
            peer(2, Phase::Commit, Some(theirs)),
        ];
        assert_eq!(
            classify_peers(mine, &peers),
            EpochVerdict::Void(vec![(2, Some(theirs))])
        );
    }

    #[test]
    fn no_peers_is_pending_not_vacuously_committed() {
        // A round that saw nobody must never satisfy the commit barrier.
        let mine = AgreedPlan::from_states(&fleet()).unwrap().epoch();
        assert_eq!(classify_peers(mine, &[]), EpochVerdict::Pending(Vec::new()));
    }

    #[test]
    fn rejoin_condition_is_self_checking() {
        // The core claim of the peer-memory design. A restarting party rebuilds
        // its own state from its DB and re-derives the epoch from its peers'
        // (unchanged) boot snapshots.
        let original = fleet();
        let in_flight = AgreedPlan::from_states(&original).unwrap().epoch();

        // Facts unchanged across the restart — converge had no work to do — so
        // the party reproduces the in-flight epoch and rejoins.
        let mut rebooted = original.clone();
        rebooted[0] = state(100, Some("0000000000000010"));
        assert_eq!(
            in_flight,
            AgreedPlan::from_states(&rebooted).unwrap().epoch(),
            "an unchanged party must reproduce the in-flight epoch"
        );

        // Facts changed across the restart — converge mutated local state, or the
        // data really did move — so the epoch differs and the fleet restarts.
        let mut rebooted = original.clone();
        rebooted[0] = state(101, Some("0000000000000011"));
        assert_ne!(
            in_flight,
            AgreedPlan::from_states(&rebooted).unwrap().epoch(),
            "a changed party must not be able to rejoin"
        );
    }

    #[test]
    fn phases_are_ordered_by_progress() {
        assert!(Phase::Discover < Phase::Propose);
        assert!(Phase::Propose < Phase::Commit);
        assert!(Phase::Commit < Phase::Converge);
        assert!(Phase::Converge < Phase::Load);
        assert!(Phase::Load < Phase::Serving);
    }
}
