//! Startup phase tracking and the data-derived *epoch* that keys it.
//!
//! # Why this exists
//!
//! Peer coordination during startup is keyed on a random per-boot UUID: the
//! heartbeat's first-contact check and [`wait_for_others_ready`] reject any peer
//! whose UUID is outside the startup-verified set. That is the right policy once
//! a node is serving — MPC session state shared with a restarted peer is
//! unrecoverable — but during startup it forces every party to restart and re-run
//! the multi-minute DB load whenever one peer bounces, even with the data
//! unchanged.
//!
//! So startup coordination is keyed on the *data* instead. An [`Epoch`] is a
//! digest of the whole starting configuration: every party's DB length,
//! ingest/queue frontier, modifications tail and common config. It is a
//! deterministic function of the exchanged [`SyncState`]s, so all parties derive
//! it with no extra round trip and no leader. A peer that restarts and recomputes
//! the same epoch is provably resuming the same startup and may rejoin; one that
//! comes back with different facts derives a different epoch, which is exactly
//! the signal that the initial sync is void and the whole fleet must restart.
//!
//! [`SyncState`] carries no party id and peers are polled in an arbitrary order,
//! so the epoch digests the *sorted multiset* of per-party fact digests rather
//! than a party-indexed vector. Duplicates are preserved, so a party's facts
//! changing from "same as peers" to "different" still moves the epoch.
//!
//! # The surviving peers are the durable record
//!
//! There is deliberately no persisted epoch table. The authority on which startup
//! is in progress is the set of peers still running it, published on
//! [`STARTUP_STATE_ROUTE`] from their live [`StartupStateHandle`]s. A restarting
//! party rebuilds its [`SyncState`] from its DB and re-derives the epoch from its
//! peers' boot snapshots; it reproduces the in-flight epoch **exactly when its own
//! facts are unchanged**, so the rejoin condition checks itself — no durable
//! state, no migration, no staleness guard. If every party restarts at once there
//! is no epoch to rejoin and all three derive a fresh one, which is the correct
//! outcome rather than a degradation.
//!
//! # What rejoin covers
//!
//! Rejoin works whenever the restarting party's facts are unchanged, which covers
//! the whole [`Phase::Load`] window — the long one, and the reason any of this
//! exists — provided [`Phase::Converge`] had no work to do, the normal case for a
//! healthy fleet.
//!
//! It does not cover a restart after a converge that actually mutated local
//! state: such a party recomputes different facts and the fleet restarts, exactly
//! as it does today unconditionally. So the crash-recovery boot, the one where
//! converge has real work, is also the one that cannot rejoin. Same for
//! `next_sns_sequence_num` with `db_backed_ingest` disabled, where it is a live
//! SQS queue head that the converge-phase trim moves.
//!
//! Replaying converge on a rejoin is safe because the facts projection was chosen
//! to *see* every converge-phase mutation, so "facts unchanged" already implies
//! "converge had no effect": `sync_modifications` shows up in `db_len` and the
//! modifications digest, `sync_graph_mutations` in that digest via
//! `graph_mutation_bytes`, the ingest frontier skip-ahead in
//! `max_persisted_sequence_number`, the queue trim in `next_sns_sequence_num`. The
//! one exception, releasing unpersisted ingest claims, touches only rows above the
//! frontier and is idempotent by construction. Anything added to converge later
//! must be fact-visible or idempotent, or a party could rejoin having silently
//! applied it twice.
//!
//! [`wait_for_others_ready`]: ampc_server_utils::wait_for_others_ready

use ampc_server_utils::shutdown_handler::ShutdownHandler;
use ampc_server_utils::{get_check_addresses, ServerCoordinationConfig};
use axum::routing::get;
use axum::Router;
use chrono::{DateTime, Utc};
use eyre::{bail, Result};
use iris_mpc_common::config::CommonConfig;
use iris_mpc_common::helpers::sha256::sha256_bytes;
use iris_mpc_common::helpers::sync::{Modification, SyncState};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use sodiumoxide::hex;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

/// Axum path serving the live startup-state document.
pub const STARTUP_STATE_ROUTE: &str = "/startup-state";

/// The same route without its leading slash, which is the form
/// `get_check_addresses` wants when building peer URLs.
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
/// answers 200 unconditionally and `/ready` only flips after the load completes.
/// Ordered by progress: a node only ever moves forward within one epoch.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    derive_more::FromStr,
    derive_more::Display,
)]
#[display(rename_all = "lowercase")]
#[from_str(rename_all = "lowercase")]
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
    /// The DB load. NOT peer-independent: `init_hawk_actor` runs
    /// `restart_from_checkpoint`, a cross-party consensus round with a 10s
    /// `PEER_ROUND_TIMEOUT`, concurrently with the iris load. A peer that vanishes
    /// here fails its peers inside the load, before the epoch layer gets to decide
    /// anything — so rejoin does not yet cover this window.
    Load,
    /// Ready, heartbeat armed, main loop running.
    Serving,
}

#[cfg(test)]
impl Phase {
    const ALL: &'static [Phase] = &[
        Phase::Discover,
        Phase::Propose,
        Phase::Commit,
        Phase::Converge,
        Phase::Load,
        Phase::Serving,
    ];
}

/// A 32-byte SHA-256 digest, carried as a lowercase hex string in JSON so the
/// document stays greppable in logs and readable with `curl`.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    derive_more::Debug,
    derive_more::Display,
)]
#[serde_as]
#[display("{}", hex::encode(_0))] // _0 accesses the inner [u8; 32]
#[debug("{self}")] // Delegates Debug directly to Display
pub struct Digest(#[serde_as(as = "Hex")] [u8; 32]);

impl Digest {
    fn of(bytes: &[u8]) -> Self {
        Digest(sha256_bytes(bytes))
    }

    /// Short form for log lines, where the full 64 hex chars are noise.
    pub fn short(&self) -> String {
        hex::encode(&self.0[..6])
    }
}

/// The epoch id: a digest over the sorted multiset of all parties' fact
/// digests. Distinct from [`Digest`] only in the type system, to keep a facts
/// digest from being passed where an epoch is expected.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    derive_more::Display,
)]
#[display("{}", _0)]
pub struct Epoch(Digest);

impl Epoch {
    pub fn short(&self) -> String {
        self.0.short()
    }
}

/// Canonical digest of one party's startup facts: everything about its local
/// state that the startup sync reads, and that must therefore be identical
/// across a restart for a rejoin to be safe.
///
/// A deliberate projection of [`SyncState`], not the whole struct:
/// `graph_mutation_bytes` is folded into the modifications digest (it is
/// parallel-by-index to `modifications` and can be megabytes), and nothing is
/// included that is not part of the startup decision.
///
/// Encoded by hand rather than via `serde_json` so the wire format of
/// [`SyncState`] can evolve (field renames, `serde(default)` additions) without
/// silently changing every epoch. Every variable-length field is length-prefixed,
/// so no two distinct fact sets can encode to the same bytes by shifting a
/// boundary.
pub fn facts_digest(state: &SyncState) -> Digest {
    let modifications = digest_modifications(&state.modifications, &state.graph_mutation_bytes);

    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(FACTS_DOMAIN);
    buf.extend_from_slice(&state.db_len.to_be_bytes());
    push_opt_u128(&mut buf, state.next_sns_sequence_num);
    push_opt_str(&mut buf, state.max_persisted_sequence_number.as_deref());
    buf.extend_from_slice(&digest_common_config(&state.common_config).0);
    buf.extend_from_slice(&modifications.0);
    Digest::of(&buf)
}

/// Derive this startup's epoch from all parties' states, including this party's
/// own.
///
/// `states` is expected to be [`SyncResult::all_states`], which is
/// `[my_state] ++ peers`.
///
/// [`SyncResult::all_states`]: iris_mpc_common::helpers::sync::SyncResult
pub fn derive_epoch(states: &[SyncState]) -> Result<Epoch> {
    if states.len() < 2 {
        bail!(
            "cannot derive a startup epoch from {} state(s): it is only meaningful over the whole \
             fleet",
            states.len()
        );
    }

    let mut facts: Vec<Digest> = states.iter().map(facts_digest).collect();
    // Sort so the epoch is independent of the order peers were polled in.
    facts.sort_unstable();

    let mut buf = Vec::with_capacity(EPOCH_DOMAIN.len() + 8 + facts.len() * 32);
    buf.extend_from_slice(EPOCH_DOMAIN);
    // Party count is part of the preimage: a two-party and a three-party fleet
    // that happen to share a digest prefix must not collide.
    buf.extend_from_slice(&(facts.len() as u64).to_be_bytes());
    for digest in &facts {
        buf.extend_from_slice(&digest.0);
    }
    Ok(Epoch(Digest::of(&buf)))
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
    /// This incarnation's coordination UUID, for correlation with `/health` and
    /// the heartbeat; deliberately *not* what peers key startup decisions on.
    ///
    /// `None` for the brief window before the coordination server is constructed,
    /// since the UUID is minted inside it while this route has to be handed to it.
    pub uuid: Option<String>,
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
    /// Test hook from `Config::startup_hold_at_phase`; see
    /// [`StartupStateHandle::new`].
    hold_at: Option<Phase>,
}

impl StartupStateHandle {
    /// Create the handle in [`Phase::Discover`].
    ///
    /// Must be called before the coordination server starts, since its router is
    /// passed in as an extra route; the UUID it mints arrives afterwards via
    /// [`StartupStateHandle::set_uuid`].
    ///
    /// `hold_at` is the test hook from `Config::startup_hold_at_phase`, `None` in
    /// every production path: the party parks forever on entering that phase,
    /// *after* publishing it, so peers keep seeing a node legitimately sitting
    /// there — which is precisely the state an e2e test wants to kill.
    pub fn new(party_id: usize, facts: Digest, hold_at: Option<Phase>) -> Self {
        StartupStateHandle {
            state: Arc::new(RwLock::new(StartupState {
                party_id,
                uuid: None,
                phase: Phase::Discover,
                phase_started_at: Utc::now(),
                facts,
                epoch: None,
            })),
            hold_at,
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

    /// Advance to `phase`, logging the transition and the time the previous phase
    /// took.
    pub async fn enter(&self, phase: Phase) {
        let now = Utc::now();
        let mut state = self.state.write().await;
        let previous = state.phase;
        let elapsed = now.signed_duration_since(state.phase_started_at);

        state.phase = phase;
        state.phase_started_at = now;
        drop(state);

        tracing::info!(
            "Startup phase {} -> {} (previous phase took {}s)",
            previous,
            phase,
            elapsed.num_seconds()
        );
        metrics::counter!("startup_phase_entered", "phase" => phase.to_string()).increment(1);

        if self.hold_at == Some(phase) {
            tracing::warn!(
                "startup_hold_at_phase={phase}: parking in this phase indefinitely. This is a \
                 test hook — the node will not start."
            );
            std::future::pending::<()>().await;
        }
    }

    /// Record the derived epoch. Called once, on entering [`Phase::Commit`].
    pub async fn set_epoch(&self, epoch: Epoch) {
        self.state.write().await.epoch = Some(epoch);
    }
}

/// Fetch and parse every peer's startup-state document, once.
///
/// Deliberately does **not** use `try_get_endpoint_other_nodes`: that helper
/// `tokio::spawn`s a retry task per peer and only aborts them along its own
/// timeout path, so bounding it against a caller's deadline with an outer
/// `tokio::time::timeout` leaks two tasks per polling round that keep hammering a
/// down peer for the next five minutes. This is a single attempt with nothing
/// spawned; the callers' loops supply the retry and can therefore stop it.
///
/// Returns what it managed to read plus a description of each peer it could not,
/// rather than failing on the first unreachable one. A down peer must not blind us
/// to the other: the reachable peer's UUID still needs recording (that is what
/// lets a *restarting* peer past its visibility barrier), and its epoch is still
/// worth checking for disagreement. Callers that need every peer — the commit
/// barrier — check `failures` themselves.
async fn poll_peer_states(
    config: &ServerCoordinationConfig,
    request_timeout: Duration,
) -> (Vec<StartupState>, Vec<String>) {
    let urls = get_check_addresses(
        &config.node_hostnames,
        &config.healthcheck_ports,
        STARTUP_STATE_ENDPOINT,
    );
    // Client-level timeout covers headers and body together, so no call site has
    // to wrap its own.
    let client = reqwest::Client::builder()
        .timeout(request_timeout)
        .build()
        .expect("reqwest client build failed");
    let mut states = Vec::with_capacity(urls.len().saturating_sub(1));
    let mut failures = Vec::new();

    for (party_id, url) in urls.iter().enumerate() {
        if party_id == config.party_id {
            continue;
        }
        match fetch_peer_state(&client, url).await {
            Ok(state) => states.push(state),
            Err(failure) => failures.push(failure),
        }
    }

    (states, failures)
}

/// One peer, one attempt. `Err` carries a ready-to-log description.
///
/// A document we cannot parse is a failure, never a skipped entry: a caller
/// counting agreeing peers must not reach quorum because a disagreeing peer was
/// quietly dropped.
async fn fetch_peer_state(
    client: &reqwest::Client,
    url: &str,
) -> std::result::Result<StartupState, String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("GET {url} failed: {err}"))?;
    let body = response
        .bytes()
        .await
        .map_err(|err| format!("reading body of {url} failed: {err}"))?;

    serde_json::from_slice(&body)
        .map_err(|err| format!("unparseable startup state from {url}: {err}"))
}

/// Record a peer incarnation as seen, so the coordination server advertises it on
/// `/health` and the `ampc-server-utils` checks downstream accept it.
///
/// Bookkeeping, not authorization — which lives entirely in the epoch comparison.
/// `wait_until_startup_visibility_is_complete` needs this from us before a
/// restarted peer can get far enough to publish an epoch at all, so withholding it
/// until the epoch matched would deadlock: the peer cannot pass its visibility
/// barrier until we list its UUID, and cannot publish an epoch until it does.
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
/// Shared by the commit barrier and the startup watch so the two cannot drift on
/// what counts as disagreement. Fails closed both ways: [`EpochVerdict::Void`]
/// beats [`EpochVerdict::Pending`], and an empty peer list is `Pending` rather
/// than a vacuous `Committed`.
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
/// A mismatch is *not* "the parties' data diverged" — divergence is normal, and is
/// what the modifications roll-forward exists to repair. The epoch digests the
/// whole starting configuration, differences included, so parties that legitimately
/// disagree about their persisted frontier still derive the same epoch. A mismatch
/// means the parties saw *different inputs*: some party's facts changed between the
/// exchange and now, in practice because it restarted with different data. The
/// agreed plan is void, and the error returned here triggers the full-fleet restart
/// that makes every party re-derive from current state.
pub async fn wait_for_epoch_commit(
    config: &ServerCoordinationConfig,
    verified_peers: &Arc<Mutex<HashSet<String>>>,
    my_epoch: Epoch,
) -> Result<()> {
    tracing::info!("Waiting for peers to commit startup epoch {}", my_epoch);

    let budget = Duration::from_secs(config.startup_sync_timeout_secs);
    let retry_delay = Duration::from_millis(config.http_query_retry_delay_ms);
    let request_timeout = Duration::from_millis(config.http_query_timeout_ms);
    let deadline = Instant::now() + budget;
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            metrics::counter!("startup_epoch_commit_timeout").increment(1);
            bail!(
                "peers did not commit startup epoch {} within {:?}",
                my_epoch,
                budget
            );
        }

        let (peers, failures) = poll_peer_states(config, request_timeout.min(remaining)).await;

        // Record before deciding: a peer that is still coming back needs its new
        // UUID published by us before it can pass its own visibility barrier.
        record_peer_uuids(verified_peers, &peers).await;

        if !failures.is_empty() {
            // Routine — a peer may be restarting. The barrier requires every peer,
            // so wait rather than classifying a partial view.
            log_poll_failure(attempt, &failures);
            tokio::time::sleep(retry_delay.min(remaining)).await;
            continue;
        }

        match classify_peers(my_epoch, &peers) {
            EpochVerdict::Void(disagreeing) => {
                metrics::counter!("startup_epoch_mismatch").increment(1);
                bail!(
                    "startup epoch mismatch: mine is {}, peers report {:?}. The agreed plan is \
                     void; every party must restart and re-derive from current state.",
                    my_epoch,
                    disagreeing
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

/// Log polling failures at WARN on the first attempt and every tenth after that,
/// DEBUG otherwise.
///
/// Mirrors `wait_for_others_unready`'s cadence: a recoverable peer outage polled
/// every second or two must not flood the log pipeline for the whole outage.
fn log_poll_failure(attempt: u32, failures: &[String]) {
    if attempt == 1 || attempt.is_multiple_of(10) {
        tracing::warn!("Could not poll peer startup state (attempt {attempt}): {failures:?}");
    } else {
        tracing::debug!("Could not poll peer startup state (attempt {attempt}): {failures:?}");
    }
}

/// Guard for the background task that watches peers across the converge and load
/// phases. Aborts the task when dropped.
pub struct StartupWatch {
    task: JoinHandle<()>,
    /// Set once, to the reason the epoch was voided.
    verdict: Arc<OnceLock<String>>,
}

impl StartupWatch {
    /// Fail if the watch observed a peer on a different epoch.
    ///
    /// Call at every point where startup can still be abandoned cheaply. In
    /// particular, call it after the DB load: `trigger_manual_shutdown` is
    /// cooperative, and the load does not poll it, so a verdict reached while
    /// loading is only acted on once the load returns.
    pub fn check(&self) -> Result<()> {
        match self.verdict.get() {
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

/// Watch peers for the duration of converge and load — the window in which no
/// *coordination* check is looking.
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
pub fn spawn_startup_watch(
    config: ServerCoordinationConfig,
    verified_peers: Arc<Mutex<HashSet<String>>>,
    shutdown_handler: Arc<ShutdownHandler>,
    epoch: Epoch,
) -> StartupWatch {
    let verdict = Arc::new(OnceLock::new());

    let task = tokio::spawn({
        let verdict = Arc::clone(&verdict);
        // Reuse the heartbeat cadence: this is the same job, over the phase of
        // startup where the heartbeat itself cannot yet run.
        let poll_interval = Duration::from_secs(config.heartbeat_interval_secs);
        let request_timeout = Duration::from_millis(config.http_query_timeout_ms);

        async move {
            let mut failures: u32 = 0;

            loop {
                tokio::time::sleep(poll_interval).await;

                if shutdown_handler.is_shutting_down() {
                    tracing::info!("Startup watch stopping: shutdown already in progress");
                    return;
                }

                let (peers, poll_failures) = poll_peer_states(&config, request_timeout).await;

                if poll_failures.is_empty() {
                    failures = 0;
                } else {
                    // Expected while a peer restarts — the case this watch exists
                    // to absorb — so it is rate-limited rather than warned on every
                    // round. The ready barrier and the heartbeat bound the terminal
                    // cases. Unlike the commit barrier we carry on with a partial
                    // view: a reachable peer on the wrong epoch is still a verdict.
                    failures += 1;
                    log_poll_failure(failures, &poll_failures);
                }

                record_peer_uuids(&verified_peers, &peers).await;

                if let EpochVerdict::Void(disagreeing) = classify_peers(epoch, &peers) {
                    let reason = format!(
                        "startup watch found a peer on a different epoch: mine is {epoch}, peers \
                         report {disagreeing:?}. The agreed plan is void; abandoning startup so \
                         every party re-derives from current state."
                    );
                    tracing::error!("{}", reason);
                    metrics::counter!("startup_epoch_mismatch").increment(1);

                    let _ = verdict.set(reason);

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

    fn epoch_of(states: &[SyncState]) -> Epoch {
        derive_epoch(states).unwrap()
    }

    #[test]
    fn epoch_is_independent_of_collection_order() {
        // The whole design rests on this: each party polls its peers in its own
        // order, yet all must derive the same epoch with no extra round trip.
        let baseline = epoch_of(&fleet());

        let mut rotated = fleet();
        rotated.rotate_left(1);
        assert_eq!(baseline, epoch_of(&rotated));

        let mut reversed = fleet();
        reversed.reverse();
        assert_eq!(baseline, epoch_of(&reversed));
    }

    #[test]
    fn epoch_changes_when_any_party_fact_changes() {
        let baseline = epoch_of(&fleet());

        // The motivating case: a peer restarts and comes back with a different
        // persisted frontier. That must void the epoch.
        let mut changed = fleet();
        changed[2].max_persisted_sequence_number = Some("0000000000000011".to_string());
        assert_ne!(
            baseline,
            epoch_of(&changed),
            "a changed persisted frontier must change the epoch"
        );

        let mut changed = fleet();
        changed[0].db_len += 1;
        assert_ne!(baseline, epoch_of(&changed));

        let mut changed = fleet();
        changed[1].modifications[0].status = "COMPLETED_WITH_ERROR".to_string();
        assert_ne!(baseline, epoch_of(&changed));

        let mut changed = fleet();
        changed[1].modifications[0].persisted = false;
        assert_ne!(baseline, epoch_of(&changed));

        let mut changed = fleet();
        changed[1].next_sns_sequence_num = None;
        assert_ne!(baseline, epoch_of(&changed));

        let mut changed = fleet();
        changed[0].graph_mutation_bytes = vec![Some(vec![0xcd; 16]), None];
        assert_ne!(baseline, epoch_of(&changed));
    }

    #[test]
    fn duplicate_facts_are_kept_as_a_multiset() {
        // Two parties in sync and one behind must not hash the same as three
        // parties in sync: collapsing duplicates would erase the difference.
        let all_agree = vec![
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
        ];
        assert_ne!(epoch_of(&fleet()), epoch_of(&all_agree));
    }

    #[test]
    fn party_count_is_part_of_the_epoch() {
        let states = fleet();
        assert_ne!(epoch_of(&states), epoch_of(&states[..2]));
    }

    #[test]
    fn a_single_state_cannot_derive_an_epoch() {
        assert!(derive_epoch(&fleet()[..1]).is_err());
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

        assert_ne!(facts_digest(&a), facts_digest(&b));
    }

    #[test]
    fn modification_order_does_not_affect_the_digest() {
        // `last_modifications` order is not a contract; the pairing with
        // `graph_mutation_bytes` is.
        let forward = state(100, None);
        let mut reversed = forward.clone();
        reversed.modifications.reverse();
        reversed.graph_mutation_bytes.reverse();

        assert_eq!(facts_digest(&forward), facts_digest(&reversed));
    }

    #[test]
    fn mispaired_graph_mutations_change_the_digest() {
        // Reversing the modifications without reversing their WAL entries
        // re-pairs them, which is a real divergence and must be visible.
        let forward = state(100, None);
        let mut mispaired = forward.clone();
        mispaired.modifications.reverse();

        assert_ne!(facts_digest(&forward), facts_digest(&mispaired));
    }

    #[test]
    fn epoch_survives_a_json_round_trip() {
        // Peers parse the epoch out of each other's documents, so the hex encoding
        // has to be exact.
        let epoch = epoch_of(&fleet());
        let json = serde_json::to_string(&epoch).unwrap();
        assert_eq!(serde_json::from_str::<Epoch>(&json).unwrap(), epoch);
    }

    #[test]
    fn phase_names_round_trip() {
        // `startup_hold_at_phase` is matched against `as_str`, so the two
        // directions must not drift apart.
        for &phase in Phase::ALL {
            assert_eq!(phase.to_string().parse::<Phase>().unwrap(), phase);
        }
        assert!("laod".parse::<Phase>().is_err());
    }

    #[tokio::test]
    async fn a_hold_publishes_its_phase_before_parking() {
        // The whole point of the hook: peers must see a party legitimately
        // sitting in the held phase, so `enter` has to publish before it parks.
        let facts = facts_digest(&state(100, None));
        let handle = StartupStateHandle::new(1, facts, Some(Phase::Propose));

        let entering = tokio::spawn({
            let handle = handle.clone();
            async move { handle.enter(Phase::Propose).await }
        });

        tokio::time::timeout(Duration::from_secs(5), async {
            while handle.snapshot().await.phase != Phase::Propose {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("held phase never became visible");

        assert!(!entering.is_finished(), "enter returned instead of parking");
        entering.abort();

        // A phase that is not the held one still advances normally.
        let handle = StartupStateHandle::new(1, facts, Some(Phase::Load));
        handle.enter(Phase::Propose).await;
        assert_eq!(handle.snapshot().await.phase, Phase::Propose);
    }

    #[tokio::test]
    async fn published_document_tracks_phase_and_epoch() {
        let facts = facts_digest(&state(100, None));
        let handle = StartupStateHandle::new(1, facts, None);

        let initial = handle.snapshot().await;
        assert_eq!(initial.phase, Phase::Discover);
        assert!(initial.epoch.is_none());
        assert!(initial.uuid.is_none());
        assert_eq!(initial.facts, facts);

        handle.set_uuid("uuid-1".to_string()).await;

        let epoch = epoch_of(&fleet());
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
        assert_eq!(parsed.uuid.as_deref(), Some("uuid-1"));
    }

    fn peer(party_id: usize, phase: Phase, epoch: Option<Epoch>) -> StartupState {
        StartupState {
            party_id,
            uuid: Some(format!("uuid-{party_id}")),
            phase,
            phase_started_at: Utc::now(),
            facts: facts_digest(&state(100, None)),
            epoch,
        }
    }

    fn other_epoch() -> Epoch {
        let mut changed = fleet();
        changed[0].db_len = 12_345;
        epoch_of(&changed)
    }

    #[test]
    fn all_peers_on_my_epoch_is_committed() {
        let mine = epoch_of(&fleet());
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
        let mine = epoch_of(&fleet());
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
        let mine = epoch_of(&fleet());
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
        let mine = epoch_of(&fleet());
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
        let mine = epoch_of(&fleet());
        assert_eq!(classify_peers(mine, &[]), EpochVerdict::Pending(Vec::new()));
    }

    #[test]
    fn rejoin_condition_is_self_checking() {
        // The core claim of the peer-memory design. A restarting party rebuilds
        // its own state from its DB and re-derives the epoch from its peers'
        // (unchanged) boot snapshots.
        let original = fleet();
        let in_flight = epoch_of(&original);

        // Facts unchanged across the restart — converge had no work to do — so
        // the party reproduces the in-flight epoch and rejoins.
        let mut rebooted = original.clone();
        rebooted[0] = state(100, Some("0000000000000010"));
        assert_eq!(
            in_flight,
            epoch_of(&rebooted),
            "an unchanged party must reproduce the in-flight epoch"
        );

        // Facts changed across the restart — converge mutated local state, or the
        // data really did move — so the epoch differs and the fleet restarts.
        let mut rebooted = original.clone();
        rebooted[0] = state(101, Some("0000000000000011"));
        assert_ne!(
            in_flight,
            epoch_of(&rebooted),
            "a changed party must not be able to rejoin"
        );
    }

    #[test]
    fn phases_are_ordered_by_progress() {
        assert!(Phase::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
