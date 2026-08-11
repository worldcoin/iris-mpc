//! Startup phase tracking and the data-derived [`SyncStateDigest`] that keys it.
//!
//! Peer coordination during startup is keyed on a random per-boot UUID: the
//! heartbeat's first-contact check and `wait_for_others_ready` reject any peer
//! whose UUID is outside the startup-verified set. That is the right policy once
//! a node is serving — MPC session state shared with a restarted peer is
//! unrecoverable — but during startup it forces every party to restart and re-run
//! the multi-minute DB load whenever one peer bounces, even with the data
//! unchanged.
//!
//! So startup coordination is keyed on the *data* instead. A fleet-wide
//! [`SyncStateDigest`] covers the whole starting configuration: every party's DB
//! length, ingest/queue frontier, modifications tail and common config. It is a
//! deterministic function of the exchanged [`SyncState`]s, so all parties derive
//! it with no extra round trip and no leader. A peer that restarts and recomputes
//! the same fleet sync-state digest is provably resuming the same startup and may
//! rejoin; one that comes back with a different [`SyncState`] computes a different
//! one, which is exactly the signal that the initial sync is void and the whole
//! fleet must restart.

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
const SYNC_STATE_DOMAIN: &[u8] = b"iris-mpc/startup-sync-state/v1";
const MODIFICATIONS_DOMAIN: &[u8] = b"iris-mpc/startup-modifications/v1";
const FLEET_DOMAIN: &[u8] = b"iris-mpc/startup-fleet-sync-state/v1";

/// Where a node is in its startup sequence.
///
/// Published to peers so they can tell "still loading" from "dead" — a
/// distinction the coordination server cannot express, since `/health`
/// answers 200 unconditionally and `/ready` only flips after the load completes.
/// Ordered by progress: a node only ever moves forward within one startup.
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
    /// Mutual visibility established; exchanging [`SyncState`]s.
    Propose,
    /// States exchanged and the fleet sync-state digest derived from them.
    Commit,
    /// Applying the synchronized [`SyncState`]s to local storage (modifications
    /// roll-forward, graph WAL, ingest frontier skip-ahead, queue trim).
    Converge,
    /// The DB load. NOT peer-independent: `init_hawk_actor` runs
    /// `restart_from_checkpoint`, a cross-party consensus round with a 10s
    /// `PEER_ROUND_TIMEOUT`, concurrently with the iris load. A peer that vanishes
    /// here fails its peers inside the load, before the fleet sync-state digest
    /// comparison gets to decide anything — so rejoin does not yet cover this window.
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

/// A 32-byte SHA-256 digest over one or more [`SyncState`]s: over a single
/// party's, from [`hash_sync_state`], or over the whole fleet's, from
/// [`hash_fleet_sync_states`].
///
/// Carried as a lowercase hex string in JSON so the document stays greppable in
/// logs and readable with `curl`.
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
pub struct SyncStateDigest(#[serde_as(as = "Hex")] [u8; 32]);

impl SyncStateDigest {
    fn from_bytes(bytes: &[u8]) -> Self {
        SyncStateDigest(sha256_bytes(bytes))
    }

    /// Short form for log lines, where the full 64 hex chars are noise.
    pub fn short(&self) -> String {
        hex::encode(&self.0[..6])
    }
}

/// Canonical digest of one party's [`SyncState`]: everything about its local
/// state that the startup sync reads, and that must therefore be identical
/// across a restart for a rejoin to be safe.
pub fn hash_sync_state(state: &SyncState) -> SyncStateDigest {
    let mut buf = Vec::with_capacity(128);
    buf.extend_from_slice(SYNC_STATE_DOMAIN);
    buf.extend_from_slice(&state.db_len.to_be_bytes());
    push_opt_u128(&mut buf, state.next_sns_sequence_num);
    push_opt_str(&mut buf, state.max_persisted_sequence_number.as_deref());
    buf.extend_from_slice(&digest_common_config(&state.common_config));
    buf.extend_from_slice(&digest_modifications(
        &state.modifications,
        &state.graph_mutation_bytes,
    ));
    SyncStateDigest::from_bytes(&buf)
}

/// Hash all parties' states, including this party's own, into the one value
/// they all key this startup on.
pub fn hash_fleet_sync_states(states: &[SyncState]) -> Result<SyncStateDigest> {
    if states.len() < 2 {
        bail!(
            "cannot digest a fleet of {} state(s): the fleet sync-state digest is only meaningful \
             over every party",
            states.len()
        );
    }

    let mut digests: Vec<SyncStateDigest> = states.iter().map(hash_sync_state).collect();
    // Sort so the fleet sync-state digest is independent of the poll order.
    digests.sort_unstable();

    let mut buf = Vec::with_capacity(
        FLEET_DOMAIN.len()
            + std::mem::size_of::<u64>()
            + (digests.len() * std::mem::size_of::<SyncStateDigest>()),
    );
    buf.extend_from_slice(FLEET_DOMAIN);
    // Party count is part of the preimage: a two-party and a three-party fleet
    // that happen to share a digest prefix must not collide.
    buf.extend_from_slice(&(digests.len() as u64).to_be_bytes());
    for digest in &digests {
        buf.extend_from_slice(&digest.0);
    }
    Ok(SyncStateDigest::from_bytes(&buf))
}

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
    /// Digest of this party's own [`SyncState`], known from the moment that state
    /// is built — i.e. before any peer has been contacted.
    pub party_sync_state_digest: SyncStateDigest,
    /// Digest over every party's [`SyncState`], `None` until the state exchange
    /// completes and it can be derived.
    pub fleet_sync_state_digest: Option<SyncStateDigest>,
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
    /// `hold_at` is the test hook from `Config::startup_hold_at_phase`,
    pub fn new(
        party_id: usize,
        party_sync_state_digest: SyncStateDigest,
        hold_at: Option<Phase>,
    ) -> Self {
        StartupStateHandle {
            state: Arc::new(RwLock::new(StartupState {
                party_id,
                uuid: None,
                phase: Phase::Discover,
                phase_started_at: Utc::now(),
                party_sync_state_digest,
                fleet_sync_state_digest: None,
            })),
            hold_at,
        }
    }

    /// Router to hand to `start_coordination_server_with_extra_routes`.
    ///
    /// Always answers 200: the document is meaningful in every phase, and a
    /// pre-agreement node is described by `fleet_sync_state_digest: null` rather than by an
    /// error status. Peers distinguish the cases by reading the fields.
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

    /// Record the derived fleet sync-state digest
    pub async fn set_fleet_sync_state_digest(&self, digest: SyncStateDigest) {
        self.state.write().await.fleet_sync_state_digest = Some(digest);
    }
}

/// Fetch and parse every peer's startup-state.
/// Returns what it managed to read plus a description of each peer it could not.
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

async fn fetch_peer_state(
    client: &reqwest::Client,
    url: &str,
) -> std::result::Result<StartupState, String> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("GET {url} failed: {err}"))?;

    if !response.status().is_success() {
        return Err(format!("GET {url} returned {}", response.status()));
    }

    let body = response
        .bytes()
        .await
        .map_err(|err| format!("reading body of {url} failed: {err}"))?;

    serde_json::from_slice(&body)
        .map_err(|err| format!("unparseable startup state from {url}: {err}"))
}

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

/// What a round of peer observations says about the fleet sync-state digest.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ConsensusVerdict {
    /// Every peer published this fleet sync-state digest.
    Committed,
    /// One or more peers have not published a fleet sync-state digest yet: either
    /// they have not reached [`Phase::Commit`], or they restarted and are on their
    /// way back.
    Pending(Vec<(usize, Phase)>),
    /// A peer published a different fleet sync-state digest, so this startup is void.
    Void(Vec<(usize, Option<SyncStateDigest>)>),
}

/// Classify one round of peer documents against our own fleet sync-state digest.
///
/// Shared by the commit barrier and the startup watch so the two cannot drift on
/// what counts as disagreement. Fails closed both ways: [`ConsensusVerdict::Void`]
/// beats [`ConsensusVerdict::Pending`], and an empty peer list is `Pending` rather
/// than a vacuous `Committed`.
fn compare_peer_states(my_digest: SyncStateDigest, peers: &[StartupState]) -> ConsensusVerdict {
    let void: Vec<_> = peers
        .iter()
        .filter(|peer| {
            peer.fleet_sync_state_digest
                .is_some_and(|digest| digest != my_digest)
        })
        .map(|peer| (peer.party_id, peer.fleet_sync_state_digest))
        .collect();

    if !void.is_empty() {
        return ConsensusVerdict::Void(void);
    }

    if peers.is_empty() {
        return ConsensusVerdict::Pending(Vec::new());
    }

    let pending: Vec<_> = peers
        .iter()
        .filter(|peer| peer.fleet_sync_state_digest.is_none())
        .map(|peer| (peer.party_id, peer.phase))
        .collect();

    if pending.is_empty() {
        ConsensusVerdict::Committed
    } else {
        ConsensusVerdict::Pending(pending)
    }
}

/// Commit barrier: hold until every peer has published *this* fleet sync-state
/// digest.
///
/// Nothing may mutate local storage before this returns. Converging over states a
/// peer never agreed to is the failure mode the barrier exists to prevent.
///
/// A mismatch is *not* "the parties' data diverged" — divergence is normal, and is
/// what the modifications roll-forward exists to repair. The fleet sync-state digest
/// covers the whole starting configuration, differences included, so parties that
/// legitimately disagree about their persisted frontier still derive the same one.
/// A mismatch means the parties saw *different inputs*: some party's
/// [`SyncState`] changed between the exchange and now, in practice because it
/// restarted with different data. This startup is void, and the error returned here
/// triggers the full-fleet restart that makes every party re-derive from current
/// state.
pub async fn wait_for_fleet_sync_state_digest_commit(
    config: &ServerCoordinationConfig,
    verified_peers: &Arc<Mutex<HashSet<String>>>,
    my_digest: SyncStateDigest,
) -> Result<()> {
    tracing::info!(
        "Waiting for peers to commit startup fleet sync-state digest {}",
        my_digest
    );

    let budget = Duration::from_secs(config.startup_sync_timeout_secs);
    let retry_delay = Duration::from_millis(config.http_query_retry_delay_ms);
    let request_timeout = Duration::from_millis(config.http_query_timeout_ms);
    let deadline = Instant::now() + budget;
    let mut attempt: u32 = 0;

    loop {
        attempt += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            metrics::counter!("startup_fleet_sync_state_digest_commit_timeout").increment(1);
            bail!(
                "peers did not commit startup fleet sync-state digest {} within {:?}",
                my_digest,
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

        match compare_peer_states(my_digest, &peers) {
            ConsensusVerdict::Void(disagreeing) => {
                metrics::counter!("startup_fleet_sync_state_digest_mismatch").increment(1);
                bail!(
                    "startup fleet sync-state digest mismatch: mine is {}, peers report {:?}. \
                     This startup is void; every party must restart and re-derive from current \
                     state.",
                    my_digest,
                    disagreeing
                );
            }
            ConsensusVerdict::Committed => {
                tracing::info!(
                    "Startup fleet sync-state digest {} committed by all {} parties",
                    my_digest,
                    peers.len() + 1
                );
                metrics::counter!("startup_fleet_sync_state_digest_agreement").increment(1);
                return Ok(());
            }
            ConsensusVerdict::Pending(pending) => {
                tracing::debug!(
                    "Peers have not committed a fleet sync-state digest yet: {:?}",
                    pending
                );
                let remaining = deadline.saturating_duration_since(Instant::now());
                tokio::time::sleep(retry_delay.min(remaining)).await;
            }
        }
    }
}

fn log_poll_failure(attempt: u32, failures: &[String]) {
    if attempt == 1 || attempt.is_multiple_of(10) {
        tracing::warn!("Could not poll peer startup state (attempt {attempt}): {failures:?}");
    } else {
        tracing::debug!("Could not poll peer startup state (attempt {attempt}): {failures:?}");
    }
}

/// Guard for the background task that watches peers across the converge and load
/// phases. Aborts the task when dropped.
pub struct DesyncWatch {
    task: JoinHandle<()>,
    /// Set once, to the reason this startup was voided.
    verdict: Arc<OnceLock<String>>,
}

impl DesyncWatch {
    /// Fail if the watch observed a peer on a different fleet sync-state digest.
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

impl Drop for DesyncWatch {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Handles the following conditions:
/// - peer republishes this digest (whatever its UUID) — it is provably resuming the
///   same startup, so it is recorded as verified and both parties carry on.
/// - peer publishes a different digest — its data changed, this startup is void, and
///   we abandon it.
/// - peer publishes no digest, or is unreachable — it is mid-restart. Tolerated: it
///   will either republish this digest or a different one, and the ready barrier
///   still bounds how long a peer may fail to arrive.
pub fn spawn_desync_watch(
    config: ServerCoordinationConfig,
    verified_peers: Arc<Mutex<HashSet<String>>>,
    shutdown_handler: Arc<ShutdownHandler>,
    fleet_sync_state_digest: SyncStateDigest,
) -> DesyncWatch {
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
                    failures += 1;
                    log_poll_failure(failures, &poll_failures);
                }

                record_peer_uuids(&verified_peers, &peers).await;

                if let ConsensusVerdict::Void(disagreeing) =
                    compare_peer_states(fleet_sync_state_digest, &peers)
                {
                    let reason = format!(
                        "startup watch found a peer on a different fleet sync-state digest: mine \
                         is {fleet_sync_state_digest}, peers report {disagreeing:?}. This startup \
                         is void; abandoning it so every party re-derives from current state."
                    );
                    tracing::error!("{}", reason);
                    metrics::counter!("startup_fleet_sync_state_digest_mismatch").increment(1);

                    let _ = verdict.set(reason);

                    if !shutdown_handler.is_shutting_down() {
                        shutdown_handler.trigger_manual_shutdown();
                    }
                    return;
                }

                for peer in &peers {
                    tracing::debug!(
                        "Startup watch: party {} in phase {} (fleet sync-state digest {})",
                        peer.party_id,
                        peer.phase,
                        peer.fleet_sync_state_digest
                            .map(|digest| digest.short())
                            .unwrap_or_else(|| "pending".to_string())
                    );
                }
            }
        }
    });

    DesyncWatch { task, verdict }
}

/// SHA-256 of the common config, as bytes to fold into [`hash_sync_state`]'s
/// preimage.
fn digest_common_config(config: &CommonConfig) -> Vec<u8> {
    let bytes = serde_json::to_vec(config).expect("CommonConfig serialization to JSON failed");
    sha256_bytes(&bytes).to_vec()
}

/// SHA-256 of the modifications tail together with its graph WAL entries, as
/// bytes to fold into [`hash_sync_state`]'s preimage.
///
/// `graph_mutation_bytes` is parallel-by-index to `modifications`, so the two
/// are zipped before being sorted by modification id — canonicalizing the order
/// without breaking the pairing. Mutation blobs are folded in by digest rather
/// than by value: they are large, and only their identity matters here.
fn digest_modifications(
    modifications: &[Modification],
    graph_mutation_bytes: &[Option<Vec<u8>>],
) -> Vec<u8> {
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
    sha256_bytes(&buf).to_vec()
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

    fn fleet_sync_state_digest_of(states: &[SyncState]) -> SyncStateDigest {
        hash_fleet_sync_states(states).unwrap()
    }

    #[test]
    fn fleet_sync_state_digest_is_independent_of_collection_order() {
        // The whole design rests on this: each party polls its peers in its own
        // order, yet all must derive the same digest with no extra round trip.
        let baseline = fleet_sync_state_digest_of(&fleet());

        let mut rotated = fleet();
        rotated.rotate_left(1);
        assert_eq!(baseline, fleet_sync_state_digest_of(&rotated));

        let mut reversed = fleet();
        reversed.reverse();
        assert_eq!(baseline, fleet_sync_state_digest_of(&reversed));
    }

    #[test]
    fn fleet_sync_state_digest_changes_when_any_party_state_changes() {
        let baseline = fleet_sync_state_digest_of(&fleet());

        // The motivating case: a peer restarts and comes back with a different
        // persisted frontier. That must void this startup.
        let mut changed = fleet();
        changed[2].max_persisted_sequence_number = Some("0000000000000011".to_string());
        assert_ne!(
            baseline,
            fleet_sync_state_digest_of(&changed),
            "a changed persisted frontier must change the fleet sync-state digest"
        );

        let mut changed = fleet();
        changed[0].db_len += 1;
        assert_ne!(baseline, fleet_sync_state_digest_of(&changed));

        let mut changed = fleet();
        changed[1].modifications[0].status = "COMPLETED_WITH_ERROR".to_string();
        assert_ne!(baseline, fleet_sync_state_digest_of(&changed));

        let mut changed = fleet();
        changed[1].modifications[0].persisted = false;
        assert_ne!(baseline, fleet_sync_state_digest_of(&changed));

        let mut changed = fleet();
        changed[1].next_sns_sequence_num = None;
        assert_ne!(baseline, fleet_sync_state_digest_of(&changed));

        let mut changed = fleet();
        changed[0].graph_mutation_bytes = vec![Some(vec![0xcd; 16]), None];
        assert_ne!(baseline, fleet_sync_state_digest_of(&changed));
    }

    #[test]
    fn duplicate_party_digests_are_kept_as_a_multiset() {
        // Two parties in sync and one behind must not hash the same as three
        // parties in sync: collapsing duplicates would erase the difference.
        let all_agree = vec![
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
            state(100, Some("0000000000000010")),
        ];
        assert_ne!(
            fleet_sync_state_digest_of(&fleet()),
            fleet_sync_state_digest_of(&all_agree)
        );
    }

    #[test]
    fn party_count_is_part_of_the_fleet_sync_state_digest() {
        let states = fleet();
        assert_ne!(
            fleet_sync_state_digest_of(&states),
            fleet_sync_state_digest_of(&states[..2])
        );
    }

    #[test]
    fn a_single_state_cannot_make_a_fleet_sync_state_digest() {
        assert!(hash_fleet_sync_states(&fleet()[..1]).is_err());
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

        assert_ne!(hash_sync_state(&a), hash_sync_state(&b));
    }

    #[test]
    fn modification_order_does_not_affect_the_digest() {
        // `last_modifications` order is not a contract; the pairing with
        // `graph_mutation_bytes` is.
        let forward = state(100, None);
        let mut reversed = forward.clone();
        reversed.modifications.reverse();
        reversed.graph_mutation_bytes.reverse();

        assert_eq!(hash_sync_state(&forward), hash_sync_state(&reversed));
    }

    #[test]
    fn mispaired_graph_mutations_change_the_digest() {
        // Reversing the modifications without reversing their WAL entries
        // re-pairs them, which is a real divergence and must be visible.
        let forward = state(100, None);
        let mut mispaired = forward.clone();
        mispaired.modifications.reverse();

        assert_ne!(hash_sync_state(&forward), hash_sync_state(&mispaired));
    }

    #[test]
    fn a_digest_survives_a_json_round_trip() {
        // Peers parse the fleet sync-state digest out of each other's documents, so
        // the hex encoding has to be exact.
        let digest = fleet_sync_state_digest_of(&fleet());
        let json = serde_json::to_string(&digest).unwrap();
        assert_eq!(
            serde_json::from_str::<SyncStateDigest>(&json).unwrap(),
            digest
        );
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
        let digest = hash_sync_state(&state(100, None));
        let handle = StartupStateHandle::new(1, digest, Some(Phase::Propose));

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
        let handle = StartupStateHandle::new(1, digest, Some(Phase::Load));
        handle.enter(Phase::Propose).await;
        assert_eq!(handle.snapshot().await.phase, Phase::Propose);
    }

    #[tokio::test]
    async fn published_document_tracks_phase_and_fleet_sync_state_digest() {
        let digest = hash_sync_state(&state(100, None));
        let handle = StartupStateHandle::new(1, digest, None);

        let initial = handle.snapshot().await;
        assert_eq!(initial.phase, Phase::Discover);
        assert!(initial.fleet_sync_state_digest.is_none());
        assert!(initial.uuid.is_none());
        assert_eq!(initial.party_sync_state_digest, digest);

        handle.set_uuid("uuid-1".to_string()).await;

        let fleet_sync_state_digest = fleet_sync_state_digest_of(&fleet());
        handle.enter(Phase::Propose).await;
        handle.enter(Phase::Commit).await;
        handle
            .set_fleet_sync_state_digest(fleet_sync_state_digest)
            .await;

        let committed = handle.snapshot().await;
        assert_eq!(committed.phase, Phase::Commit);
        assert_eq!(
            committed.fleet_sync_state_digest,
            Some(fleet_sync_state_digest)
        );

        // The document must survive the wire: peers parse exactly this.
        let json = serde_json::to_string(&committed).unwrap();
        let parsed: StartupState = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.fleet_sync_state_digest,
            Some(fleet_sync_state_digest)
        );
        assert_eq!(parsed.phase, Phase::Commit);
        assert_eq!(parsed.party_id, 1);
        assert_eq!(parsed.uuid.as_deref(), Some("uuid-1"));
    }

    fn peer(
        party_id: usize,
        phase: Phase,
        fleet_sync_state_digest: Option<SyncStateDigest>,
    ) -> StartupState {
        StartupState {
            party_id,
            uuid: Some(format!("uuid-{party_id}")),
            phase,
            phase_started_at: Utc::now(),
            party_sync_state_digest: hash_sync_state(&state(100, None)),
            fleet_sync_state_digest,
        }
    }

    fn other_fleet_sync_state_digest() -> SyncStateDigest {
        let mut changed = fleet();
        changed[0].db_len = 12_345;
        fleet_sync_state_digest_of(&changed)
    }

    #[test]
    fn all_peers_on_my_fleet_sync_state_digest_is_committed() {
        let mine = fleet_sync_state_digest_of(&fleet());
        let peers = [
            peer(1, Phase::Commit, Some(mine)),
            peer(2, Phase::Load, Some(mine)),
        ];
        assert_eq!(
            compare_peer_states(mine, &peers),
            ConsensusVerdict::Committed
        );
    }

    #[test]
    fn a_peer_without_a_fleet_sync_state_digest_is_pending_not_void() {
        // Either it has not reached Commit, or it restarted and is on its way
        // back. Neither is a disagreement.
        let mine = fleet_sync_state_digest_of(&fleet());
        let peers = [
            peer(1, Phase::Commit, Some(mine)),
            peer(2, Phase::Discover, None),
        ];
        assert_eq!(
            compare_peer_states(mine, &peers),
            ConsensusVerdict::Pending(vec![(2, Phase::Discover)])
        );
    }

    #[test]
    fn a_peer_on_a_different_fleet_sync_state_digest_voids_the_startup() {
        // The motivating failure: a peer restarted with changed data.
        let mine = fleet_sync_state_digest_of(&fleet());
        let theirs = other_fleet_sync_state_digest();
        let peers = [
            peer(1, Phase::Commit, Some(mine)),
            peer(2, Phase::Commit, Some(theirs)),
        ];
        assert_eq!(
            compare_peer_states(mine, &peers),
            ConsensusVerdict::Void(vec![(2, Some(theirs))])
        );
    }

    #[test]
    fn void_takes_precedence_over_pending() {
        // Fail closed: one peer still arriving must not mask another peer that
        // has already proven this startup void.
        let mine = fleet_sync_state_digest_of(&fleet());
        let theirs = other_fleet_sync_state_digest();
        let peers = [
            peer(1, Phase::Discover, None),
            peer(2, Phase::Commit, Some(theirs)),
        ];
        assert_eq!(
            compare_peer_states(mine, &peers),
            ConsensusVerdict::Void(vec![(2, Some(theirs))])
        );
    }

    #[test]
    fn no_peers_is_pending_not_vacuously_committed() {
        // A round that saw nobody must never satisfy the commit barrier.
        let mine = fleet_sync_state_digest_of(&fleet());
        assert_eq!(
            compare_peer_states(mine, &[]),
            ConsensusVerdict::Pending(Vec::new())
        );
    }

    #[test]
    fn rejoin_condition_is_self_checking() {
        // The core claim of the peer-memory design. A restarting party rebuilds
        // its own state from its DB and re-derives the fleet sync-state digest from
        // its peers' (unchanged) boot snapshots.
        let original = fleet();
        let in_flight = fleet_sync_state_digest_of(&original);

        // State unchanged across the restart — converge had no work to do — so
        // the party reproduces the in-flight digest and rejoins.
        let mut rebooted = original.clone();
        rebooted[0] = state(100, Some("0000000000000010"));
        assert_eq!(
            in_flight,
            fleet_sync_state_digest_of(&rebooted),
            "an unchanged party must reproduce the in-flight digest"
        );

        // State changed across the restart — converge mutated local state, or the
        // data really did move — so the digest differs and the fleet restarts.
        let mut rebooted = original.clone();
        rebooted[0] = state(101, Some("0000000000000011"));
        assert_ne!(
            in_flight,
            fleet_sync_state_digest_of(&rebooted),
            "a changed party must not be able to rejoin"
        );
    }

    #[test]
    fn phases_are_ordered_by_progress() {
        assert!(Phase::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
