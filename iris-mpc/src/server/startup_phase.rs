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
//! # Current status: observation only
//!
//! Nothing here fails a startup yet. [`StartupStateHandle`] publishes the live
//! document on [`STARTUP_STATE_ROUTE`] and [`observe_peer_epochs`] logs whether
//! the parties agree, so epoch determinism can be confirmed on a real cluster
//! before any teardown decision depends on it. Enforcement (the commit barrier
//! and the rejoin acceptance path) comes next, and needs the durable per-party
//! generation counter that [`StartupState::generation`] is reserved for.
//!
//! [`wait_for_others_ready`]: ampc_server_utils::wait_for_others_ready

use ampc_server_utils::{try_get_endpoint_other_nodes, ServerCoordinationConfig};
use axum::routing::get;
use axum::Router;
use chrono::{DateTime, Utc};
use eyre::{bail, Result};
use iris_mpc_common::config::CommonConfig;
use iris_mpc_common::helpers::sha256::sha256_bytes;
use iris_mpc_common::helpers::sync::{Modification, SyncState};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sodiumoxide::hex;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

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
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s)
            .map_err(|_| D::Error::custom(format!("digest is not valid hex: {s}")))?;
        let bytes: [u8; 32] = bytes.try_into().map_err(|_| {
            D::Error::custom(format!("digest must be 32 bytes, got {} hex chars", s.len()))
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
    /// Durable, monotonic per party: bumped whenever a party abandons an epoch,
    /// so a stale document from a previous attempt cannot be mistaken for the
    /// current one (the ABA guard). Always 0 until the epoch record is
    /// persisted, which enforcement will add.
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

/// Poll peers' startup-state documents and report whether they derived the same
/// epoch.
///
/// Observation only: every outcome — disagreement, unreachable peer, timeout —
/// is logged and counted, never returned as an error. The point is to confirm
/// on a live cluster that the canonicalization really is deterministic across
/// parties before anything is allowed to fail on it.
pub async fn observe_peer_epochs(
    config: &ServerCoordinationConfig,
    my_epoch: Epoch,
    budget: Duration,
    retry_delay: Duration,
) {
    let deadline = Instant::now() + budget;

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            tracing::warn!(
                "Could not confirm peer startup epochs within {:?}; proceeding (observation only)",
                budget
            );
            metrics::counter!("startup_epoch_observation_timeout").increment(1);
            return;
        }

        // `try_get_endpoint_other_nodes` retries internally against its own
        // `startup_sync_timeout_secs` budget, which is unrelated to ours, so
        // cap it here rather than inheriting it.
        let responses = match tokio::time::timeout(
            remaining,
            try_get_endpoint_other_nodes(config, STARTUP_STATE_ENDPOINT),
        )
        .await
        {
            Ok(Ok(responses)) => responses,
            Ok(Err(err)) => {
                tracing::warn!("Failed to poll peer startup state: {:?}", err);
                tokio::time::sleep(retry_delay.min(remaining)).await;
                continue;
            }
            Err(_) => continue, // deadline hit; handled at the top of the loop
        };

        let mut peers = Vec::with_capacity(responses.len());
        for (status, body) in responses {
            match serde_json::from_slice::<StartupState>(&body) {
                Ok(state) => peers.push(state),
                Err(err) => {
                    tracing::warn!(
                        "Failed to deserialize peer startup state (status {}): {:?}",
                        status,
                        err
                    );
                }
            }
        }

        // A peer that has not reached Commit yet has no epoch to compare;
        // that is expected, since the three parties get here independently.
        let pending: Vec<_> = peers
            .iter()
            .filter(|peer| peer.epoch.is_none())
            .map(|peer| (peer.party_id, peer.phase))
            .collect();

        if !pending.is_empty() {
            tracing::debug!("Peers have not derived an epoch yet: {:?}", pending);
            let remaining = deadline.saturating_duration_since(Instant::now());
            tokio::time::sleep(retry_delay.min(remaining)).await;
            continue;
        }

        let disagreeing: Vec<_> = peers
            .iter()
            .filter(|peer| peer.epoch != Some(my_epoch))
            .map(|peer| {
                (
                    peer.party_id,
                    peer.epoch.map(|e| e.to_string()),
                    peer.facts.short(),
                )
            })
            .collect();

        if disagreeing.is_empty() {
            tracing::info!(
                "Startup epoch {} agreed by all {} parties",
                my_epoch,
                peers.len() + 1
            );
            metrics::counter!("startup_epoch_agreement").increment(1);
        } else {
            // Under enforcement this is the "initial sync is void, restart the
            // fleet" trigger. For now it is loud but harmless.
            tracing::error!(
                "Startup epoch MISMATCH: mine is {}, peers report {:?}. Under enforcement this \
                 would void the epoch and restart the fleet.",
                my_epoch,
                disagreeing
            );
            metrics::counter!("startup_epoch_mismatch").increment(1);
        }
        return;
    }
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

    #[test]
    fn phases_are_ordered_by_progress() {
        assert!(Phase::Discover < Phase::Propose);
        assert!(Phase::Propose < Phase::Commit);
        assert!(Phase::Commit < Phase::Converge);
        assert!(Phase::Converge < Phase::Load);
        assert!(Phase::Load < Phase::Serving);
    }
}
