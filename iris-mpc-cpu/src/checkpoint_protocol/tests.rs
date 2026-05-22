//! Mocked-trait protocol tests for `run_cycle`. The three-party tests at
//! the bottom drive a real in-memory ring transport.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use iris_mpc_common::vector_id::VectorId;

use crate::checkpoint_protocol::{
    run_cycle, Blake3Hash, CheckpointMeta, ConsensusMessage, ConsensusTransport, CycleConfig,
    CycleError, FreezeHeight, Graph, GraphHasher, GraphMutationId, Materializer, MostRecentCommon,
    MutationStore, Outcome, SkipReason, StrictLatest, TerminalAction,
};
use crate::execution::hawk_main::BothEyes;
use crate::hnsw::{graph::mutation::GraphMutation, GraphMem};

// ── shared fixtures ──────────────────────────────────────────────────────

fn empty_graph() -> Graph {
    [GraphMem::<VectorId>::new(), GraphMem::<VectorId>::new()]
}

fn cfg(min: u64) -> CycleConfig {
    CycleConfig {
        min_mutations_to_apply: min,
        peer_round_timeout: Duration::from_millis(100),
        checkpoint_window: 10,
    }
}

fn meta(id: i64, mut_id: Option<i64>) -> CheckpointMeta {
    CheckpointMeta {
        checkpoint_id: id,
        s3_key: format!("cp/{id}"),
        last_indexed_iris_id: 0,
        last_indexed_modification_id: 0,
        graph_mutation_id: mut_id,
        blake3_hash: "deadbeef".into(),
        graph_version: 1,
    }
}

// ── mock MutationStore ───────────────────────────────────────────────────

#[derive(Clone)]
struct MockStore {
    /// Newest first.
    recent: Vec<CheckpointMeta>,
    max_id: GraphMutationId,
}

impl MockStore {
    fn with_latest(latest: CheckpointMeta, max_id: GraphMutationId) -> Self {
        Self {
            recent: vec![latest],
            max_id,
        }
    }
}

#[async_trait]
impl MutationStore for MockStore {
    async fn recent_checkpoints(&self, window: usize) -> Result<Vec<CheckpointMeta>, CycleError> {
        Ok(self.recent.iter().take(window).cloned().collect())
    }
    async fn mutations_in_range(
        &self,
        _lo_exclusive: GraphMutationId,
        _hi_inclusive: GraphMutationId,
    ) -> Result<BoxStream<'_, Result<BothEyes<Vec<GraphMutation<VectorId>>>, CycleError>>, CycleError>
    {
        Ok(Box::pin(stream::empty()))
    }
    async fn current_max_mutation_id(&self) -> Result<GraphMutationId, CycleError> {
        Ok(self.max_id)
    }
}

// ── mock ConsensusTransport ──────────────────────────────────────────────

#[derive(Clone)]
struct ExchangeCall {
    msg_variant: &'static str,
    nonce: u128,
    timeout: Duration,
}

#[derive(Clone)]
struct MockTransport {
    calls: Arc<Mutex<Vec<ExchangeCall>>>,
    // FIFO queue of canned peer responses; each phase pops one entry.
    canned: Arc<Mutex<Vec<Vec<ConsensusMessage>>>>,
}

impl MockTransport {
    fn new(canned: Vec<Vec<ConsensusMessage>>) -> Self {
        Self {
            calls: Arc::new(Mutex::new(vec![])),
            canned: Arc::new(Mutex::new(canned)),
        }
    }
}

fn variant(m: &ConsensusMessage) -> &'static str {
    match m {
        ConsensusMessage::BaseProposal { .. } => "Base",
        ConsensusMessage::HeightProposal { .. } => "Height",
        ConsensusMessage::HashProposal { .. } => "Hash",
    }
}

#[async_trait]
impl ConsensusTransport for MockTransport {
    async fn exchange<T: Send + 'static>(
        &self,
        msg: ConsensusMessage,
        expect: fn(ConsensusMessage) -> Option<T>,
        cycle_nonce: u128,
        timeout: Duration,
    ) -> Result<Vec<T>, CycleError> {
        self.calls.lock().unwrap().push(ExchangeCall {
            msg_variant: variant(&msg),
            nonce: cycle_nonce,
            timeout,
        });

        let mut canned = self.canned.lock().unwrap();
        if canned.is_empty() {
            return Err(CycleError::Fatal(
                "mock transport: no canned response".into(),
            ));
        }
        let responses = canned.remove(0);

        let mut projected = Vec::with_capacity(responses.len());
        for r in responses {
            match expect(r) {
                Some(v) => projected.push(v),
                None => {
                    return Err(CycleError::Fatal(
                        "mock transport: peer returned wrong variant".into(),
                    ))
                }
            }
        }
        Ok(projected)
    }
}

// ── mock Materializer ────────────────────────────────────────────────────

#[derive(Clone)]
struct MockMaterializer {
    /// Records the `freeze` passed on each invocation. (Base is discarded —
    /// no test inspects it; existing assertions only look at length / freeze.)
    freezes: Arc<Mutex<Vec<FreezeHeight>>>,
}

impl MockMaterializer {
    fn new() -> Self {
        Self {
            freezes: Arc::new(Mutex::new(vec![])),
        }
    }
}

#[async_trait]
impl Materializer for MockMaterializer {
    async fn snapshot(
        &mut self,
        _base: CheckpointMeta,
        freeze: FreezeHeight,
    ) -> Result<Graph, CycleError> {
        self.freezes.lock().unwrap().push(freeze);
        Ok(empty_graph())
    }
}

// ── mock TerminalAction ──────────────────────────────────────────────────

#[derive(Clone)]
struct MockFinalizer {
    /// Records (hash, freeze) on each invocation.
    calls: Arc<Mutex<Vec<(Blake3Hash, GraphMutationId)>>>,
}

impl MockFinalizer {
    fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(vec![])),
        }
    }
}

#[async_trait]
impl TerminalAction for MockFinalizer {
    async fn finalize(
        &mut self,
        _base: CheckpointMeta,
        freeze: FreezeHeight,
        _graph: Graph,
        hash: Blake3Hash,
    ) -> Result<(), CycleError> {
        self.calls.lock().unwrap().push((hash, freeze.0));
        Ok(())
    }
}

// ── deterministic GraphHasher ────────────────────────────────────────────

struct ConstHasher(Blake3Hash);

impl GraphHasher for ConstHasher {
    fn hash_canonical(&self, _graph: &Graph) -> Blake3Hash {
        self.0
    }
}

fn hash_a() -> Blake3Hash {
    [0xAA; 32]
}
fn hash_b() -> Blake3Hash {
    [0xBB; 32]
}

// ── happy path ───────────────────────────────────────────────────────────

#[tokio::test]
async fn happy_path_finalizes() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(0)), 100);
    let canned = vec![
        // Phase 1 base: peers agree
        vec![
            ConsensusMessage::BaseProposal {
                recent: vec![meta(1, Some(0))],
            },
            ConsensusMessage::BaseProposal {
                recent: vec![meta(1, Some(0))],
            },
        ],
        // Phase 2 height: peers propose same height
        vec![
            ConsensusMessage::HeightProposal { height: 100 },
            ConsensusMessage::HeightProposal { height: 100 },
        ],
        // Phase 5 hash: peers agree
        vec![
            ConsensusMessage::HashProposal { hash: hash_a() },
            ConsensusMessage::HashProposal { hash: hash_a() },
        ],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(10),
    )
    .await
    .expect("happy path");

    match outcome {
        Outcome::Finalized { hash, height } => {
            assert_eq!(hash, hash_a());
            assert_eq!(height, 100);
        }
        other => panic!("expected Finalized, got {other:?}"),
    }
    assert_eq!(fin.calls.lock().unwrap().len(), 1);
    assert_eq!(mat.freezes.lock().unwrap().len(), 1);

    // Phase ordering: Base, Height, Hash.
    let calls = transport.calls.lock().unwrap();
    let variants: Vec<_> = calls.iter().map(|c| c.msg_variant).collect();
    assert_eq!(variants, vec!["Base", "Height", "Hash"]);

    // Phase 1 uses the fixed BASE_PHASE_NONCE sentinel (the agreed base
    // isn't known yet); Phases 2-5 share a nonce derived from the agreed
    // base's checkpoint_id. Two distinct nonces overall.
    let nonces: std::collections::HashSet<u128> = calls.iter().map(|c| c.nonce).collect();
    assert_eq!(nonces.len(), 2, "phase 1 nonce differs from phases 2-5");
    assert_eq!(
        calls[1].nonce, calls[2].nonce,
        "phases 2 and 5 must share the cycle nonce"
    );

    // Timeout is propagated from config.
    for c in calls.iter() {
        assert_eq!(c.timeout, Duration::from_millis(100));
    }
}

// ── gating: not enough mutations → Skipped, no materialize/finalize ──────

#[tokio::test]
async fn skips_when_not_enough_mutations() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(95)), 100);
    let canned = vec![
        vec![
            ConsensusMessage::BaseProposal {
                recent: vec![meta(1, Some(95))],
            },
            ConsensusMessage::BaseProposal {
                recent: vec![meta(1, Some(95))],
            },
        ],
        vec![
            ConsensusMessage::HeightProposal { height: 100 },
            ConsensusMessage::HeightProposal { height: 100 },
        ],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(100),
    )
    .await
    .expect("should skip cleanly");

    match outcome {
        Outcome::Skipped(SkipReason::NotEnoughMutations {
            available,
            required,
        }) => {
            assert_eq!(available, 5);
            assert_eq!(required, 100);
        }
        other => panic!("expected Skipped, got {other:?}"),
    }
    assert_eq!(
        mat.freezes.lock().unwrap().len(),
        0,
        "materializer must not run on skip"
    );
    assert_eq!(
        fin.calls.lock().unwrap().len(),
        0,
        "finalizer must not run on skip"
    );
}

/// Restart caller passes `min=0`; an empty WAL (height = base) still runs.
#[tokio::test]
async fn restart_with_min_zero_runs_even_with_no_new_mutations() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(100)), 100);
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            recent: vec![meta(1, Some(100))],
        }],
        vec![ConsensusMessage::HeightProposal { height: 100 }],
        vec![ConsensusMessage::HashProposal { hash: hash_a() }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(0),
    )
    .await
    .expect("min=0 should always run");

    assert!(matches!(outcome, Outcome::Finalized { .. }));
    assert_eq!(mat.freezes.lock().unwrap().len(), 1);
    assert_eq!(fin.calls.lock().unwrap().len(), 1);
}

// ── base disagreement: StrictLatest skips; MostRecentCommon falls back ──

#[tokio::test]
async fn base_mismatch_skips_with_strict_latest_selector() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(0)), 100);
    let canned = vec![vec![
        ConsensusMessage::BaseProposal {
            recent: vec![meta(2, Some(50))],
        },
        ConsensusMessage::BaseProposal {
            recent: vec![meta(1, Some(0))],
        },
    ]];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(0),
    )
    .await
    .expect("disagreement is Skipped(NoCommonBase), not Fatal");
    assert!(matches!(
        outcome,
        Outcome::Skipped(SkipReason::NoCommonBase)
    ));
    assert_eq!(
        mat.freezes.lock().unwrap().len(),
        0,
        "must not materialize after base mismatch"
    );
}

/// When one peer is one row behind on the newest checkpoint,
/// `MostRecentCommon` finds the deepest entry everyone shares and the cycle
/// proceeds against that fallback base.
#[tokio::test]
async fn most_recent_common_falls_back_to_shared_ancestor() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    // My recent list: newest first [cp(5), cp(4)].
    let store = MockStore {
        recent: vec![meta(5, Some(50)), meta(4, Some(40))],
        max_id: 200,
    };
    let canned = vec![
        // Phase 1: peer 0 has the full list; peer 1 is missing cp(5).
        vec![
            ConsensusMessage::BaseProposal {
                recent: vec![meta(5, Some(50)), meta(4, Some(40))],
            },
            ConsensusMessage::BaseProposal {
                recent: vec![meta(4, Some(40))],
            },
        ],
        // Phase 2: all parties report the same height.
        vec![
            ConsensusMessage::HeightProposal { height: 200 },
            ConsensusMessage::HeightProposal { height: 200 },
        ],
        // Phase 5: hash agreement.
        vec![
            ConsensusMessage::HashProposal { hash: hash_a() },
            ConsensusMessage::HashProposal { hash: hash_a() },
        ],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &MostRecentCommon,
        &cfg(0),
    )
    .await
    .expect("MostRecentCommon should pick cp(4)");

    assert!(matches!(outcome, Outcome::Finalized { .. }));
    assert_eq!(mat.freezes.lock().unwrap().len(), 1);
    assert_eq!(fin.calls.lock().unwrap().len(), 1);
}

/// If no entry appears in every peer's list, even `MostRecentCommon` skips.
#[tokio::test]
async fn most_recent_common_skips_when_no_overlap() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore {
        recent: vec![meta(5, Some(50)), meta(4, Some(40))],
        max_id: 200,
    };
    let canned = vec![vec![
        ConsensusMessage::BaseProposal {
            recent: vec![meta(10, Some(100))],
        },
        ConsensusMessage::BaseProposal {
            recent: vec![meta(11, Some(110))],
        },
    ]];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &MostRecentCommon,
        &cfg(0),
    )
    .await
    .expect("no overlap → Skipped");
    assert!(matches!(
        outcome,
        Outcome::Skipped(SkipReason::NoCommonBase)
    ));
    assert_eq!(mat.freezes.lock().unwrap().len(), 0);
    assert_eq!(fin.calls.lock().unwrap().len(), 0);
}

// ── height = min across parties ──────────────────────────────────────────

#[tokio::test]
async fn height_is_min_across_parties() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(0)), 200);
    let canned = vec![
        vec![
            ConsensusMessage::BaseProposal {
                recent: vec![meta(1, Some(0))],
            },
            ConsensusMessage::BaseProposal {
                recent: vec![meta(1, Some(0))],
            },
        ],
        vec![
            ConsensusMessage::HeightProposal { height: 80 }, // smaller peer
            ConsensusMessage::HeightProposal { height: 150 },
        ],
        vec![
            ConsensusMessage::HashProposal { hash: hash_a() },
            ConsensusMessage::HashProposal { hash: hash_a() },
        ],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(0),
    )
    .await
    .expect("should finalize");

    let freeze = mat.freezes.lock().unwrap()[0];
    assert_eq!(
        freeze.0, 80,
        "freeze height must be min(local=200, peers={{80,150}})"
    );

    if let Outcome::Finalized { height, .. } = outcome {
        assert_eq!(height, 80);
    } else {
        panic!("expected Finalized");
    }
}

// ── fatal: hash disagreement ─────────────────────────────────────────────

#[tokio::test]
async fn hash_mismatch_is_fatal_and_skips_finalize() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(0)), 100);
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            recent: vec![meta(1, Some(0))],
        }],
        vec![ConsensusMessage::HeightProposal { height: 100 }],
        vec![ConsensusMessage::HashProposal { hash: hash_b() }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let err = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(0),
    )
    .await
    .expect_err("hash mismatch must be fatal");
    assert!(matches!(err, CycleError::Fatal(_)));
    assert_eq!(
        mat.freezes.lock().unwrap().len(),
        1,
        "materializer ran before hash phase"
    );
    assert_eq!(
        fin.calls.lock().unwrap().len(),
        0,
        "finalizer must not run on hash mismatch"
    );
}

// ── fatal: peer returns wrong variant ────────────────────────────────────

#[tokio::test]
async fn peer_returning_wrong_variant_is_fatal() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(0)), 100);
    // Phase 1 asks for BaseProposal; peer replies with HeightProposal.
    let canned = vec![vec![ConsensusMessage::HeightProposal { height: 100 }]];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let err = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(0),
    )
    .await
    .expect_err("wrong variant must be fatal");
    assert!(matches!(err, CycleError::Fatal(_)));
}

// ── freeze < base.graph_mutation_id surfaces as PeerBehindBase ───────────
//
// Critical for restart (min_mutations_to_apply=0): without this gate, a
// peer behind the base would slip through `available < min` and run_cycle
// would materialize the *base* graph while reporting a freeze height below
// the base — corrupting the next checkpoint's WAL anchor.

#[tokio::test]
async fn freeze_below_base_skips_as_peer_behind_even_when_min_is_zero() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, Some(500)), 100);
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            recent: vec![meta(1, Some(500))],
        }],
        vec![ConsensusMessage::HeightProposal { height: 100 }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    // min=0 mimics the restart path; the PeerBehindBase check must still fire.
    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(0),
    )
    .await
    .expect("freeze < base must not error, just skip");

    match outcome {
        Outcome::Skipped(SkipReason::PeerBehindBase { freeze, base }) => {
            assert_eq!(freeze, 100);
            assert_eq!(base, 500);
        }
        other => panic!("expected Skipped::PeerBehindBase, got {other:?}"),
    }

    // The materializer and finalizer must not have been called — the cycle
    // bails before phase 3.
    assert_eq!(mat.freezes.lock().unwrap().len(), 0);
    assert_eq!(fin.calls.lock().unwrap().len(), 0);
}

// ── base.graph_mutation_id == None treated as lo=0 ───────────────────────

#[tokio::test]
async fn fresh_base_with_no_graph_mutation_id_uses_zero_as_lo() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore::with_latest(meta(1, None), 42);
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            recent: vec![meta(1, None)],
        }],
        vec![ConsensusMessage::HeightProposal { height: 42 }],
        vec![ConsensusMessage::HashProposal { hash: hash_a() }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(
        &mut mat,
        &mut fin,
        &transport,
        &store,
        &hasher,
        &StrictLatest,
        &cfg(40),
    )
    .await
    .expect("None base → lo=0 should let 42 mutations through");
    assert!(matches!(outcome, Outcome::Finalized { .. }));
}

// ── 3-party end-to-end against the in-memory ring ────────────────────────

use crate::checkpoint_protocol::{transport::test_ring::triangle, RingConsensusTransport};

/// Drives three concurrent `run_cycle` instances against a triangle ring.
/// All three see the same base / height / hash, so all three must Finalize
/// with the same outcome. This is the cross-party determinism story end-to-end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_parties_finalize_in_lockstep() {
    let [r0, r1, r2] = triangle(8);
    let t0 = RingConsensusTransport::new(Box::new(r0));
    let t1 = RingConsensusTransport::new(Box::new(r1));
    let t2 = RingConsensusTransport::new(Box::new(r2));

    let base = meta(7, Some(0));
    let store_for_party = |b: CheckpointMeta| MockStore::with_latest(b, 100);
    let config = CycleConfig {
        min_mutations_to_apply: 1,
        peer_round_timeout: Duration::from_secs(2),
        checkpoint_window: 10,
    };

    let h0 = tokio::spawn({
        let cfg = config.clone();
        let store = store_for_party(base.clone());
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t0,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });
    let h1 = tokio::spawn({
        let cfg = config.clone();
        let store = store_for_party(base.clone());
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t1,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });
    let h2 = tokio::spawn({
        let cfg = config.clone();
        let store = store_for_party(base.clone());
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t2,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });

    let o0 = h0.await.expect("p0 join").expect("p0 cycle");
    let o1 = h1.await.expect("p1 join").expect("p1 cycle");
    let o2 = h2.await.expect("p2 join").expect("p2 cycle");

    for (i, o) in [&o0, &o1, &o2].iter().enumerate() {
        match o {
            Outcome::Finalized { hash, height } => {
                assert_eq!(*hash, hash_a(), "party {i} hash");
                assert_eq!(*height, 100, "party {i} height");
            }
            other => panic!("party {i}: expected Finalized, got {other:?}"),
        }
    }
}

/// One party disagrees on the base — every party should `Skipped(NoCommonBase)`
/// under `StrictLatest`. Operator decides whether to retry.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_parties_skip_on_base_disagreement_with_strict_latest() {
    let [r0, r1, r2] = triangle(8);
    let t0 = RingConsensusTransport::new(Box::new(r0));
    let t1 = RingConsensusTransport::new(Box::new(r1));
    let t2 = RingConsensusTransport::new(Box::new(r2));

    let config = CycleConfig {
        min_mutations_to_apply: 0,
        peer_round_timeout: Duration::from_secs(2),
        checkpoint_window: 10,
    };

    let h0 = tokio::spawn({
        let cfg = config.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let store = MockStore::with_latest(meta(1, Some(0)), 50);
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t0,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });
    let h1 = tokio::spawn({
        let cfg = config.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            // Different base — id=9 vs the others' id=1.
            let store = MockStore::with_latest(meta(9, Some(0)), 50);
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t1,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });
    let h2 = tokio::spawn({
        let cfg = config.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let store = MockStore::with_latest(meta(1, Some(0)), 50);
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t2,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });

    let r0 = h0.await.expect("p0 join");
    let r1 = h1.await.expect("p1 join");
    let r2 = h2.await.expect("p2 join");

    for (i, r) in [&r0, &r1, &r2].iter().enumerate() {
        match r {
            Ok(Outcome::Skipped(SkipReason::NoCommonBase)) => {}
            other => panic!("party {i}: expected Skipped(NoCommonBase), got {other:?}"),
        }
    }
}

/// One party computes a different hash — all three see Fatal at phase 5.
/// Finalize never runs on any party (correctness invariant).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_parties_fail_on_hash_disagreement() {
    let [r0, r1, r2] = triangle(8);
    let t0 = RingConsensusTransport::new(Box::new(r0));
    let t1 = RingConsensusTransport::new(Box::new(r1));
    let t2 = RingConsensusTransport::new(Box::new(r2));

    let base = meta(1, Some(0));
    let config = CycleConfig {
        min_mutations_to_apply: 0,
        peer_round_timeout: Duration::from_secs(2),
        checkpoint_window: 10,
    };

    let fin0 = MockFinalizer::new();
    let fin1 = MockFinalizer::new();
    let fin2 = MockFinalizer::new();
    let fin0_handle = fin0.calls.clone();
    let fin1_handle = fin1.calls.clone();
    let fin2_handle = fin2.calls.clone();

    let h0 = tokio::spawn({
        let cfg = config.clone();
        let base = base.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = fin0;
            let store = MockStore::with_latest(base, 50);
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t0,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });
    let h1 = tokio::spawn({
        let cfg = config.clone();
        let base = base.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = fin1;
            let store = MockStore::with_latest(base, 50);
            let hasher = ConstHasher(hash_b()); // dissident
            run_cycle(
                &mut mat,
                &mut fin,
                &t1,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });
    let h2 = tokio::spawn({
        let cfg = config.clone();
        let base = base.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = fin2;
            let store = MockStore::with_latest(base, 50);
            let hasher = ConstHasher(hash_a());
            run_cycle(
                &mut mat,
                &mut fin,
                &t2,
                &store,
                &hasher,
                &StrictLatest,
                &cfg,
            )
            .await
        }
    });

    let r0 = h0.await.expect("p0 join");
    let r1 = h1.await.expect("p1 join");
    let r2 = h2.await.expect("p2 join");

    for (i, r) in [&r0, &r1, &r2].iter().enumerate() {
        match r {
            Err(CycleError::Fatal(_)) => {}
            other => panic!("party {i}: expected Fatal, got {other:?}"),
        }
    }
    assert_eq!(fin0_handle.lock().unwrap().len(), 0, "p0 must not finalize");
    assert_eq!(fin1_handle.lock().unwrap().len(), 0, "p1 must not finalize");
    assert_eq!(fin2_handle.lock().unwrap().len(), 0, "p2 must not finalize");
}
