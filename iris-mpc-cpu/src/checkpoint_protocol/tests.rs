//! Protocol-level unit tests using mocks for all five traits.
//!
//! Goal: exercise `run_cycle` in isolation — every phase, every fatal path,
//! every gating condition — without S3, Postgres, or peers. The mocks below
//! are deliberately dumb; their only job is to record what `run_cycle`
//! called on them and to return canned responses.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use iris_mpc_common::vector_id::VectorId;

use crate::checkpoint_protocol::{
    run_cycle, Blake3Hash, CheckpointMeta, ConsensusMessage, ConsensusTransport, CycleConfig,
    CycleError, FreezeHeight, Graph, GraphHasher, GraphMutationId, Materializer, MutationStore,
    Outcome, PeerResponses, SkipReason, TerminalAction,
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
        cycle_nonce: 0xC0FFEE_u128,
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
    latest: CheckpointMeta,
    max_id: GraphMutationId,
}

#[async_trait]
impl MutationStore for MockStore {
    async fn latest_checkpoint(&self) -> Result<CheckpointMeta, CycleError> {
        Ok(self.latest.clone())
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
//
// Records every `exchange` call (msg variant + nonce + timeout) and returns
// canned peer responses. Each canned response is a `ConsensusMessage` —
// the caller's `expect` closure projects it. Returning a wrong variant
// drives the "variant mismatch is fatal" path.

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
    ) -> Result<PeerResponses<T>, CycleError> {
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
        Ok(PeerResponses {
            responses: projected,
        })
    }
}

// ── mock Materializer ────────────────────────────────────────────────────

#[derive(Clone)]
struct MockMaterializer {
    /// Records (base, freeze) on each invocation.
    calls: Arc<Mutex<Vec<(CheckpointMeta, FreezeHeight)>>>,
}

impl MockMaterializer {
    fn new() -> Self {
        Self {
            calls: Arc::new(Mutex::new(vec![])),
        }
    }
}

#[async_trait]
impl Materializer for MockMaterializer {
    async fn snapshot(
        &mut self,
        base: CheckpointMeta,
        freeze: FreezeHeight,
    ) -> Result<Graph, CycleError> {
        self.calls.lock().unwrap().push((base, freeze));
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
    let store = MockStore {
        latest: meta(1, Some(0)),
        max_id: 100,
    };
    let canned = vec![
        // Phase 1 base: peers agree
        vec![
            ConsensusMessage::BaseProposal {
                checkpoint: meta(1, Some(0)),
            },
            ConsensusMessage::BaseProposal {
                checkpoint: meta(1, Some(0)),
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

    let outcome = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(10))
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
    assert_eq!(mat.calls.lock().unwrap().len(), 1);

    // Phase ordering: Base, Height, Hash.
    let calls = transport.calls.lock().unwrap();
    let variants: Vec<_> = calls.iter().map(|c| c.msg_variant).collect();
    assert_eq!(variants, vec!["Base", "Height", "Hash"]);

    // All exchanges share one cycle nonce.
    let nonces: std::collections::HashSet<u128> = calls.iter().map(|c| c.nonce).collect();
    assert_eq!(nonces.len(), 1, "all phases must reuse the same nonce");

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
    let store = MockStore {
        latest: meta(1, Some(95)),
        max_id: 100,
    };
    let canned = vec![
        vec![
            ConsensusMessage::BaseProposal {
                checkpoint: meta(1, Some(95)),
            },
            ConsensusMessage::BaseProposal {
                checkpoint: meta(1, Some(95)),
            },
        ],
        vec![
            ConsensusMessage::HeightProposal { height: 100 },
            ConsensusMessage::HeightProposal { height: 100 },
        ],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(100))
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
        mat.calls.lock().unwrap().len(),
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
    let store = MockStore {
        latest: meta(1, Some(100)),
        max_id: 100,
    };
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            checkpoint: meta(1, Some(100)),
        }],
        vec![ConsensusMessage::HeightProposal { height: 100 }],
        vec![ConsensusMessage::HashProposal { hash: hash_a() }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(0))
        .await
        .expect("min=0 should always run");

    assert!(matches!(outcome, Outcome::Finalized { .. }));
    assert_eq!(mat.calls.lock().unwrap().len(), 1);
    assert_eq!(fin.calls.lock().unwrap().len(), 1);
}

// ── fatal: base disagreement ─────────────────────────────────────────────

#[tokio::test]
async fn base_mismatch_is_fatal() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore {
        latest: meta(1, Some(0)),
        max_id: 100,
    };
    let canned = vec![vec![
        ConsensusMessage::BaseProposal {
            checkpoint: meta(2, Some(50)),
        },
        ConsensusMessage::BaseProposal {
            checkpoint: meta(1, Some(0)),
        },
    ]];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let err = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(0))
        .await
        .expect_err("must be fatal");
    assert!(matches!(err, CycleError::Fatal(_)));
    assert_eq!(
        mat.calls.lock().unwrap().len(),
        0,
        "must not materialize after base mismatch"
    );
}

// ── height = min across parties ──────────────────────────────────────────

#[tokio::test]
async fn height_is_min_across_parties() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore {
        latest: meta(1, Some(0)),
        max_id: 200, // local is highest
    };
    let canned = vec![
        vec![
            ConsensusMessage::BaseProposal {
                checkpoint: meta(1, Some(0)),
            },
            ConsensusMessage::BaseProposal {
                checkpoint: meta(1, Some(0)),
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

    let outcome = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(0))
        .await
        .expect("should finalize");

    let (_base, freeze) = mat.calls.lock().unwrap()[0].clone();
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
    let store = MockStore {
        latest: meta(1, Some(0)),
        max_id: 100,
    };
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            checkpoint: meta(1, Some(0)),
        }],
        vec![ConsensusMessage::HeightProposal { height: 100 }],
        vec![ConsensusMessage::HashProposal { hash: hash_b() }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let err = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(0))
        .await
        .expect_err("hash mismatch must be fatal");
    assert!(matches!(err, CycleError::Fatal(_)));
    assert_eq!(
        mat.calls.lock().unwrap().len(),
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
    let store = MockStore {
        latest: meta(1, Some(0)),
        max_id: 100,
    };
    // Phase 1 asks for BaseProposal; peer replies with HeightProposal.
    let canned = vec![vec![ConsensusMessage::HeightProposal { height: 100 }]];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let err = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(0))
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
    let store = MockStore {
        latest: meta(1, Some(500)),
        max_id: 100,
    };
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            checkpoint: meta(1, Some(500)),
        }],
        vec![ConsensusMessage::HeightProposal { height: 100 }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    // min=0 mimics the restart path; the PeerBehindBase check must still fire.
    let outcome = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(0))
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
    assert_eq!(mat.calls.lock().unwrap().len(), 0);
    assert_eq!(fin.calls.lock().unwrap().len(), 0);
}

// ── base.graph_mutation_id == None treated as lo=0 ───────────────────────

#[tokio::test]
async fn fresh_base_with_no_graph_mutation_id_uses_zero_as_lo() {
    let mut mat = MockMaterializer::new();
    let mut fin = MockFinalizer::new();
    let store = MockStore {
        latest: meta(1, None), // brand-new checkpoint with no WAL anchor yet
        max_id: 42,
    };
    let canned = vec![
        vec![ConsensusMessage::BaseProposal {
            checkpoint: meta(1, None),
        }],
        vec![ConsensusMessage::HeightProposal { height: 42 }],
        vec![ConsensusMessage::HashProposal { hash: hash_a() }],
    ];
    let transport = MockTransport::new(canned);
    let hasher = ConstHasher(hash_a());

    let outcome = run_cycle(&mut mat, &mut fin, &transport, &store, &hasher, &cfg(40))
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
    let t0 = RingConsensusTransport::new(r0);
    let t1 = RingConsensusTransport::new(r1);
    let t2 = RingConsensusTransport::new(r2);

    let base = meta(7, Some(0));
    let store_for_party = |b: CheckpointMeta| MockStore {
        latest: b,
        max_id: 100,
    };
    let config = CycleConfig {
        min_mutations_to_apply: 1,
        peer_round_timeout: Duration::from_secs(2),
        cycle_nonce: 0xABCDEF_u128,
    };

    let h0 = tokio::spawn({
        let cfg = config.clone();
        let store = store_for_party(base.clone());
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t0, &store, &hasher, &cfg).await
        }
    });
    let h1 = tokio::spawn({
        let cfg = config.clone();
        let store = store_for_party(base.clone());
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t1, &store, &hasher, &cfg).await
        }
    });
    let h2 = tokio::spawn({
        let cfg = config.clone();
        let store = store_for_party(base.clone());
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t2, &store, &hasher, &cfg).await
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

/// One party disagrees on the base — every party should report a Fatal.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_parties_fail_on_base_disagreement() {
    let [r0, r1, r2] = triangle(8);
    let t0 = RingConsensusTransport::new(r0);
    let t1 = RingConsensusTransport::new(r1);
    let t2 = RingConsensusTransport::new(r2);

    let config = CycleConfig {
        min_mutations_to_apply: 0,
        peer_round_timeout: Duration::from_secs(2),
        cycle_nonce: 0x11_u128,
    };

    let h0 = tokio::spawn({
        let cfg = config.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let store = MockStore {
                latest: meta(1, Some(0)),
                max_id: 50,
            };
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t0, &store, &hasher, &cfg).await
        }
    });
    let h1 = tokio::spawn({
        let cfg = config.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            // Different base — id=9 vs the others' id=1.
            let store = MockStore {
                latest: meta(9, Some(0)),
                max_id: 50,
            };
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t1, &store, &hasher, &cfg).await
        }
    });
    let h2 = tokio::spawn({
        let cfg = config.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = MockFinalizer::new();
            let store = MockStore {
                latest: meta(1, Some(0)),
                max_id: 50,
            };
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t2, &store, &hasher, &cfg).await
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
}

/// One party computes a different hash — all three see Fatal at phase 5.
/// Finalize never runs on any party (correctness invariant).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn three_parties_fail_on_hash_disagreement() {
    let [r0, r1, r2] = triangle(8);
    let t0 = RingConsensusTransport::new(r0);
    let t1 = RingConsensusTransport::new(r1);
    let t2 = RingConsensusTransport::new(r2);

    let base = meta(1, Some(0));
    let config = CycleConfig {
        min_mutations_to_apply: 0,
        peer_round_timeout: Duration::from_secs(2),
        cycle_nonce: 0x22_u128,
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
            let store = MockStore {
                latest: base,
                max_id: 50,
            };
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t0, &store, &hasher, &cfg).await
        }
    });
    let h1 = tokio::spawn({
        let cfg = config.clone();
        let base = base.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = fin1;
            let store = MockStore {
                latest: base,
                max_id: 50,
            };
            let hasher = ConstHasher(hash_b()); // dissident
            run_cycle(&mut mat, &mut fin, &t1, &store, &hasher, &cfg).await
        }
    });
    let h2 = tokio::spawn({
        let cfg = config.clone();
        let base = base.clone();
        async move {
            let mut mat = MockMaterializer::new();
            let mut fin = fin2;
            let store = MockStore {
                latest: base,
                max_id: 50,
            };
            let hasher = ConstHasher(hash_a());
            run_cycle(&mut mat, &mut fin, &t2, &store, &hasher, &cfg).await
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
