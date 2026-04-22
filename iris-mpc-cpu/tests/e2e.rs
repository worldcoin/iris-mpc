use eyre::Result;
use iris_mpc_common::{
    helpers::smpc_request::{REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
    iris_db::{db::IrisDB, iris::IrisCode},
    job::{BatchMetadata, BatchQuery, JobSubmissionHandle, ServerJobResult},
    test::{generate_full_test_db, prepare_batch, E2ETemplate, TestCase, TestCaseGenerator},
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{HawkActor, HawkArgs, HawkHandle, HawkMutation},
    hawkers::{
        aby3::aby3_store::{Aby3SharedIrises, Aby3Store, Aby3VectorRef, FhdOps},
        plaintext_store::PlaintextStore,
        shared_irises::SharedIrises,
    },
    hnsw::{GraphMem, HnswSearcher, LayerDistribution},
    protocol::shared_iris::GaloisRingSharedIris,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

const DB_SIZE: usize = 1000;
const DB_RNG_SEED: u64 = 0xfeedface;
const INTERNAL_RNG_SEED: u64 = 0xcafef00d;
// Bumped for coverage: with 14 always-available TestCase variants (coupon-
// collector expectation ≈ 45 trials) plus dynamically-gated variants that
// need prior state, we need well over 50 queries to cover every variant in
// `TestCase::default_test_set()`.
const NUM_BATCHES: usize = 20;
const MAX_BATCH_SIZE: usize = 10;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const HAWK_CONNECTION_PARALLELISM: usize = 1;
// Deletions and reset_updates are supported in the actor (hawk_main.rs +
// identity_update.rs). The earlier `= 0` TODOs are stale; unblock them here
// so PreviouslyDeleted / MatchAfterResetUpdate actually get exercised.
const MAX_DELETIONS_PER_BATCH: usize = 3;
const MAX_RESET_UPDATES_PER_BATCH: usize = 3;

const HNSW_EF_CONSTR: usize = 320;
const HNSW_M: usize = 256;
const HNSW_EF_SEARCH: usize = 256;
// Must match HawkActor's LINEAR_SCAN_MAX_GRAPH_LAYER (hawk_main.rs:227).
// The actor builds its runtime searcher with `new_linear_scan(ef_constr,
// ef_search, M, 1)`; our plaintext bootstrap graph is later loaded into that
// actor, so it must be built with the same layer mode or the graph shape and
// entry points will mismatch at runtime.
const LINEAR_SCAN_MAX_GRAPH_LAYER: usize = 1;
// Layer density: controls geometric distribution `layer_probability = 1/D`.
// With LINEAR_SCAN_MAX_GRAPH_LAYER=1, roughly 1/D of nodes become entry
// points (layer 1) and the rest sit on layer 0. Both bootstrap and actor
// runtime must agree on this or the bootstrap graph's entry-point density
// won't match what the runtime searcher expects.
const HNSW_LAYER_DENSITY: usize = 20;

// Uniqueness test constants
const UNIQUENESS_REQUEST_PARALLELISM: usize = 4;
const UNIQUENESS_TEST_RNG_SEED: u64 = 0x5ca1ab1e;

fn install_tracing() {
    // try_init so running both tests in the same binary (each calling
    // install_tracing) is idempotent — the second call no-ops instead of
    // panicking with "a global default trace dispatcher has already been set".
    let _ = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}

async fn create_graph_from_plain_dbs(
    player_index: usize,
    left_db: &IrisDB,
    right_db: &IrisDB,
    searcher: &HnswSearcher,
) -> Result<([GraphMem<Aby3VectorRef>; 2], [Aby3SharedIrises; 2])> {
    let mut rng = StdRng::seed_from_u64(DB_RNG_SEED);
    let left_points: HashMap<VectorId, Arc<IrisCode>> = left_db
        .db
        .iter()
        .enumerate()
        .map(|(idx, iris)| (VectorId::from_0_index(idx as u32), Arc::new(iris.clone())))
        .collect();
    let left_storage = SharedIrises::new(left_points, Default::default());

    let right_points: HashMap<VectorId, Arc<IrisCode>> = right_db
        .db
        .iter()
        .enumerate()
        .map(|(idx, iris)| (VectorId::from_0_index(idx as u32), Arc::new(iris.clone())))
        .collect();
    let right_storage = SharedIrises::new(right_points, Default::default());

    let mut left_store = PlaintextStore::<FhdOps>::with_storage(left_storage);
    let mut right_store = PlaintextStore::<FhdOps>::with_storage(right_storage);

    let left_graph = left_store
        .generate_graph(&mut rng, DB_SIZE, searcher)
        .await?;
    let right_graph = right_store
        .generate_graph(&mut rng, DB_SIZE, searcher)
        .await?;

    let left_mpc_graph: GraphMem<Aby3VectorRef> = left_graph;
    let right_mpc_graph: GraphMem<Aby3VectorRef> = right_graph;

    let mut left_shared_irises = HashMap::new();
    let mut right_shared_irises = HashMap::new();

    // sort the points by serial id to ensure consistent ordering
    let left_points_sorted: Vec<_> = left_store.storage.get_sorted_serial_ids();

    let right_points_sorted: Vec<_> = right_store.storage.get_sorted_serial_ids();

    for serial_id in left_points_sorted {
        let vector_id: VectorId = VectorId::from_serial_id(serial_id);
        let shares = GaloisRingSharedIris::generate_shares_locally(
            &mut rng,
            left_store
                .storage
                .get_vector_by_serial_id(serial_id)
                .unwrap()
                .as_ref()
                .clone(),
        );
        left_shared_irises.insert(vector_id, Arc::new(shares[player_index].clone()));
    }
    for serial_id in right_points_sorted {
        let vector_id: VectorId = VectorId::from_serial_id(serial_id);
        let shares = GaloisRingSharedIris::generate_shares_locally(
            &mut rng,
            right_store
                .storage
                .get_vector_by_serial_id(serial_id)
                .unwrap()
                .as_ref()
                .clone(),
        );
        right_shared_irises.insert(vector_id, Arc::new(shares[player_index].clone()));
    }

    let left_iris_store = Aby3Store::<FhdOps>::new_storage(Some(left_shared_irises));
    let right_iris_store = Aby3Store::<FhdOps>::new_storage(Some(right_shared_irises));

    Ok((
        [left_mpc_graph, right_mpc_graph],
        [left_iris_store, right_iris_store],
    ))
}

async fn start_hawk_node(
    args: &HawkArgs,
    left_db: &IrisDB,
    right_db: &IrisDB,
) -> Result<HawkHandle> {
    tracing::info!("🦅 Starting Hawk node {}", args.party_index);

    // Match HawkActor::from_cli_with_graph_and_store which builds its runtime
    // searcher with new_linear_scan and LINEAR_SCAN_MAX_GRAPH_LAYER=1
    // (hawk_main.rs:488-493). The bootstrap graph must be built with the same
    // layer mode so the entry points align with what the actor expects.
    let mut searcher = HnswSearcher::new_linear_scan(
        args.hnsw_param_ef_constr,
        args.hnsw_param_ef_search,
        args.hnsw_param_m,
        LINEAR_SCAN_MAX_GRAPH_LAYER,
    );
    // Override the default geometric distribution (which uses M=256) so the
    // bootstrap agrees with the actor's `hnsw_layer_density` override.
    searcher.layer_distribution = LayerDistribution::new_geometric_from_M(HNSW_LAYER_DENSITY);

    let (graph, iris_store) =
        create_graph_from_plain_dbs(args.party_index, left_db, right_db, &searcher).await?;
    let hawk_actor =
        HawkActor::from_cli_with_graph_and_store(args, CancellationToken::new(), graph, iris_store)
            .await?;

    let handle = HawkHandle::new(hawk_actor).await?;

    Ok(handle)
}

fn make_args(party_index: usize, addresses: Vec<String>, request_parallelism: usize) -> HawkArgs {
    HawkArgs {
        party_index,
        addresses: addresses.clone(),
        outbound_addrs: addresses,
        request_parallelism,
        connection_parallelism: HAWK_CONNECTION_PARALLELISM,
        hnsw_param_ef_constr: HNSW_EF_CONSTR,
        hnsw_param_m: HNSW_M,
        hnsw_param_ef_search: HNSW_EF_SEARCH,
        hnsw_param_ef_supermatch: 4000,
        hnsw_param_ef_saturation_margin: 0,
        hnsw_layer_density: Some(HNSW_LAYER_DENSITY),
        hnsw_fixed_layer_search_batch_size: None,
        hnsw_prf_key: None,
        disable_persistence: false,
        hnsw_disable_memory_persistence: false,
        tls: None,
        numa: true,
    }
}

async fn spawn_three_hawk_nodes(
    db_left: &IrisDB,
    db_right: &IrisDB,
    addresses: Vec<String>,
    request_parallelism: usize,
) -> Result<(HawkHandle, HawkHandle, HawkHandle)> {
    let args0 = make_args(0, addresses.clone(), request_parallelism);
    let args1 = make_args(1, addresses.clone(), request_parallelism);
    let args2 = make_args(2, addresses.clone(), request_parallelism);
    let (h0, h1, h2) = tokio::join!(
        start_hawk_node(&args0, db_left, db_right),
        start_hawk_node(&args1, db_left, db_right),
        start_hawk_node(&args2, db_left, db_right),
    );
    Ok((h0?, h1?, h2?))
}

#[ignore = "Takes long time to run, in CI this is selected in a separate step"]
#[test]
fn e2e_test() -> Result<()> {
    // This test is stack-hungry in release mode; run it on a larger stack to
    // avoid platform-dependent stack overflows.
    std::thread::Builder::new()
        .name("e2e_test".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(e2e_test_async())
        })
        .expect("failed to spawn e2e_test thread")
        .join()
        .expect("e2e_test thread panicked")
}

async fn e2e_test_async() -> Result<()> {
    install_tracing();

    let test_db = generate_full_test_db(DB_SIZE, DB_RNG_SEED, false);
    let db_left = test_db.plain_dbs(0);
    let db_right = test_db.plain_dbs(1);

    let addresses = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let (mut handle0, mut handle1, mut handle2) =
        spawn_three_hawk_nodes(db_left, db_right, addresses, HAWK_REQUEST_PARALLELISM).await?;

    let mut test_case_generator = TestCaseGenerator::new_with_db(test_db, INTERNAL_RNG_SEED, true);

    test_case_generator
        .run_n_batches(
            NUM_BATCHES,
            MAX_BATCH_SIZE,
            MAX_DELETIONS_PER_BATCH,
            MAX_RESET_UPDATES_PER_BATCH,
            [&mut handle0, &mut handle1, &mut handle2],
        )
        .await?;

    // Coverage check: every variant listed in `TestCase::default_test_set()`
    // should have been picked at least once over the run. If the seed + batch
    // counts don't cover something, this fails loudly rather than silently
    // under-covering. Dynamic-gated variants (PreviouslyInserted,
    // PreviouslyDeleted, EnrollmentAfterResetCheckNonMatch, MatchAfterResetUpdate,
    // MatchAfterReauthSkipPersistence) require prior state; bumping batch
    // counts increases the chance they fire.
    let picked = test_case_generator.picked_test_cases();
    tracing::info!("TestCase coverage: {:?}", picked);
    let missing: Vec<TestCase> = TestCase::default_test_set()
        .into_iter()
        .filter(|tc| !picked.contains_key(tc))
        .collect();
    assert!(
        missing.is_empty(),
        "e2e_test did not cover these TestCase variants: {missing:?}. \
         Picked counts: {picked:?}. \
         Bump NUM_BATCHES / MAX_BATCH_SIZE or change INTERNAL_RNG_SEED."
    );

    drop(handle0);
    drop(handle1);
    drop(handle2);

    // TODO: ATM we have no real way to wait for the actors to finish, so just sleep
    // a bit for now
    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}

// ======================================================================
// Uniqueness test
//
// The test drives the live actor pipeline through four batches built on a
// single canonical iris X. Each request we submit is a `Variant` carrying
// its own `purpose` (human narrative of its role) and `expect` (the
// `Expectation` it must satisfy). After every batch we call
// `check_expectations`, which walks every (variant, observation) pair,
// collects ALL violations, and panics once with a self-contained report
// that points at the offending slot, its role, what we expected, what we
// observed, and a pointer into the source that explains why.
//
// Batches:
//   Batch 1: seed X + 30 non-mirror rotations of X          → slot 0 UniqueInsert; rotations all IntraBatchBlocked by slot 0
//   Batch 2: 62 rotation/mirror variants of X               → DbMatchAt / MirrorAttack (X now in DB)
//   Batch 3: [X' reauth, Y=X+30, Z=X-15]                    → Reauth succeeds; Y blocked intra-batch; Z blocked by DB match on original X
//   Batch 4: [Y=X+30, Z=X-15] (after reauth persists X')    → Y matches X' in DB; Z is fresh insert (X' out of rotation window)
//   Batch 5: [delete(seed_db_index), uniqueness(X')]        → deletion acts before same-batch search; X' re-enrolls at fresh serial
// ======================================================================

/// The outcome a single request must produce, recorded at variant
/// construction and checked after the batch returns. Each arm carries the
/// evidence that should appear in `ServerJobResult` for the variant's
/// slot, so the evaluator can point at the specific field that disagrees.
#[derive(Clone, Debug)]
enum Expectation {
    /// Uniqueness accepted as a brand-new iris. `min_index` is the smallest
    /// acceptable `merged_results` (0-indexed) — the insertion must be above
    /// everything we've previously placed in the DB.
    UniqueInsert { min_index: u32 },
    /// Uniqueness rejected because search hit a specific stored iris.
    DbMatchAt { expected_match_index: u32 },
    /// Uniqueness rejected because the slot intra-batch-matched an earlier
    /// mutating request. `earlier_request_id` must appear in
    /// `matched_batch_request_ids` for this slot.
    IntraBatchBlockedBy {
        earlier_request_id: String,
        earlier_role: String,
    },
    /// Full-face mirror attack: normal `match_ids` empty, mirror match
    /// present at `expected_mirror_match_index`,
    /// `full_face_mirror_attack_detected` raised.
    MirrorAttackDetected { expected_mirror_match_index: u32 },
    /// Reauth against `target_index` succeeded — `successful_reauths[slot]`
    /// true and `merged_results[slot]` equals the target (index, 0-based).
    ReauthSucceeds { target_index: u32 },
}

impl Expectation {
    /// One-line source pointer so the failure message says WHY we think
    /// this outcome should hold.
    fn diagnosis(&self) -> &'static str {
        use Expectation::*;
        match self {
            UniqueInsert { .. } => {
                "hawk_main.rs::HawkResult::merged_results: fresh uniqueness inserts surface as inserted_id.index()"
            }
            DbMatchAt { .. } => {
                "hawk_main/matching.rs:289-297: uniqueness with any Search/Luc/Reauth/Supermatch → NoMutation"
            }
            IntraBatchBlockedBy { .. } => {
                "hawk_main/matching.rs:289-297 + intra_batch.rs:76-84: later slot's rotated query vs earlier slot's centered query, ANY-rotation/ANY-eye; if earlier decision is a mutation, later uniqueness is blocked"
            }
            MirrorAttackDetected { .. } => {
                "hawk_main.rs:1557-1559: full_face_mirror_attack_detected = (normal match_ids empty) && (mirror match_ids non-empty)"
            }
            ReauthSucceeds { .. } => {
                "hawk_main/matching.rs:310-315: Reauth with reauth_rule(or_rule, matches) → ReauthUpdate; mutation applied only in handle_mutations AFTER all searches"
            }
        }
    }

    /// Short human-readable expected-outcome summary.
    fn describe(&self) -> String {
        use Expectation::*;
        match self {
            UniqueInsert { min_index } => {
                format!("UniqueInsert (merged_results ≥ {min_index}, no match, no reauth)")
            }
            DbMatchAt {
                expected_match_index,
            } => format!("DbMatch at index {expected_match_index} (match_ids contains it)"),
            IntraBatchBlockedBy {
                earlier_role,
                earlier_request_id,
            } => format!(
                "IntraBatchBlocked by earlier slot (role: {earlier_role}; request_id={earlier_request_id}) — match_ids empty, matched_batch_request_ids contains the earlier request_id"
            ),
            MirrorAttackDetected {
                expected_mirror_match_index,
            } => format!(
                "MirrorAttack flagged (mirror_match_ids contains {expected_mirror_match_index}, normal match_ids empty)"
            ),
            ReauthSucceeds { target_index } => {
                format!("ReauthSucceeds (successful_reauths=true, merged_results={target_index})")
            }
        }
    }
}

/// A single request in the batch, fully self-describing: the transformation
/// applied to X (`rotation`, `mirror`), the narrative role it plays
/// (`purpose`), and the outcome we want to see (`expect`). Debug
/// diagnostics print all four.
#[derive(Clone)]
struct Variant {
    rotation: isize,
    mirror: bool,
    template: E2ETemplate,
    request_id: String,
    message_type: String,
    reauth_target_index: Option<u32>,
    purpose: String,
    expect: Expectation,
}

impl Variant {
    fn label(&self) -> String {
        format!("rot={:+} mirror={}", self.rotation, self.mirror)
    }
}

fn apply_rotation(code: &IrisCode, rotation: isize) -> IrisCode {
    if rotation == 0 {
        code.clone()
    } else {
        code.rotations_from_range(rotation..rotation + 1)
            .into_iter()
            .next()
            .expect("rotations_from_range returns one element")
    }
}

/// Construct the (left, right) pair submitted as a rotation+mirror variant of
/// the seed (x_left, x_right).
///
/// When `mirror` is false this is a simple rotation of each eye. When `mirror`
/// is true the transformation models a physical full-face mirror attack: the
/// eyes swap sides AND each eye is mirrored in place. See
/// `iris-mpc-common/src/test.rs:1075-1079` (`FullFaceMirrorAttack` case) for
/// the canonical construction — the Mirror-orientation search path in
/// `iris-mpc-cpu/src/execution/hawk_main.rs:1227-1231` swaps eyes when
/// looking for an attack, so the query must be swapped symmetrically or the
/// match will never land.
fn build_mirror_variant_pair(
    x_left: &IrisCode,
    x_right: &IrisCode,
    rotation: isize,
    mirror: bool,
) -> (IrisCode, IrisCode) {
    let (base_l, base_r) = if mirror {
        (x_right.mirrored(), x_left.mirrored())
    } else {
        (x_left.clone(), x_right.clone())
    };
    (
        apply_rotation(&base_l, rotation),
        apply_rotation(&base_r, rotation),
    )
}

/// Factory for a uniqueness `Variant` with a given rotation+mirror of X.
fn uniqueness_variant(
    x_left: &IrisCode,
    x_right: &IrisCode,
    rotation: isize,
    mirror: bool,
    purpose: String,
    expect: Expectation,
) -> Variant {
    let (l, r) = build_mirror_variant_pair(x_left, x_right, rotation, mirror);
    Variant {
        rotation,
        mirror,
        template: E2ETemplate::new(l, r),
        request_id: Uuid::new_v4().to_string(),
        message_type: UNIQUENESS_MESSAGE_TYPE.to_string(),
        reauth_target_index: None,
        purpose,
        expect,
    }
}

/// Factory for a reauth `Variant`: submits rotation+mirror of X as a reauth
/// targeting `target_index`.
fn reauth_variant(
    x_left: &IrisCode,
    x_right: &IrisCode,
    rotation: isize,
    mirror: bool,
    target_index: u32,
    purpose: String,
) -> Variant {
    let (l, r) = build_mirror_variant_pair(x_left, x_right, rotation, mirror);
    Variant {
        rotation,
        mirror,
        template: E2ETemplate::new(l, r),
        request_id: Uuid::new_v4().to_string(),
        message_type: REAUTH_MESSAGE_TYPE.to_string(),
        reauth_target_index: Some(target_index),
        purpose,
        expect: Expectation::ReauthSucceeds { target_index },
    }
}

/// Build the 62 rotation/mirror variants (identity slot first). Each variant
/// carries its own purpose + expectation tailored to the batch-2 scenario
/// where the seed iris X is already persisted at `seed_db_index`.
fn build_batch2_variants(
    x_left: &IrisCode,
    x_right: &IrisCode,
    seed_db_index: u32,
) -> Vec<Variant> {
    let mut variants = Vec::with_capacity(62);
    // Identity slot first so the batch's canonical non-rotated X sits at
    // slot 0 — intra-batch detection compares later slots' rotated queries
    // against slot 0's centered raw query.
    let mut push = |rotation: isize, mirror: bool| {
        let (purpose, expect) = if mirror {
            (
                format!(
                    "batch2: rot={rotation:+} MIRROR — full-face mirror of X; must trip mirror-attack detector against seed at idx {seed_db_index}"
                ),
                Expectation::MirrorAttackDetected {
                    expected_mirror_match_index: seed_db_index,
                },
            )
        } else if rotation == 0 {
            (
                format!("batch2: rot=0 — X itself; exact DB match on seed at idx {seed_db_index}"),
                Expectation::DbMatchAt {
                    expected_match_index: seed_db_index,
                },
            )
        } else {
            (
                format!(
                    "batch2: rot={rotation:+} non-mirror — rotation of seed X within [-15,+15] search window; must match seed at idx {seed_db_index}"
                ),
                Expectation::DbMatchAt {
                    expected_match_index: seed_db_index,
                },
            )
        };
        variants.push(uniqueness_variant(
            x_left, x_right, rotation, mirror, purpose, expect,
        ));
    };
    push(0, false);
    for rotation in -15isize..=15 {
        for mirror in [false, true] {
            if rotation == 0 && !mirror {
                continue;
            }
            push(rotation, mirror);
        }
    }
    variants
}

async fn submit_variants_batch(
    variants: &[Variant],
    deletions: &[u32],
    h0: &mut HawkHandle,
    h1: &mut HawkHandle,
    h2: &mut HawkHandle,
    rng: &mut StdRng,
) -> Result<[ServerJobResult<HawkMutation>; 3]> {
    let mut batches = [
        BatchQuery::default(),
        BatchQuery::default(),
        BatchQuery::default(),
    ];
    for b in batches.iter_mut() {
        b.full_face_mirror_attacks_detection_enabled = true;
    }

    // Deletions: same 0-based index on all three party batches.
    // These are applied at the START of batch processing (hawk_main.rs:1815),
    // BEFORE any search, so same-batch searches see the deletion as done.
    for &del_idx in deletions {
        for batch in batches.iter_mut() {
            batch.push_deletion_request(
                format!("sns_del_{del_idx}"),
                del_idx,
                BatchMetadata::default(),
            );
        }
    }

    for v in variants {
        let shared = v.template.to_shared_template(true, rng);
        for (party, batch) in batches.iter_mut().enumerate() {
            prepare_batch(
                batch,
                true,
                v.request_id.clone(),
                party,
                shared.clone(),
                vec![],
                v.reauth_target_index.as_ref(),
                false,
                v.message_type.clone(),
            )?;
        }
    }

    let [b0, b1, b2] = batches;
    let (f0, f1, f2) = tokio::join!(
        h0.submit_batch_query(b0),
        h1.submit_batch_query(b1),
        h2.submit_batch_query(b2),
    );
    let res0 = f0.await?;
    let res1 = f1.await?;
    let res2 = f2.await?;
    Ok([res0, res1, res2])
}

/// Asserts the three parties returned byte-identical per-request results for
/// every field we care about. Returns the party-0 result for inspection.
fn assert_three_parties_agree<'a>(
    batch_label: &str,
    results: &'a [ServerJobResult<HawkMutation>; 3],
) -> &'a ServerJobResult<HawkMutation> {
    let [r0, r1, r2] = results;
    macro_rules! agree {
        ($field:ident) => {
            assert_eq!(
                r0.$field,
                r1.$field,
                "{}: party 0 vs 1 disagree on {}",
                batch_label,
                stringify!($field)
            );
            assert_eq!(
                r0.$field,
                r2.$field,
                "{}: party 0 vs 2 disagree on {}",
                batch_label,
                stringify!($field)
            );
        };
    }
    agree!(request_ids);
    agree!(matches);
    agree!(merged_results);
    agree!(successful_reauths);
    agree!(full_face_mirror_attack_detected);
    agree!(matched_batch_request_ids);
    agree!(match_ids);
    agree!(full_face_mirror_match_ids);
    r0
}

/// Per-slot result extracted from the batch's `ServerJobResult`.
#[derive(Clone, Debug)]
struct Observation {
    slot: usize,
    label: String,
    purpose: String,
    expect: Expectation,
    was_match: bool,
    successful_reauth: bool,
    mirror_attack: bool,
    merged_results: u32,
    match_ids: Vec<u32>,
    full_face_mirror_match_ids: Vec<u32>,
    matched_batch_request_ids: Vec<String>,
}

/// Evaluate one variant's expectation against its observation. Returns a
/// list of problems (empty = pass).
fn evaluate_expectation(obs: &Observation) -> Vec<String> {
    use Expectation::*;
    let mut p: Vec<String> = Vec::new();
    match &obs.expect {
        UniqueInsert { min_index } => {
            if obs.was_match {
                p.push("was_match=true (wanted false)".into());
            }
            if obs.mirror_attack {
                p.push("mirror_attack=true (wanted false)".into());
            }
            if obs.successful_reauth {
                p.push("successful_reauth=true (wanted false)".into());
            }
            if obs.merged_results < *min_index {
                p.push(format!(
                    "merged_results={} < min_index={} (wanted a fresh serial_id above all previously-seen indices)",
                    obs.merged_results, min_index
                ));
            }
            if !obs.match_ids.is_empty() {
                p.push(format!("match_ids={:?} (wanted empty)", obs.match_ids));
            }
            if !obs.matched_batch_request_ids.is_empty() {
                p.push(format!(
                    "matched_batch_request_ids={:?} (wanted empty)",
                    obs.matched_batch_request_ids
                ));
            }
        }
        DbMatchAt {
            expected_match_index,
        } => {
            if !obs.was_match {
                p.push("was_match=false (wanted true)".into());
            }
            if obs.mirror_attack {
                p.push("mirror_attack=true (wanted false)".into());
            }
            if obs.successful_reauth {
                p.push("successful_reauth=true (wanted false)".into());
            }
            if obs.merged_results != *expected_match_index {
                p.push(format!(
                    "merged_results={} (wanted {})",
                    obs.merged_results, expected_match_index
                ));
            }
            if !obs.match_ids.contains(expected_match_index) {
                p.push(format!(
                    "match_ids={:?} missing expected index {}",
                    obs.match_ids, expected_match_index
                ));
            }
        }
        IntraBatchBlockedBy {
            earlier_request_id,
            earlier_role: _,
        } => {
            if !obs.was_match {
                p.push("was_match=false (wanted true — intra-batch match counts as match)".into());
            }
            if obs.mirror_attack {
                p.push("mirror_attack=true (wanted false)".into());
            }
            if obs.successful_reauth {
                p.push("successful_reauth=true (wanted false)".into());
            }
            if !obs.match_ids.is_empty() {
                p.push(format!(
                    "match_ids={:?} (wanted empty — there should be no direct DB match, only intra-batch)",
                    obs.match_ids
                ));
            }
            if !obs.matched_batch_request_ids.contains(earlier_request_id) {
                p.push(format!(
                    "matched_batch_request_ids={:?} missing expected earlier request_id={}",
                    obs.matched_batch_request_ids, earlier_request_id
                ));
            }
        }
        MirrorAttackDetected {
            expected_mirror_match_index,
        } => {
            if !obs.mirror_attack {
                p.push("mirror_attack=false (wanted true)".into());
            }
            if obs.successful_reauth {
                p.push("successful_reauth=true (wanted false)".into());
            }
            if !obs.match_ids.is_empty() {
                p.push(format!(
                    "normal match_ids={:?} non-empty (mirror attack requires normal empty)",
                    obs.match_ids
                ));
            }
            if !obs
                .full_face_mirror_match_ids
                .contains(expected_mirror_match_index)
            {
                p.push(format!(
                    "full_face_mirror_match_ids={:?} missing expected index {}",
                    obs.full_face_mirror_match_ids, expected_mirror_match_index
                ));
            }
        }
        ReauthSucceeds { target_index } => {
            if !obs.successful_reauth {
                p.push("successful_reauth=false (wanted true)".into());
            }
            if obs.mirror_attack {
                p.push("mirror_attack=true (wanted false)".into());
            }
            if obs.merged_results != *target_index {
                p.push(format!(
                    "merged_results={} (wanted reauth target {})",
                    obs.merged_results, target_index
                ));
            }
        }
    }
    p
}

/// Walk every (variant, observation) pair, collect ALL violations, panic
/// once with a rich multi-line report if any exist. Returns observations
/// for subsequent use (e.g., extracting a newly-inserted index).
fn check_expectations(
    batch_label: &str,
    variants: &[Variant],
    results: &[ServerJobResult<HawkMutation>; 3],
) -> Vec<Observation> {
    let agreed = assert_three_parties_agree(batch_label, results);
    assert_eq!(
        agreed.request_ids.len(),
        variants.len(),
        "{}: expected {} results, got {}",
        batch_label,
        variants.len(),
        agreed.request_ids.len()
    );

    let observations: Vec<Observation> = variants
        .iter()
        .map(|v| {
            let slot = agreed
                .request_ids
                .iter()
                .position(|r| r == &v.request_id)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: variant '{}' request_id {} not in batch result",
                        batch_label, v.purpose, v.request_id
                    )
                });
            Observation {
                slot,
                label: v.label(),
                purpose: v.purpose.clone(),
                expect: v.expect.clone(),
                was_match: agreed.matches[slot],
                successful_reauth: agreed.successful_reauths[slot],
                mirror_attack: agreed.full_face_mirror_attack_detected[slot],
                merged_results: agreed.merged_results[slot],
                match_ids: agreed.match_ids[slot].clone(),
                full_face_mirror_match_ids: agreed.full_face_mirror_match_ids[slot].clone(),
                matched_batch_request_ids: agreed.matched_batch_request_ids[slot].clone(),
            }
        })
        .collect();

    // Always log the per-slot table so a tracing-enabled run has the full
    // picture even on success.
    tracing::info!("=== {} — per-slot observations ===", batch_label);
    for obs in &observations {
        tracing::info!(
            "  [{:>2}] {:<18}  was_match={}  reauth_ok={}  mirror_attack={}  merged={:>4}  match_ids={:?}  mirror_match_ids={:?}  intra_batch_ids={:?}  purpose: {}",
            obs.slot,
            obs.label,
            obs.was_match,
            obs.successful_reauth,
            obs.mirror_attack,
            obs.merged_results,
            obs.match_ids,
            obs.full_face_mirror_match_ids,
            obs.matched_batch_request_ids,
            obs.purpose,
        );
    }

    let mut failures: Vec<String> = Vec::new();
    for obs in &observations {
        let problems = evaluate_expectation(obs);
        if !problems.is_empty() {
            failures.push(format!(
                "  slot {:>2} [{}]\n    purpose : {}\n    expected: {}\n    observed: was_match={}, successful_reauth={}, mirror_attack={}, merged_results={}, match_ids={:?}, mirror_match_ids={:?}, matched_batch_request_ids={:?}\n    problems: {}\n    diagnosis: {}",
                obs.slot,
                obs.label,
                obs.purpose,
                obs.expect.describe(),
                obs.was_match,
                obs.successful_reauth,
                obs.mirror_attack,
                obs.merged_results,
                obs.match_ids,
                obs.full_face_mirror_match_ids,
                obs.matched_batch_request_ids,
                problems.join("; "),
                obs.expect.diagnosis(),
            ));
        }
    }

    if !failures.is_empty() {
        let mut report = format!(
            "\n\n=== {} FAILED: {} expectation(s) violated ===\n",
            batch_label,
            failures.len()
        );
        for f in &failures {
            report.push_str(f);
            report.push('\n');
        }
        report.push_str("\n--- Full result table for this batch ---\n");
        for obs in &observations {
            report.push_str(&format!(
                "  [{:>2}] {:<18}  was_match={}  reauth_ok={}  mirror_attack={}  merged={:>4}  match_ids={:?}  mirror_match_ids={:?}  intra_batch_ids={:?}  purpose: {}\n",
                obs.slot,
                obs.label,
                obs.was_match,
                obs.successful_reauth,
                obs.mirror_attack,
                obs.merged_results,
                obs.match_ids,
                obs.full_face_mirror_match_ids,
                obs.matched_batch_request_ids,
                obs.purpose,
            ));
        }
        panic!("{}", report);
    }

    observations
}

#[ignore = "Takes long time to run, in CI this is selected in a separate step"]
#[test]
fn e2e_uniqueness_test() -> Result<()> {
    std::thread::Builder::new()
        .name("e2e_uniqueness_test".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(e2e_uniqueness_test_async())
        })
        .expect("failed to spawn e2e_uniqueness_test thread")
        .join()
        .expect("e2e_uniqueness_test thread panicked")
}

async fn e2e_uniqueness_test_async() -> Result<()> {
    install_tracing();

    let test_db = generate_full_test_db(DB_SIZE, DB_RNG_SEED, false);
    let db_left = test_db.plain_dbs(0);
    let db_right = test_db.plain_dbs(1);

    // Ports distinct from e2e_test to avoid TIME_WAIT conflicts when both
    // tests run back-to-back in the same CI step.
    let addresses = ["127.0.0.1:16300", "127.0.0.1:16400", "127.0.0.1:16500"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let (mut h0, mut h1, mut h2) =
        spawn_three_hawk_nodes(db_left, db_right, addresses, UNIQUENESS_REQUEST_PARALLELISM)
            .await?;

    let mut rng = StdRng::seed_from_u64(UNIQUENESS_TEST_RNG_SEED);

    // Fresh seed iris X. With 12800-bit random codes, collision with the
    // 1000-entry plaintext DB is astronomically unlikely.
    let x_left = IrisCode::random_rng(&mut rng);
    let x_right = IrisCode::random_rng(&mut rng);

    // --- Batch 1: seed X plus its 30 non-mirror rotations -----------------
    // Slot 0 is X (rot=0) — this is the "new enrollment" that gets UniqueInsert.
    // Slots 1..30 are X rotated by each non-zero r ∈ [-15,+15], same iris
    // viewed through the rotation window. DB is still empty of X at search
    // time (mutations apply AFTER search), so none of them hit the DB.
    // Each one's rotation window includes X: slot_k at rot=-k brings X+k
    // back to X, which matches slot 0's centered raw iris via the intra-batch
    // path. Slot 0's decision is UniqueInsert (mutation), so each rotation
    // slot is blocked with `IntraBatchBlockedBy(slot 0)`. Confirms intra-batch
    // de-duplication when the "primary" slot is itself a new insertion.
    let seed = uniqueness_variant(
        &x_left,
        &x_right,
        0,
        false,
        format!("batch1 slot 0: seed — insert X into DB (expect serial_id ≥ {DB_SIZE})"),
        Expectation::UniqueInsert {
            min_index: DB_SIZE as u32,
        },
    );
    let seed_request_id = seed.request_id.clone();
    let mut batch1 = vec![seed];
    for rot in -15isize..=15 {
        if rot == 0 {
            continue;
        }
        batch1.push(uniqueness_variant(
            &x_left,
            &x_right,
            rot,
            false,
            format!(
                "batch1: rot={rot:+} non-mirror — same iris as slot 0 under rotation; must intra-batch match slot 0 (its rot=-({rot:+}) hits X) and be blocked because slot 0 decides UniqueInsert (mutation)"
            ),
            Expectation::IntraBatchBlockedBy {
                earlier_request_id: seed_request_id.clone(),
                earlier_role: "batch1 slot 0 — seed X UniqueInsert".to_string(),
            },
        ));
    }
    assert_eq!(batch1.len(), 31);
    tracing::info!(
        "Batch 1: submitting seed X + {} non-mirror rotation variants",
        batch1.len() - 1
    );
    let results = submit_variants_batch(&batch1, &[], &mut h0, &mut h1, &mut h2, &mut rng).await?;
    let obs = check_expectations("batch1 (seed + 30 rotations)", &batch1, &results);
    let seed_db_index = obs[0].merged_results;
    tracing::info!(
        "Batch 1 OK — seed persisted at index {}; 30 rotation variants all blocked intra-batch by slot 0",
        seed_db_index
    );

    // --- Batch 2: 62 rotation/mirror variants of X ------------------------
    let batch2 = build_batch2_variants(&x_left, &x_right, seed_db_index);
    assert_eq!(batch2.len(), 62);
    tracing::info!("Batch 2: submitting {} variants", batch2.len());
    let results = submit_variants_batch(&batch2, &[], &mut h0, &mut h1, &mut h2, &mut rng).await?;
    check_expectations("batch2 (62 variants)", &batch2, &results);
    tracing::info!("Batch 2 OK");

    // --- Batch 3: [X' reauth, Y=X+30 uniqueness, Z=X-15 uniqueness] -------
    // X' = X rotated +15 on both eyes, submitted as a reauth of seed_db_index.
    // Y = X rotated +30:
    //   DB search (vs stored X): Y window [X+15..X+45], no rotation hits X → no DB match.
    //   Intra-batch vs X' center X+15: Y at rot=-15 lands on X+15 → IntraBatch match with slot 0.
    //   Slot 0 decision is ReauthUpdate (mutation) → blocks Y. (matching.rs:289-297)
    // Z = X rotated -15:
    //   DB search (vs stored X, reauth NOT yet applied to store): Z window [X-30..X+0], rot=+15 hits X → DB match.
    //   Mutations are applied only in handle_mutations AFTER the batch's searches (hawk_main.rs:1797..).
    let x_prime = reauth_variant(
        &x_left,
        &x_right,
        15,
        false,
        seed_db_index,
        format!(
            "batch3 slot 0: REAUTH X'=X+15 targeting seed idx {seed_db_index} — must succeed; mutation applied only AFTER batch searches"
        ),
    );
    let y_b3 = uniqueness_variant(
        &x_left,
        &x_right,
        30,
        false,
        "batch3 slot 1: Y=X+30 uniqueness — blocked intra-batch by X' reauth (Y rot=-15 → X+15 = X' center). Reauth is a mutation → NoMutation.".to_string(),
        Expectation::IntraBatchBlockedBy {
            earlier_request_id: x_prime.request_id.clone(),
            earlier_role: "X'=X+15 reauth at batch3 slot 0".to_string(),
        },
    );
    let z_b3 = uniqueness_variant(
        &x_left,
        &x_right,
        -15,
        false,
        format!(
            "batch3 slot 2: Z=X-15 uniqueness — DB still has ORIGINAL X during search (reauth not yet applied); Z window [X-30..X+0] hits X at rot=+15 → DbMatch at idx {seed_db_index}"
        ),
        Expectation::DbMatchAt {
            expected_match_index: seed_db_index,
        },
    );
    let batch3 = vec![x_prime, y_b3, z_b3];
    tracing::info!("Batch 3: submitting [X' reauth, Y=X+30, Z=X-15]");
    let results = submit_variants_batch(&batch3, &[], &mut h0, &mut h1, &mut h2, &mut rng).await?;
    check_expectations("batch3 (reauth + Y + Z)", &batch3, &results);
    tracing::info!("Batch 3 OK — X' has now replaced X at seed idx");

    // --- Batch 4: [Y=X+30, Z=X-15] after reauth persists ------------------
    // DB at seed_db_index now holds X' = X+15 (shared_irises.rs:73-80 overwrites in place).
    // Y = X+30 vs X': Y window [X+15..X+45], rot=-15 hits X+15 → DbMatch.
    // Z = X-15 vs X': Z window [X-30..X+0], closest delta to X+15 is 15 (at rot=+15 → X+0 vs X+15, delta=15... wait, need |delta|>15 for no-match).
    //   Actually: Z at rot r: X-15+r. Match X+15 ⇒ r=30, outside ±15 window. No match.
    // Intra-batch within batch4: Y center X+30 vs Z window [X-30..X+0] — no hit.
    // ⇒ Y blocked by DB match; Z is a UniqueInsert at the next fresh serial_id.
    let y_b4 = uniqueness_variant(
        &x_left,
        &x_right,
        30,
        false,
        format!(
            "batch4 slot 0: Y=X+30 uniqueness — DB now has X'=X+15 at idx {seed_db_index}; Y rot=-15 hits X+15 → DbMatch at idx {seed_db_index}"
        ),
        Expectation::DbMatchAt {
            expected_match_index: seed_db_index,
        },
    );
    let z_b4 = uniqueness_variant(
        &x_left,
        &x_right,
        -15,
        false,
        format!(
            "batch4 slot 1: Z=X-15 uniqueness — vs stored X'=X+15, |delta|=30 is out of ±15 window; must insert as NEW serial_id > {seed_db_index}"
        ),
        Expectation::UniqueInsert {
            min_index: seed_db_index + 1,
        },
    );
    let batch4 = vec![y_b4, z_b4];
    tracing::info!("Batch 4: submitting [Y=X+30, Z=X-15] post-reauth");
    let results = submit_variants_batch(&batch4, &[], &mut h0, &mut h1, &mut h2, &mut rng).await?;
    let batch4_obs = check_expectations("batch4 (post-reauth Y + Z)", &batch4, &results);
    let z_db_index = batch4_obs[1].merged_results;
    tracing::info!(
        "Batch 4 OK — reauth displacement verified; Z inserted at index {}",
        z_db_index
    );

    // --- Batch 5: [delete(seed_db_index), uniqueness(X')] -----------------
    // Exercises the timing asymmetry: deletions run BEFORE same-batch searches
    // (hawk_main.rs:1815 / identity_update.rs:57-78), whereas uniqueness
    // mutations run AFTER. Effect: by the time the uniqueness slot searches,
    // serial @ seed_db_index has been overwritten with a dummy iris — so X'
    // has nothing to match against and re-enrolls at a fresh serial. Nothing
    // exotic, just a straightforward demonstration that deletion is
    // immediately visible within the batch.
    //
    // DB state entering batch 5: X'=X+15 at seed_db_index, Z=X-15 at z_db_index.
    // X' window [X+0..X+30] vs stored X' @ seed_db_index: delete→dummy, no match.
    // X' window vs stored Z=X-15 @ z_db_index: closest rotation delta is 30, out
    // of window. No match. Hence UniqueInsert at (z_db_index + 1).
    let x_prime_reenroll = uniqueness_variant(
        &x_left,
        &x_right,
        15,
        false,
        format!(
            "batch5: uniqueness X'=X+15 — submitted in same batch as delete({seed_db_index}). By search time, seed idx is a dummy (deletion acts before search); X' has no DB match (Z at idx {z_db_index} is out of rotation window). Must UniqueInsert at serial > {z_db_index}"
        ),
        Expectation::UniqueInsert {
            min_index: z_db_index + 1,
        },
    );
    let batch5 = vec![x_prime_reenroll];
    let deletions5 = vec![seed_db_index];
    tracing::info!("Batch 5: submitting [delete({seed_db_index}), uniqueness(X')]");
    let results =
        submit_variants_batch(&batch5, &deletions5, &mut h0, &mut h1, &mut h2, &mut rng).await?;
    let batch5_obs =
        check_expectations("batch5 (delete + uniqueness re-enroll)", &batch5, &results);
    tracing::info!(
        "Batch 5 OK — X' re-enrolled at index {} after deletion cleared the slot",
        batch5_obs[0].merged_results
    );

    tracing::info!("e2e_uniqueness_test: all 5 batches passed");

    drop(h0);
    drop(h1);
    drop(h2);

    // TODO: ATM we have no real way to wait for the actors to finish, so just
    // sleep a bit.
    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}
