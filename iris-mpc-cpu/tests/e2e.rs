#![recursion_limit = "256"]

use eyre::Result;
use iris_mpc_common::{
    helpers::smpc_request::{REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
    iris_db::{db::IrisDB, iris::IrisCode},
    job::{BatchMetadata, BatchQuery, JobSubmissionHandle, ServerJobResult},
    test::{generate_full_test_db, prepare_batch, E2ETemplate, TestCaseGenerator},
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
use std::{collections::HashMap, future::Future, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

const DB_SIZE: usize = 1000;
const DB_RNG_SEED: u64 = 0xfeedface;
const INTERNAL_RNG_SEED: u64 = 0xcafef00d;
const NUM_BATCHES: usize = 20;
const MAX_BATCH_SIZE: usize = 10;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const HAWK_CONNECTION_PARALLELISM: usize = 1;
const MAX_DELETIONS_PER_BATCH: usize = 3;
const MAX_RESET_UPDATES_PER_BATCH: usize = 3;

const HNSW_EF_CONSTR: usize = 320;
const HNSW_M: usize = 256;
const HNSW_EF_SEARCH: usize = 256;
// Must match HawkActor's LINEAR_SCAN_MAX_GRAPH_LAYER (hawk_main.rs).
const LINEAR_SCAN_MAX_GRAPH_LAYER: usize = 1;
const HNSW_LAYER_DENSITY: usize = 20;

const UNIQUENESS_REQUEST_PARALLELISM: usize = 4;
const UNIQUENESS_TEST_RNG_SEED: u64 = 0x5ca1ab1e;

fn install_tracing() {
    // try_init: both tests live in the same binary and each call this.
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

    // Bootstrap graph must match the actor's runtime searcher
    // (HawkActor::from_cli_with_graph_and_store) or entry points won't align.
    let mut searcher = HnswSearcher::new_linear_scan(
        args.hnsw_param_ef_constr,
        args.hnsw_param_ef_search,
        args.hnsw_param_m,
        LINEAR_SCAN_MAX_GRAPH_LAYER,
    );
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

/// Run an async test on a 64 MB stack to avoid platform-dependent stack
/// overflows in release mode.
fn run_async_on_big_stack<F, Fut>(name: &'static str, main: F) -> Result<()>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = Result<()>>,
{
    std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            rt.block_on(main())
        })
        .expect("failed to spawn test thread")
        .join()
        .expect("test thread panicked")
}

#[ignore = "Takes long time to run, in CI this is selected in a separate step"]
#[test]
fn e2e_test() -> Result<()> {
    run_async_on_big_stack("e2e_test", e2e_test_async)
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
// Drives the actor through five batches built on a single canonical iris X.
// Each request is a `Variant` carrying its own `purpose` (narrative role
// surfaced in failure messages) and `expect` (the `Expectation` it must
// satisfy). `check_expectations` walks every (variant, observation) pair
// after each batch, collects all violations, and panics once with a
// self-contained report.
//
//   Batch 1: seed X + 61 rotation/mirror variants → slot 0 UniqueInsert; every other slot intra-batch-blocked (mirror variants too — see note below)
//   Batch 2: same 62 variants resubmitted        → DbMatchAt / MirrorAttack (X now in DB; this is where the mirror flag actually fires)
//   Batch 3: [X' reauth, Y=X+30, Z=X-15]       → reauth OK; Y blocked intra-batch; Z matches original X (reauth not yet applied)
//   Batch 4: [Y=X+30, Z=X-15] (post-reauth)    → Y matches X' in DB; Z is out of rotation window → UniqueInsert
//   Batch 5: [delete(seed_idx), uniqueness(X')] → deletion acts before search; X' re-enrolls at fresh serial
// ======================================================================

#[derive(Clone, Debug)]
enum Expectation {
    /// Fresh insert; `merged_results` must be ≥ `min_index`.
    UniqueInsert { min_index: u32 },
    /// Matched a specific stored iris.
    DbMatchAt { expected_match_index: u32 },
    /// Blocked by an earlier mutating slot in the same batch.
    IntraBatchBlockedBy { earlier_request_id: String },
    /// `full_face_mirror_attack_detected` raised, mirror match at the given index.
    MirrorAttackDetected { expected_mirror_match_index: u32 },
    /// Reauth target updated.
    ReauthSucceeds { target_index: u32 },
}

impl Expectation {
    /// Source file printed in failure messages, to orient debugging.
    fn diagnosis(&self) -> &'static str {
        use Expectation::*;
        match self {
            UniqueInsert { .. } => "hawk_main::HawkResult::merged_results",
            DbMatchAt { .. } => "hawk_main/matching.rs: uniqueness with any match → NoMutation",
            IntraBatchBlockedBy { .. } => {
                "hawk_main/matching.rs + intra_batch.rs: later uniqueness is blocked iff earlier slot's decision is a mutation"
            }
            MirrorAttackDetected { .. } => {
                "hawk_main: full_face_mirror_attack_detected = normal empty && mirror non-empty"
            }
            ReauthSucceeds { .. } => {
                "hawk_main/matching.rs: Reauth → ReauthUpdate; mutation applied in handle_mutations after searches"
            }
        }
    }

    fn describe(&self) -> String {
        use Expectation::*;
        match self {
            UniqueInsert { min_index } => {
                format!("UniqueInsert (merged_results ≥ {min_index}, no match, no reauth)")
            }
            DbMatchAt {
                expected_match_index,
            } => format!("DbMatch at index {expected_match_index} (match_ids contains it)"),
            IntraBatchBlockedBy { earlier_request_id } => format!(
                "IntraBatchBlocked — match_ids empty, matched_batch_request_ids contains {earlier_request_id}"
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

/// Rotation+mirror variant of (x_left, x_right). For `mirror=true` the eyes
/// swap sides AND each is mirrored — the canonical full-face-mirror-attack
/// construction, matching Mirror-orientation search expectations.
fn build_iris_pair(
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

fn uniqueness_variant(
    x_left: &IrisCode,
    x_right: &IrisCode,
    rotation: isize,
    mirror: bool,
    purpose: String,
    expect: Expectation,
) -> Variant {
    let (l, r) = build_iris_pair(x_left, x_right, rotation, mirror);
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

fn reauth_variant(
    x_left: &IrisCode,
    x_right: &IrisCode,
    rotation: isize,
    mirror: bool,
    target_index: u32,
    purpose: String,
) -> Variant {
    let (l, r) = build_iris_pair(x_left, x_right, rotation, mirror);
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

/// All 62 rotation/mirror variants of X, to be submitted after X is already
/// persisted at `seed_db_index`. Identity slot (rot=0, non-mirror) is pushed
/// first so the centered raw iris for intra-batch comparisons is X itself.
fn build_batch2_variants(
    x_left: &IrisCode,
    x_right: &IrisCode,
    seed_db_index: u32,
) -> Vec<Variant> {
    let mut variants = Vec::with_capacity(62);
    let mut push = |rotation: isize, mirror: bool| {
        let (purpose, expect) = if mirror {
            (
                format!("batch2: rot={rotation:+} mirror — mirror-attack vs seed"),
                Expectation::MirrorAttackDetected {
                    expected_mirror_match_index: seed_db_index,
                },
            )
        } else {
            (
                format!("batch2: rot={rotation:+} non-mirror — matches seed at {seed_db_index}"),
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

    // Deletions are applied before any search within the batch.
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

/// Require byte-identical results on the fields we inspect. Returns party 0.
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

/// Returns the list of problems; empty means the expectation held.
fn evaluate_expectation(obs: &Observation) -> Vec<String> {
    use Expectation::*;
    let mut p: Vec<String> = Vec::new();
    let mut chk = |name: &str, got: bool, want: bool| {
        if got != want {
            p.push(format!("{name}={got} (wanted {want})"));
        }
    };
    match &obs.expect {
        UniqueInsert { min_index } => {
            chk("was_match", obs.was_match, false);
            chk("mirror_attack", obs.mirror_attack, false);
            chk("successful_reauth", obs.successful_reauth, false);
            if obs.merged_results < *min_index {
                p.push(format!(
                    "merged_results={} < min_index={min_index}",
                    obs.merged_results
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
            chk("was_match", obs.was_match, true);
            chk("mirror_attack", obs.mirror_attack, false);
            chk("successful_reauth", obs.successful_reauth, false);
            if obs.merged_results != *expected_match_index {
                p.push(format!(
                    "merged_results={} (wanted {expected_match_index})",
                    obs.merged_results
                ));
            }
            if !obs.match_ids.contains(expected_match_index) {
                p.push(format!(
                    "match_ids={:?} missing expected {expected_match_index}",
                    obs.match_ids
                ));
            }
        }
        IntraBatchBlockedBy { earlier_request_id } => {
            chk("was_match", obs.was_match, true);
            chk("mirror_attack", obs.mirror_attack, false);
            chk("successful_reauth", obs.successful_reauth, false);
            if !obs.match_ids.is_empty() {
                p.push(format!(
                    "match_ids={:?} (wanted empty — no direct DB match expected)",
                    obs.match_ids
                ));
            }
            if !obs.matched_batch_request_ids.contains(earlier_request_id) {
                p.push(format!(
                    "matched_batch_request_ids={:?} missing expected {earlier_request_id}",
                    obs.matched_batch_request_ids
                ));
            }
        }
        MirrorAttackDetected {
            expected_mirror_match_index,
        } => {
            chk("mirror_attack", obs.mirror_attack, true);
            chk("successful_reauth", obs.successful_reauth, false);
            if !obs.match_ids.is_empty() {
                p.push(format!(
                    "normal match_ids={:?} non-empty (wanted empty)",
                    obs.match_ids
                ));
            }
            if !obs
                .full_face_mirror_match_ids
                .contains(expected_mirror_match_index)
            {
                p.push(format!(
                    "full_face_mirror_match_ids={:?} missing expected {expected_mirror_match_index}",
                    obs.full_face_mirror_match_ids
                ));
            }
        }
        ReauthSucceeds { target_index } => {
            chk("successful_reauth", obs.successful_reauth, true);
            chk("mirror_attack", obs.mirror_attack, false);
            if obs.merged_results != *target_index {
                p.push(format!(
                    "merged_results={} (wanted reauth target {target_index})",
                    obs.merged_results
                ));
            }
        }
    }
    p
}

/// Submit `variants` + `deletions` as a single batch and return the per-slot
/// observations; panics if any expectation is violated.
async fn submit_and_check(
    label: &str,
    variants: &[Variant],
    deletions: &[u32],
    h0: &mut HawkHandle,
    h1: &mut HawkHandle,
    h2: &mut HawkHandle,
    rng: &mut StdRng,
) -> Result<Vec<Observation>> {
    let results = submit_variants_batch(variants, deletions, h0, h1, h2, rng).await?;
    Ok(check_expectations(label, variants, &results))
}

/// Assert every variant's expectation. On any failure, panic once with a
/// full per-slot table so the run is self-diagnosing.
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

    tracing::info!("=== {} — per-slot observations ===", batch_label);
    for obs in &observations {
        tracing::info!("{}", format_observation_row(obs));
    }

    let failures: Vec<String> = observations
        .iter()
        .filter_map(|obs| {
            let problems = evaluate_expectation(obs);
            if problems.is_empty() {
                return None;
            }
            Some(format!(
                "  slot {:>2} [{}]\n    purpose : {}\n    expected: {}\n    problems: {}\n    diagnosis: {}",
                obs.slot,
                obs.label,
                obs.purpose,
                obs.expect.describe(),
                problems.join("; "),
                obs.expect.diagnosis(),
            ))
        })
        .collect();

    if !failures.is_empty() {
        let mut report = format!(
            "\n\n=== {} FAILED: {} expectation(s) violated ===\n{}\n\n--- Full result table for this batch ---\n",
            batch_label,
            failures.len(),
            failures.join("\n"),
        );
        for obs in &observations {
            report.push_str(&format_observation_row(obs));
            report.push('\n');
        }
        panic!("{}", report);
    }

    observations
}

fn format_observation_row(obs: &Observation) -> String {
    format!(
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
    )
}

#[ignore = "Takes long time to run, in CI this is selected in a separate step"]
#[test]
fn e2e_uniqueness_test() -> Result<()> {
    run_async_on_big_stack("e2e_uniqueness_test", e2e_uniqueness_test_async)
}

async fn e2e_uniqueness_test_async() -> Result<()> {
    install_tracing();

    let test_db = generate_full_test_db(DB_SIZE, DB_RNG_SEED, false);
    let db_left = test_db.plain_dbs(0);
    let db_right = test_db.plain_dbs(1);

    // Distinct ports from e2e_test to avoid TIME_WAIT conflicts when both run.
    let addresses = ["127.0.0.1:16300", "127.0.0.1:16400", "127.0.0.1:16500"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let (mut h0, mut h1, mut h2) =
        spawn_three_hawk_nodes(db_left, db_right, addresses, UNIQUENESS_REQUEST_PARALLELISM)
            .await?;

    let mut rng = StdRng::seed_from_u64(UNIQUENESS_TEST_RNG_SEED);

    let x_left = IrisCode::random_rng(&mut rng);
    let x_right = IrisCode::random_rng(&mut rng);

    // The X'/Y/Z family of variants used across batches 3–5, defined as
    // rotations of the seed X. The rotation window is ±15, so |X'−X|=15 sits
    // at the edge, |Y−X|=30 is out of window, and |Z−X|=15 is also edge.
    let x_prime_rot: isize = 15; // X' = X rotated +15 (reauth's new iris)
    let y_rot: isize = 30; // Y  = X rotated +30
    let z_rot: isize = -15; // Z  = X rotated −15

    // Batch 1: seed X + 61 rotation/mirror variants. Slot 0 inserts; every
    // other slot is intra-batch-blocked by slot 0. Mirror variants are also
    // blocked here (not flagged as mirror-attacks): the CPU actor only sets
    // `full_face_mirror_attack_detected` on a DB mirror hit, not on an
    // intra-batch one — so with X not yet in the DB, mirror slots look the
    // same as non-mirror slots. Batch 2 resubmits these same variants once
    // X is in the DB and exercises the actual mirror-attack path.
    let seed = uniqueness_variant(
        &x_left,
        &x_right,
        0,
        false,
        "batch1 slot 0: seed X — UniqueInsert".to_string(),
        Expectation::UniqueInsert {
            min_index: DB_SIZE as u32,
        },
    );
    let seed_request_id = seed.request_id.clone();
    let mut batch1 = vec![seed];
    let push_blocked = |rotation: isize, mirror: bool| {
        let blocked = Expectation::IntraBatchBlockedBy {
            earlier_request_id: seed_request_id.clone(),
        };
        let purpose =
            format!("batch1: rot={rotation:+} mirror={mirror} — blocked intra-batch by slot 0");
        uniqueness_variant(&x_left, &x_right, rotation, mirror, purpose, blocked)
    };
    for rotation in -15isize..=15 {
        for mirror in [false, true] {
            if rotation == 0 && !mirror {
                continue;
            }
            batch1.push(push_blocked(rotation, mirror));
        }
    }
    assert_eq!(batch1.len(), 62);
    let obs = submit_and_check(
        "batch1 (seed + 61 rotation/mirror variants)",
        &batch1,
        &[],
        &mut h0,
        &mut h1,
        &mut h2,
        &mut rng,
    )
    .await?;
    // The slot at this index is the "primary" slot for the rest of the
    // test: batch 3's reauth overwrites the iris stored here, batch 5's
    // delete replaces it with a dummy, but the serial ID itself is stable.
    let seed_db_index = obs[0].merged_results;

    // Batch 2: every rotation/mirror variant matches the stored seed.
    let batch2 = build_batch2_variants(&x_left, &x_right, seed_db_index);
    assert_eq!(batch2.len(), 62);
    submit_and_check(
        "batch2 (62 variants)",
        &batch2,
        &[],
        &mut h0,
        &mut h1,
        &mut h2,
        &mut rng,
    )
    .await?;

    // Batch 3: reauth X' = X+15, then Y = X+30 and Z = X-15 as uniqueness.
    // Reauth's store mutation lands only after the batch's searches, so Z
    // still sees the original X; Y is blocked intra-batch by the reauth slot.
    let x_prime = reauth_variant(
        &x_left,
        &x_right,
        x_prime_rot,
        false,
        seed_db_index,
        format!("batch3 slot 0: reauth X'=X+15 target={seed_db_index}"),
    );
    let y_b3 = uniqueness_variant(
        &x_left,
        &x_right,
        y_rot,
        false,
        "batch3 slot 1: Y=X+30 — blocked intra-batch by X' reauth".to_string(),
        Expectation::IntraBatchBlockedBy {
            earlier_request_id: x_prime.request_id.clone(),
        },
    );
    let z_b3 = uniqueness_variant(
        &x_left,
        &x_right,
        z_rot,
        false,
        format!(
            "batch3 slot 2: Z=X-15 — matches original X at {seed_db_index} (reauth not yet applied)"
        ),
        Expectation::DbMatchAt {
            expected_match_index: seed_db_index,
        },
    );
    let batch3 = vec![x_prime, y_b3, z_b3];
    submit_and_check(
        "batch3 (reauth + Y + Z)",
        &batch3,
        &[],
        &mut h0,
        &mut h1,
        &mut h2,
        &mut rng,
    )
    .await?;

    // Batch 4: same (Y, Z) irises as batch 3, but the DB now holds X' at
    // seed_idx. Y matches X'; Z is out of rotation window vs X' and inserts
    // fresh. Proves reauth displaced X.
    let y_b4 = uniqueness_variant(
        &x_left,
        &x_right,
        y_rot,
        false,
        format!("batch4 slot 0: Y=X+30 — matches X' in DB at {seed_db_index}"),
        Expectation::DbMatchAt {
            expected_match_index: seed_db_index,
        },
    );
    let z_b4 = uniqueness_variant(
        &x_left,
        &x_right,
        z_rot,
        false,
        format!("batch4 slot 1: Z=X-15 — UniqueInsert at serial > {seed_db_index}"),
        Expectation::UniqueInsert {
            min_index: seed_db_index + 1,
        },
    );
    let batch4 = vec![y_b4, z_b4];
    let batch4_obs = submit_and_check(
        "batch4 (post-reauth Y + Z)",
        &batch4,
        &[],
        &mut h0,
        &mut h1,
        &mut h2,
        &mut rng,
    )
    .await?;
    let z_db_index = batch4_obs[1].merged_results;

    // Batch 5: same-batch delete of seed_idx + uniqueness of X'. Deletion
    // acts before search, so X' re-enrolls at a fresh serial.
    let x_prime_reenroll = uniqueness_variant(
        &x_left,
        &x_right,
        x_prime_rot,
        false,
        format!("batch5: uniqueness X'=X+15 after same-batch delete({seed_db_index})"),
        // Serials are monotonic and never reused; Z at z_db_index was the
        // last fresh insert, so anything new here must land above it.
        Expectation::UniqueInsert {
            min_index: z_db_index + 1,
        },
    );
    submit_and_check(
        "batch5 (delete + uniqueness re-enroll)",
        &[x_prime_reenroll],
        &[seed_db_index],
        &mut h0,
        &mut h1,
        &mut h2,
        &mut rng,
    )
    .await?;

    drop(h0);
    drop(h1);
    drop(h2);

    // TODO: ATM we have no real way to wait for the actors to finish, so just
    // sleep a bit.
    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}
