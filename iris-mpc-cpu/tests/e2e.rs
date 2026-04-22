use eyre::Result;
use iris_mpc_common::{
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
    iris_db::{db::IrisDB, iris::IrisCode},
    job::{BatchQuery, JobSubmissionHandle, ServerJobResult},
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
    hnsw::{GraphMem, HnswSearcher},
    protocol::shared_iris::GaloisRingSharedIris,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

const DB_SIZE: usize = 1000;
const DB_RNG_SEED: u64 = 0xdeadbeef;
const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
const NUM_BATCHES: usize = 5;
const MAX_BATCH_SIZE: usize = 5;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const HAWK_CONNECTION_PARALLELISM: usize = 1;
const MAX_DELETIONS_PER_BATCH: usize = 0; // TODO: set back to 10 or so once deletions are supported
const MAX_RESET_UPDATES_PER_BATCH: usize = 0; // TODO: set back to 10 or so once reset is supported

const HNSW_EF_CONSTR: usize = 320;
const HNSW_M: usize = 256;
const HNSW_EF_SEARCH: usize = 256;

// Uniqueness test constants
const UNIQUENESS_REQUEST_PARALLELISM: usize = 4;
const UNIQUENESS_TEST_RNG_SEED: u64 = 0xbeefdead;

fn install_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
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

    let searcher = HnswSearcher::new_standard(
        args.hnsw_param_ef_constr,
        args.hnsw_param_ef_search,
        args.hnsw_param_m,
    );

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
        hnsw_layer_density: None,
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
// ======================================================================

#[derive(Clone)]
struct Variant {
    rotation: isize,
    mirror: bool,
    template: E2ETemplate,
    request_id: String,
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

fn build_variants(x_left: &IrisCode, x_right: &IrisCode) -> Vec<Variant> {
    let mut variants = Vec::with_capacity(62);
    // Put the identity combo (rotation=0, mirror=false) first so the batch's
    // canonical non-rotated X sits at slot 0. Intra-batch match/mirror-attack
    // detection compares later slots' rotated queries against slot 0's
    // *centered* raw query — alignment only works cleanly when that centered
    // raw query is the unrotated X itself.
    let mut push = |rotation: isize, mirror: bool| {
        let (l, r) = build_mirror_variant_pair(x_left, x_right, rotation, mirror);
        variants.push(Variant {
            rotation,
            mirror,
            template: E2ETemplate::new(l, r),
            request_id: Uuid::new_v4().to_string(),
        });
    };
    push(0, false);
    for rotation in -15isize..=15 {
        for mirror in [false, true] {
            if rotation == 0 && !mirror {
                continue; // already pushed as slot 0
            }
            push(rotation, mirror);
        }
    }
    variants
}

async fn submit_variants_batch(
    variants: &[Variant],
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

    for v in variants {
        let shared = v.template.to_shared_template(true, rng);
        for (party, batch) in batches.iter_mut().enumerate() {
            prepare_batch(
                batch,
                true, // is_valid
                v.request_id.clone(),
                party,
                shared.clone(),
                vec![], // or_rule_indices
                None,   // maybe_reauth_target_index
                false,  // skip_persistence
                UNIQUENESS_MESSAGE_TYPE.to_string(),
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
/// every field we care about in this test. Returns the party-0 result for
/// subsequent inspection.
fn assert_three_parties_agree(
    results: &[ServerJobResult<HawkMutation>; 3],
) -> &ServerJobResult<HawkMutation> {
    let [r0, r1, r2] = results;
    assert_eq!(r0.request_ids, r1.request_ids);
    assert_eq!(r0.request_ids, r2.request_ids);
    assert_eq!(r0.matches, r1.matches);
    assert_eq!(r0.matches, r2.matches);
    assert_eq!(r0.merged_results, r1.merged_results);
    assert_eq!(r0.merged_results, r2.merged_results);
    assert_eq!(
        r0.full_face_mirror_attack_detected,
        r1.full_face_mirror_attack_detected
    );
    assert_eq!(
        r0.full_face_mirror_attack_detected,
        r2.full_face_mirror_attack_detected
    );
    assert_eq!(r0.matched_batch_request_ids, r1.matched_batch_request_ids);
    assert_eq!(r0.matched_batch_request_ids, r2.matched_batch_request_ids);
    r0
}

fn assert_seed_inserted(
    seed_request_id: &str,
    results: &[ServerJobResult<HawkMutation>; 3],
) -> Result<u32> {
    let r = assert_three_parties_agree(results);
    assert_eq!(r.request_ids.len(), 1, "seed batch should have 1 result");
    assert_eq!(r.request_ids[0], seed_request_id);
    assert!(
        !r.matches[0],
        "seed X should not match anything in the initial DB"
    );
    assert!(
        !r.full_face_mirror_attack_detected[0],
        "seed X should not be reported as a mirror attack"
    );
    let db_index = r.merged_results[0];
    assert!(
        db_index >= DB_SIZE as u32,
        "seed insertion index should be beyond initial DB (got {}, DB_SIZE={})",
        db_index,
        DB_SIZE,
    );
    Ok(db_index)
}

/// Observed outcome of a single variant's uniqueness query against a DB that
/// already contains the seed X. Used for per-variant diagnostics.
#[derive(Debug)]
#[allow(dead_code)]
struct SingletonObservation {
    label: String,
    was_match: bool,
    mirror_attack: bool,
    merged_results: u32,
    match_ids: Vec<u32>,
    full_face_mirror_match_ids: Vec<u32>,
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

    // Use ports distinct from e2e_test to avoid TIME_WAIT conflicts when both
    // tests run back-to-back in the same CI step.
    let addresses = ["127.0.0.1:16300", "127.0.0.1:16400", "127.0.0.1:16500"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let (mut h0, mut h1, mut h2) =
        spawn_three_hawk_nodes(db_left, db_right, addresses, UNIQUENESS_REQUEST_PARALLELISM)
            .await?;

    let mut rng = StdRng::seed_from_u64(UNIQUENESS_TEST_RNG_SEED);

    // Generate a fresh seed iris X. With 12800-bit random codes, collision with
    // the 1000-entry plaintext DB is astronomically unlikely.
    let x_left = IrisCode::random_rng(&mut rng);
    let x_right = IrisCode::random_rng(&mut rng);

    // Build all 62 variants: rotation ∈ [-15, +15] × mirror ∈ {false, true}.
    // Same rotation applied to both eyes. (rotation=0, mirror=false) is X itself.
    let variants = build_variants(&x_left, &x_right);
    assert_eq!(variants.len(), 62);

    // --- Batch 1: X alone. Seeds the DB with the canonical iris. ---
    let seed_variant = Variant {
        rotation: 0,
        mirror: false,
        template: E2ETemplate::new(x_left.clone(), x_right.clone()),
        request_id: Uuid::new_v4().to_string(),
    };
    tracing::info!("Batch 1: submitting seed iris X alone");
    let results = submit_variants_batch(
        std::slice::from_ref(&seed_variant),
        &mut h0,
        &mut h1,
        &mut h2,
        &mut rng,
    )
    .await?;
    let seed_db_index = assert_seed_inserted(&seed_variant.request_id, &results)?;
    tracing::info!(
        "Batch 1 OK, seed iris persisted at DB index {}",
        seed_db_index
    );

    // --- Batch 2: all 62 variants in a single batch.
    // With X already persisted, each variant's DB search has a target to hit.
    // Submitting them in one batch avoids 62 rounds of setup/tear-down but
    // introduces intra-batch matches between variants — see the assertion
    // comments below for how we handle that.
    tracing::info!(
        "Batch 2: submitting {} variants in one batch",
        variants.len()
    );
    let results =
        submit_variants_batch(variants.as_slice(), &mut h0, &mut h1, &mut h2, &mut rng).await?;
    let agreed = assert_three_parties_agree(&results);
    assert_eq!(agreed.request_ids.len(), variants.len());

    // Re-associate each variant with its slot in the batch result.
    let mut observations: Vec<SingletonObservation> = Vec::with_capacity(variants.len());
    for v in &variants {
        let slot = agreed
            .request_ids
            .iter()
            .position(|r| r == &v.request_id)
            .expect("variant's request_id must appear in batch result");
        observations.push(SingletonObservation {
            label: v.label(),
            was_match: agreed.matches[slot],
            mirror_attack: agreed.full_face_mirror_attack_detected[slot],
            merged_results: agreed.merged_results[slot],
            match_ids: agreed.match_ids[slot].clone(),
            full_face_mirror_match_ids: agreed.full_face_mirror_match_ids[slot].clone(),
        });
    }

    // Summarize per (rotation, mirror). Output two rows per rotation for easy scanning.
    tracing::info!(
        "=== Singleton outcomes (seed at DB idx {}) ===",
        seed_db_index
    );
    for obs in &observations {
        tracing::info!(
            "  {:<22} was_match={} mirror_attack={} merged_results={} match_ids={:?} mirror_match_ids={:?}",
            obs.label,
            obs.was_match,
            obs.mirror_attack,
            obs.merged_results,
            obs.match_ids,
            obs.full_face_mirror_match_ids,
        );
    }

    // Uniqueness invariant: after X is in the DB, no rotation/mirror variant
    // should be accepted as a brand-new unique iris.
    let leaked: Vec<&SingletonObservation> = observations
        .iter()
        .filter(|o| !o.was_match && !o.mirror_attack)
        .collect();
    assert!(
        leaked.is_empty(),
        "variants that bypassed uniqueness as new inserts: {:?}",
        leaked.iter().map(|o| &o.label).collect::<Vec<_>>(),
    );

    // Per-category expectations:
    //   non-mirror variant → must match the seed at its DB index.
    //   mirror variant    → must be flagged as a full-face mirror attack.
    for (v, obs) in variants.iter().zip(observations.iter()) {
        if v.mirror {
            assert!(
                obs.mirror_attack,
                "mirror variant {} should be flagged as mirror attack; got {:?}",
                v.label(),
                obs,
            );
        } else {
            assert!(
                obs.was_match && !obs.mirror_attack,
                "non-mirror variant {} should match normally; got {:?}",
                v.label(),
                obs,
            );
            assert_eq!(
                obs.merged_results,
                seed_db_index,
                "non-mirror variant {} should resolve to seed DB index {}",
                v.label(),
                seed_db_index,
            );
        }
    }

    tracing::info!("e2e_uniqueness_test: seed + 62-variant batch passed");

    drop(h0);
    drop(h1);
    drop(h2);

    // TODO: ATM we have no real way to wait for the actors to finish, so just sleep
    // a bit for now.
    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}
