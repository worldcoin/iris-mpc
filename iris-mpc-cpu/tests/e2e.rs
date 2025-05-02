use eyre::Result;
use iris_mpc_common::{
    iris_db::db::IrisDB,
    test::{generate_full_test_db, TestCase, TestCaseGenerator},
};
use iris_mpc_cpu::{
    execution::hawk_main::{HawkActor, HawkArgs, HawkHandle, VectorId},
    hawkers::{
        aby3::aby3_store::{Aby3Store, SharedIrises},
        plaintext_store::PlaintextStore,
    },
    hnsw::{graph::layered_graph::migrate, GraphMem, HnswParams},
    protocol::shared_iris::GaloisRingSharedIris,
};
use rand::{rngs::StdRng, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const DB_SIZE: usize = 1000;
const DB_RNG_SEED: u64 = 0xdeadbeef;
const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
const NUM_BATCHES: usize = 5;
const MAX_BATCH_SIZE: usize = 5;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const HAWK_CONNECTION_PARALLELISM: usize = 1;
const MAX_DELETIONS_PER_BATCH: usize = 0; // TODO: set back to 10 or so once deletions are supported
const MAX_RESET_UPDATES_PER_BATCH: usize = 0; // TODO: set back to 10 or so once reset is supported

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
    params: &HnswParams,
) -> Result<([GraphMem<Aby3Store>; 2], [SharedIrises; 2])> {
    let mut rng = StdRng::seed_from_u64(DB_RNG_SEED);
    let mut left_store = PlaintextStore::create_random_store_with_db(left_db.db.clone()).await?;
    let mut right_store = PlaintextStore::create_random_store_with_db(right_db.db.clone()).await?;
    let left_graph = left_store.create_graph(&mut rng, DB_SIZE, params).await?;
    let right_graph = right_store.create_graph(&mut rng, DB_SIZE, params).await?;

    let left_mpc_graph: GraphMem<Aby3Store> = migrate(left_graph, |v| v);
    let right_mpc_graph: GraphMem<Aby3Store> = migrate(right_graph, |v| v);

    let mut left_shared_irises = HashMap::new();
    let mut right_shared_irises = HashMap::new();

    for (vector_id, iris) in left_store.points.iter().enumerate() {
        let vector_id: VectorId = VectorId::from_0_index(vector_id as u32);
        let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, (**iris).clone());
        left_shared_irises.insert(vector_id, Arc::new(shares[player_index].clone()));
    }
    for (vector_id, iris) in right_store.points.iter().enumerate() {
        let vector_id: VectorId = VectorId::from_0_index(vector_id as u32);
        let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, (**iris).clone());
        right_shared_irises.insert(vector_id, Arc::new(shares[player_index].clone()));
    }

    let left_iris_store = SharedIrises::new(left_shared_irises);
    let right_iris_store = SharedIrises::new(right_shared_irises);

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

    // TODO: replace with: `HnswParams::new(args.hnsw_ef_search, args.hnsw_ef_constr, args.hnsw_M)`
    let params = HnswParams::new(320, 256, 256);
    let (graph, iris_store) =
        create_graph_from_plain_dbs(args.party_index, left_db, right_db, &params).await?;
    let hawk_actor = HawkActor::from_cli_with_graph_and_store(args, graph, iris_store).await?;

    let handle = HawkHandle::new(hawk_actor).await?;

    Ok(handle)
}

#[ignore = "Expected to fail for now"]
#[tokio::test]
async fn e2e_test() -> Result<()> {
    install_tracing();

    let test_db = generate_full_test_db(DB_SIZE, DB_RNG_SEED);
    let db_left = test_db.plain_dbs(0);
    let db_right = test_db.plain_dbs(1);

    let addresses = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let args0 = HawkArgs {
        party_index: 0,
        addresses,
        request_parallelism: HAWK_REQUEST_PARALLELISM,
        connection_parallelism: HAWK_CONNECTION_PARALLELISM,
        hnsw_prng_seed: None,
        disable_persistence: false,
        match_distances_buffer_size: 64,
        n_buckets: 10,
    };
    let args1 = HawkArgs {
        party_index: 1,
        ..args0.clone()
    };
    let args2 = HawkArgs {
        party_index: 2,
        ..args0.clone()
    };
    let (handle0, handle1, handle2) = tokio::join!(
        start_hawk_node(&args0, db_left, db_right),
        start_hawk_node(&args1, db_left, db_right),
        start_hawk_node(&args2, db_left, db_right),
    );
    let mut handle0 = handle0?;
    let mut handle1 = handle1?;
    let mut handle2 = handle2?;

    let mut test_case_generator = TestCaseGenerator::new_with_db(test_db, INTERNAL_RNG_SEED, true);

    // Disable test cases that are not yet supported
    // TODO: enable these once supported

    test_case_generator.disable_test_case(TestCase::FullFaceMirrorAttack);
    test_case_generator.disable_test_case(TestCase::MatchSkipPersistence);
    test_case_generator.disable_test_case(TestCase::NonMatchSkipPersistence);
    test_case_generator.disable_test_case(TestCase::CloseToThreshold);
    test_case_generator.disable_test_case(TestCase::PreviouslyDeleted);
    test_case_generator.disable_test_case(TestCase::WithOrRuleSet);
    test_case_generator.disable_test_case(TestCase::ReauthMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthNonMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthOrRuleMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthOrRuleNonMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ResetCheckMatch);
    test_case_generator.disable_test_case(TestCase::ResetCheckNonMatch);

    // TODO: enable this once supported
    // test_case_generator.enable_bucket_statistic_checks(
    //     N_BUCKETS,
    //     num_devices,
    //     MATCH_DISTANCES_BUFFER_SIZE,
    // );

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
