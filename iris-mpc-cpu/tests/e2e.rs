use eyre::Result;
use iris_mpc_common::{
    iris_db::db::IrisDB,
    test::{TestCase, TestCaseGenerator},
};
use iris_mpc_cpu::{
    execution::hawk_main::{HawkActor, HawkArgs, HawkHandle, VectorId},
    hawkers::{
        aby3::{
            aby3_store::{Aby3Store, SharedIrisesRef},
            test_utils::get_trivial_share,
        },
        plaintext_store::PlaintextStore,
    },
    hnsw::{graph::layered_graph::migrate, GraphMem},
    protocol::shared_iris::GaloisRingSharedIris,
    shares::share::DistanceShare,
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
const MAX_DELETIONS_PER_BATCH: usize = 0; // TODO: set back to 10 or so once deletions are supported

fn install_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
}

async fn create_graph_from_plain_db(
    player_index: usize,
    db: &IrisDB,
) -> eyre::Result<(GraphMem<Aby3Store>, [SharedIrisesRef; 2])> {
    let mut rng = StdRng::seed_from_u64(DB_RNG_SEED);
    let mut store = PlaintextStore::create_random_store_with_db(db.db.clone()).await?;
    let graph = store.create_graph(&mut rng, DB_SIZE).await?;

    let mpc_graph: GraphMem<Aby3Store> = migrate(
        graph,
        |v| v.into(),
        |(c, m)| {
            DistanceShare::new(
                get_trivial_share(c, player_index),
                get_trivial_share(m, player_index),
            )
        },
    );

    let mut shared_irises = HashMap::new();

    for (vector_id, iris) in store.points.iter().enumerate() {
        let vector_id: VectorId = VectorId::from_serial_id(vector_id as u32);
        let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, iris.data.0.clone());
        shared_irises.insert(vector_id, Arc::new(shares[player_index].clone()));
    }

    let aby3_store = SharedIrisesRef::new(shared_irises);

    Ok((mpc_graph, [(); 2].map(|_| aby3_store.clone())))
}

async fn start_hawk_node(args: &HawkArgs, db: &mut IrisDB) -> Result<HawkHandle> {
    tracing::info!("🦅 Starting Hawk node {}", args.party_index);

    let (mpc_graph, aby3_store) = create_graph_from_plain_db(args.party_index, db).await?;
    let hawk_actor = HawkActor::from_cli_with_graph_and_store(args, mpc_graph, aby3_store).await?;

    let handle = HawkHandle::new(hawk_actor, HAWK_REQUEST_PARALLELISM).await?;

    Ok(handle)
}

#[ignore = "Expected to fail for now"]
#[tokio::test]
async fn e2e_test() -> Result<()> {
    install_tracing();

    let mut db = IrisDB::new_random_rng(DB_SIZE, &mut StdRng::seed_from_u64(DB_RNG_SEED));
    let mut db0 = db.clone();
    let mut db1 = db.clone();
    let mut db2 = db.clone();

    let addresses = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let args0 = HawkArgs {
        party_index: 0,
        addresses,
        request_parallelism: HAWK_REQUEST_PARALLELISM,
        disable_persistence: false,
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
        start_hawk_node(&args0, &mut db0),
        start_hawk_node(&args1, &mut db1),
        start_hawk_node(&args2, &mut db2)
    );
    let mut handle0 = handle0?;
    let mut handle1 = handle1?;
    let mut handle2 = handle2?;

    let mut test_case_generator = TestCaseGenerator::new_with_db(&mut db, INTERNAL_RNG_SEED, true);

    // Disable test cases that are not yet supported
    // TODO: enable these once supported

    test_case_generator.disable_test_case(TestCase::MatchSkipPersistence);
    test_case_generator.disable_test_case(TestCase::NonMatchSkipPersistence);
    test_case_generator.disable_test_case(TestCase::CloseToThreshold);
    test_case_generator.disable_test_case(TestCase::PreviouslyDeleted);
    test_case_generator.disable_test_case(TestCase::WithOrRuleSet);
    test_case_generator.disable_test_case(TestCase::ReauthMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthNonMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthOrRuleMatchingTarget);
    test_case_generator.disable_test_case(TestCase::ReauthOrRuleNonMatchingTarget);

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
