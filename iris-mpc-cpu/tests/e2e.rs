use eyre::Result;
use iris_mpc_common::{
    iris_db::{db::IrisDB, iris::IrisCode},
    test::{generate_full_test_db, TestCaseGenerator},
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{HawkActor, HawkArgs, HawkHandle},
    hawkers::{
        aby3::aby3_store::{Aby3SharedIrises, Aby3Store, Aby3VectorRef},
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

    let mut left_store = PlaintextStore::with_storage(left_storage);
    let mut right_store = PlaintextStore::with_storage(right_storage);

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

    let left_iris_store = Aby3Store::new_storage(Some(left_shared_irises));
    let right_iris_store = Aby3Store::new_storage(Some(right_shared_irises));

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
        args.hnsw_param_M,
    );

    let (graph, iris_store) =
        create_graph_from_plain_dbs(args.party_index, left_db, right_db, &searcher).await?;
    let hawk_actor =
        HawkActor::from_cli_with_graph_and_store(args, CancellationToken::new(), graph, iris_store)
            .await?;

    let handle = HawkHandle::new(hawk_actor).await?;

    Ok(handle)
}

#[ignore = "Takes long time to run, in CI this is selected in a separate step"]
#[tokio::test]
async fn e2e_test() -> Result<()> {
    install_tracing();

    let test_db = generate_full_test_db(DB_SIZE, DB_RNG_SEED, false);
    let db_left = test_db.plain_dbs(0);
    let db_right = test_db.plain_dbs(1);

    let addresses = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let args0 = HawkArgs {
        party_index: 0,
        addresses: addresses.clone(),
        outbound_addrs: addresses,
        request_parallelism: HAWK_REQUEST_PARALLELISM,
        connection_parallelism: HAWK_CONNECTION_PARALLELISM,
        hnsw_param_ef_constr: HNSW_EF_CONSTR,
        hnsw_param_M: HNSW_M,
        hnsw_param_ef_search: HNSW_EF_SEARCH,
        hnsw_prf_key: None,
        disable_persistence: false,
        match_distances_buffer_size: 64,
        n_buckets: 10,
        tls: None,
        numa: true,
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
