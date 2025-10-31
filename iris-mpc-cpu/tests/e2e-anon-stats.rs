use eyre::Result;
use iris_mpc_common::{
    iris_db::{db::IrisDB, iris::IrisCode},
    test::{generate_full_test_db, SimpleAnonStatsTestGenerator},
    IrisVectorId as VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{HawkActor, HawkArgs, HawkHandle},
    hawkers::{
        aby3::aby3_store::{Aby3SharedIrises, Aby3Store, Aby3VectorRef},
        plaintext_store::PlaintextStore,
        shared_irises::SharedIrises,
    },
    hnsw::{GraphMem, HnswParams, HnswSearcher},
    protocol::shared_iris::GaloisRingSharedIris,
};
use rand::{random, rngs::StdRng, SeedableRng};
use std::{collections::HashMap, env, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt};

const DB_SIZE: usize = 1000;
const NUM_BATCHES: usize = 30;
const N_BUCKETS: usize = 8;
const MATCH_DISTANCE_BUFFER_SIZE: usize = 64;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const HAWK_CONNECTION_PARALLELISM: usize = 1;

const HNSW_EF_CONSTR: usize = 320;
const HNSW_M: usize = 256;
const HNSW_EF_SEARCH: usize = 256;

fn install_tracing() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
                .with_target(true),
        )
        .init();
}

async fn create_graph_from_plain_dbs(
    player_index: usize,
    db_seed: u64,
    left_db: &IrisDB,
    right_db: &IrisDB,
    params: &HnswParams,
) -> Result<([GraphMem<Aby3VectorRef>; 2], [Aby3SharedIrises; 2])> {
    let mut rng = StdRng::seed_from_u64(db_seed);
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

    let searcher = HnswSearcher {
        params: params.clone(),
    };
    let left_graph = left_store
        .generate_graph(&mut rng, DB_SIZE, &searcher)
        .await?;
    let right_graph = right_store
        .generate_graph(&mut rng, DB_SIZE, &searcher)
        .await?;

    let left_mpc_graph: GraphMem<Aby3VectorRef> = left_graph;
    let right_mpc_graph: GraphMem<Aby3VectorRef> = right_graph;

    let mut left_shared_irises = HashMap::new();
    let mut right_shared_irises = HashMap::new();

    // sort the points by serial id to ensure consistent ordering
    let mut left_points_sorted = left_store.storage.get_sorted_serial_ids();
    left_points_sorted.sort();

    let mut right_points_sorted = right_store.storage.get_sorted_serial_ids();
    right_points_sorted.sort();

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
    db_seed: u64,
    left_db: &IrisDB,
    right_db: &IrisDB,
) -> Result<HawkHandle> {
    tracing::info!("🦅 Starting Hawk node {}", args.party_index);

    let params = HnswParams::new(
        args.hnsw_param_ef_constr,
        args.hnsw_param_ef_search,
        args.hnsw_param_M,
    );
    let (graph, iris_store) =
        create_graph_from_plain_dbs(args.party_index, db_seed, left_db, right_db, &params).await?;
    let hawk_actor =
        HawkActor::from_cli_with_graph_and_store(args, CancellationToken::new(), graph, iris_store)
            .await?;

    let handle = HawkHandle::new(hawk_actor).await?;

    Ok(handle)
}

#[ignore = "Takes long time to run, in CI this is selected in a separate step"]
#[tokio::test]
async fn e2e_anon_stats_test() -> Result<()> {
    install_tracing();

    let internal_seed = match env::var("INTERNAL_SEED") {
        Ok(seed) => {
            tracing::info!("Internal SEED was passed: {}", seed);
            seed.parse::<u64>()?
        }
        Err(_) => {
            tracing::info!("Internal SEED not set, using random seed");
            random()
        }
    };
    let db_seed = match env::var("DB_SEED") {
        Ok(seed) => {
            tracing::info!("DB SEED was passed: {}", seed);
            seed.parse::<u64>()?
        }
        Err(_) => {
            tracing::info!("DB SEED not set, using random seed");
            random()
        }
    };
    tracing::info!(
        "Seeds for this test run. DB: {}, Internal: {}",
        db_seed,
        internal_seed
    );

    let addresses = ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    let test_db = generate_full_test_db(DB_SIZE, db_seed, true);
    let db_left = test_db.plain_dbs(0);
    let db_right = test_db.plain_dbs(1);

    let args0 = HawkArgs {
        party_index: 0,
        addresses,
        request_parallelism: HAWK_REQUEST_PARALLELISM,
        connection_parallelism: HAWK_CONNECTION_PARALLELISM,
        hnsw_param_ef_constr: HNSW_EF_CONSTR,
        hnsw_param_M: HNSW_M,
        hnsw_param_ef_search: HNSW_EF_SEARCH,
        hnsw_prf_key: None,
        numa: true,
        disable_persistence: false,
        match_distances_buffer_size: MATCH_DISTANCE_BUFFER_SIZE,
        n_buckets: N_BUCKETS,
        tls: None,
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
        start_hawk_node(&args0, db_seed, db_left, db_right),
        start_hawk_node(&args1, db_seed, db_left, db_right),
        start_hawk_node(&args2, db_seed, db_left, db_right),
    );
    let mut handle0 = handle0?;
    let mut handle1 = handle1?;
    let mut handle2 = handle2?;
    let mut test_case_generator =
        SimpleAnonStatsTestGenerator::new(test_db, internal_seed, N_BUCKETS, true, 0.0);

    tracing::info!("Setup done, starting tests");
    test_case_generator
        .run_n_batches(NUM_BATCHES, [&mut handle0, &mut handle1, &mut handle2])
        .await?;

    drop(handle0);
    drop(handle1);
    drop(handle2);

    // TODO: ATM we have no real way to wait for the actors to finish, so just sleep
    // a bit for now
    tokio::time::sleep(Duration::from_secs(5)).await;

    Ok(())
}
