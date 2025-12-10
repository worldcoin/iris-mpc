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
    hnsw::{GraphMem, HnswParams, HnswSearcher},
    // import FSS flags
    protocol::{
        fss_traffic_totals, msb_fss_total_inputs, ops::USE_FSS,
        shared_iris::GaloisRingSharedIris, USE_PARALLEL_THRESH,
    },
};
use rand::{rngs::StdRng, SeedableRng};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const DB_SIZE: usize = 1000; // original value was 1000
const DB_RNG_SEED: u64 = 0xdeadbeef;
const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
const NUM_BATCHES: usize = 5;
const MAX_BATCH_SIZE: usize = 5;
const HAWK_REQUEST_PARALLELISM: usize = 1;
const HAWK_CONNECTION_PARALLELISM: usize = 8; // Increased from 1 to 8 for better network parallelism
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
    params: &HnswParams,
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

    let mut left_store = PlaintextStore {
        storage: left_storage,
    };
    let mut right_store = PlaintextStore {
        storage: right_storage,
    };

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

    let params = HnswParams::new(
        args.hnsw_param_ef_constr,
        args.hnsw_param_ef_search,
        args.hnsw_param_M,
    );
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

    let test_db = generate_full_test_db(DB_SIZE, DB_RNG_SEED, false);
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

    // -- FSS metrics & stats starts -- //

    // Metrics: cross_compare calls
    let (scheme, calls) = if iris_mpc_cpu::protocol::ops::USE_FSS {
        (
            "FSS",
            iris_mpc_cpu::protocol::ops::cross_compare_calls_fss(),
        )
    } else {
        (
            "RSS",
            iris_mpc_cpu::protocol::ops::cross_compare_calls_rss(),
        )
    };

    println!("\n{scheme}: cross_compare was called {calls} times\n");

    // Metrics: total duration per party for cross_compare.extract_open
    let [p0, p1, p2] =
        iris_mpc_cpu::protocol::perf_stats::total_duration_per_party("cross_compare.extract_open");

    println!("extract+open_bin totals — p0: {p0:?}, p1: {p1:?}, p2: {p2:?}\n");

    if USE_FSS && USE_PARALLEL_THRESH {
        // times for FSS evaluation parties
        println!("\nTimers for parties 0 and 1:");

        let evaluator_stats = vec![
            (
                "fss.network.start_recv_keylen",
                "Receive key length from the dealer",
            ),
            (
                "fss.network.start_recv_keys",
                "Receive keys from the dealer",
            ),
            ("fss.network.recon.send", "Reconstruct d+r send"),
            ("fss.network.recon.recv", "Reconstruct d+r recv"),
            ("fss.add3.non-parallel", "FSS add3 non-parallel"),
            ("fss.add3.icf.eval", "FSS add3 ICF eval"),
            ("fss.network.post-icf.send_prev", "Post-ICF send_prev"),
            ("fss.network.post-icf.send_next", "Post-ICF send_next"),
            ("fss.network.post-icf.recv_to_eval", "Post-ICF recv_to_eval"),
        ];

        for (key, message) in evaluator_stats {
            let [p0, p1, _] = iris_mpc_cpu::protocol::perf_stats::total_duration_per_party(key);
            println!("{} p0: {:?}", message, p0);
            println!("{} p1: {:?}", message, p1);
            println!();
        }

        // times for FSS dealer
        println!("\nTimers for FSS dealer:");

        let dealer_stats = vec![
            (
                "fss.network.dealer.send_P0a",
                "Dealer send FSS keylen to p0",
            ),
            ("fss.network.dealer.send_P0b", "Dealer send FSS keys to p0"),
            (
                "fss.network.dealer.send_P1a",
                "Dealer send FSS keylen to p1",
            ),
            ("fss.network.dealer.send_P1b", "Dealer send FSS keys to p1"),
            (
                "fss.network.dealer.recv_P0",
                "Dealer recv FSS shares from p0",
            ),
            (
                "fss.network.dealer.recv_P1",
                "Dealer recv FSS shares from p1",
            ),
            ("fss.dealer.genkeys", "Dealer generate keys"),
        ];

        for (key, label) in dealer_stats {
            let [_, _, p2] = iris_mpc_cpu::protocol::perf_stats::total_duration_per_party(key);
            println!("{}: {:?}", label, p2);
        }
        println!("");
    } // if USE_FSS && USE_PARALLEL_THRESH

    // Since the code differs between the parallel and non-parallel version,
    // some timers are the same while the rest differ, but there is a lot of repetition
    if USE_FSS && !USE_PARALLEL_THRESH {
        // times for FSS evaluation parties
        println!("\nTimers for parties 0 and 1:");

        let evaluator_stats = vec![
            ("fss.network.recon.send", "Reconstruct d+r send"),
            ("fss.network.recon.recv", "Reconstruct d+r recv"),
            (
                "fss.network.start_recv_keys",
                "Receive FSS keys from the dealer",
            ),
            ("fss.network.post-icf.send_next", "Post-ICF send_next"),
            ("fss.network.post-icf.send_prev", "Post-ICF send_prev"),
            ("fss.network.post-icf.recv", "Post-ICF recv"),
        ];

        for (key, message) in evaluator_stats {
            let [p0, p1, _] = iris_mpc_cpu::protocol::perf_stats::total_duration_per_party(key);
            println!("{} p0: {:?}", message, p0);
            println!("{} p1: {:?}", message, p1);
            println!();
        }

        // times for FSS dealer
        println!("\nTimers for FSS dealer:");

        let dealer_stats = vec![
            ("fss.network.dealer.send_P0", "Dealer send FSS keys to p0"),
            ("fss.network.dealer.send_P1", "Dealer send FSS keys to p1"),
            (
                "fss.network.dealer.recv_P0",
                "Dealer receive FSS shares from p0",
            ),
            (
                "fss.network.dealer.recv_P1",
                "Dealer receive FSS shares from p1",
            ),
            ("fss.dealer.genkeys", "Dealer generate keys"),
        ];

        for (key, label) in dealer_stats {
            let [_, _, p2] = iris_mpc_cpu::protocol::perf_stats::total_duration_per_party(key);
            println!("{}: {:?}", label, p2);
        }

        println!("");
    } // if USE_FSS && !USE_PARALLEL_THRESH

    // Print total FSS traffic once at the end
    let (sent_bytes, recv_bytes) = fss_traffic_totals();
    println!(
        "FSS total traffic: sent ~{} bytes, received ~{} bytes",
        sent_bytes, recv_bytes
    );
    println!(
        "FSS total inputs processed: {}",
        msb_fss_total_inputs()
    );

    Ok(())
}
