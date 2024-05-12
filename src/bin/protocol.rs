///! End-to-end example implementation of the MPC v1.5 protocol
///! This requires three individual nodes. It can be run like this:
///! Node 0: cargo run --release --bin protocol 0
///! Node 1: cargo run --release --bin protocol 1 [NODE_0_IP]
///! Node 2: cargo run --release --bin protocol 2 [NODE_0_IP]
use std::{env, time::Instant};

use float_eq::assert_float_eq;
use gpu_iris_mpc::{
    device_manager::DeviceManager,
    preprocess_query,
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, iris::IrisCode, shamir_db::ShamirIrisDB, shamir_iris::ShamirIris},
        shamir::Shamir,
    },
    DistanceComparator, ShareDB,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::time;

const DB_SIZE: usize = 8 * 125_000;
const QUERIES: usize = 930;
const RNG_SEED: u64 = 42;
const N_BATCHES: usize = 10; // We expect 10 batches with each QUERIES/ROTATIONS
const MAX_CONCURRENT_REQUESTS: usize = 10;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    // TODO
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let seed0 = rng.gen::<[u32; 8]>();
    let seed1 = rng.gen::<[u32; 8]>();
    let seed2 = rng.gen::<[u32; 8]>();

    let args = env::args().collect::<Vec<_>>();
    let party_id: usize = args[1].parse().unwrap();
    let url = args.get(2);

    // Init RNGs
    let chacha_seeds = match party_id {
        0 => (seed0, seed2),
        1 => (seed1, seed0),
        2 => (seed2, seed1),
        _ => unimplemented!(),
    };

    // Init DB
    let db = IrisDB::new_random_par(DB_SIZE, &mut rng);
    let shamir_db = ShamirIrisDB::share_db_par(&db, &mut rng);
    let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(party_id as u8).unwrap());

    println!("Random shared DB generated!");

    // Import masks to GPU DB
    let codes_db = shamir_db[party_id]
        .db
        .iter()
        .flat_map(|entry| entry.code)
        .collect::<Vec<_>>();

    let masks_db = shamir_db[party_id]
        .db
        .iter()
        .flat_map(|entry| entry.mask)
        .collect::<Vec<_>>();

    println!("Starting engines...");

    let device_manager = DeviceManager::init();

    let mut codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        &codes_db,
        QUERIES,
        chacha_seeds,
        url.clone(),
        Some(true),
        Some(3000),
    );
    let mut masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        &masks_db,
        QUERIES,
        chacha_seeds,
        url.clone(),
        Some(true),
        Some(3001),
    );
    let mut distance_comparator = DistanceComparator::init(DB_SIZE, QUERIES);

    println!("Engines ready!");

    // Prepare streams and cuBLAS handles
    // They will be reused for multiple requests since construction and destruction is costly
    let mut streams = vec![];
    let mut cublas_handles = vec![];
    for i in 0..MAX_CONCURRENT_REQUESTS {
        let tmp_streams = device_manager.fork_streams();
        cublas_handles.push(device_manager.create_cublas(&tmp_streams));
        streams.push(tmp_streams);
    }

    // Entrypoint for incoming request
    let mut request_batches = vec![];
    let mut dot_events = vec![];
    let mut exchange_events = vec![];

    while request_batches.len() < MAX_CONCURRENT_REQUESTS {
        let query = random_query(party_id, &mut rng); // TODO: fetch from queue

        request_batches.push(query);
        dot_events.push(device_manager.create_events());
        exchange_events.push(device_manager.create_events());
    }

    let total_time = Instant::now();

    for i in 0..request_batches.len() {
        let now = Instant::now();

        let (code_query, mask_query) = request_batches[i].clone();
        let request_streams = &streams[i];
        let request_cublas_handles = &cublas_handles[i];

        println!("1: {:?}", now.elapsed());

        // First stream doesn't need to wait on anyone
        if i == 0 {
            device_manager.record_event(request_streams, &dot_events[0]);
            device_manager.record_event(request_streams, &exchange_events[0]);
        }

        println!("2: {:?}", now.elapsed());

        // BLOCK 1: calculate individual dot products
        device_manager.await_event(request_streams, &dot_events[i]);
        codes_engine.dot(&code_query, request_streams, request_cublas_handles);
        masks_engine.dot(&mask_query, request_streams, request_cublas_handles);
        // skip last
        if i < request_batches.len()-1 {
            device_manager.record_event(request_streams, &dot_events[i + 1]);
        }

        println!("3: {:?}", now.elapsed());

        // BLOCK 2: calculate final dot product result, exchange and compare 
        device_manager.await_event(request_streams, &exchange_events[i]);
        codes_engine.dot_reduce(request_streams);
        masks_engine.dot_reduce(request_streams);
        codes_engine.exchange_results(request_streams);
        masks_engine.exchange_results(request_streams);
        distance_comparator.reconstruct_and_compare(
            &codes_engine.results_peers,
            &masks_engine.results_peers,
            request_streams,
        );
        // skip last
        if i < request_batches.len()-1 {
            device_manager.record_event(request_streams, &exchange_events[i + 1]);
        }

        println!("Loop time: {:?}", now.elapsed());
    }

    // Now all streams are running, we need to await each on CPU
    for i in 0..request_batches.len() {
        device_manager.await_streams(&streams[i]);
        // let results = distance_comparator.fetch_results();
    }

    println!(
        "Total time for {} samples: {:?}",
        request_batches.len(),
        total_time.elapsed()
    );

    // let reference_dists = db.calculate_distances(&query_template);

    // for i in 0..DB_SIZE / n_devices {
    //     assert_float_eq!(dists[i], reference_dists[i], abs <= 1e-6);
    // }

    println!("Distances match the reference!");

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}

fn random_query(party_id: usize, rng: &mut StdRng) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let query_template = IrisCode::random_rng(rng);
    let random_query = ShamirIris::share_iris(&query_template, rng);
    let mut code_queries = vec![vec![], vec![], vec![]];
    let mut mask_queries = vec![vec![], vec![], vec![]];

    for i in 0..QUERIES {
        // TODO: rotate
        let tmp: [ShamirIris; 3] = random_query.clone();
        code_queries[0].push(tmp[0].code.to_vec());
        code_queries[1].push(tmp[1].code.to_vec());
        code_queries[2].push(tmp[2].code.to_vec());

        mask_queries[0].push(tmp[0].mask.to_vec());
        mask_queries[1].push(tmp[1].mask.to_vec());
        mask_queries[2].push(tmp[2].mask.to_vec());
    }

    println!("Starting query...");
    let code_query = preprocess_query(
        &code_queries[party_id]
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    );
    let mask_query = preprocess_query(
        &mask_queries[party_id]
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    );
    (code_query, mask_query)
}
