///! End-to-end example implementation of the MPC v1.5 protocol
///! This requires three individual nodes. It can be run like this:
///! Node 0: cargo run --release --bin protocol 0
///! Node 1: cargo run --release --bin protocol 1 [NODE_0_IP]
///! Node 2: cargo run --release --bin protocol 2 [NODE_0_IP]

use std::{env, time::Instant};

use cudarc::driver::CudaDevice;
use float_eq::assert_float_eq;
use gpu_iris_mpc::{
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, shamir_db::ShamirIrisDB, shamir_iris::ShamirIris},
        shamir::Shamir,
    },
    DistanceComparator, ShareDB,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use tokio::time;

const DB_SIZE: usize = 8 * 1000;
const QUERIES: usize = 31;
const RNG_SEED: u64 = 42;

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
    let n_devices = CudaDevice::count().unwrap() as usize;

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

    let mut codes_engine = ShareDB::init(
        party_id,
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
        l_coeff,
        &masks_db,
        QUERIES,
        chacha_seeds,
        url.clone(),
        Some(true),
        Some(3001),
    );
    let mut distance_comparator = DistanceComparator::init(n_devices, DB_SIZE, QUERIES);

    println!("Engines ready!");

    // Prepare queries
    let query_template = db.db[0].get_similar_iris(&mut rng);
    let random_query = ShamirIris::share_iris(&query_template, &mut rng);
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
    let code_query = codes_engine.preprocess_query(
        &code_queries[party_id]
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    );
    let mask_query = masks_engine.preprocess_query(
        &mask_queries[party_id]
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    );

    for _ in 0..1 {
        let now = Instant::now();

        codes_engine.dot(&code_query);
        println!("Dot codes took: {:?}", now.elapsed());

        codes_engine.exchange_results();
        println!("Exchange codes took: {:?}", now.elapsed());

        masks_engine.dot(&mask_query);
        println!("Dot masks took: {:?}", now.elapsed());

        masks_engine.exchange_results();
        println!("Exchange masks took: {:?}", now.elapsed());

        distance_comparator
            .reconstruct_and_compare(&codes_engine.results_peers, &masks_engine.results_peers);

        println!("Total time: {:?}", now.elapsed());
    }

    let mut results_codes = vec![0u16; DB_SIZE / n_devices * QUERIES];
    codes_engine.fetch_results(&mut results_codes, 0);
    let mut results_masks = vec![0u16; DB_SIZE / n_devices * QUERIES];
    masks_engine.fetch_results(&mut results_masks, 0);

    // Sanity check: compare results against reference (debug only)
    let (dists, _) = distance_comparator
        .reconstruct_distances_debug(&codes_engine.results_peers, &masks_engine.results_peers);

    let reference_dists = db.calculate_distances(&query_template);

    for i in 0..DB_SIZE / n_devices {
        assert_float_eq!(dists[i], reference_dists[i], abs <= 1e-6);
    }

    println!("Distances match the reference!");

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}
