use core::slice;
use std::{env, ffi::c_void, thread, time::Instant};

use cudarc::driver::{sys::lib, CudaDevice};
use gpu_iris_mpc::{
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, iris::IrisCode, shamir_db::ShamirIrisDB, shamir_iris::ShamirIris},
        shamir::Shamir,
    },
    IrisCodeDB,
};
use rand::{rngs::StdRng, SeedableRng};
use tokio::time;

const DB_SIZE: usize = 1_000_000;
const QUERIES: usize = 320;
const RNG_SEED: u64 = 1337;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let args = env::args().collect::<Vec<_>>();
    let party_id: usize = args[1].parse().unwrap();
    let url = args.get(2);
    let n_devices = CudaDevice::count().unwrap() as usize;
    let local_db_size = DB_SIZE / n_devices;

    // Init DB

    println!("1");
    let db = IrisDB::new_random_seed(DB_SIZE, RNG_SEED);
    println!("2");
    let shamir_db = ShamirIrisDB::share_db_seed(&db, RNG_SEED);

    let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(party_id as u8).unwrap());
    println!("3");
    // Import masks to GPU DB
    let codes_db = shamir_db[party_id]
        .db
        .iter()
        .flat_map(|entry| entry.code)
        .collect::<Vec<_>>();

    println!("Starting engine...");

    let mut engine = IrisCodeDB::init(party_id, l_coeff, &codes_db, url.clone(), false);

    println!("Engine ready!");

    // Prepare query
    let random_query = ShamirIris::share_iris(&IrisCode::random_rng(&mut rng), &mut rng);
    let mut queries = vec![vec![], vec![], vec![]];

    for i in 0..QUERIES {
        // TODO: rotate
        let tmp: [ShamirIris; 3] = random_query.clone();
        queries[0].push(tmp[0].code.to_vec());
        queries[1].push(tmp[1].code.to_vec());
        queries[2].push(tmp[2].code.to_vec());
    }

    println!("Starting query...");
    let query =
        engine.preprocess_query(&queries[0].clone().into_iter().flatten().collect::<Vec<_>>());

    for i in 0..10 {
        let now = Instant::now();
    
        engine.dot(&query);
        engine.exchange_results();
    
        println!("Time elapsed: {:?}", now.elapsed());
    }
    
    let mut gpu_result = vec![0u16; local_db_size * QUERIES];
    engine.fetch_results_peer(&mut gpu_result, 0, 0);
    println!("REMOTE RESULT: {:?}", gpu_result[0]);

    time::sleep(time::Duration::from_secs(5)).await;
    Ok(())
}
