use aws_sdk_sqs::{config::Region, Client, Error};
use base64::{engine::general_purpose, Engine};
use clap::Parser;
use cudarc::driver::{
    result::{
        event::{self, elapsed},
        memcpy_dtoh_async,
        stream::synchronize,
    },
    sys::lib,
    DevicePtr,
};
use gpu_iris_mpc::{
    device_ptrs,
    setup::iris_db::{iris::IrisCodeArray, shamir_iris::ShamirIris},
    sqs::{SMPCRequest, SQSMessage},
};
use std::{
    env,
    fs::metadata,
    time::{Duration, Instant},
};
use tokio::time::sleep;

use gpu_iris_mpc::{
    device_manager::DeviceManager,
    mmap::{read_mmap_file, write_mmap_file},
    preprocess_query,
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, iris::IrisCode, shamir_db::ShamirIrisDB},
        shamir::Shamir,
    },
    DistanceComparator, ShareDB,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const ENABLE_QUERY_DEDUP: bool = true;
const REGION: &str = "us-east-2";
const DB_SIZE: usize = 8 * 50_000;
const QUERIES: usize = 930;
const RNG_SEED: u64 = 42;
const N_BATCHES: usize = 10;
const MAX_CONCURRENT_REQUESTS: usize = 5;
const DB_CODE_FILE: &str = "/opt/dlami/nvme/codes.db";
const DB_MASK_FILE: &str = "/opt/dlami/nvme/masks.db";

macro_rules! debug_record_event {
    ($manager:expr, $streams:expr, $timers:expr) => {
        let evts = $manager.create_events();
        $manager.record_event($streams, &evts);
        $timers.push(evts);
    };
}

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    queue: String,

    #[structopt(short, long)]
    party_id: usize,

    #[structopt(short, long)]
    bootstrap_url: Option<String>,
}

async fn receive_batch(client: &Client, queue_url: &String) -> eyre::Result<Vec<ShamirIris>> {
    let mut batch = vec![];

    while batch.len() < QUERIES {
        let rcv_message_output = client
            .receive_message()
            .max_number_of_messages(1i32)
            .queue_url(queue_url)
            .send()
            .await?;

        for sns_message in rcv_message_output.messages.unwrap_or_default() {
            let message: SQSMessage = serde_json::from_str(sns_message.body().unwrap())?;
            let message: SMPCRequest = serde_json::from_str(&message.message)?;

            let iris: ShamirIris = message.into();

            batch.extend(iris.all_rotations());
            // TODO: we should only delete after processing
            client
                .delete_message()
                .queue_url(queue_url)
                .receipt_handle(sns_message.receipt_handle.unwrap())
                .send()
                .await?;
        }
    }

    Ok(batch)
}

fn prepare_query_batch(batch: Vec<ShamirIris>) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let (code_queries, mask_queries): (Vec<Vec<u16>>, Vec<Vec<u16>>) = batch
        .iter()
        .map(|iris| (iris.code.to_vec(), iris.mask.to_vec()))
        .unzip();

    let code_query = preprocess_query(&code_queries.into_iter().flatten().collect::<Vec<_>>());
    let mask_query = preprocess_query(&mask_queries.into_iter().flatten().collect::<Vec<_>>());
    (code_query, mask_query)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let Opt {
        queue,
        party_id,
        bootstrap_url,
    } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let seed0 = rng.gen::<[u32; 8]>();
    let seed1 = rng.gen::<[u32; 8]>();
    let seed2 = rng.gen::<[u32; 8]>();

    // Init RNGs
    let chacha_seeds = match party_id {
        0 => (seed0, seed2),
        1 => (seed1, seed0),
        2 => (seed2, seed1),
        _ => unimplemented!(),
    };

    let l_coeff = Shamir::my_lagrange_coeff_d2(PartyID::try_from(party_id as u8).unwrap());

    // Generate or load DB
    let (codes_db, masks_db) = if metadata(DB_CODE_FILE).is_ok() && metadata(DB_MASK_FILE).is_ok() {
        (read_mmap_file(DB_CODE_FILE)?, read_mmap_file(DB_MASK_FILE)?)
    } else {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        let shamir_db = ShamirIrisDB::share_db_par(&db, &mut rng);

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

        write_mmap_file(DB_CODE_FILE, &codes_db)?;
        write_mmap_file(DB_MASK_FILE, &masks_db)?;
        (codes_db, masks_db)
    };

    println!("Starting engines...");

    let device_manager = DeviceManager::init();

    let mut codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        DB_SIZE,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3000),
    );

    let code_db_slices = codes_engine.load_db(&codes_db);

    println!("Codes Engines ready!");

    let mut masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        DB_SIZE,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3001),
    );

    let mask_db_slices = masks_engine.load_db(&masks_db);

    let mut distance_comparator = DistanceComparator::init(DB_SIZE, QUERIES, true);

    println!("Engines ready!");

    // Engines for inflight queries
    let mut batch_codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        QUERIES,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3002),
    );
    let mut batch_masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        QUERIES,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3003),
    );
    let mut batch_distance_comparator = DistanceComparator::init(QUERIES, QUERIES, false);

    // Prepare streams etc.
    let mut streams = vec![];
    let mut cublas_handles = vec![];
    let mut results = vec![];
    let mut batch_results = vec![];
    let mut query_results_mask = vec![];
    for _ in 0..MAX_CONCURRENT_REQUESTS {
        let tmp_streams = device_manager.fork_streams();
        cublas_handles.push(device_manager.create_cublas(&tmp_streams));
        streams.push(tmp_streams);
        results.push(distance_comparator.prepare_results());
        batch_results.push(batch_distance_comparator.prepare_results());
        query_results_mask.push(batch_distance_comparator.prepare_results_mask());
    }

    // Main Loop
    let mut current_dot_event = device_manager.create_events();
    let mut next_dot_event = device_manager.create_events();
    let mut current_exchange_event = device_manager.create_events();
    let mut next_exchange_event = device_manager.create_events();
    let mut request_counter = 0;
    let mut timer_events = vec![];
    let start_timer = device_manager.create_events();
    let end_timer = device_manager.create_events();

    // loop {
    for _ in 0..N_BATCHES {
        let now = Instant::now();
        let batch = receive_batch(&client, &queue).await?;
        println!("Received batch in {:?}", now.elapsed());
        let (code_query, mask_query) = prepare_query_batch(batch);

        let mut timers = vec![];

        let request_streams = &streams[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_cublas_handles = &cublas_handles[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_results = &results[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_batch_results = &batch_results[request_counter % MAX_CONCURRENT_REQUESTS];

        // First stream doesn't need to wait on anyone
        if request_counter == 0 {
            device_manager.record_event(request_streams, &current_dot_event);
            device_manager.record_event(request_streams, &current_exchange_event);
            device_manager.record_event(request_streams, &start_timer);
        }

        // Transfer queries to device
        // TODO: free all of this!
        let code_query = device_manager.htod_transfer_query(&code_query, request_streams);
        let mask_query = device_manager.htod_transfer_query(&mask_query, request_streams);
        let code_query_sums =
            codes_engine.query_sums(&code_query, request_streams, request_cublas_handles);
        let mask_query_sums =
            masks_engine.query_sums(&mask_query, request_streams, request_cublas_handles);

        if ENABLE_QUERY_DEDUP {
            batch_codes_engine.dot(
                &code_query,
                &code_query,
                request_streams,
                request_cublas_handles,
            );
    
            batch_masks_engine.dot(
                &code_query,
                &code_query,
                request_streams,
                request_cublas_handles,
            );
    
            batch_codes_engine.dot_reduce(&code_query_sums, &code_query_sums, request_streams);
            batch_masks_engine.dot_reduce(&code_query_sums, &code_query_sums, request_streams);
    
            batch_codes_engine.exchange_results(request_streams);
            batch_masks_engine.exchange_results(request_streams);
    
            batch_distance_comparator.reconstruct_and_compare(
                &batch_codes_engine.results_peers,
                &batch_masks_engine.results_peers,
                request_streams,
                device_ptrs(request_batch_results),
            );

            // filter out dups
            // TODO:
        }


        // BLOCK 1: calculate individual dot products
        device_manager.await_event(request_streams, &current_dot_event);

        debug_record_event!(device_manager, request_streams, timers);

        codes_engine.dot(
            &code_query,
            &(
                device_ptrs(&code_db_slices.0 .0),
                device_ptrs(&code_db_slices.0 .1),
            ),
            request_streams,
            request_cublas_handles,
        );
        masks_engine.dot(
            &mask_query,
            &(
                device_ptrs(&mask_db_slices.0 .0),
                device_ptrs(&mask_db_slices.0 .1),
            ),
            request_streams,
            request_cublas_handles,
        );

        debug_record_event!(device_manager, request_streams, timers);

        // BLOCK 2: calculate final dot product result, exchange and compare
        device_manager.await_event(request_streams, &current_exchange_event);

        codes_engine.dot_reduce(
            &code_query_sums,
            &(
                device_ptrs(&code_db_slices.1 .0),
                device_ptrs(&code_db_slices.1 .1),
            ),
            request_streams,
        );
        masks_engine.dot_reduce(
            &mask_query_sums,
            &(
                device_ptrs(&mask_db_slices.1 .0),
                device_ptrs(&mask_db_slices.1 .1),
            ),
            request_streams,
        );

        device_manager.record_event(request_streams, &next_dot_event);

        debug_record_event!(device_manager, request_streams, timers);

        codes_engine.exchange_results(request_streams);
        masks_engine.exchange_results(request_streams);

        debug_record_event!(device_manager, request_streams, timers);

        distance_comparator.reconstruct_and_compare(
            &codes_engine.results_peers,
            &masks_engine.results_peers,
            request_streams,
            device_ptrs(request_results),
        );

        // TODO: filter out dups from query and append query to DB

        device_manager.record_event(request_streams, &next_exchange_event);

        // Start thread to wait for the results
        let tmp_streams = request_streams
            .iter()
            .map(|s| s.stream as u64)
            .collect::<Vec<_>>();
        let tmp_devs = distance_comparator.devs.clone();
        let tmp_results = device_ptrs(request_results);
        let tmp_evts = end_timer.iter().map(|e| *e as u64).collect::<Vec<_>>();

        tokio::spawn(async move {
            let mut index_results = vec![];
            for i in 0..tmp_devs.len() {
                tmp_devs[i].bind_to_thread().unwrap();
                let mut tmp_result = vec![0u32; QUERIES];
                unsafe {
                    lib()
                        .cuMemcpyDtoHAsync_v2(
                            tmp_result.as_mut_ptr() as *mut _,
                            tmp_results[i],
                            QUERIES * std::mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .result()
                        .unwrap();

                    event::record(tmp_evts[i] as *mut _, tmp_streams[i] as *mut _).unwrap();
                    synchronize(tmp_streams[i] as *mut _).unwrap();
                }
                index_results.push(tmp_result);
            }

            let mut found = false;
            for j in 0..8 {
                for i in 0..QUERIES {
                    if index_results[j][i] != u32::MAX {
                        println!(
                            "Found query {} at index {:?}",
                            i,
                            (DB_SIZE / 8 * j) as u32 + index_results[j][i]
                        );
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                println!("Not found in DB.");
            }
            println!("----");
        });

        // Prepare for next batch
        timer_events.push(timers);

        request_counter += 1;
        current_dot_event = next_dot_event;
        current_exchange_event = next_exchange_event;
        next_dot_event = device_manager.create_events();
        next_exchange_event = device_manager.create_events();
        println!("CPU time of one iteration {:?}", now.elapsed());
    }

    sleep(Duration::from_secs(5)).await;

    for timers in timer_events {
        unsafe {
            device_manager.device(0).bind_to_thread().unwrap();
            let dot_time = elapsed(timers[0][0], timers[1][0]).unwrap();
            let exchange_time = elapsed(timers[2][0], timers[3][0]).unwrap();
            println!(
                "Dot time: {:?}, Exchange time: {:?}",
                dot_time, exchange_time
            );
        }
    }

    for i in 0..8 {
        unsafe {
            device_manager.device(i).bind_to_thread().unwrap();
            let total_time = elapsed(start_timer[i], end_timer[i]).unwrap();
            println!("Total time: {:?}", total_time);
        }
    }

    Ok(())
}
