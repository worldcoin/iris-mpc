use aws_sdk_sqs::{config::Region, Client};
use clap::Parser;
use cudarc::driver::{
    result::{
        self,
        event::{self, elapsed},
        stream::synchronize,
    },
    sys::lib,
    CudaSlice,
};
use gpu_iris_mpc::{
    device_manager, device_ptrs,
    setup::iris_db::shamir_iris::ShamirIris,
    sqs::{SMPCRequest, SQSMessage},
    IRIS_CODE_LENGTH, ROTATIONS,
};
use std::{
    fs::metadata,
    mem,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::time::sleep;

use gpu_iris_mpc::{
    device_manager::DeviceManager,
    mmap::{read_mmap_file, write_mmap_file},
    preprocess_query,
    setup::{
        id::PartyID,
        iris_db::{db::IrisDB, shamir_db::ShamirIrisDB},
        shamir::Shamir,
    },
    DistanceComparator, ShareDB,
};
use rand::prelude::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};

const ENABLE_DEDUP_QUERY: bool = true;
const ENABLE_WRITE_DB: bool = true;
const REGION: &str = "eu-north-1";
const DB_SIZE: usize = 8 * 1_000;
const DB_BUFFER: usize = 8 * 1_000;
const QUERIES: usize = 496;
const RNG_SEED: u64 = 42;
const SHUFFLE_SEED: u64 = 42;
const N_BATCHES: usize = 10;
const MAX_CONCURRENT_REQUESTS: usize = 5;
const DB_CODE_FILE: &str = "codes.db";
const DB_MASK_FILE: &str = "masks.db";
const DEFAULT_PATH: &str = "/opt/dlami/nvme/";

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

    #[structopt(short, long)]
    path: Option<String>,
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

fn slice_tuples_to_ptrs(
    tuple: &(
        (Vec<CudaSlice<i8>>, Vec<CudaSlice<i8>>),
        (Vec<CudaSlice<u32>>, Vec<CudaSlice<u32>>),
    ),
) -> ((Vec<u64>, Vec<u64>), (Vec<u64>, Vec<u64>)) {
    (
        (device_ptrs(&tuple.0 .0), device_ptrs(&tuple.0 .1)),
        (device_ptrs(&tuple.1 .0), device_ptrs(&tuple.1 .1)),
    )
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let Opt {
        queue,
        party_id,
        bootstrap_url,
        path,
    } = Opt::parse();
    let path = path.unwrap_or(DEFAULT_PATH.to_string());

    let code_db_path = format!("{}/{}", path, DB_CODE_FILE);
    let mask_db_path = format!("{}/{}", path, DB_MASK_FILE);

    let shuffle_rng = StdRng::seed_from_u64(SHUFFLE_SEED);

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
    let (codes_db, masks_db) = if metadata(&code_db_path).is_ok() && metadata(&mask_db_path).is_ok() {
        (read_mmap_file(&code_db_path)?, read_mmap_file(&mask_db_path)?)
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

        write_mmap_file(&code_db_path, &codes_db)?;
        write_mmap_file(&mask_db_path, &masks_db)?;
        (codes_db, masks_db)
    };

    println!("Starting engines...");

    let device_manager = DeviceManager::init();

    let mut codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        DB_SIZE + DB_BUFFER,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3000),
    );

    let code_db_slices = codes_engine.load_db(&codes_db, DB_SIZE, DB_SIZE + DB_BUFFER);

    let mut masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        DB_SIZE + DB_BUFFER,
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3001),
    );

    let mask_db_slices = masks_engine.load_db(&masks_db, DB_SIZE, DB_SIZE + DB_BUFFER);

    let mut distance_comparator = DistanceComparator::init(QUERIES);

    // Engines for inflight queries
    let mut batch_codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        l_coeff,
        QUERIES * device_manager.device_count(),
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
        QUERIES * device_manager.device_count(),
        QUERIES,
        chacha_seeds,
        bootstrap_url.clone(),
        Some(true),
        Some(3003),
    );

    // Prepare streams etc.
    let mut streams = vec![];
    let mut cublas_handles = vec![];
    let mut results = vec![];
    let mut batch_results = vec![];
    let mut final_results = vec![];
    for _ in 0..MAX_CONCURRENT_REQUESTS {
        let tmp_streams = device_manager.fork_streams();
        cublas_handles.push(device_manager.create_cublas(&tmp_streams));
        streams.push(tmp_streams);
        results.push(distance_comparator.prepare_results());
        batch_results.push(distance_comparator.prepare_results());
        final_results.push(distance_comparator.prepare_final_results());
    }

    // Main Loop
    let mut previous_previous_stream_event = device_manager.create_events();
    let mut previous_stream_event = device_manager.create_events();
    let mut current_stream_event = device_manager.create_events();

    let mut current_dot_event = device_manager.create_events();
    let mut next_dot_event = device_manager.create_events();
    let mut current_exchange_event = device_manager.create_events();
    let mut next_exchange_event = device_manager.create_events();
    let mut request_counter = 0;
    let mut timer_events = vec![];
    let start_timer = device_manager.create_events();
    let end_timer = device_manager.create_events();

    let mut code_db_sizes = vec![];
    let mut mask_db_sizes = vec![];
    let mut query_db_sizes = vec![];
    for i in 0..device_manager.device_count() {
        code_db_sizes.push(
            device_manager
                .device(i)
                .htod_copy(vec![(DB_SIZE / device_manager.device_count()) as u32; 1])
                .unwrap(),
        );
        mask_db_sizes.push(
            device_manager
                .device(i)
                .htod_copy(vec![(DB_SIZE / device_manager.device_count()) as u32; 1])
                .unwrap(),
        );
        query_db_sizes.push(
            device_manager
                .device(i)
                .htod_copy(vec![QUERIES as u32; 1])
                .unwrap(),
        );
    }

    let current_db_size: Vec<usize> =
        vec![DB_SIZE / device_manager.device_count(); device_manager.device_count()];
    let query_db_size = vec![QUERIES; device_manager.device_count()];
    let current_db_size_mutex = current_db_size
        .iter()
        .map(|&s| Arc::new(Mutex::new(s)))
        .collect::<Vec<_>>();

    println!("All systems ready.");

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
        let request_final_results = &final_results[request_counter % MAX_CONCURRENT_REQUESTS];

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

        // update the db size, skip this for the first two
        if request_counter > 2 {
            // We have two streams working concurrently, we'll await the stream before previous one
            let previous_streams = &streams[(request_counter - 2) % MAX_CONCURRENT_REQUESTS];
            device_manager.await_event(previous_streams, &previous_previous_stream_event);
            device_manager.await_streams(previous_streams);
        }

        let current_db_size_stream = current_db_size_mutex
            .iter()
            .map(|e| *e.lock().unwrap())
            .collect::<Vec<_>>();

        // BLOCK 1: calculate individual dot products
        device_manager.await_event(request_streams, &current_dot_event);

        if ENABLE_DEDUP_QUERY {
            batch_codes_engine.dot(
                &code_query,
                &code_query,
                &query_db_size,
                request_streams,
                request_cublas_handles,
            );

            batch_masks_engine.dot(
                &code_query,
                &code_query,
                &query_db_size,
                request_streams,
                request_cublas_handles,
            );

            batch_codes_engine.dot_reduce(
                &code_query_sums,
                &code_query_sums,
                &query_db_size,
                request_streams,
            );
            batch_masks_engine.dot_reduce(
                &code_query_sums,
                &code_query_sums,
                &query_db_size,
                request_streams,
            );

            batch_codes_engine.exchange_results(&query_db_size, request_streams);
            batch_masks_engine.exchange_results(&query_db_size, request_streams);

            distance_comparator.reconstruct_and_compare(
                &batch_codes_engine.results_peers,
                &batch_masks_engine.results_peers,
                &query_db_size,
                request_streams,
                device_ptrs(request_batch_results),
            );
        }

        debug_record_event!(device_manager, request_streams, timers);

        codes_engine.dot(
            &code_query,
            &(
                device_ptrs(&code_db_slices.0 .0),
                device_ptrs(&code_db_slices.0 .1),
            ),
            &current_db_size_stream,
            request_streams,
            request_cublas_handles,
        );

        masks_engine.dot(
            &mask_query,
            &(
                device_ptrs(&mask_db_slices.0 .0),
                device_ptrs(&mask_db_slices.0 .1),
            ),
            &current_db_size_stream,
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
            &current_db_size_stream,
            request_streams,
        );
        masks_engine.dot_reduce(
            &mask_query_sums,
            &(
                device_ptrs(&mask_db_slices.1 .0),
                device_ptrs(&mask_db_slices.1 .1),
            ),
            &current_db_size_stream,
            request_streams,
        );

        device_manager.record_event(request_streams, &next_dot_event);

        debug_record_event!(device_manager, request_streams, timers);

        codes_engine.exchange_results(&current_db_size_stream, request_streams);
        masks_engine.exchange_results(&current_db_size_stream, request_streams);

        debug_record_event!(device_manager, request_streams, timers);

        distance_comparator.reconstruct_and_compare(
            &codes_engine.results_peers,
            &masks_engine.results_peers,
            &current_db_size_stream,
            request_streams,
            device_ptrs(request_results),
        );

        device_manager.await_streams(request_streams);
        let xx = device_manager.device(0).dtoh_sync_copy(&request_results[0]).unwrap();
        println!("xxxx: {:?}", xx);


        if ENABLE_DEDUP_QUERY && ENABLE_WRITE_DB {
            distance_comparator.dedup_and_append(
                &device_ptrs(request_batch_results),
                &device_ptrs(request_results),
                &code_query.0,
                &code_query.1,
                &code_query_sums.0,
                &code_query_sums.1,
                &device_ptrs(&request_final_results),
                &device_ptrs(&code_db_sizes),
                request_streams,
            );

            distance_comparator.dedup_and_append(
                &device_ptrs(request_batch_results),
                &device_ptrs(request_results),
                &mask_query.0,
                &mask_query.1,
                &mask_query_sums.0,
                &mask_query_sums.1,
                &device_ptrs(&request_final_results),
                &device_ptrs(&mask_db_sizes),
                request_streams,
            );
        }

        device_manager.record_event(request_streams, &next_exchange_event);

        // Start thread to wait for the results
        let tmp_streams = request_streams
            .iter()
            .map(|s| s.stream as u64)
            .collect::<Vec<_>>();
        let tmp_devs = distance_comparator.devs.clone();
        let tmp_final_results = device_ptrs(request_final_results);
        let tmp_code_db_sizes = device_ptrs(&code_db_sizes);
        let tmp_code_db_slices = slice_tuples_to_ptrs(&code_db_slices);
        let tmp_mask_db_slices = slice_tuples_to_ptrs(&mask_db_slices);
        let tmp_evts = end_timer.iter().map(|e| *e as u64).collect::<Vec<_>>();
        let mut shuffle_rng = shuffle_rng.clone();
        let current_stream_event_tmp = current_stream_event
            .iter()
            .map(|e| *e as u64)
            .collect::<Vec<_>>();
        let current_db_size_mutex_clone = current_db_size_mutex
            .iter()
            .map(|e| Arc::clone(e))
            .collect::<Vec<_>>();

        tokio::spawn(async move {
            let mut host_results = vec![];
            // Step 1: Fetch the uniqueness results for each query for each device
            for i in 0..tmp_devs.len() {
                tmp_devs[i].bind_to_thread().unwrap();
                host_results.push(vec![u32::MAX; QUERIES / ROTATIONS]);

                unsafe {
                    lib()
                        .cuMemcpyDtoHAsync_v2(
                            host_results[i].as_ptr() as *mut _,
                            tmp_final_results[i],
                            host_results[i].len() * std::mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .result()
                        .unwrap();
                }
            }

            // Step 2: Evaluate the results across devices
            let mut insertion_list = vec![];
            for j in 0..host_results[0].len() {
                let mut match_entry = u32::MAX;
                for i in 0..tmp_devs.len() {
                    if host_results[i][j] != u32::MAX {
                        match_entry = host_results[i][j];
                        break;
                    }
                }

                if match_entry == u32::MAX {
                    insertion_list.push(j);
                }

                println!(
                    "Query {}: unique={} [index: {}]",
                    j,
                    match_entry == u32::MAX,
                    match_entry
                );
            }

            let mut insertion_list = insertion_list
                .chunks(QUERIES / ROTATIONS / tmp_devs.len())
                .collect::<Vec<_>>();

            println!("insertion_list: {:?}", insertion_list);
            insertion_list.shuffle(&mut shuffle_rng);
            println!("insertion_list: {:?}", insertion_list);

            for i in 0..tmp_devs.len() {
                tmp_devs[i].bind_to_thread().unwrap();
                let mut old_size = *current_db_size_mutex_clone[i].lock().unwrap() as u64;
                for insertion_idx in insertion_list[i] {
                    unsafe {
                        // Step 4: fetch and update db counters
                        // Append to codes db
                        result::memcpy_dtod_async(
                            tmp_code_db_slices.0 .0[i] + old_size,
                            code_query.0[i] + (insertion_idx * IRIS_CODE_LENGTH * ROTATIONS) as u64,
                            IRIS_CODE_LENGTH,
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        result::memcpy_dtod_async(
                            tmp_code_db_slices.0 .1[i] + old_size,
                            code_query.1[i] + (insertion_idx * IRIS_CODE_LENGTH * ROTATIONS) as u64,
                            IRIS_CODE_LENGTH,
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        result::memcpy_dtod_async(
                            tmp_code_db_slices.1 .0[i] + (old_size * mem::size_of::<u32>() as u64),
                            code_query_sums.0[i]
                                + (insertion_idx * ROTATIONS * mem::size_of::<u32>()) as u64,
                            mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        result::memcpy_dtod_async(
                            tmp_code_db_slices.1 .1[i] + (old_size * mem::size_of::<u32>() as u64),
                            code_query_sums.1[i]
                                + (insertion_idx * ROTATIONS * mem::size_of::<u32>()) as u64,
                            mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        // Append to masks db
                        result::memcpy_dtod_async(
                            tmp_mask_db_slices.0 .0[i] + old_size,
                            mask_query.0[i] + (insertion_idx * IRIS_CODE_LENGTH * ROTATIONS) as u64,
                            IRIS_CODE_LENGTH,
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        result::memcpy_dtod_async(
                            tmp_mask_db_slices.0 .1[i] + old_size,
                            mask_query.1[i] + (insertion_idx * IRIS_CODE_LENGTH * ROTATIONS) as u64,
                            IRIS_CODE_LENGTH,
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        result::memcpy_dtod_async(
                            tmp_mask_db_slices.1 .0[i] + (old_size * mem::size_of::<u32>() as u64),
                            mask_query_sums.0[i]
                                + (insertion_idx * ROTATIONS * mem::size_of::<u32>()) as u64,
                            mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();

                        result::memcpy_dtod_async(
                            tmp_mask_db_slices.1 .1[i] + (old_size * mem::size_of::<u32>() as u64),
                            mask_query_sums.1[i]
                                + (insertion_idx * ROTATIONS * mem::size_of::<u32>()) as u64,
                            mem::size_of::<u32>(),
                            tmp_streams[i] as *mut _,
                        )
                        .unwrap();
                    }
                    old_size += 1;
                }
                unsafe {
                    // Step 3: write new db sizes to device
                    // *current_db_size_mutex_clone[i].lock().unwrap() +=
                    //     insertion_list[i].len() as usize;
                    // println!(
                    //     "Updating DB size on device {}: {:?}",
                    //     i,
                    //     *current_db_size_mutex_clone[i].lock().unwrap()
                    // );

                    // let tmp_host = vec![*current_db_size_mutex_clone[i].lock().unwrap() as u32; 1];
                    // lib()
                    //     .cuMemcpyHtoDAsync_v2(
                    //         tmp_code_db_sizes[i],
                    //         tmp_host.as_ptr() as *mut _,
                    //         mem::size_of::<u32>(),
                    //         tmp_streams[i] as *mut _,
                    //     )
                    //     .result()
                    //     .unwrap();

                    // Step 5: emit stream finished event to unblock the stream after the following
                    event::record(
                        current_stream_event_tmp[i] as *mut _,
                        tmp_streams[i] as *mut _,
                    )
                    .unwrap();

                    // Emit debug event to measure time for e2e process
                    event::record(tmp_evts[i] as *mut _, tmp_streams[i] as *mut _).unwrap();
                }
            }
        });

        // Prepare for next batch
        timer_events.push(timers);

        request_counter += 1;
        previous_previous_stream_event = previous_stream_event;
        previous_stream_event = current_stream_event;
        current_stream_event = device_manager.create_events();
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

    for i in 0..device_manager.device_count() {
        unsafe {
            device_manager.device(i).bind_to_thread().unwrap();
            let total_time = elapsed(start_timer[i], end_timer[i]).unwrap();
            println!("Total time: {:?}", total_time);
        }
    }

    Ok(())
}
