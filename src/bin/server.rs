#![allow(clippy::needless_range_loop)]
use aws_sdk_sns::Client as SNSClient;
use aws_sdk_sqs::{config::Region, Client};
use clap::Parser;
use core::sync::atomic::Ordering::SeqCst;
use cudarc::driver::{
    result::{
        self,
        event::{self, elapsed},
        stream::synchronize,
    },
    CudaDevice, CudaSlice,
};
use gpu_iris_mpc::{
    dot::{
        device_manager::DeviceManager,
        distance_comparator::DistanceComparator,
        share_db::{preprocess_query, ShareDB},
        IRIS_CODE_LENGTH, ROTATIONS,
    },
    helpers::{
        device_ptrs, device_ptrs_to_slices,
        kms_dh::derive_shared_secret,
        mmap::{read_mmap_file, write_mmap_file},
        sqs::{ResultEvent, SMPCRequest, SQSMessage},
    },
    setup::{galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::db::IrisDB},
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use lazy_static::lazy_static;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::{
    fs::metadata,
    mem,
    sync::{atomic::AtomicUsize, Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{sync::mpsc, task::spawn_blocking, time::sleep};

const REGION: &str = "eu-north-1";
const DB_SIZE: usize = 8 * 1_000;
const DB_BUFFER: usize = 8 * 1_000;
const N_QUERIES: usize = 32;
const N_BATCHES: usize = 10;
const RNG_SEED: u64 = 42;
const SHUFFLE_SEED: u64 = 42;
const MAX_CONCURRENT_REQUESTS: usize = 5;
const DB_CODE_FILE: &str = "codes.db";
const DB_MASK_FILE: &str = "masks.db";
const DEFAULT_PATH: &str = "/opt/dlami/nvme/";
const QUERIES: usize = ROTATIONS * N_QUERIES;
const KMS_KEY_IDS: [&str; 3] = [
    "077788e2-9eeb-4044-859b-34496cfd500b",
    "896353dc-5ea5-42d4-9e4e-f65dd8169dee",
    "42bb01f5-8380-48b4-b1f1-929463a587fb",
];

lazy_static! {
    static ref KDF_NONCE: AtomicUsize = AtomicUsize::new(0);
    static ref KDF_SALT: Salt = Salt::new(HKDF_SHA256, b"IRIS_MPC");
}

macro_rules! debug_record_event {
    ($manager:expr, $streams:expr, $timers:expr) => {
        let evts = $manager.create_events();
        $manager.record_event($streams, &evts);
        $timers.push(evts);
    };
}

macro_rules! forget_vec {
    ($vec:expr) => {
        while let Some(item) = $vec.pop() {
            std::mem::forget(item);
        }
    };
}

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    queue: String,

    #[structopt(short, long)]
    results_topic_arn: String,

    #[structopt(short, long)]
    party_id: usize,

    #[structopt(short, long)]
    bootstrap_url: Option<String>,

    #[structopt(short, long)]
    path: Option<String>,
}

#[derive(Default)]
struct BatchQueryEntries {
    pub code: Vec<GaloisRingIrisCodeShare>,
    pub mask: Vec<GaloisRingIrisCodeShare>,
}

#[derive(Default)]
struct BatchQuery {
    pub query: BatchQueryEntries,
    pub db:    BatchQueryEntries,
}

async fn receive_batch(
    party_id: usize,
    client: &Client,
    queue_url: &String,
) -> eyre::Result<BatchQuery> {
    let mut batch_query = BatchQuery::default();

    while batch_query.db.code.len() < QUERIES {
        let rcv_message_output = client
            .receive_message()
            .max_number_of_messages(1i32)
            .queue_url(queue_url)
            .send()
            .await?;

        for sns_message in rcv_message_output.messages.unwrap_or_default() {
            let message: SQSMessage = serde_json::from_str(sns_message.body().unwrap())?;
            let message: SMPCRequest = serde_json::from_str(&message.message)?;

            let (db_iris_shares, db_mask_shares, iris_shares, mask_shares) =
                spawn_blocking(move || {
                    let mut iris_share =
                        GaloisRingIrisCodeShare::new(party_id + 1, message.get_iris_shares());
                    let mut mask_share =
                        GaloisRingIrisCodeShare::new(party_id + 1, message.get_mask_shares());

                    let db_iris_shares = iris_share.all_rotations();
                    let db_mask_shares = mask_share.all_rotations();

                    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut iris_share);
                    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut mask_share);

                    (
                        db_iris_shares,
                        db_mask_shares,
                        iris_share.all_rotations(),
                        mask_share.all_rotations(),
                    )
                })
                .await?;

            batch_query.db.code.extend(db_iris_shares);
            batch_query.db.mask.extend(db_mask_shares);
            batch_query.query.code.extend(iris_shares);
            batch_query.query.mask.extend(mask_shares);

            // TODO: we should only delete after processing
            client
                .delete_message()
                .queue_url(queue_url)
                .receipt_handle(sns_message.receipt_handle.unwrap())
                .send()
                .await?;
        }
    }

    Ok(batch_query)
}

fn prepare_query_shares(shares: Vec<GaloisRingIrisCodeShare>) -> Vec<Vec<u8>> {
    preprocess_query(
        &shares
            .into_iter()
            .map(|e| e.coefs)
            .flatten()
            .collect::<Vec<_>>(),
    )
}

#[allow(clippy::type_complexity)]
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

fn open(
    party: &mut Circuits,
    x: &[ChunkShare<u64>],
    distance_comparator: &DistanceComparator,
    results_ptrs: &[CudaSlice<u32>],
    chunk_size: usize,
    db_sizes: &[usize],
) {
    let n_devices = x.len();
    let mut a = Vec::with_capacity(n_devices);
    let mut b = Vec::with_capacity(n_devices);
    let mut c = Vec::with_capacity(n_devices);

    cudarc::nccl::result::group_start().unwrap();
    for (idx, res) in x.iter().enumerate() {
        // Result is in bit 0
        let res = res.get_offset(0, chunk_size);
        party.send_view(&res.b, party.next_id(), idx);
        a.push(res.a);
        b.push(res.b);
    }
    for (idx, res) in x.iter().enumerate() {
        let mut res = res.get_offset(1, chunk_size);
        party.receive_view(&mut res.a, party.prev_id(), idx);
        c.push(res.a);
    }
    cudarc::nccl::result::group_end().unwrap();

    distance_comparator.open_results(&a, &b, &c, results_ptrs, db_sizes);
}

fn get_merged_results(host_results: &[Vec<u32>], db_sizes: &[usize]) -> Vec<u32> {
    let cumsums = db_sizes
        .iter()
        .scan(0, |acc, &x| {
            let res = *acc;
            *acc += x;
            Some(res as u32)
        })
        .collect::<Vec<_>>();

    let mut results = vec![];
    for j in 0..host_results[0].len() {
        let mut match_entry = u32::MAX;
        for i in 0..host_results.len() {
            if host_results[i][j] != u32::MAX {
                match_entry = cumsums[i] + host_results[i][j];
                break;
            }
        }

        results.push(match_entry);

        // DEBUG
        println!(
            "Query {}: match={} [index: {}]",
            j,
            match_entry != u32::MAX,
            match_entry
        );
    }
    results
}

fn await_streams(streams: &[u64]) {
    for i in 0..streams.len() {
        unsafe {
            synchronize(streams[i] as *mut _).unwrap();
        }
    }
}

fn dtod_at_offset(
    dst: u64,
    dst_offset: usize,
    src: u64,
    src_offset: usize,
    len: usize,
    stream_ptr: u64,
) {
    unsafe {
        result::memcpy_dtod_async(
            dst + dst_offset as u64,
            src + src_offset as u64,
            len,
            stream_ptr as *mut _,
        )
        .unwrap();
    }
}

fn device_ptrs_to_shares<T>(
    a: &[u64],
    b: &[u64],
    lens: &[usize],
    devs: &[Arc<CudaDevice>],
) -> Vec<ChunkShare<T>> {
    let a = device_ptrs_to_slices(a, lens, devs);
    let b = device_ptrs_to_slices(b, lens, devs);

    a.into_iter()
        .zip(b.into_iter())
        .map(|(a, b)| ChunkShare::new(a, b))
        .collect::<Vec<_>>()
}

/// Internal helper function to derive a new seed from the given seed and nonce.
fn derive_seed(seed: [u32; 8], nonce: usize) -> eyre::Result<[u32; 8]> {
    let pseudo_rand_key = KDF_SALT.extract(bytemuck::cast_slice(&seed));
    let nonce = nonce.to_be_bytes();
    let context = vec![nonce.as_slice()];
    let output_key_material: Okm<Algorithm> =
        pseudo_rand_key.expand(&context, HKDF_SHA256).unwrap();
    let mut result = [0u32; 8];
    output_key_material
        .fill(bytemuck::cast_slice_mut(&mut result))
        .unwrap();
    Ok(result)
}

/// Applies a KDF to the given seeds to derive new seeds.
fn next_chacha_seeds(seeds: ([u32; 8], [u32; 8])) -> eyre::Result<([u32; 8], [u32; 8])> {
    let nonce = KDF_NONCE.fetch_add(1, SeqCst);
    Ok((derive_seed(seeds.0, nonce)?, derive_seed(seeds.1, nonce)?))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let Opt {
        queue,
        party_id,
        bootstrap_url,
        path,
        results_topic_arn,
    } = Opt::parse();
    let path = path.unwrap_or(DEFAULT_PATH.to_string());

    let code_db_path = format!("{}/{}", path, DB_CODE_FILE);
    let mask_db_path = format!("{}/{}", path, DB_MASK_FILE);

    let shuffle_rng = StdRng::seed_from_u64(SHUFFLE_SEED);

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let sqs_client = Client::new(&shared_config);
    let sns_client = SNSClient::new(&shared_config);

    // Init RNGs
    let own_key_id = KMS_KEY_IDS[party_id];
    let dh_pairs = match party_id {
        0 => (1usize, 2usize),
        1 => (2usize, 0usize),
        2 => (0usize, 1usize),
        _ => unimplemented!(),
    };

    let chacha_seeds = (
        bytemuck::cast(derive_shared_secret(own_key_id, KMS_KEY_IDS[dh_pairs.0]).await?),
        bytemuck::cast(derive_shared_secret(own_key_id, KMS_KEY_IDS[dh_pairs.1]).await?),
    );

    // Generate or load DB
    let (codes_db, masks_db) = if metadata(&code_db_path).is_ok() && metadata(&mask_db_path).is_ok()
    {
        (
            read_mmap_file(&code_db_path)?,
            read_mmap_file(&mask_db_path)?,
        )
    } else {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        let codes_db = db
            .db
            .iter()
            .map(|iris| {
                GaloisRingIrisCodeShare::encode_iris_code(
                    &iris.code,
                    &iris.mask,
                    &mut StdRng::seed_from_u64(RNG_SEED),
                )[party_id]
                    .coefs
            })
            .flatten()
            .collect::<Vec<_>>();

        let masks_db = db
            .db
            .iter()
            .map(|iris| {
                GaloisRingIrisCodeShare::encode_mask_code(
                    &iris.mask,
                    &mut StdRng::seed_from_u64(RNG_SEED),
                )[party_id]
                    .coefs
            })
            .flatten()
            .collect::<Vec<_>>();

        write_mmap_file(&code_db_path, &codes_db)?;
        write_mmap_file(&mask_db_path, &masks_db)?;
        (codes_db, masks_db)
    };

    println!("Starting engines...");

    let device_manager = Arc::new(DeviceManager::init());

    // Phase 1 Setup
    let mut codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        DB_SIZE + DB_BUFFER,
        QUERIES,
        next_chacha_seeds(chacha_seeds)?,
        bootstrap_url.clone(),
        Some(true),
        Some(4000),
    );

    let mut masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        DB_SIZE + DB_BUFFER,
        QUERIES,
        next_chacha_seeds(chacha_seeds)?,
        bootstrap_url.clone(),
        Some(true),
        Some(4001),
    );

    let code_db_slices = codes_engine.load_db(&codes_db, DB_SIZE, DB_SIZE + DB_BUFFER);
    let mask_db_slices = masks_engine.load_db(&masks_db, DB_SIZE, DB_SIZE + DB_BUFFER);

    // Engines for inflight queries
    let mut batch_codes_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        QUERIES * device_manager.device_count(),
        QUERIES,
        next_chacha_seeds(chacha_seeds)?,
        bootstrap_url.clone(),
        Some(true),
        Some(4002),
    );
    let mut batch_masks_engine = ShareDB::init(
        party_id,
        device_manager.clone(),
        QUERIES * device_manager.device_count(),
        QUERIES,
        next_chacha_seeds(chacha_seeds)?,
        bootstrap_url.clone(),
        Some(true),
        Some(4003),
    );

    // Phase 2 Setup
    let phase2_chunk_size =
        (QUERIES * DB_SIZE / device_manager.device_count()).div_ceil(2048) * 2048;
    let phase2_chunk_size_max =
        (QUERIES * (DB_SIZE + DB_BUFFER) / device_manager.device_count()).div_ceil(2048) * 2048;
    let phase2_batch_chunk_size = (QUERIES * QUERIES).div_ceil(2048) * 2048;

    let phase2_batch = Arc::new(Mutex::new(Circuits::new(
        party_id,
        phase2_batch_chunk_size,
        phase2_batch_chunk_size,
        next_chacha_seeds(chacha_seeds)?,
        bootstrap_url.clone(),
        Some(4004),
    )));

    let phase2 = Arc::new(Mutex::new(Circuits::new(
        party_id,
        phase2_chunk_size,
        phase2_chunk_size_max / 64,
        next_chacha_seeds(chacha_seeds)?,
        bootstrap_url.clone(),
        Some(4005),
    )));

    let distance_comparator = Arc::new(Mutex::new(DistanceComparator::init(QUERIES)));

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
        results.push(distance_comparator.lock().unwrap().prepare_results());
        batch_results.push(distance_comparator.lock().unwrap().prepare_results());
        final_results.push(distance_comparator.lock().unwrap().prepare_final_results());
    }

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

    let current_db_size: Vec<usize> =
        vec![DB_SIZE / device_manager.device_count(); device_manager.device_count()];
    let query_db_size = vec![QUERIES; device_manager.device_count()];
    let current_db_size_mutex = current_db_size
        .iter()
        .map(|&s| Arc::new(Mutex::new(s)))
        .collect::<Vec<_>>();

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<Vec<u32>>(32); // TODO: pick some buffer value
    let rx_sns_client = sns_client.clone();
    tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            for id in message {
                // TODO: write each result to postgres

                // Notify consumers about result
                println!("Sending results back to SNS...");
                let (db_index, is_match) = match id {
                    u32::MAX => (None, false),
                    _ => (Some(id), true),
                };
                let result_event =
                    ResultEvent::new(party_id, db_index, is_match, "dummy".to_string(), 0); // TODO

                rx_sns_client
                    .publish()
                    .topic_arn(&results_topic_arn)
                    .message(serde_json::to_string(&result_event).unwrap())
                    .send()
                    .await
                    .unwrap();
            }
        }
    });

    println!("All systems ready.");

    let mut total_time = Instant::now();
    let mut batch_times = Duration::from_secs(0);

    // Main loop
    for _i in 0..N_BATCHES {
        // Skip first iteration
        if _i == 1 {
            total_time = Instant::now();
            batch_times = Duration::from_secs(0);
        }
        let now = Instant::now();
        let batch = receive_batch(party_id, &sqs_client, &queue).await?;
        println!("Received batch in {:?}", now.elapsed());
        batch_times += now.elapsed();

        let (code_query, mask_query, code_query_insert, mask_query_insert) =
            spawn_blocking(move || {
                let code_query = prepare_query_shares(batch.query.code);
                let mask_query = prepare_query_shares(batch.query.mask);
                let code_query_insert = prepare_query_shares(batch.db.code);
                let mask_query_insert = prepare_query_shares(batch.db.mask);
                (code_query, mask_query, code_query_insert, mask_query_insert)
            })
            .await?;

        let mut timers = vec![];

        let request_streams = &streams[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_cublas_handles = &cublas_handles[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_results = &results[request_counter % MAX_CONCURRENT_REQUESTS];
        let request_results_batch = &batch_results[request_counter % MAX_CONCURRENT_REQUESTS];
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
        let code_query_insert =
            device_manager.htod_transfer_query(&code_query_insert, request_streams);
        let mask_query_insert =
            device_manager.htod_transfer_query(&mask_query_insert, request_streams);
        let code_query_sums =
            codes_engine.query_sums(&code_query, request_streams, request_cublas_handles);
        let mask_query_sums =
            masks_engine.query_sums(&mask_query, request_streams, request_cublas_handles);
        let code_query_insert_sums =
            codes_engine.query_sums(&code_query_insert, request_streams, request_cublas_handles);
        let mask_query_insert_sums =
            masks_engine.query_sums(&mask_query_insert, request_streams, request_cublas_handles);

        // update the db size, skip this for the first two
        if request_counter > 2 {
            // We have two streams working concurrently, we'll await the stream before
            // previous one
            let previous_previous_streams =
                &streams[(request_counter - 2) % MAX_CONCURRENT_REQUESTS];
            device_manager.await_event(previous_previous_streams, &previous_previous_stream_event);
            device_manager.await_streams(previous_previous_streams);
        }

        let current_db_size_stream = current_db_size_mutex
            .iter()
            .map(|e| *e.lock().unwrap())
            .collect::<Vec<_>>();

        // BLOCK 1: calculate individual dot products
        device_manager.await_event(request_streams, &current_dot_event);

        // ---- START BATCH DEDUP ----

        batch_codes_engine.dot(
            &code_query,
            &code_query_insert,
            &query_db_size,
            request_streams,
            request_cublas_handles,
        );

        batch_masks_engine.dot(
            &mask_query,
            &mask_query_insert,
            &query_db_size,
            request_streams,
            request_cublas_handles,
        );

        batch_codes_engine.dot_reduce(
            &code_query_sums,
            &code_query_insert_sums,
            &query_db_size,
            request_streams,
        );

        batch_masks_engine.dot_reduce(
            &mask_query_sums,
            &mask_query_insert_sums,
            &query_db_size,
            request_streams,
        );

        batch_codes_engine.reshare_results(&query_db_size, request_streams);
        batch_masks_engine.reshare_results(&query_db_size, request_streams);

        // ---- END BATCH DEDUP ----

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

        codes_engine.reshare_results(&current_db_size_stream, request_streams);
        masks_engine.reshare_results(&current_db_size_stream, request_streams);

        debug_record_event!(device_manager, request_streams, timers);

        device_manager.record_event(request_streams, &next_exchange_event);

        println!("phase 1 done");

        // Convert a bunch of objects to device pointers to not have copies of memory
        let thread_streams = request_streams
            .iter()
            .map(|s| s.stream as u64)
            .collect::<Vec<_>>();
        let thread_device_manager = device_manager.clone();
        // let thread_devs = thread_device_manager.devices();
        let thread_evts = end_timer.iter().map(|e| *e as u64).collect::<Vec<_>>();
        let mut thread_shuffle_rng = shuffle_rng.clone();
        let thread_current_stream_event = current_stream_event
            .iter()
            .map(|e| *e as u64)
            .collect::<Vec<_>>();
        let thread_current_db_size_mutex = current_db_size_mutex
            .iter()
            .map(Arc::clone)
            .collect::<Vec<_>>();
        let db_sizes_batch = query_db_size.clone();
        let thread_request_results_batch = device_ptrs(&request_results_batch);
        let thread_request_results = device_ptrs(&request_results);
        let thread_request_final_results = device_ptrs(&request_final_results);

        // Batch phase 1 results
        let thread_code_results_batch = device_ptrs(&batch_codes_engine.results);
        let thread_code_results_peer_batch = device_ptrs(&batch_codes_engine.results_peer);
        let thread_mask_results_batch = device_ptrs(&batch_masks_engine.results);
        let thread_mask_results_peer_batch = device_ptrs(&batch_masks_engine.results_peer);

        // DB phase 1 results
        let thread_code_results = device_ptrs(&codes_engine.results);
        let thread_code_results_peer = device_ptrs(&codes_engine.results_peer);
        let thread_mask_results = device_ptrs(&masks_engine.results);
        let thread_mask_results_peer = device_ptrs(&masks_engine.results_peer);

        let thread_phase2 = phase2.clone();
        let thread_phase2_batch = phase2_batch.clone();
        let thread_distance_comparator = distance_comparator.clone();
        let thread_code_db_slices = slice_tuples_to_ptrs(&code_db_slices);
        let thread_mask_db_slices = slice_tuples_to_ptrs(&mask_db_slices);

        let thread_sender = tx.clone();

        let handle = spawn_blocking(move || {
            // Wait for Phase 1 to finish
            await_streams(&thread_streams);

            let thread_devs = thread_device_manager.devices();
            let mut thread_phase2_batch = thread_phase2_batch.lock().unwrap();
            let mut thread_phase2 = thread_phase2.lock().unwrap();
            let tmp_distance_comparator = thread_distance_comparator.lock().unwrap();
            let (result_sizes, db_sizes): (Vec<_>, Vec<_>) = thread_current_db_size_mutex
                .iter()
                .map(|e| {
                    let db_size = *e.lock().unwrap();
                    (db_size * QUERIES, db_size)
                })
                .unzip();

            let result_sizes_batch = db_sizes_batch
                .iter()
                .map(|&e| e * QUERIES)
                .collect::<Vec<_>>();

            let mut code_dots_batch: Vec<ChunkShare<u16>> = device_ptrs_to_shares(
                &thread_code_results_batch,
                &thread_code_results_peer_batch,
                &result_sizes_batch,
                &thread_devs,
            );
            let mut mask_dots_batch: Vec<ChunkShare<u16>> = device_ptrs_to_shares(
                &thread_mask_results_batch,
                &thread_mask_results_peer_batch,
                &result_sizes_batch,
                &thread_devs,
            );

            let mut code_dots: Vec<ChunkShare<u16>> = device_ptrs_to_shares(
                &thread_code_results,
                &thread_code_results_peer,
                &result_sizes,
                &thread_devs,
            );
            let mut mask_dots: Vec<ChunkShare<u16>> = device_ptrs_to_shares(
                &thread_mask_results,
                &thread_mask_results_peer,
                &result_sizes,
                &thread_devs,
            );

            // We only use the default streams of the devices, therefore Phase 2's are never
            // running concurrently
            let streams = thread_phase2
                .get_devices()
                .iter()
                .map(|d| *d.cu_stream() as u64)
                .collect::<Vec<_>>();

            // Phase 2 [Batch]: compare each result against threshold
            thread_phase2_batch.compare_threshold_masked_many(&code_dots_batch, &mask_dots_batch);

            // Phase 2 [Batch]: Reveal the binary results
            let res = thread_phase2_batch.take_result_buffer();
            let mut thread_request_results_slice_batch: Vec<CudaSlice<u32>> = device_ptrs_to_slices(
                &thread_request_results_batch,
                &vec![QUERIES; thread_devs.len()],
                &thread_devs,
            );

            let chunk_size_batch = thread_phase2_batch.chunk_size();
            open(
                &mut thread_phase2_batch,
                &res,
                &tmp_distance_comparator,
                &thread_request_results_slice_batch,
                chunk_size_batch,
                &db_sizes_batch,
            );
            thread_phase2_batch.return_result_buffer(res);

            // Phase 2 [DB]: compare each result against threshold
            thread_phase2.compare_threshold_masked_many(&code_dots, &mask_dots);

            // Phase 2 [DB]: Reveal the binary results
            let res = thread_phase2.take_result_buffer();
            let mut thread_request_results_slice: Vec<CudaSlice<u32>> = device_ptrs_to_slices(
                &thread_request_results,
                &vec![QUERIES; thread_devs.len()],
                &thread_devs,
            );

            let chunk_size = thread_phase2.chunk_size();
            open(
                &mut thread_phase2,
                &res,
                &tmp_distance_comparator,
                &thread_request_results_slice,
                chunk_size,
                &db_sizes,
            );
            thread_phase2.return_result_buffer(res);

            // Merge results and fetch matching indices
            let host_results = tmp_distance_comparator.merge_results(
                &thread_request_results_batch,
                &thread_request_results,
                &thread_request_final_results,
                &streams,
            );

            // Evaluate the results across devices
            let merged_results = get_merged_results(&host_results, &db_sizes);
            let insertion_list = merged_results
                .iter()
                .enumerate()
                .filter(|&(_idx, &num)| num == u32::MAX)
                .map(|(idx, _num)| idx)
                .collect::<Vec<_>>();

            let mut insertion_list = insertion_list
                .chunks(QUERIES / ROTATIONS / thread_devs.len())
                .collect::<Vec<_>>();
            insertion_list.shuffle(&mut thread_shuffle_rng);

            // DEBUG
            println!("Insertion list: {:?}", insertion_list);

            for i in 0..thread_devs.len() {
                thread_devs[i].bind_to_thread().unwrap();
                if insertion_list.len() > i {
                    let mut old_db_size = *thread_current_db_size_mutex[i].lock().unwrap();
                    for insertion_idx in insertion_list[i] {
                        // Append to codes and masks db
                        for (db, query, sums) in [
                            (
                                &thread_code_db_slices,
                                &code_query_insert,
                                &code_query_insert_sums,
                            ),
                            (
                                &thread_mask_db_slices,
                                &mask_query_insert,
                                &mask_query_insert_sums,
                            ),
                        ] {
                            dtod_at_offset(
                                db.0 .0[i],
                                old_db_size * IRIS_CODE_LENGTH,
                                query.0[i],
                                insertion_idx * IRIS_CODE_LENGTH * ROTATIONS,
                                IRIS_CODE_LENGTH,
                                streams[i],
                            );

                            dtod_at_offset(
                                db.0 .1[i],
                                old_db_size * IRIS_CODE_LENGTH,
                                query.1[i],
                                insertion_idx * IRIS_CODE_LENGTH * ROTATIONS,
                                IRIS_CODE_LENGTH,
                                streams[i],
                            );

                            dtod_at_offset(
                                db.1 .0[i],
                                old_db_size * mem::size_of::<u32>(),
                                sums.0[i],
                                insertion_idx * mem::size_of::<u32>() * ROTATIONS,
                                mem::size_of::<u32>(),
                                streams[i],
                            );

                            dtod_at_offset(
                                db.1 .1[i],
                                old_db_size * mem::size_of::<u32>(),
                                sums.1[i],
                                insertion_idx * mem::size_of::<u32>() * ROTATIONS,
                                mem::size_of::<u32>(),
                                streams[i],
                            );
                        }
                        old_db_size += 1;
                    }

                    // Write new db sizes to device
                    *thread_current_db_size_mutex[i].lock().unwrap() +=
                        insertion_list[i].len() as usize;

                    // DEBUG
                    println!(
                        "Updating DB size on device {}: {:?}",
                        i,
                        *thread_current_db_size_mutex[i].lock().unwrap()
                    );
                }

                // Update Phase 2 chunk size to max all db sizes on all devices
                let max_db_size = thread_current_db_size_mutex
                    .iter()
                    .map(|e| *e.lock().unwrap())
                    .max()
                    .unwrap();
                let new_chunk_size = (QUERIES * max_db_size).div_ceil(2048) * 2048;
                assert!(new_chunk_size <= phase2_chunk_size_max);
                thread_phase2.set_chunk_size(new_chunk_size / 64);

                // Emit stream finished event to unblock the stream after the following
                unsafe {
                    event::record(
                        thread_current_stream_event[i] as *mut _,
                        thread_streams[i] as *mut _,
                    )
                    .unwrap();

                    // DEBUG: emit event to measure time for e2e process
                    event::record(thread_evts[i] as *mut _, thread_streams[i] as *mut _).unwrap();
                }
            }

            // Pass to internal sender thread
            thread_sender.try_send(merged_results).unwrap();

            // Make sure to not call `Drop` on those
            forget_vec!(code_dots);
            forget_vec!(mask_dots);
            forget_vec!(code_dots_batch);
            forget_vec!(mask_dots_batch);
            forget_vec!(thread_request_results_slice);
            forget_vec!(thread_request_results_slice_batch);
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

        // Await the last one for benching
        if _i == N_BATCHES - 1 {
            handle.await?;
        }
    }

    println!(
        "Total time for {} iterations: {:?}",
        N_BATCHES - 1,
        total_time.elapsed() - batch_times
    );

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
