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
    sys::{CUdeviceptr, CUstream, CUstream_st},
    CudaDevice, CudaSlice,
};
use gpu_iris_mpc::{
    config::Config,
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
    store::Store,
    threshold_ring::protocol::{ChunkShare, Circuits},
};
use lazy_static::lazy_static;
use rand::{rngs::StdRng, SeedableRng};
use ring::hkdf::{Algorithm, Okm, Salt, HKDF_SHA256};
use std::{
    fs::metadata,
    mem,
    path::PathBuf,
    sync::{atomic::AtomicUsize, Arc, Mutex},
    time::{Duration, Instant},
};
use telemetry_batteries::metrics::statsd::StatsdBattery;
use telemetry_batteries::tracing::datadog::DatadogBattery;
use telemetry_batteries::tracing::TracingShutdownHandle;
use tokio::{
    runtime,
    sync::mpsc,
    task::{spawn_blocking, JoinHandle},
    time::sleep,
};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use gpu_iris_mpc::config;
use gpu_iris_mpc::helpers::aws::{
    NODE_ID_MESSAGE_ATTRIBUTE_NAME, SPAN_ID_MESSAGE_ATTRIBUTE_NAME, TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
};

const REGION: &str = "eu-north-1";
const DB_SIZE: usize = 8 * 1_000;
const DB_BUFFER: usize = 8 * 1_000;
const N_QUERIES: usize = 32;
const N_BATCHES: usize = 100;
const RNG_SEED: u64 = 42;
/// The number of batches before a stream is re-used.
const MAX_BATCHES_BEFORE_REUSE: usize = 5;

/// The number of batches that are launched concurrently.
///
/// Code can run concurrently in:
/// - requests `N` and `N - 1`,
/// - requests `N` and `N - MAX_CONCURRENT_BATCHES`, because the next request
///   phase 1 awaits completion of the request `MAX_CONCURRENT_BATCHES` behind
///   it, or
/// - request `N` only, because phase 2 limits the critical section to one
///   batch.
/// - requests `N` and `N - 1`,
/// - requests `N` and `N - MAX_CONCURRENT_BATCHES`, because the next request
///   phase 1 awaits completion of the request `MAX_CONCURRENT_BATCHES` behind
///   it, or
/// - request `N` only, because phase 2 limits the critical section to one
///   batch.
const MAX_CONCURRENT_BATCHES: usize = 2;

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
    static ref RESULTS_INIT_HOST: Vec<u32> = vec![u32::MAX; N_QUERIES * ROTATIONS];
    static ref FINAL_RESULTS_INIT_HOST: Vec<u32> = vec![u32::MAX; N_QUERIES];
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
struct BatchMetadata {
    node_id: String,
    trace_id: String,
    span_id: String,
}

#[derive(Default)]
struct BatchQuery {
    pub request_ids: Vec<String>,
    pub metadata: Vec<BatchMetadata>,
    pub query: BatchQueryEntries,
    pub db: BatchQueryEntries,
    pub store: BatchQueryEntries,
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
            .max_number_of_messages(10i32)
            .queue_url(queue_url)
            .send()
            .await?;

        if let Some(messages) = rcv_message_output.messages {
            for sqs_message in messages {
                let message: SQSMessage = serde_json::from_str(sqs_message.body().unwrap())?;
                let message: SMPCRequest = serde_json::from_str(&message.message)?;

                let message_attributes = sqs_message.message_attributes.unwrap_or_default();

                let mut batch_metadata = BatchMetadata::default();

                if let Some(node_id) = message_attributes.get(NODE_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let node_id = node_id.string_value().unwrap();
                    batch_metadata.node_id = node_id.to_string();
                }
                if let Some(trace_id) = message_attributes.get(TRACE_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let trace_id = trace_id.string_value().unwrap();
                    batch_metadata.trace_id = trace_id.to_string();
                }
                if let Some(span_id) = message_attributes.get(SPAN_ID_MESSAGE_ATTRIBUTE_NAME) {
                    let span_id = span_id.string_value().unwrap();
                    batch_metadata.span_id = span_id.to_string();
                }

                batch_query.request_ids.push(message.clone().request_id);
                batch_query.metadata.push(batch_metadata);

            let (
                store_iris_shares,
                store_mask_shares,
                db_iris_shares,
                db_mask_shares,
                iris_shares,
                mask_shares,
            ) = spawn_blocking(move || {
                let mut iris_share =
                    GaloisRingIrisCodeShare::new(party_id + 1, message.get_iris_shares());
                let mut mask_share =
                    GaloisRingIrisCodeShare::new(party_id + 1, message.get_mask_shares());

                // Original for storage.
                let store_iris_shares = iris_share.clone();
                let store_mask_shares = mask_share.clone();

                // With rotations for in-memory database.
                let db_iris_shares = iris_share.all_rotations();
                let db_mask_shares = mask_share.all_rotations();

                // With Lagrange interpolation.
                GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut iris_share);
                GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut mask_share);

                (
                    store_iris_shares,
                    store_mask_shares,
                    db_iris_shares,
                    db_mask_shares,
                    iris_share.all_rotations(),
                    mask_share.all_rotations(),
                )
            })
            .await?;

            batch_query.store.code.push(store_iris_shares);
            batch_query.store.mask.push(store_mask_shares);
            batch_query.db.code.extend(db_iris_shares);
            batch_query.db.mask.extend(db_mask_shares);
            batch_query.query.code.extend(iris_shares);
            batch_query.query.mask.extend(mask_shares);

                // TODO: we should only delete after processing
                client
                    .delete_message()
                    .queue_url(queue_url)
                    .receipt_handle(sqs_message.receipt_handle.unwrap())
                    .send()
                    .await?;
            }
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
) -> (
    (Vec<CUdeviceptr>, Vec<CUdeviceptr>),
    (Vec<CUdeviceptr>, Vec<CUdeviceptr>),
) {
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

fn get_merged_results(host_results: &[Vec<u32>], n_devices: usize) -> Vec<u32> {
    let mut results = vec![];
    for j in 0..host_results[0].len() {
        let mut match_entry = u32::MAX;
        for i in 0..host_results.len() {
            let match_idx = host_results[i][j] * n_devices as u32 + i as u32;
            if host_results[i][j] != u32::MAX && match_idx < match_entry {
                match_entry = match_idx;
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

fn await_streams(streams: &mut [&mut CUstream_st]) {
    for i in 0..streams.len() {
        // SAFETY: these streams have already been created, and the caller holds a
        // reference to their CudaDevice, which makes sure they aren't dropped.
        unsafe {
            synchronize(streams[i]).unwrap();
        }
    }
}

fn dtod_at_offset(
    dst: CUdeviceptr,
    dst_offset: usize,
    src: CUdeviceptr,
    src_offset: usize,
    len: usize,
    stream_ptr: CUstream,
) {
    unsafe {
        result::memcpy_dtod_async(
            dst + dst_offset as CUdeviceptr,
            src + src_offset as CUdeviceptr,
            len,
            stream_ptr,
        )
        .unwrap();
    }
}

fn device_ptrs_to_shares<T>(
    a: &[CUdeviceptr],
    b: &[CUdeviceptr],
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

fn distribute_insertions(results: &[usize], db_sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut ret = vec![vec![]; db_sizes.len()];
    let start = db_sizes
        .iter()
        .position(|&x| x == *db_sizes.iter().min().unwrap())
        .unwrap();

    let mut c = start;
    for &r in results {
        ret[c].push(r);
        c = (c + 1) % db_sizes.len();
    }
    ret
}

fn reset_results(
    devs: &[Arc<CudaDevice>],
    dst: &[CUdeviceptr],
    src: &[u32],
    streams: &mut [*mut CUstream_st],
) {
    for i in 0..devs.len() {
        devs[i].bind_to_thread().unwrap();
        unsafe { result::memcpy_htod_async(dst[i], src, streams[i]) }.unwrap();
    }
}

pub fn calculate_insertion_indices(
    merged_results: &mut [u32],
    insertion_list: &[Vec<usize>],
    db_sizes: &[usize],
) -> Vec<bool> {
    let mut matches = vec![true; N_QUERIES];
    let mut last_db_index = db_sizes.iter().sum::<usize>() as u32;
    let (mut min_index, mut min_index_val) = (0, usize::MAX);
    for (i, list) in insertion_list.iter().enumerate() {
        if let Some(&first_val) = list.first() {
            if first_val < min_index_val {
                min_index_val = first_val;
                min_index = i;
            }
        }
    }
    let mut c: usize = 0;
    loop {
        for i in 0..insertion_list.len() {
            let idx = (i + min_index) % insertion_list.len();

            if c >= insertion_list[idx].len() {
                return matches;
            }

            let insert_idx = insertion_list[idx][c];
            merged_results[insert_idx] = last_db_index;
            matches[insert_idx] = false;
            last_db_index += 1;
        }
        c += 1;
    }
}

#[derive(Parser)]
#[clap(version)]
pub struct Args {
    #[clap(short, long, env)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    dotenvy::dotenv().ok();

    let args = Args::parse();

    let config: Config = config::load_config("SMPC", args.config.as_deref())?;

    let _tracing_shutdown_handle = if let Some(service) = &config.service {
        let tracing_shutdown_handle = DatadogBattery::init(
            service.traces_endpoint.as_deref(),
            &service.service_name,
            None,
            true,
        );

        if let Some(metrics_config) = &service.metrics {
            StatsdBattery::init(
                &metrics_config.host,
                metrics_config.port,
                metrics_config.queue_size,
                metrics_config.buffer_size,
                Some(&metrics_config.prefix),
            )?;
        }

        tracing_shutdown_handle
    } else {
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().pretty().compact())
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .init();

        TracingShutdownHandle
    };

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

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let sqs_client = Client::new(&shared_config);
    let sns_client = SNSClient::new(&shared_config);
    let store = Store::new_from_env().await?;

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
    let (mut codes_db, mut masks_db) =
        if metadata(&code_db_path).is_ok() && metadata(&mask_db_path).is_ok() {
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

    // Load DB from persistent storage.
    for iris in store.iter_irises().await? {
        codes_db.extend(iris.code());
        masks_db.extend(iris.mask());
    }

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

    let code_db_slices = codes_engine.load_db(&codes_db, DB_SIZE, DB_SIZE + DB_BUFFER, true);
    let mask_db_slices = masks_engine.load_db(&masks_db, DB_SIZE, DB_SIZE + DB_BUFFER, true);

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
    for _ in 0..MAX_BATCHES_BEFORE_REUSE {
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

    let mut previous_thread_handle: Option<JoinHandle<()>> = None;
    let mut current_dot_event = device_manager.create_events();
    let mut next_dot_event = device_manager.create_events();
    let mut current_exchange_event = device_manager.create_events();
    let mut next_exchange_event = device_manager.create_events();
    let mut timer_events = vec![];
    let start_timer = device_manager.create_events();
    let mut end_timer = device_manager.create_events();

    let current_db_size: Vec<usize> =
        vec![DB_SIZE / device_manager.device_count(); device_manager.device_count()];
    let query_db_size = vec![QUERIES; device_manager.device_count()];
    let current_db_size_mutex = current_db_size
        .iter()
        .map(|&s| Arc::new(Mutex::new(s)))
        .collect::<Vec<_>>();

    // Start thread that will be responsible for communicating back the results
    let (tx, mut rx) = mpsc::channel::<(Vec<u32>, Vec<String>, Vec<bool>, BatchQueryEntries)>(32); // TODO: pick some buffer value
    let rx_sns_client = sns_client.clone();

    tokio::spawn(async move {
        while let Some((merged_results, request_ids, matches, query_store)) = rx.recv().await {
            for (i, &idx_result) in merged_results.iter().enumerate() {
                // Insert non-matching queries into the persistent store.
                {
                    let codes_and_masks: Vec<(&[u16], &[u16])> = matches
                        .iter()
                        .enumerate()
                        .filter_map(
                            // Find the indices of non-matching queries in the batch.
                            |(query_idx, is_match)| if !is_match { Some(query_idx) } else { None },
                        )
                        .map(|query_idx| {
                            // Get the original vectors from `receive_batch`.
                            let code = &query_store.code[query_idx].coefs[..];
                            let mask = &query_store.mask[query_idx].coefs[..];
                            (code, mask)
                        })
                        .collect();

                    store
                        .insert_irises(&codes_and_masks)
                        .await
                        .expect("failed to persist queries");
                }

                // Notify consumers about result
                println!("Sending results back to SNS...");
                let result_event =
                    ResultEvent::new(party_id, idx_result, matches[i], request_ids[i].clone());

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
    for request_counter in 0..N_BATCHES {
        // **Tensor format of queries**
        //
        // The functions `receive_batch` and `prepare_query_shares` will prepare the _query_ variables as `Vec<Vec<u8>>` formatted as follows:
        //
        // - The inner Vec is a flattening of these dimensions (inner to outer):
        //   - One u8 limb of one iris bit.
        //   - One code: 12800 coefficients.
        //   - One query: all rotated variants of a code.
        //   - One batch: many queries.
        // - The outer Vec is the dimension of the Galois Ring (2):
        //   - A decomposition of each iris bit into two u8 limbs.

        // Skip first iteration
        if request_counter == 1 {
            total_time = Instant::now();
            batch_times = Duration::from_secs(0);
        }
        let now = Instant::now();

        //This batch can consist of N sets of iris_share + mask
        //It also includes a vector of request ids, mapping to the sets above
        let batch = receive_batch(party_id, &sqs_client, &queue).await?;

        // Iterate over a list of tracing payloads, and create logs with mappings to payloads
        // Log at least a "start" event using a log with trace.id and parent.trace.id
        for tracing_payload in batch.metadata.iter() {
            tracing::info!(
                node_id = tracing_payload.node_id,
                dd.trace_id = tracing_payload.trace_id,
                dd.span_id = tracing_payload.span_id,
                "Started processing share",
            );
        }

        // start trace span - with single TraceId and single ParentTraceID
        println!("Received batch in {:?}", now.elapsed());
        batch_times += now.elapsed();

        let (code_query, mask_query, code_query_insert, mask_query_insert, query_store) =
            spawn_blocking(move || {
                // *Query* variant including Lagrange interpolation.
                let code_query = prepare_query_shares(batch.query.code);
                let mask_query = prepare_query_shares(batch.query.mask);
                // *Storage* variant (no interpolation).
                let code_query_insert = prepare_query_shares(batch.db.code);
                let mask_query_insert = prepare_query_shares(batch.db.mask);
                (
                    code_query,
                    mask_query,
                    code_query_insert,
                    mask_query_insert,
                    batch.store,
                )
            })
            .await?;

        let mut timers = vec![];

        let request_streams = &streams[request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_cublas_handles = &cublas_handles[request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_results = &results[request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_results_batch = &batch_results[request_counter % MAX_BATCHES_BEFORE_REUSE];
        let request_final_results = &final_results[request_counter % MAX_BATCHES_BEFORE_REUSE];

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
        if request_counter > MAX_CONCURRENT_BATCHES {
            // We have two streams working concurrently, we'll await the stream before
            // previous one.
            // SAFETY:
            let previous_previous_streams =
                &streams[(request_counter - MAX_CONCURRENT_BATCHES) % MAX_BATCHES_BEFORE_REUSE];
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

        // SAFETY:
        // - We are sending these streams and events to a single thread (without cloning
        //   them), then dropping our references to them (without destroying them).
        // - These pointers are aligned, dereferencable, and initialized.
        // Unique usage:
        // - Streams are re-used after MAX_BATCHES_BEFORE_REUSE threads, but we only
        //   launch MAX_CONCURRENT_BATCHES threads at a time. So this reference performs
        //   the only accesses to its memory across both C and Rust.
        // - New current stream events are created for each batch. They are only re-used
        //   after MAX_CONCURRENT_BATCHES, but we wait for the previous batch to finish
        //   before running that code.
        // - End events are re-used in each thread, but we only end one thread at a
        //   time.
        assert!(MAX_BATCHES_BEFORE_REUSE > MAX_CONCURRENT_BATCHES);
        // into_iter() makes the Rust compiler check that the streams are not re-used.
        let mut thread_streams = request_streams
            .into_iter()
            .map(|s| unsafe { s.stream.as_mut().unwrap() })
            .collect::<Vec<_>>();
        // The compiler can't tell that we wait for the previous batch before re-using
        // these events.
        let mut thread_current_stream_event = current_stream_event
            .iter()
            .map(|e| unsafe { e.as_mut().unwrap() })
            .collect::<Vec<_>>();
        let mut thread_end_timer = end_timer
            .iter()
            .map(|e| unsafe { e.as_mut().unwrap() })
            .collect::<Vec<_>>();

        let thread_device_manager = device_manager.clone();
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

        // SAFETY: phase2 and phase2_batch are only used in one spawned threat at a
        // time.
        let thread_phase2 = phase2.clone();
        let thread_phase2_batch = phase2_batch.clone();
        let thread_distance_comparator = distance_comparator.clone();
        let thread_code_db_slices = slice_tuples_to_ptrs(&code_db_slices);
        let thread_mask_db_slices = slice_tuples_to_ptrs(&mask_db_slices);
        let thread_request_ids = batch.request_ids.clone();
        let thread_sender = tx.clone();


        previous_thread_handle = Some(spawn_blocking(move || {
            // Wait for Phase 1 to finish
            await_streams(&mut thread_streams);

            // Iterate over a list of tracing payloads, and create logs with mappings to payloads
            // Log at least a "start" event using a log with trace.id and parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Phase 1 finished",
                );
            }

            // Wait for Phase 2 of previous round to finish in order to not have them
            // overlapping. SAFETY: waiting here makes sure we don't access
            // these mutable streams or events concurrently:
            // - CUstream: thread_streams (only re-used after MAX_BATCHES_BEFORE_REUSE
            //   batches),
            // - CUevent: thread_current_stream_event, thread_end_timer,
            // - Comm: phase2, phase2_batch.
            if previous_thread_handle.is_some() {
                runtime::Handle::current()
                    .block_on(previous_thread_handle.unwrap())
                    .unwrap();
            }
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

            // Iterate over a list of tracing payloads, and create logs with mappings to payloads
            // Log at least a "start" event using a log with trace.id and parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Phase 2 finished",
                );
            }

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

            // SAFETY:
            // - We only use the default streams of the devices, therefore Phase 2's are
            //   never running concurrently.
            // - We only use the default streams of the devices, therefore Phase 2's are
            //   never running concurrently.
            // - These pointers are aligned, dereferencable, and initialized.
            let mut phase2_streams = thread_phase2
                .get_devices()
                .iter()
                .map(|d| *d.cu_stream())
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

            // Iterate over a list of tracing payloads, and create logs with mappings to payloads
            // Log at least a "start" event using a log with trace.id and parent.trace.id
            for tracing_payload in batch.metadata.iter() {
                tracing::info!(
                    node_id = tracing_payload.node_id,
                    dd.trace_id = tracing_payload.trace_id,
                    dd.span_id = tracing_payload.span_id,
                    "Phase 2 finished",
                );
            }

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
            // Format: host_results[device_index][query_index]
            let host_results = tmp_distance_comparator.merge_results(
                &thread_request_results_batch,
                &thread_request_results,
                &thread_request_final_results,
                &phase2_streams,
            );

            // Evaluate the results across devices
            // Format: merged_results[query_index]
            let mut merged_results = get_merged_results(&host_results, thread_devs.len());

            // List the indices of the queries that did not match.
            let insertion_list = merged_results
                .iter()
                .enumerate()
                .filter(|&(_idx, &num)| num == u32::MAX)
                .map(|(idx, _num)| idx)
                .collect::<Vec<_>>();

            // Spread the insertions across devices.
            let insertion_list = distribute_insertions(&insertion_list, &db_sizes);

            // Calculate the new indices for the inserted queries
            let matches =
                calculate_insertion_indices(&mut merged_results, &insertion_list, &db_sizes);


            for i in 0..thread_devs.len() {
                thread_devs[i].bind_to_thread().unwrap();
                let mut old_db_size = *thread_current_db_size_mutex[i].lock().unwrap();
                for insertion_idx in insertion_list[i].clone() {
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
                            IRIS_CODE_LENGTH * 15 + insertion_idx * IRIS_CODE_LENGTH * ROTATIONS,
                            IRIS_CODE_LENGTH,
                            *&mut phase2_streams[i],
                        );

                        dtod_at_offset(
                            db.0 .1[i],
                            old_db_size * IRIS_CODE_LENGTH,
                            query.1[i],
                            IRIS_CODE_LENGTH * 15 + insertion_idx * IRIS_CODE_LENGTH * ROTATIONS,
                            IRIS_CODE_LENGTH,
                            *&mut phase2_streams[i],
                        );

                        dtod_at_offset(
                            db.1 .0[i],
                            old_db_size * mem::size_of::<u32>(),
                            sums.0[i],
                            mem::size_of::<u32>() * 15
                                + insertion_idx * mem::size_of::<u32>() * ROTATIONS,
                            mem::size_of::<u32>(),
                            *&mut phase2_streams[i],
                        );

                        dtod_at_offset(
                            db.1 .1[i],
                            old_db_size * mem::size_of::<u32>(),
                            sums.1[i],
                            mem::size_of::<u32>() * 15
                                + insertion_idx * mem::size_of::<u32>() * ROTATIONS,
                            mem::size_of::<u32>(),
                            *&mut phase2_streams[i],
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

                // Update Phase 2 chunk size to max all db sizes on all devices
                let max_db_size = thread_current_db_size_mutex
                    .iter()
                    .map(|e| *e.lock().unwrap())
                    .max()
                    .unwrap();
                let new_chunk_size = (QUERIES * max_db_size).div_ceil(2048) * 2048;
                assert!(new_chunk_size <= phase2_chunk_size_max);
                thread_phase2.set_chunk_size(new_chunk_size / 64);

                // Emit stream finished event to unblock the stream after the following stream.
                // Since previous timers are overwritten, only the final end timers are used to
                // calculate the total time.
                //
                // SAFETY:
                // - the events are created before launching the thread, so they are never null.
                // - the streams have already been created, and we hold a reference to their
                //   CudaDevice, which makes sure they aren't dropped.
                unsafe {
                    event::record(
                        *&mut thread_current_stream_event[i],
                        *&mut thread_streams[i],
                    )
                    .unwrap();

                    // DEBUG: emit event to measure time for e2e process
                    event::record(*&mut thread_end_timer[i], *&mut thread_streams[i]).unwrap();
                }
            }

            // Pass to internal sender thread
            thread_sender
                .try_send((merged_results, thread_request_ids, matches, query_store))
                .unwrap();

            // Reset the results buffers for reuse
            reset_results(
                &thread_device_manager.devices(),
                &thread_request_results,
                &RESULTS_INIT_HOST,
                &mut phase2_streams,
            );
            reset_results(
                &thread_device_manager.devices(),
                &thread_request_results_batch,
                &RESULTS_INIT_HOST,
                &mut phase2_streams,
            );
            reset_results(
                &thread_device_manager.devices(),
                &thread_request_final_results,
                &FINAL_RESULTS_INIT_HOST,
                &mut phase2_streams,
            );

            // Make sure to not call `Drop` on those
            forget_vec!(code_dots);
            forget_vec!(mask_dots);
            forget_vec!(code_dots_batch);
            forget_vec!(mask_dots_batch);
            forget_vec!(thread_request_results_slice);
            forget_vec!(thread_request_results_slice_batch);
        }));

        // Prepare for next batch
        timer_events.push(timers);

        previous_previous_stream_event = previous_stream_event;
        previous_stream_event = current_stream_event;
        current_stream_event = device_manager.create_events();
        current_dot_event = next_dot_event;
        current_exchange_event = next_exchange_event;
        next_dot_event = device_manager.create_events();
        next_exchange_event = device_manager.create_events();

        println!("CPU time of one iteration {:?}", now.elapsed());

        // wrap up span context
    }

    // Await the last thread for benching
    previous_thread_handle.unwrap().await?;

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
            let total_time = elapsed(start_timer[i], *&mut end_timer[i]).unwrap();
            println!("Total time: {:?}", total_time);
        }
    }

    Ok(())
}
