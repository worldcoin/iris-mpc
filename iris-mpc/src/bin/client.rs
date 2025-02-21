#![allow(clippy::needless_range_loop)]

use aws_config::retry::RetryConfig;
use aws_sdk_sns::{config::Region, Client};
use base64::{engine::general_purpose, Engine};
use clap::Parser;
use eyre::{Context, ContextCompat};
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare,
    helpers::{
        key_pair::download_public_key,
        sha256::sha256_as_hex_string,
        smpc_request::{IrisCodeSharesJSON, UniquenessRequest, UNIQUENESS_MESSAGE_TYPE},
        smpc_response::create_message_type_attribute_map,
        sqs_s3_helper::upload_file_and_generate_presigned_url,
    },
    iris_db::{db::IrisDB, iris::IrisCode},
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::to_string;
use sodiumoxide::crypto::{box_::PublicKey, sealedbox};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    sync::{Mutex, Semaphore},
    time::sleep,
};
use uuid::Uuid;

const MAX_CONCURRENT_REQUESTS: usize = 16;
const BATCH_SIZE: usize = 64;
const N_BATCHES: usize = 100;
const WAIT_AFTER_BATCH: Duration = Duration::from_secs(2);
const RNG_SEED_SERVER: u64 = 42;
const DB_SIZE: usize = 8 * 1_000;
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

#[derive(Debug, Parser, Clone)]
pub struct Opt {
    #[arg(long, env, required = true)]
    pub request_topic_arn: String,

    #[arg(long, env, required = true)]
    pub request_topic_region: String,

    #[arg(long, env, required = true)]
    pub requests_bucket_name: String,

    #[arg(long, env, required = true)]
    pub public_key_base_url: String,

    #[arg(long, env, required = true)]
    pub requests_bucket_region: String,

    #[arg(long, env)]
    pub db_index: Option<usize>,

    #[arg(long, env)]
    pub rng_seed: Option<u64>,

    #[arg(long, env)]
    pub n_repeat: Option<usize>,

    #[arg(long, env)]
    pub random: Option<bool>,
}

/// The core client functionality is moved into an async function so it can be
/// called from a benchmark harness without spawning a new process.
pub async fn run_client(opts: Opt) -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let Opt {
        public_key_base_url,
        requests_bucket_name,
        requests_bucket_region,
        request_topic_arn,
        request_topic_region,
        db_index,
        rng_seed,
        n_repeat,
        random,
    } = opts;

    // Download public keys for each participant.
    let mut shares_encryption_public_keys: Vec<PublicKey> = Vec::new();
    for i in 0..3 {
        let public_key_string =
            download_public_key(public_key_base_url.to_string(), i.to_string()).await?;
        let public_key_bytes = general_purpose::STANDARD
            .decode(public_key_string)
            .context("Failed to decode public key")?;
        let public_key =
            PublicKey::from_slice(&public_key_bytes).context("Failed to parse public key")?;
        shares_encryption_public_keys.push(public_key);
    }

    let n_repeat = n_repeat.unwrap_or(0);

    let region_provider = Region::new(request_topic_region);
    let requests_sns_config = aws_config::from_env()
        .region(region_provider)
        .retry_config(RetryConfig::standard().with_max_attempts(5))
        .load()
        .await;
    let requests_sns_client = Client::new(&requests_sns_config);

    let db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(RNG_SEED_SERVER));

    let expected_results: Arc<Mutex<HashMap<String, Option<u32>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let requests: Arc<Mutex<HashMap<String, IrisCode>>> = Arc::new(Mutex::new(HashMap::new()));
    let responses: Arc<Mutex<HashMap<u32, IrisCode>>> = Arc::new(Mutex::new(HashMap::new()));
    let db: Arc<Mutex<IrisDB>> = Arc::new(Mutex::new(db));
    let requests_sns_client: Arc<Client> = Arc::new(requests_sns_client);

    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_REQUESTS));

    for batch_idx in 0..N_BATCHES {
        let mut handles = Vec::new();
        for batch_query_idx in 0..BATCH_SIZE {
            let shares_encryption_public_keys2 = shares_encryption_public_keys.clone();
            let requests_sns_client2 = requests_sns_client.clone();
            let thread_db2 = db.clone();
            let thread_expected_results2 = expected_results.clone();
            let thread_requests2 = requests.clone();
            let thread_responses2 = responses.clone();
            let request_topic_arn = request_topic_arn.clone();
            let requests_bucket_region = requests_bucket_region.clone();
            let requests_bucket_name = requests_bucket_name.clone();
            let semaphore = Arc::clone(&semaphore);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await;

                let mut rng = if let Some(rng_seed) = rng_seed {
                    StdRng::seed_from_u64(rng_seed)
                } else {
                    StdRng::from_entropy()
                };

                let request_id = Uuid::new_v4();

                let template = if random.unwrap_or(false) {
                    // Automatic random tests
                    let responses_len = {
                        let tmp = thread_responses2.lock().await;
                        tmp.len()
                    };

                    let options = if responses_len == 0 { 2 } else { 3 };

                    match rng.gen_range(0..options) {
                        0 => {
                            println!("Sending new iris code");
                            {
                                let mut tmp = thread_expected_results2.lock().await;
                                tmp.insert(request_id.to_string(), None);
                            }
                            IrisCode::random_rng(&mut rng)
                        }
                        1 => {
                            println!("Sending iris code from db");
                            let db_len = {
                                let tmp = thread_db2.lock().await;
                                tmp.db.len()
                            };
                            let db_index = rng.gen_range(0..db_len);
                            {
                                let mut tmp = thread_expected_results2.lock().await;
                                tmp.insert(request_id.to_string(), Some(db_index as u32 + 1));
                            }
                            {
                                let tmp = thread_db2.lock().await;
                                tmp.db[db_index].clone()
                            }
                        }
                        2 => {
                            println!("Sending freshly inserted iris code");
                            let keys_vec = {
                                let tmp = thread_responses2.lock().await;
                                tmp.keys().cloned().collect::<Vec<_>>()
                            };
                            let keys_idx = rng.gen_range(0..keys_vec.len());
                            let iris_code = {
                                let tmp = thread_responses2.lock().await;
                                tmp.get(&keys_vec[keys_idx]).unwrap().clone()
                            };
                            {
                                let mut tmp = thread_expected_results2.lock().await;
                                tmp.insert(request_id.to_string(), Some(keys_vec[keys_idx]));
                            }
                            iris_code
                        }
                        _ => unreachable!(),
                    }
                } else {
                    // Manually passed CLI arguments
                    if let Some(db_index) = db_index {
                        if batch_query_idx * batch_idx < n_repeat {
                            let tmp = thread_db2.lock().await;
                            tmp.db[db_index].clone()
                        } else {
                            IrisCode::random_rng(&mut rng)
                        }
                    } else {
                        let mut rng = StdRng::seed_from_u64(1337);
                        IrisCode::random_rng(&mut rng)
                    }
                };

                {
                    let mut tmp = thread_requests2.lock().await;
                    tmp.insert(request_id.to_string(), template.clone());
                }

                let shared_code = GaloisRingIrisCodeShare::encode_iris_code(
                    &template.code,
                    &template.mask,
                    &mut rng,
                );
                let shared_mask =
                    GaloisRingIrisCodeShare::encode_mask_code(&template.mask, &mut rng);

                let mut iris_shares_file_hashes: [String; 3] = Default::default();
                let mut iris_codes_shares_base64: [String; 3] = Default::default();

                for i in 0..3 {
                    let iris_code_shares_json = IrisCodeSharesJSON {
                        iris_version:           "1.0".to_string(),
                        iris_shares_version:    "1.3".to_string(),
                        right_iris_code_shares: shared_code[i].to_base64(),
                        right_mask_code_shares: shared_mask[i].to_base64(),
                        left_iris_code_shares:  shared_code[i].to_base64(),
                        left_mask_code_shares:  shared_mask[i].to_base64(),
                    };
                    let serialized_iris_codes_json = to_string(&iris_code_shares_json)
                        .expect("Serialization failed")
                        .clone();

                    let hash_string = sha256_as_hex_string(&serialized_iris_codes_json);

                    let encrypted_bytes = sealedbox::seal(
                        serialized_iris_codes_json.as_bytes(),
                        &shares_encryption_public_keys2[i],
                    );

                    iris_codes_shares_base64[i] =
                        general_purpose::STANDARD.encode(&encrypted_bytes);
                    iris_shares_file_hashes[i] = hash_string;
                }

                let contents = serde_json::to_vec(&iris_codes_shares_base64)?;
                let presigned_url = match upload_file_and_generate_presigned_url(
                    &requests_bucket_name,
                    &request_id.to_string(),
                    Box::leak(requests_bucket_region.clone().into_boxed_str()),
                    &contents,
                )
                .await
                {
                    Ok(url) => url,
                    Err(e) => {
                        eprintln!("Failed to upload file: {}", e);
                        // ignore the error and continue
                        return Ok(());
                    }
                };

                let request_message = UniquenessRequest {
                    batch_size:         None,
                    signup_id:          request_id.to_string(),
                    s3_key:             presigned_url,
                    or_rule_serial_ids: None,
                };

                let message_attributes = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);

                requests_sns_client2
                    .publish()
                    .topic_arn(request_topic_arn.clone())
                    .message_group_id(ENROLLMENT_REQUEST_TYPE)
                    .message(to_string(&request_message)?)
                    .set_message_attributes(Some(message_attributes))
                    .send()
                    .await?;

                Ok::<(), eyre::Error>(())
            });
            handles.push(handle);
        }

        // Wait for all tasks in this batch to complete.
        for handle in handles {
            handle.await??;
        }

        println!("Batch {} sent!", batch_idx);
        sleep(WAIT_AFTER_BATCH).await;
    }

    Ok(())
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let opts = Opt::parse();
    run_client(opts).await
}
