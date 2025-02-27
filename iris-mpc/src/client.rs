#![allow(clippy::needless_range_loop)]

use aws_config::retry::RetryConfig;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::{config::Region, types::MessageAttributeValue, Client};
use base64::{engine::general_purpose, Engine};
use clap::Parser;
use eyre::{Context, ContextCompat};
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare,
    helpers::{
        aws::{
            NODE_ID_MESSAGE_ATTRIBUTE_NAME, SPAN_ID_MESSAGE_ATTRIBUTE_NAME,
            TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
        },
        key_pair::download_public_key,
        sha256::sha256_as_hex_string,
        smpc_request::{
            IrisCodeSharesJSON, SharesS3Object, UniquenessRequest, UNIQUENESS_MESSAGE_TYPE,
        },
        smpc_response::create_message_type_attribute_map,
        sqs_s3_helper::upload_file_to_s3,
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

const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 16;
const DEFAULT_BATCH_SIZE: usize = 64;
const DEFAULT_N_BATCHES: usize = 5;
const DEFAULT_N_REPEAT: usize = 0;

const WAIT_AFTER_BATCH: Duration = Duration::from_secs(2);
const RNG_SEED_SERVER: u64 = 42;
const DB_SIZE: usize = 8 * 1_000;
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

#[derive(Debug, Parser, Clone)]
pub struct Opt {
    #[arg(long, env, required = true)]
    pub request_topic_arn: String,

    #[arg(long, env, required = true)]
    pub requests_bucket_name: String,

    #[arg(long, env, required = true)]
    pub public_key_base_url: String,

    #[arg(long, env, required = true)]
    pub region: String,

    #[arg(long, env)]
    pub db_index: Option<usize>,

    #[arg(long, env)]
    pub rng_seed: Option<u64>,

    #[arg(long, env)]
    pub random: Option<bool>,

    #[arg(long, env)]
    pub endpoint_url: Option<String>,

    #[arg(long, env, default_value_t = DEFAULT_N_REPEAT)]
    pub n_repeat: usize,

    #[arg(long, env, default_value_t = DEFAULT_N_BATCHES)]
    pub n_batches: usize,

    #[arg(long, env, default_value_t = DEFAULT_BATCH_SIZE )]
    pub batch_size: usize,

    #[arg(long, env, default_value_t = DEFAULT_MAX_CONCURRENT_REQUESTS)]
    pub max_concurrent_requests: usize,
}

/// The core client functionality is moved into an async function so it can be
/// called from a benchmark harness without spawning a new process.
pub async fn run_client(opts: Opt) -> eyre::Result<()> {
    let Opt {
        public_key_base_url,
        requests_bucket_name,
        region,
        request_topic_arn,
        db_index,
        rng_seed,
        n_repeat,
        random,
        endpoint_url,
        n_batches,
        batch_size,
        max_concurrent_requests,
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

    let region = Region::new(region);
    let shared_config = aws_config::from_env()
        .region(region)
        .retry_config(RetryConfig::standard().with_max_attempts(5))
        .load()
        .await;

    let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&shared_config);
    let mut sns_config_builder = aws_sdk_sns::config::Builder::from(&shared_config);

    if let Some(endpoint_url) = endpoint_url.as_ref() {
        s3_config_builder = s3_config_builder.endpoint_url(endpoint_url);
        s3_config_builder = s3_config_builder.force_path_style(true);
        sns_config_builder = sns_config_builder.endpoint_url(endpoint_url);
    }

    let s3_client = S3Client::from_conf(s3_config_builder.build());
    let sns_client = Client::from_conf(sns_config_builder.build());

    let db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(RNG_SEED_SERVER));

    let expected_results: Arc<Mutex<HashMap<String, Option<u32>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let requests: Arc<Mutex<HashMap<String, IrisCode>>> = Arc::new(Mutex::new(HashMap::new()));
    let responses: Arc<Mutex<HashMap<u32, IrisCode>>> = Arc::new(Mutex::new(HashMap::new()));
    let db: Arc<Mutex<IrisDB>> = Arc::new(Mutex::new(db));
    let requests_sns_client: Arc<Client> = Arc::new(sns_client);

    let semaphore = Arc::new(Semaphore::new(max_concurrent_requests));

    // let recv_thread = spawn(async move {
    //     let region_provider = Region::new(response_queue_region);
    //     let results_sqs_config =
    // aws_config::from_env().region(region_provider).load().await;
    //     let results_sqs_client = SqsClient::new(&results_sqs_config);
    //     let mut counter = 0;
    //     while counter < N_QUERIES * 3 {
    //         // Receive responses
    //         let msg = results_sqs_client
    //             .receive_message()
    //             .max_number_of_messages(1)
    //             .queue_url(response_queue_url.clone())
    //             .send()
    //             .await
    //             .context("Failed to receive message")?;
    //
    //         for msg in msg.messages.unwrap_or_default() {
    //             counter += 1;
    //
    //             let result: UniquenessResult =
    //                 serde_json::from_str(&msg.body.context("No body found")?)
    //                     .context("Failed to parse message body")?;
    //
    //             println!("Received result: {:?}", result);
    //
    //             let expected_result_option = {
    //                 let tmp = thread_expected_results.lock().await;
    //                 tmp.get(&result.signup_id).cloned()
    //             };
    //             if expected_result_option.is_none() {
    //                 eprintln!(
    //                     "No expected result found for request_id: {}, the SQS
    // message is likely \                      stale, clear the queue",
    //                     result.signup_id
    //                 );
    //
    //                 results_sqs_client
    //                     .delete_message()
    //                     .queue_url(response_queue_url.clone())
    //                     .receipt_handle(msg.receipt_handle.unwrap())
    //                     .send()
    //                     .await
    //                     .context("Failed to delete message")?;
    //
    //                 continue;
    //             }
    //             let expected_result = expected_result_option.unwrap();
    //
    //             if expected_result.is_none() {
    //                 // New insertion
    //                 assert!(!result.is_match);
    //                 let request = {
    //                     let tmp = thread_requests.lock().await;
    //                     tmp.get(&result.signup_id).unwrap().clone()
    //                 };
    //                 {
    //                     let mut tmp = thread_responses.lock().await;
    //                     tmp.insert(result.serial_id.unwrap(), request);
    //                 }
    //             } else {
    //                 // Existing entry
    //                 assert!(result.is_match);
    //                 assert!(result.matched_serial_ids.is_some());
    //                 let matched_ids = result.matched_serial_ids.unwrap();
    //                 assert!(matched_ids.len() == 1);
    //                 assert_eq!(expected_result.unwrap(), matched_ids[0]);
    //             }
    //
    //             results_sqs_client
    //                 .delete_message()
    //                 .queue_url(response_queue_url.clone())
    //                 .receipt_handle(msg.receipt_handle.unwrap())
    //                 .send()
    //                 .await
    //                 .context("Failed to delete message")?;
    //         }
    //     }
    //     eyre::Ok(())
    // });

    for batch_idx in 0..n_batches {
        let mut handles = Vec::new();
        for batch_query_idx in 0..batch_size {
            let shares_encryption_public_keys2 = shares_encryption_public_keys.clone();
            let requests_sns_client2 = requests_sns_client.clone();
            let thread_db2 = db.clone();
            let thread_expected_results2 = expected_results.clone();
            let thread_requests2 = requests.clone();
            let thread_responses2 = responses.clone();
            let request_topic_arn = request_topic_arn.clone();
            let requests_bucket_name = requests_bucket_name.clone();
            let semaphore = Arc::clone(&semaphore);
            let s3_client = s3_client.clone();
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

                let shares_s3_object = SharesS3Object {
                    iris_share_0:  iris_codes_shares_base64[0].clone(),
                    iris_share_1:  iris_codes_shares_base64[1].clone(),
                    iris_share_2:  iris_codes_shares_base64[2].clone(),
                    iris_hashes_0: iris_shares_file_hashes[0].clone(),
                    iris_hashes_1: iris_shares_file_hashes[1].clone(),
                    iris_hashes_2: iris_shares_file_hashes[2].clone(),
                };

                let contents = serde_json::to_vec(&shares_s3_object)?;
                let bucket_key = match upload_file_to_s3(
                    &requests_bucket_name,
                    &request_id.to_string(),
                    s3_client.clone(),
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
                    s3_key:             bucket_key,
                    or_rule_serial_ids: None,
                    skip_persistence:   None,
                };

                let message_attributes = {
                    let mut attrs = create_message_type_attribute_map(UNIQUENESS_MESSAGE_TYPE);
                    attrs.extend(
                        [
                            TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
                            SPAN_ID_MESSAGE_ATTRIBUTE_NAME,
                            NODE_ID_MESSAGE_ATTRIBUTE_NAME,
                        ]
                        .iter()
                        .map(|key| {
                            (
                                key.to_string(),
                                MessageAttributeValue::builder()
                                    .data_type("String")
                                    .string_value("TEST")
                                    .build()
                                    .unwrap(),
                            )
                        }),
                    );
                    attrs
                };

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
