#![allow(clippy::needless_range_loop)]
use crate::client::iris_data::{
    generate_party_shares, read_iris_data_from_file, IrisCodePartyShares,
};
use aws_config::retry::RetryConfig;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sns::{config::Region, types::MessageAttributeValue, Client as SnsClient};
use aws_sdk_sqs::Client as SqsClient;
use base64::{engine::general_purpose, Engine};
use clap::Parser;
use eyre::Result;
use eyre::{Context, ContextCompat};
use iris_mpc_common::helpers::{
    aws::{
        NODE_ID_MESSAGE_ATTRIBUTE_NAME, SPAN_ID_MESSAGE_ATTRIBUTE_NAME,
        TRACE_ID_MESSAGE_ATTRIBUTE_NAME,
    },
    key_pair::download_public_key,
    sha256::sha256_as_hex_string,
    smpc_request::{SharesS3Object, UniquenessRequest, UNIQUENESS_MESSAGE_TYPE},
    smpc_response::{create_message_type_attribute_map, UniquenessResult},
    sqs_s3_helper::upload_file_to_s3,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::to_string;
use sodiumoxide::crypto::{box_::PublicKey, sealedbox};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{Mutex, Semaphore},
    task::spawn,
    time::sleep,
};
use uuid::Uuid;

const DEFAULT_MAX_CONCURRENT_REQUESTS: usize = 5;
const DEFAULT_BATCH_SIZE: usize = 10;
const DEFAULT_N_BATCHES: usize = 5;

const WAIT_AFTER_BATCH: Duration = Duration::from_secs(10);
const RECEIVER_POLL_INTERVAL: Duration = Duration::from_millis(500);
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

#[derive(Debug, Parser, Clone)]
pub struct Opt {
    #[arg(long, env, required = true)]
    pub request_topic_arn: String,

    #[arg(long, env, required = true)]
    pub requests_bucket_name: String,

    #[arg(long, env, required = true)]
    pub public_key_base_url: String,

    #[arg(long, env = "AWS_REGION", required = true)]
    pub region: String,

    #[arg(long, env, required = true)]
    pub response_queue_url: String,

    #[arg(long, env)]
    pub rng_seed: Option<u64>,

    #[arg(long, env = "AWS_ENDPOINT_URL")]
    pub endpoint_url: Option<String>,

    #[arg(long, env, default_value_t = DEFAULT_N_BATCHES)]
    pub n_batches: usize,

    #[arg(long, env, default_value_t = DEFAULT_BATCH_SIZE )]
    pub batch_size: usize,

    #[arg(long, env, default_value_t = DEFAULT_MAX_CONCURRENT_REQUESTS)]
    pub max_concurrent_requests: usize,

    #[arg(long, env)]
    pub data_from_file: Option<String>,

    #[arg(long, env)]
    pub populate_file_data_limit: Option<usize>,
}

/// The core client functionality is moved into an async function so it can be
/// called from a benchmark harness without spawning a new process.
pub async fn run_client(opts: Opt) -> Result<()> {
    // Create the client and await its creation
    let mut client = E2EClient::new(opts.clone()).await;

    // Initialize AWS resources and encryption keys
    client.initialize(opts.public_key_base_url).await?;

    // Run the client
    client.run().await?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct E2EClient {
    // Configuration for client
    request_topic_arn: String,
    requests_bucket_name: String,
    response_queue_url: String,
    rng_seed: Option<u64>,
    n_batches: usize,
    batch_size: usize,
    data_from_file: Option<String>,
    populate_file_data_limit: Option<usize>,

    // AWS clients
    s3_client: S3Client,
    sns_client: Arc<SnsClient>,
    sqs_client: SqsClient,

    // Shared state for ensuring results are correct
    // The expected results contain what the request id and then serial id to match
    expected_results: Arc<Mutex<HashMap<String, Option<u32>>>>,
    // the requests contain the request id and the iris code shares
    requests: Arc<Mutex<HashMap<String, IrisCodePartyShares>>>,
    // the responses contain the serial id and the iris code shares
    responses: Arc<Mutex<HashMap<u32, IrisCodePartyShares>>>,
    encryption_public_keys: Vec<PublicKey>,
    semaphore: Arc<Semaphore>,
    file_data: Arc<Vec<IrisCodePartyShares>>,
}

impl E2EClient {
    pub async fn new(opts: Opt) -> Self {
        // Initialize AWS clients
        let region = Region::new(opts.region.clone());
        let shared_config = aws_config::from_env()
            .region(region)
            .retry_config(RetryConfig::standard().with_max_attempts(20))
            .load()
            .await;

        let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&shared_config);
        let mut sns_config_builder = aws_sdk_sns::config::Builder::from(&shared_config);

        if let Some(endpoint_url) = opts.endpoint_url.as_ref() {
            s3_config_builder = s3_config_builder.endpoint_url(endpoint_url);
            s3_config_builder = s3_config_builder.force_path_style(true);
            sns_config_builder = sns_config_builder.endpoint_url(endpoint_url);
        }
        let s3_client = S3Client::from_conf(s3_config_builder.build());
        let sns_client = Arc::new(SnsClient::from_conf(sns_config_builder.build()));
        let sqs_client = SqsClient::new(&shared_config);

        Self {
            request_topic_arn: opts.request_topic_arn,
            requests_bucket_name: opts.requests_bucket_name,
            response_queue_url: opts.response_queue_url,
            rng_seed: opts.rng_seed,

            n_batches: opts.n_batches,
            batch_size: opts.batch_size,
            data_from_file: opts.data_from_file,
            populate_file_data_limit: opts.populate_file_data_limit,

            s3_client,
            sns_client,
            sqs_client,

            expected_results: Arc::new(Mutex::new(HashMap::new())),
            requests: Arc::new(Mutex::new(HashMap::new())),
            responses: Arc::new(Mutex::new(HashMap::new())),
            encryption_public_keys: Vec::new(),
            semaphore: Arc::new(Semaphore::new(opts.max_concurrent_requests)),
            file_data: Arc::new(Vec::new()),
        }
    }

    pub async fn initialize(&mut self, public_key_base_url: String) -> Result<()> {
        // Download public keys for each participant
        self.encryption_public_keys = Vec::new();
        for i in 0..3 {
            let public_key_string =
                download_public_key(public_key_base_url.clone(), i.to_string()).await?;
            let public_key_bytes = general_purpose::STANDARD
                .decode(public_key_string)
                .context("Failed to decode public key")?;
            let public_key =
                PublicKey::from_slice(&public_key_bytes).context("Failed to parse public key")?;
            self.encryption_public_keys.push(public_key);
        }

        self.sqs_client
            .purge_queue()
            .queue_url(self.response_queue_url.clone())
            .send()
            .await?;

        // Load data from File if configured
        if let Some(file_path) = &self.data_from_file {
            let data = read_iris_data_from_file(file_path).await?;
            self.file_data = Arc::new(data);
        }

        Ok(())
    }

    pub async fn run(&self) -> Result<()> {
        let n_queries = if self.data_from_file.is_some() && self.populate_file_data_limit.is_some()
        {
            let limit = self.populate_file_data_limit.unwrap_or(0);

            if limit == 0 || limit > self.file_data.len() {
                self.file_data.len()
            } else {
                limit
            }
        } else {
            self.batch_size * self.n_batches
        };

        let recv_thread = self.spawn_receiver_thread(n_queries, self.sqs_client.clone());

        if self.data_from_file.is_some() && self.populate_file_data_limit.is_some() {
            self.populate_only_file_data(n_queries).await?;
        } else {
            self.run_e2e_test().await?;
        }

        recv_thread.await??;
        Ok(())
    }

    async fn populate_only_file_data(&self, limit: usize) -> Result<()> {
        let mut party_shares_index: usize = 0;
        let mut batch_idx = 0;
        while party_shares_index < limit {
            let mut handles = Vec::new();
            let batch_size = std::cmp::min(self.batch_size, limit - party_shares_index);
            for _ in 0..batch_size {
                let semaphore = self.semaphore.clone();
                let client = self.clone();
                let requests = self.requests.clone();
                let party_shares = self.file_data[party_shares_index].clone();
                let expected_results = self.expected_results.clone();
                let handle = tokio::spawn(async move {
                    let _permit = semaphore.acquire().await;
                    let request_id = party_shares.signup_id.clone();
                    println!(
                        "Sending iris code {} from file {}",
                        party_shares_index, request_id
                    );
                    requests
                        .lock()
                        .await
                        .insert(request_id.clone(), party_shares.clone());
                    expected_results
                        .lock()
                        .await
                        .insert(request_id.clone(), None);
                    client.send_enrollment_request(party_shares.clone()).await?;
                    Ok::<(), eyre::Report>(())
                });
                handles.push(handle);
                party_shares_index += 1;
            }
            for handle in handles {
                handle.await??;
            }
            println!("Batch {} sent!", batch_idx + 1);
            batch_idx += 1;
            sleep(WAIT_AFTER_BATCH).await;
        }

        Ok(())
    }

    async fn run_e2e_test(&self) -> Result<()> {
        let used_file_indices = Arc::new(Mutex::new(HashSet::new()));
        let used_response_indices = Arc::new(Mutex::new(HashSet::new()));
        for batch_idx in 0..self.n_batches {
            let mut handles = Vec::new();
            for _batch_query_idx in 0..self.batch_size {
                let expected_results = self.expected_results.clone();
                let requests = self.requests.clone();
                let responses = self.responses.clone();
                let semaphore = self.semaphore.clone();
                let file_data = self.file_data.clone();
                let rng_seed = self.rng_seed;
                let has_file_data_loaded = self.data_from_file.is_some();
                let client = self.clone();
                let used_indices = used_file_indices.clone();
                let used_response_indices = used_response_indices.clone();
                let handle = tokio::spawn(async move {
                    let _permit = semaphore.acquire().await;

                    let mut rng = if let Some(rng_seed) = rng_seed {
                        StdRng::seed_from_u64(rng_seed)
                    } else {
                        StdRng::from_entropy()
                    };
                    let mut request_id = Uuid::new_v4().to_string();

                    let party_shares: IrisCodePartyShares = {
                        // Automatic random tests
                        let responses_len = {
                            let tmp = responses.lock().await;
                            tmp.len()
                        };
                        let used_responses_indices_len = {
                            let indices = used_response_indices.lock().await;
                            indices.len()
                        };

                        let options =
                            if responses_len == 0 || used_responses_indices_len == responses_len {
                                1
                            } else {
                                2
                            };

                        match rng.gen_range(0..options) {
                            0 => {
                                let party_share_data = if has_file_data_loaded {
                                    let mut index = rng.gen_range(0..file_data.len());
                                    let mut used_indices_locked = used_indices.lock().await;
                                    while used_indices_locked.contains(&index) {
                                        index = rng.gen_range(0..file_data.len());
                                    }
                                    used_indices_locked.insert(index);
                                    file_data[index].clone()
                                } else {
                                    generate_party_shares(rng)
                                };
                                request_id = party_share_data.signup_id.clone();
                                println!("Sending new iris code for request id {}", request_id);
                                {
                                    let mut tmp = expected_results.lock().await;
                                    tmp.insert(request_id.clone(), None);
                                }
                                party_share_data
                            }
                            1 => {
                                let new_request_id_for_duplicate = request_id.clone(); // Capture new request_id for clarity
                                let original_serial_id_to_duplicate;
                                let duplicated_party_shares;

                                {
                                    // Scope for responses and used_response_indices locks
                                    let locked_responses = responses.lock().await;
                                    let mut locked_response_indices =
                                        used_response_indices.lock().await;

                                    let keys_vec =
                                        locked_responses.keys().cloned().collect::<Vec<_>>();
                                    let mut keys_idx = rng.gen_range(0..keys_vec.len());
                                    while locked_response_indices.contains(&keys_vec[keys_idx]) {
                                        keys_idx = rng.gen_range(0..keys_vec.len());
                                    }
                                    original_serial_id_to_duplicate = keys_vec[keys_idx];

                                    let original_party_shares_to_duplicate = locked_responses
                                        .get(&original_serial_id_to_duplicate)
                                        .unwrap();
                                    duplicated_party_shares = original_party_shares_to_duplicate
                                        .create_duplicate_party_shares(
                                            new_request_id_for_duplicate.clone(),
                                        );

                                    locked_response_indices.insert(original_serial_id_to_duplicate);
                                } // `locked_responses` and `locked_response_indices` are released here

                                {
                                    let mut tmp = expected_results.lock().await;
                                    tmp.insert(
                                        new_request_id_for_duplicate.to_string(),
                                        Some(original_serial_id_to_duplicate),
                                    );
                                }
                                println!(
                                    "Sending freshly inserted iris code for request id {} - this is a duplicate of data originally from serial id {:?}",
                                    new_request_id_for_duplicate, Some(original_serial_id_to_duplicate)
                                );
                                // The party_shares variable for the outer scope should be the duplicated_party_shares
                                request_id = new_request_id_for_duplicate; // Ensure outer request_id is the new one
                                duplicated_party_shares // This becomes the party_shares for the current iteration
                            }
                            _ => unreachable!(),
                        }
                    };

                    {
                        let mut tmp = requests.lock().await;
                        tmp.insert(request_id.to_string(), party_shares.clone());
                    }
                    client.send_enrollment_request(party_shares).await?;
                    Ok::<(), eyre::Report>(())
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
        println!("All batches sent!");
        Ok(())
    }

    fn spawn_receiver_thread(
        &self,
        n_queries: usize,
        sqs_client: SqsClient,
    ) -> tokio::task::JoinHandle<Result<()>> {
        let response_queue_url = self.response_queue_url.clone();
        let expected_results = self.expected_results.clone();
        let requests = self.requests.clone();
        let responses = self.responses.clone();

        spawn(async move {
            let total_messages = n_queries * 3;
            println!(
                "Receiver thread started: expecting {} messages total",
                total_messages
            );
            let mut counter = 0;
            while counter < total_messages {
                // Receive responses with 2s long polling
                let msg = sqs_client
                    .receive_message()
                    .wait_time_seconds(2)
                    .max_number_of_messages(1)
                    .queue_url(response_queue_url.clone())
                    .send()
                    .await
                    .context("Failed to receive a message")?;

                for msg in msg.messages.unwrap_or_default() {
                    counter += 1;
                    let remaining = total_messages - counter;
                    // print only every
                    println!(
                        "Received message {}/{} ({} remaining)",
                        counter, total_messages, remaining
                    );

                    let sns_notification: serde_json::Value =
                        serde_json::from_str(&msg.body.context("No body found")?)
                            .context("Failed to parse SNS notification")?;

                    let sqs_message = sns_notification["Message"]
                        .as_str()
                        .context("No Message field in SNS notification")?;
                    let result: UniquenessResult = serde_json::from_str(sqs_message)
                        .context("Failed to parse UniquenessResult from Message")?;

                    let expected_result_option = {
                        let tmp = expected_results.lock().await;
                        tmp.get(&result.signup_id).cloned()
                    };
                    assert!(expected_result_option.is_some());
                    let expected_result = expected_result_option.unwrap();

                    if expected_result.is_none() {
                        // New insertion
                        assert!(!result.is_match);
                        let request = {
                            let tmp = requests.lock().await;
                            tmp.get(&result.signup_id).unwrap().clone()
                        };
                        {
                            let mut tmp = responses.lock().await;
                            tmp.insert(result.serial_id.unwrap(), request);
                        }
                    } else {
                        // Existing entry
                        assert!(result.is_match);
                        assert!(result.matched_serial_ids.is_some());
                        let matched_ids = result.matched_serial_ids.unwrap();
                        assert_eq!(matched_ids.len(), 1);
                        assert_eq!(expected_result.unwrap(), matched_ids[0]);
                    }

                    sqs_client
                        .delete_message()
                        .queue_url(response_queue_url.clone())
                        .receipt_handle(msg.receipt_handle.unwrap())
                        .send()
                        .await
                        .context("Failed to delete message")?;
                }
                // throttle polling interval
                sleep(RECEIVER_POLL_INTERVAL).await;
            }
            eyre::Ok(())
        })
    }

    async fn send_enrollment_request(&self, party_shares: IrisCodePartyShares) -> Result<()> {
        let mut iris_shares_file_hashes: [String; 3] = Default::default();
        let mut iris_codes_shares_base64: [String; 3] = Default::default();

        for i in 0..3 {
            let iris_code_shares_json = party_shares.party(i);
            let serialized_iris_codes_json = to_string(&iris_code_shares_json)
                .expect("Serialization failed")
                .clone();

            let hash_string = sha256_as_hex_string(&serialized_iris_codes_json);

            let encrypted_bytes = sealedbox::seal(
                serialized_iris_codes_json.as_bytes(),
                &self.encryption_public_keys[i],
            );

            iris_codes_shares_base64[i] = general_purpose::STANDARD.encode(&encrypted_bytes);
            iris_shares_file_hashes[i] = hash_string;
        }

        let shares_s3_object = SharesS3Object {
            iris_share_0: iris_codes_shares_base64[0].clone(),
            iris_share_1: iris_codes_shares_base64[1].clone(),
            iris_share_2: iris_codes_shares_base64[2].clone(),
            iris_hashes_0: iris_shares_file_hashes[0].clone(),
            iris_hashes_1: iris_shares_file_hashes[1].clone(),
            iris_hashes_2: iris_shares_file_hashes[2].clone(),
        };

        let contents = serde_json::to_vec(&shares_s3_object)?;
        let bucket_key = match upload_file_to_s3(
            &self.requests_bucket_name,
            &party_shares.signup_id,
            self.s3_client.clone(),
            &contents,
        )
        .await
        {
            Ok(url) => url,
            Err(e) => {
                eprintln!(
                    "Failed to upload file for signup_id {}: {}",
                    party_shares.signup_id, e
                );
                return Err(eyre::eyre!(
                    "S3 upload failed for signup_id {}: {}",
                    party_shares.signup_id,
                    e
                ));
            }
        };

        let request_message = UniquenessRequest {
            // TODO: in future use the batch size from the request
            batch_size: Some(1),
            signup_id: party_shares.signup_id.clone(),
            s3_key: bucket_key,
            or_rule_serial_ids: None,
            skip_persistence: None,
            full_face_mirror_attacks_detection_enabled: Some(true),
            disable_anonymized_stats: None,
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

        self.sns_client
            .clone()
            .publish()
            .topic_arn(self.request_topic_arn.clone())
            .message_group_id(ENROLLMENT_REQUEST_TYPE)
            .message(to_string(&request_message)?)
            .set_message_attributes(Some(message_attributes))
            .send()
            .await?;

        Ok(())
    }
}
