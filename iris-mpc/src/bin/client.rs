#![allow(clippy::needless_range_loop)]
use std::{collections::HashMap, sync::Arc, time::Duration};

use aws_sdk_sns::{Client, config::Region};
use aws_sdk_sqs::Client as SqsClient;
use base64::{Engine, engine::general_purpose};
use clap::Parser;
use eyre::{Context, ContextCompat};
use rand::{Rng, rngs::StdRng, SeedableRng, thread_rng};
use serde_json::to_string;
use sha2::{Digest, Sha256};
use sodiumoxide::crypto::{box_::PublicKey, sealedbox};
use tokio::{spawn, sync::Mutex, time::sleep};
use uuid::Uuid;

use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare,
    helpers::{
        key_pair::download_public_key,
        smpc_request::{IrisCodesJSON, ResultEvent, SMPCRequest},
        sqs_s3_helper::upload_file_and_generate_presigned_url,
    },
    iris_db::{db::IrisDB, iris::IrisCode},
};

const N_QUERIES: usize = 64 * 5;
const RESULT_SQS_AWS_REGION: &str = "eu-central-1";
const RNG_SEED_SERVER: u64 = 42;
const DB_SIZE: usize = 8 * 1_000;
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";
const PUBLIC_KEY_BASE_URL: &str = "https://d24uxaabh702ht.cloudfront.net";
const SQS_REQUESTS_BUCKET_NAME: &str = "wf-mpc-stage-smpcv2-sns-requests";
const SQS_REQUESTS_BUCKET_REGION: &str = "eu-north-1";
const SNS_TOPIC_REGION: &str = "eu-north-1";

#[derive(Debug, Parser)]
struct Opt {
    #[arg(long, env)]
    request_topic_arn: String,

    #[arg(long, env)]
    response_queue_url: String,

    #[arg(long, env)]
    db_index: Option<usize>,

    #[arg(long, env)]
    rng_seed: Option<u64>,

    #[arg(long, env)]
    n_repeat: Option<usize>,

    #[arg(long, env)]
    random: Option<bool>,

    #[arg(long, env)]
    encrypt_shares: Option<bool>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let Opt {
        request_topic_arn,
        response_queue_url,
        db_index,
        rng_seed,
        n_repeat,
        random,
        encrypt_shares,
    } = Opt::parse();

    let mut rng = if let Some(rng_seed) = rng_seed {
        StdRng::seed_from_u64(rng_seed)
    } else {
        StdRng::from_entropy()
    };

    let mut shares_encryption_public_keys: Vec<PublicKey> = vec![];
    let _ = encrypt_shares.is_some();

    for i in 0..3 {
        let public_key_string =
            download_public_key(PUBLIC_KEY_BASE_URL.to_string(), i.to_string()).await?;
        let public_key_bytes = general_purpose::STANDARD
            .decode(public_key_string)
            .context("Failed to decode public key")?;
        let public_key =
            PublicKey::from_slice(&public_key_bytes).context("Failed to parse public key")?;
        shares_encryption_public_keys.push(public_key);
    }

    let n_repeat = n_repeat.unwrap_or(0);

    // THIS IS REQUIRED TO USE THE SQS FROM SECONDARY REGION, URL DOES NOT SUFFICE
    let region_provider = Region::new(RESULT_SQS_AWS_REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;

    let sqs_config = aws_config::from_env().region(SNS_TOPIC_REGION).load().await;
    let client = Client::new(&sqs_config);

    let db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(RNG_SEED_SERVER));

    let mut choice_rng = thread_rng();

    let expected_results: Arc<Mutex<HashMap<String, Option<u32>>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let requests: Arc<Mutex<HashMap<String, IrisCode>>> = Arc::new(Mutex::new(HashMap::new()));
    let responses: Arc<Mutex<HashMap<u32, IrisCode>>> = Arc::new(Mutex::new(HashMap::new()));

    let thread_expected_results = expected_results.clone();
    let thread_requests = requests.clone();
    let thread_responses = responses.clone();

    let recv_thread = spawn(async move {
        let sqs_client = SqsClient::new(&shared_config);
        let mut counter = 0;
        while counter < N_QUERIES * 3 {
            // Receive responses
            let msg = sqs_client
                .receive_message()
                .max_number_of_messages(1)
                .queue_url(response_queue_url.clone())
                .send()
                .await
                .context("Failed to receive message")?;

            for msg in msg.messages.unwrap_or_default() {
                counter += 1;
                let result: ResultEvent = serde_json::from_str(&msg.body.context("No body found")?)
                    .context("Failed to parse message body")?;

                println!("Received result: {:?}", result);

                let tmp = thread_expected_results.lock().await;
                let expected_result = tmp.get(&result.signup_id);
                if expected_result.is_none() {
                    eprintln!(
                        "No expected result found for request_id: {}, the SQS message is likely \
                         stale, clear the queue",
                        result.signup_id
                    );

                    sqs_client
                        .delete_message()
                        .queue_url(response_queue_url.clone())
                        .receipt_handle(msg.receipt_handle.unwrap())
                        .send()
                        .await
                        .context("Failed to delete message")?;

                    continue;
                }
                let expected_result = expected_result.unwrap();

                if expected_result.is_none() {
                    // New insertion
                    assert!(!result.is_match);
                    let request = thread_requests
                        .lock()
                        .await
                        .get(&result.signup_id)
                        .unwrap()
                        .clone();
                    thread_responses
                        .lock()
                        .await
                        .insert(result.serial_id.unwrap(), request);
                } else {
                    // Existing entry
                    println!(
                        "Expected: {:?} Got: {:?}",
                        expected_result, result.serial_id
                    );
                    assert!(result.is_match);
                    assert_eq!(result.serial_id.unwrap(), expected_result.unwrap());
                }

                sqs_client
                    .delete_message()
                    .queue_url(response_queue_url.clone())
                    .receipt_handle(msg.receipt_handle.unwrap())
                    .send()
                    .await
                    .context("Failed to delete message")?;
            }
        }
        eyre::Ok(())
    });

    // Prepare query
    for query_idx in 0..N_QUERIES {
        let request_id = Uuid::new_v4();
        println!("Building request: {}", request_id);

        let template = if random.is_some() {
            // Automatic random tests
            let options = if responses.lock().await.len() == 0 {
                2
            } else {
                3
            };
            match choice_rng.gen_range(0..options) {
                0 => {
                    println!("Sending new iris code");
                    expected_results
                        .lock()
                        .await
                        .insert(request_id.to_string(), None);
                    IrisCode::random_rng(&mut rng)
                }
                1 => {
                    println!("Sending iris code from db");
                    let db_index = rng.gen_range(0..db.db.len());
                    expected_results
                        .lock()
                        .await
                        .insert(request_id.to_string(), Some(db_index as u32));
                    db.db[db_index].clone()
                }
                2 => {
                    println!("Sending freshly inserted iris code");
                    let tmp = responses.lock().await;
                    let keys = tmp.keys().collect::<Vec<_>>();
                    let idx = rng.gen_range(0..keys.len());
                    let iris_code = tmp.get(keys[idx]).unwrap().clone();
                    expected_results
                        .lock()
                        .await
                        .insert(request_id.to_string(), Some(*keys[idx]));
                    iris_code
                }
                _ => unreachable!(),
            }
        } else {
            // Manually passed cli arguments
            if let Some(db_index) = db_index {
                if query_idx < n_repeat {
                    db.db[db_index].clone()
                } else {
                    IrisCode::random_rng(&mut rng)
                }
            } else {
                let mut rng = StdRng::seed_from_u64(1337); // TODO
                IrisCode::random_rng(&mut rng)
            }
        };

        requests
            .lock()
            .await
            .insert(request_id.to_string(), template.clone());

        let shared_code = GaloisRingIrisCodeShare::encode_iris_code(
            &template.code,
            &template.mask,
            &mut StdRng::seed_from_u64(RNG_SEED_SERVER),
        );
        let shared_mask = GaloisRingIrisCodeShare::encode_mask_code(
            &template.mask,
            &mut StdRng::seed_from_u64(RNG_SEED_SERVER),
        );

        let mut iris_shares_file_hashes: [String; 3] = Default::default();
        let mut iris_codes_shares_base64: [String; 3] = Default::default();

        for i in 0..3 {
            let iris_code_coefs_base64 =
                general_purpose::STANDARD.encode(bytemuck::cast_slice(&shared_code[i].coefs));
            let mask_code_coefs_base64 =
                general_purpose::STANDARD.encode(bytemuck::cast_slice(&shared_mask[i].coefs));

            let iris_codes_json = IrisCodesJSON {
                iris_version: "1.0".to_string(),
                right_iris_code: iris_code_coefs_base64,
                right_iris_mask: mask_code_coefs_base64,
                left_iris_code: "nan".to_string(),
                left_iris_mask: "nan".to_string(),
            };
            let serialized_iris_codes_json = to_string(&iris_codes_json)
                .expect("Serialization failed")
                .clone();

            // calculate hash of the object
            let mut hasher = Sha256::new();
            hasher.update(serialized_iris_codes_json.clone());
            let result = hasher.finalize();
            let hash_string = format!("{:x}", result);

            // encrypt the object using sealed box and public key
            let encrypted_bytes = sealedbox::seal(
                serialized_iris_codes_json.as_bytes(),
                &shares_encryption_public_keys[i],
            );

            iris_codes_shares_base64[i] = general_purpose::STANDARD.encode(&encrypted_bytes);
            iris_shares_file_hashes[i] = hash_string;
        }

        let contents = serde_json::to_vec(&iris_codes_shares_base64)?;
        let presigned_url = match upload_file_and_generate_presigned_url(
            SQS_REQUESTS_BUCKET_NAME,
            &request_id.to_string(),
            SQS_REQUESTS_BUCKET_REGION,
            &contents,
        )
            .await
        {
            Ok(url) => url,
            Err(e) => {
                eprintln!("Failed to upload file: {}", e);
                continue;
            }
        };

        let request_message = SMPCRequest {
            batch_size: None,
            signup_id: request_id.to_string(),
            s3_presigned_url: presigned_url,
            iris_shares_file_hashes,
        };

        // Send all messages in batch
        client
            .publish()
            .topic_arn(request_topic_arn.clone())
            .message_group_id(ENROLLMENT_REQUEST_TYPE)
            .message(to_string(&request_message)?)
            .send()
            .await?;

        if (query_idx + 1) % 32 == 0 {
            sleep(Duration::from_secs(1)).await;
        }
    }

    sleep(Duration::from_secs(10)).await;

    // Receive all messages
    recv_thread.await??;

    Ok(())
}
