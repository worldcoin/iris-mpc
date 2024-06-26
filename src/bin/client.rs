#![allow(clippy::needless_range_loop)]
use aws_sdk_sns::{
    config::Region,
    types::{MessageAttributeValue, PublishBatchRequestEntry},
    Client,
};
use aws_sdk_sqs::Client as SqsClient;
use base64::{engine::general_purpose, Engine};
use clap::Parser;
use eyre::ContextCompat;
use gpu_iris_mpc::{
    helpers::sqs::{ResultEvent, SMPCRequest},
    setup::{
        galois_engine::degree4::GaloisRingIrisCodeShare,
        iris_db::{db::IrisDB, iris::IrisCode},
    },
};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::to_string;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

const N_QUERIES: usize = 32;
const REGION: &str = "eu-north-1";
const RNG_SEED_SERVER: u64 = 42;
const DB_SIZE: usize = 8 * 1_000;
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    request_topic_arn: String,

    #[structopt(short, long)]
    response_queue_url: String,

    #[structopt(short, long)]
    db_index: Option<usize>,

    #[structopt(short, long)]
    rng_seed: Option<u64>,

    #[structopt(short, long)]
    n_repeat: Option<usize>,
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
    } = Opt::parse();

    let mut rng = if let Some(rng_seed) = rng_seed {
        StdRng::seed_from_u64(rng_seed)
    } else {
        StdRng::from_entropy()
    };

    let n_repeat = n_repeat.unwrap_or(0);

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    let db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(RNG_SEED_SERVER));

    // Prepare query
    for query_idx in 0..N_QUERIES {
        let template = if let Some(db_index) = db_index {
            if query_idx < n_repeat {
                db.db[db_index].clone()
            } else {
                IrisCode::random_rng(&mut rng)
            }
        } else {
            let mut rng = StdRng::seed_from_u64(1337); // TODO
            IrisCode::random_rng(&mut rng)
        };

        let shared_code = GaloisRingIrisCodeShare::encode_iris_code(
            &template.code,
            &template.mask,
            &mut StdRng::seed_from_u64(RNG_SEED_SERVER),
        );
        let shared_mask = GaloisRingIrisCodeShare::encode_mask_code(
            &template.mask,
            &mut StdRng::seed_from_u64(RNG_SEED_SERVER),
        );
        let request_id = Uuid::new_v4();

        let mut messages = vec![];
        for i in 0..3 {
            let sns_id = Uuid::new_v4();
            let iris_code =
                general_purpose::STANDARD.encode(bytemuck::cast_slice(&shared_code[i].coefs));
            let mask_code =
                general_purpose::STANDARD.encode(bytemuck::cast_slice(&shared_mask[i].coefs));

            let request_message = SMPCRequest {
                request_type: ENROLLMENT_REQUEST_TYPE.to_string(),
                request_id: request_id.to_string(),
                iris_code,
                mask_code,
            };

            messages.push(
                PublishBatchRequestEntry::builder()
                    .message(to_string(&request_message)?)
                    .id(sns_id.to_string())
                    .message_group_id(ENROLLMENT_REQUEST_TYPE)
                    .message_attributes(
                        "nodeId",
                        MessageAttributeValue::builder()
                            .set_string_value(Some(i.to_string()))
                            .set_data_type(Some("String".to_string()))
                            .build()?,
                    )
                    .build()
                    .unwrap(),
            );
        }

        // Send all messages in batch
        client
            .publish_batch()
            .topic_arn(request_topic_arn.clone())
            .set_publish_batch_request_entries(Some(messages))
            .send()
            .await?;

        println!("Enrollment request batch {} published.", query_idx);
    }

    let sqs_client = SqsClient::new(&shared_config);

    loop {
        sleep(Duration::from_secs(1)).await;

        // Receive responses
        let msg = sqs_client
            .receive_message()
            .max_number_of_messages(10)
            .queue_url(response_queue_url.clone())
            .send()
            .await?;

        for msg in msg.messages.unwrap_or_default() {
            let result: ResultEvent = serde_json::from_str(msg.body().context("No body found")?)?;

            println!("Received response: {:?}", result);

            sqs_client
                .delete_message()
                .queue_url(response_queue_url.clone())
                .receipt_handle(msg.receipt_handle.unwrap())
                .send()
                .await?;
        }
    }

    Ok(())
}
