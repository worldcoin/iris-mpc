use aws_sdk_sns::{
    config::Region,
    types::{MessageAttributeValue, PublishBatchRequestEntry},
    Client,
};
use base64::{engine::general_purpose, Engine};
use clap::Parser;
use gpu_iris_mpc::{
    setup::iris_db::{db::IrisDB, iris::IrisCode, shamir_iris::ShamirIris},
    sqs::SMPCRequest,
};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::to_string;
use uuid::Uuid;

const N_QUERIES: usize = 30;
const REGION: &str = "us-east-2";
const RNG_SEED: u64 = 42;
const RNG_SEED_SERVER: u64 = 42;
const DB_SIZE: usize = 100;
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    topic_arn: String,

    #[structopt(short, long)]
    db_index: Option<usize>,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let Opt { topic_arn, db_index } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    let db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(RNG_SEED_SERVER));

    // Prepare query
    for _i in 0..N_QUERIES {

        let template = if let Some(db_index) = db_index {
            db.db[db_index].clone()
        } else {
            IrisCode::random_rng(&mut rng)
        };

        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let shared_template = ShamirIris::share_iris(&template, &mut rng);

        let mut messages = vec![];
        for i in 0..3 {
            let request_id = Uuid::new_v4();
            let sns_id = Uuid::new_v4();
            let iris_code =
                general_purpose::STANDARD.encode(bytemuck::cast_slice(&shared_template[i].code));
            let mask_code =
                general_purpose::STANDARD.encode(bytemuck::cast_slice(&shared_template[i].mask));

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
            .topic_arn(topic_arn.clone())
            .set_publish_batch_request_entries(Some(messages))
            .send()
            .await?;

        println!("Enrollment request batch {} published.", _i);
    }

    Ok(())
}
