use aws_sdk_sns::{config::Region, types::PublishBatchRequestEntry, Client};
use clap::Parser;
use gpu_iris_mpc::{
    setup::iris_db::{iris::IrisCode, shamir_iris::ShamirIris},
    sqs::SMPCRequest,
};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::to_string;
use uuid::Uuid;

const N_QUERIES: usize = 30;
const REGION: &str = "us-east-2";
const RNG_SEED: u64 = 42;
const ENROLLMENT_REQUEST_TYPE: &str = "enrollment";

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    topic_arn: String,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let Opt { topic_arn } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    // Prepare query
    for _ in 0..N_QUERIES {
        let template = IrisCode::random_rng(&mut rng);
        let shared_template = ShamirIris::share_iris(&template, &mut rng);
        let id = Uuid::new_v4();

        let mut messages = vec![];
        for i in 0..3 {
            let request_message = SMPCRequest {
                request_type: ENROLLMENT_REQUEST_TYPE.to_string(),
                request_id: id.to_string(),
                iris_code: shared_template[i].code.to_vec(),
                mask_code: shared_template[i].mask.to_vec(),
            };

            messages.push(
                PublishBatchRequestEntry::builder()
                    .message(to_string(&request_message)?)
                    .message_group_id(format!("node{}", i))
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
    }

    Ok(())
}
