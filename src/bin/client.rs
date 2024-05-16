use aws_sdk_sns::{config::Region, Client};
use clap::Parser;
use gpu_iris_mpc::setup::iris_db::iris::IrisCode;
use rand::{rngs::StdRng, SeedableRng};

const N_QUERIES: usize = 30;
const N_ROTATIONS: usize = 31;
const REGION: &str = "us-east-2";
const RNG_SEED: u64 = 42;

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    topic_arn: String,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let Opt {
        topic_arn,
    } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    IrisCode::random_rng(&mut rng);

    let rsp = client
        .publish()
        .topic_arn(topic_arn)
        .message("hello sns!")
        .message_group_id("abc")
        .send()
        .await?;

    Ok(())
}