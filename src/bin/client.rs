use aws_sdk_sns::{config::Region, Client};
use clap::Parser;

const N_QUERIES: usize = 30;
const N_ROTATIONS: usize = 31;
const REGION: &str = "us-east-2";

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    topic_arn: String,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();

    let Opt {
        topic_arn,
    } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    let rsp = client
        .publish()
        .topic_arn(topic_arn)
        .message("hello sns!")
        .message_group_id("abc")
        .send()
        .await?;

    Ok(())
}