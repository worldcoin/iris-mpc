use aws_config::meta::region::RegionProviderChain;
use aws_sdk_sqs::{config::Region, meta::PKG_VERSION, Client, Error};
use clap::Parser;
use gpu_iris_mpc::{setup::iris_db::shamir_iris::ShamirIris, sqs::SQSMessage};
use serde::{Deserialize, Serialize};

const REGION: &str = "us-east-2";
const QUERY_SIZE: usize = 30;

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    queue: Option<String>,
}

async fn receive(client: &Client, queue_url: &String) -> eyre::Result<()> {
    let rcv_message_output = client.receive_message().queue_url(queue_url).send().await?;

    for message in rcv_message_output.messages.unwrap_or_default() {
        let messsage: SQSMessage = serde_json::from_str(message.body().unwrap())?;
        let iris: ShamirIris = messsage.message.into();

        // put in batch

    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let Opt {
        queue,
    } = Opt::parse();

    let region_provider = Region::new(REGION);
    let shared_config = aws_config::from_env().region(region_provider).load().await;
    let client = Client::new(&shared_config);

    // let first_queue_url = find_first_queue(&client).await?;
    let queue_url = queue.unwrap();

    let message = SQSMessage {
        body: "hello from my queue".to_owned(),
    };

    receive(&client, &queue_url).await?;

    Ok(())
}