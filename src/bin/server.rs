use aws_config::meta::region::RegionProviderChain;
use aws_sdk_sqs::{config::Region, meta::PKG_VERSION, Client, Error};
use clap::Parser;

const REGION: &str = "us-east-2";

#[derive(Debug, Parser)]
struct Opt {
    #[structopt(short, long)]
    queue: Option<String>,
}

#[derive(Debug)]
struct SQSMessage {
    body: String,
}

async fn receive(client: &Client, queue_url: &String) -> Result<(), Error> {
    let rcv_message_output = client.receive_message().queue_url(queue_url).send().await?;

    for message in rcv_message_output.messages.unwrap_or_default() {
        println!("Got the message: {:#?}", message);
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