mod server;

use clap::Parser;
use iris_mpc::client::{run_client, Opt};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let opt = Opt::parse();
    println!("Running with options: {:?}", opt);
    run_client(opt).await
}
