use clap::Parser;
use iris_mpc::client::e2e::{run_client, Opt};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let opts = Opt::parse();
    println!("Running with options: {:?}", opts);
    run_client(opts).await
}
