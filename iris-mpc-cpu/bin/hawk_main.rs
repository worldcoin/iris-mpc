use clap::Parser;
use eyre::Result;
use iris_mpc_cpu::execution::hawk_main::{hawk_main, HawkArgs};

#[tokio::main]
async fn main() -> Result<()> {
    hawk_main(HawkArgs::parse()).await
}
