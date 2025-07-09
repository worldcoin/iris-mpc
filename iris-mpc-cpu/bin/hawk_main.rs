use clap::Parser;
use eyre::Result;
use iris_mpc_cpu::execution::hawk_main::{hawk_main, HawkArgs};
use std::process::exit;

#[tokio::main]
async fn main() -> Result<()> {
    match hawk_main(HawkArgs::parse()).await {
        Ok(_) => tracing::info!("Hawk main execution completed successfully!"),
        Err(e) => {
            tracing::error!("Encountered an error during hawk_main processing: {}", e);
            exit(1);
        }
    };
    Ok(())
}
