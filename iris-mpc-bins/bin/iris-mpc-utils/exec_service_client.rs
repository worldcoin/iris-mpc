use std::{fmt, path::PathBuf};

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, SeedableRng};

use iris_mpc_utils::{
    client::{ServiceClient, ServiceClientConfiguration},
    fsys::reader::read_toml_config,
};

#[tokio::main]
pub async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let options = CliOptions::parse();
    tracing::info!("{}", options);

    let mut client = ServiceClient::<StdRng>::async_from(options.clone()).await;
    if let Err(e) = client.init().await {
        tracing::error!("Initialisation failure: {}", e);
        return Err(e.into());
    }

    client.exec().await?;

    Ok(())
}

#[derive(Debug, Parser, Clone)]
struct CliOptions {
    /// Path to service client configuration file.
    #[clap(long)]
    path_to_config_file: String,

    /// A random number generator seed for upstream entropy.
    #[clap(long)]
    rng_seed: Option<u64>,
}

impl CliOptions {
    #[allow(dead_code)]
    fn rng_seed(&self) -> StdRng {
        if self.rng_seed.is_some() {
            StdRng::seed_from_u64(self.rng_seed.unwrap())
        } else {
            StdRng::from_entropy()
        }
    }
}

impl fmt::Display for CliOptions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
------------------------------------------------------------------------
Iris-MPC Service Client Options:
    path_to_config_file
        {}
    rng_seed
        {:?}
------------------------------------------------------------------------
                ",
            self.path_to_config_file, self.rng_seed,
        )
    }
}

#[async_from::async_trait]
impl AsyncFrom<CliOptions> for ServiceClient<StdRng> {
    async fn async_from(options: CliOptions) -> Self {
        ServiceClient::<StdRng>::new(
            ServiceClientConfiguration::from(&options),
            options.rng_seed(),
        )
        .await
    }
}

impl From<&CliOptions> for ServiceClientConfiguration {
    fn from(options: &CliOptions) -> Self {
        read_toml_config::<ServiceClientConfiguration>(
            PathBuf::from(&options.path_to_config_file).as_path(),
        )
        .expect("Failed to read service client configuration file")
    }
}
