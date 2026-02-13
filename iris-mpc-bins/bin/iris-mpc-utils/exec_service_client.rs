use std::{fmt, path::PathBuf};

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

use iris_mpc_utils::{
    client::{AwsOptions, ServiceClient, ServiceClientOptions},
    fsys::reader::read_toml,
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
    /// Path to service client options.
    #[clap(long)]
    path_to_opts: String,

    /// Path to AWS options.
    #[clap(long)]
    path_to_opts_aws: String,
}

impl CliOptions {
    fn path_to_opts(&self) -> PathBuf {
        PathBuf::from(self.path_to_opts.clone())
    }

    fn path_to_opts_aws(&self) -> PathBuf {
        PathBuf::from(self.path_to_opts_aws.clone())
    }
}

impl fmt::Display for CliOptions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
------------------------------------------------------------------------
Iris-MPC Service Client Options:
    aws options
        {}
    exec options
        {}
------------------------------------------------------------------------
                ",
            self.path_to_opts_aws, self.path_to_opts,
        )
    }
}

#[async_from::async_trait]
impl<R: Rng + CryptoRng + SeedableRng + Send> AsyncFrom<CliOptions> for ServiceClient<R> {
    async fn async_from(options: CliOptions) -> Self {
        ServiceClient::<R>::new(
            ServiceClientOptions::from(&options),
            AwsOptions::from(&options),
        )
        .await
        .unwrap()
    }
}

impl From<&CliOptions> for ServiceClientOptions {
    fn from(options: &CliOptions) -> Self {
        read_toml::<Self>(options.path_to_opts().as_path())
            .expect("Failed to read service client configuration file")
    }
}

impl From<&CliOptions> for AwsOptions {
    fn from(options: &CliOptions) -> Self {
        read_toml::<Self>(options.path_to_opts_aws().as_path())
            .expect("Failed to read service client AWS configuration file")
    }
}
