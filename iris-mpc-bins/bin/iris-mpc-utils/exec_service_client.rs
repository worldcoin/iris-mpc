use std::{fmt, path::PathBuf};

use async_from::{self, AsyncFrom};
use clap::Parser;
use eyre::Result;
use rand::{rngs::StdRng, CryptoRng, Rng, SeedableRng};

use iris_mpc_utils::{
    client::{
        AwsConfiguration, ServiceClient as Client,
        ServiceClientConfiguration as ClientConfiguration,
    },
    fsys::reader::read_toml,
};

#[tokio::main]
pub async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    let options = CliOptions::parse();
    tracing::info!("{}", options);

    let mut client = Client::<StdRng>::async_from(options.clone()).await;
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
    path_to_config: String,

    /// Path to AWS configuration file.
    #[clap(long)]
    path_to_config_aws: String,
}

impl CliOptions {
    fn path_to_config(&self) -> PathBuf {
        PathBuf::from(self.path_to_config.clone())
    }

    fn path_to_config_aws(&self) -> PathBuf {
        PathBuf::from(self.path_to_config_aws.clone())
    }
}

impl fmt::Display for CliOptions {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
------------------------------------------------------------------------
Iris-MPC Service Client Options:
    path_to_config
        {}
    path_to_config_aws
        {}
------------------------------------------------------------------------
                ",
            self.path_to_config, self.path_to_config_aws,
        )
    }
}

#[async_from::async_trait]
impl<R: Rng + CryptoRng + SeedableRng + Send> AsyncFrom<CliOptions> for Client<R> {
    async fn async_from(options: CliOptions) -> Self {
        Client::<R>::new(
            ClientConfiguration::from(&options),
            AwsConfiguration::from(&options),
        )
        .await
    }
}

impl From<&CliOptions> for ClientConfiguration {
    fn from(options: &CliOptions) -> Self {
        read_toml::<ClientConfiguration>(options.path_to_config().as_path())
            .expect("Failed to read service client configuration file")
    }
}

impl From<&CliOptions> for AwsConfiguration {
    fn from(options: &CliOptions) -> Self {
        read_toml::<AwsConfiguration>(options.path_to_config_aws().as_path())
            .expect("Failed to read service client AWS configuration file")
    }
}
