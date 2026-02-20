use std::{fmt, path::PathBuf, time::UNIX_EPOCH};

use clap::Parser;
use eyre::Result;

use iris_mpc_utils::{
    client::{AwsOptions, ServiceClient2, ServiceClientOptions},
    fsys::reader::read_toml,
};

struct UtcHms;

impl tracing_subscriber::fmt::time::FormatTime for UtcHms {
    fn format_time(&self, w: &mut tracing_subscriber::fmt::format::Writer<'_>) -> fmt::Result {
        let secs = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        write!(
            w,
            "{:02}:{:02}:{:02}",
            (secs / 3600) % 24,
            (secs / 60) % 60,
            secs % 60
        )
    }
}

#[tokio::main]
pub async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_timer(UtcHms)
        .with_ansi(false)
        .with_target(false)
        .init();

    let options = CliOptions::parse();
    tracing::info!("{}", options);

    let mut opts = ServiceClientOptions::from(&options);
    if let Some(ref iris_path) = options.path_to_iris_shares {
        opts.set_iris_shares_path(iris_path);
    }

    let client = ServiceClient2::new(
        AwsOptions::from(&options),
        opts.request_batch,
        opts.shares_generator,
    )
    .await?;

    ServiceClient2::run(client).await;

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

    /// Path to iris shares NDJSON file (required when shares_generator is FromFile).
    #[clap(long)]
    path_to_iris_shares: Option<String>,
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
    iris shares
        {}
------------------------------------------------------------------------
                ",
            self.path_to_opts_aws,
            self.path_to_opts,
            self.path_to_iris_shares.as_deref().unwrap_or("(none)"),
        )
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
