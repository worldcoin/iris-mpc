use crate::config::json_wrapper::JsonStrWrapper;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fmt;

pub mod json_wrapper;

#[derive(Debug, Parser)]
pub struct Opt {
    #[structopt(long)]
    requests_queue_url: Option<String>,

    #[structopt(long)]
    results_topic_arn: Option<String>,

    #[structopt(long)]
    party_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub environment: String,

    #[serde(default)]
    pub party_id: usize,

    #[serde(default)]
    pub requests_queue_url: String,

    #[serde(default)]
    pub results_topic_arn: String,

    #[serde(default)]
    pub kms_key_arns: JsonStrWrapper<Vec<String>>,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub database: Option<DbConfig>,

    #[serde(default)]
    pub aws: Option<AwsConfig>,

    #[serde(default = "default_processing_timeout_secs")]
    pub processing_timeout_secs: u64,

    #[serde(default)]
    pub public_key_base_url: String,
}

fn default_processing_timeout_secs() -> u64 {
    60
}

impl Config {
    pub fn load_config(prefix: &str) -> eyre::Result<Config> {
        let settings = config::Config::builder();
        let settings = settings
            .add_source(
                config::Environment::with_prefix(prefix)
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;

        let config: Config = settings.try_deserialize::<Config>()?;

        Ok(config)
    }

    pub fn overwrite_defaults_with_cli_args(&mut self, opts: Opt) {
        if let Some(requests_queue_url) = opts.requests_queue_url {
            self.requests_queue_url = requests_queue_url;
        }

        if let Some(results_topic_arn) = opts.results_topic_arn {
            self.results_topic_arn = results_topic_arn;
        }

        if let Some(party_id) = opts.party_id {
            self.party_id = party_id;
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct DbConfig {
    pub url: String,

    #[serde(default)]
    pub migrate: bool,

    #[serde(default)]
    pub create: bool,

    #[serde(default = "default_load_parallelism")]
    pub load_parallelism: usize,
}

fn default_load_parallelism() -> usize {
    8
}

impl fmt::Debug for DbConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DbConfig")
            .field("url", &"********") // Mask the URL
            .field("migrate", &self.migrate)
            .field("create", &self.create)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AwsConfig {
    /// Useful when using something like LocalStack
    pub endpoint: Option<String>,

    #[serde(default)]
    pub region: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    // Service name - used for logging, metrics and tracing
    pub service_name:    String,
    // Traces
    pub traces_endpoint: Option<String>,
    // Metrics
    pub metrics:         Option<MetricsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub host:        String,
    pub port:        u16,
    pub queue_size:  usize,
    pub buffer_size: usize,
    pub prefix:      String,
}
