pub mod json_wrapper;

use crate::config::json_wrapper::JsonStrWrapper;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Parser)]
pub struct Opt {
    #[structopt(long)]
    requests_queue_url: Option<String>,

    #[structopt(long)]
    results_topic_arn: Option<String>,

    #[structopt(long)]
    party_id: Option<usize>,

    #[structopt(long)]
    bootstrap_url: Option<String>,

    #[structopt(long)]
    path: Option<String>,
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
    pub path: String,

    #[serde(default)]
    pub kms_key_arns: JsonStrWrapper<Vec<String>>,

    #[serde(default)]
    pub servers: ServersConfig,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub database: Option<DbConfig>,

    #[serde(default)]
    pub aws: Option<AwsConfig>,
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

        dbg!("Debug config");
        dbg!(&config);

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

        if let Some(bootstrap_url) = opts.bootstrap_url {
            self.servers.bootstrap_url = Some(bootstrap_url);
        }

        if let Some(path) = opts.path {
            self.path = path;
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServersConfig {
    pub codes_engine_port:       u16,
    pub masks_engine_port:       u16,
    pub batch_codes_engine_port: u16,
    pub batch_masks_engine_port: u16,
    pub phase_2_batch_port:      u16,
    pub phase_2_port:            u16,
    pub sync_port:               u16,
    #[serde(default)]
    pub bootstrap_url:           Option<String>,
}
