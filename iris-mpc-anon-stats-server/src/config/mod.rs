use clap::Parser;
use iris_mpc_common::config::{AwsConfig, ServiceConfig};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Parser)]
pub struct Opt {
    #[clap(long)]
    pub party_id: Option<usize>,

    /// The addresses for the networking parties.
    #[clap(long)]
    pub addresses: Option<Vec<String>>,

    #[clap(long)]
    pub healthcheck_port: Option<usize>,

    #[clap(long)]
    pub results_topic_arn: Option<String>,
}

/// CLI configuration for the anon stats server.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonStatsServerConfig {
    /// The addresses for the networking parties.
    #[serde(default, deserialize_with = "deserialize_yaml_json_string")]
    pub addresses: Vec<String>,

    #[serde(default = "default_healthcheck_port")]
    pub healthcheck_port: usize,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub aws: Option<AwsConfig>,

    #[serde(default)]
    pub party_id: usize,

    #[serde(default)]
    pub environment: String,

    #[serde(default)]
    pub results_topic_arn: String,

    #[serde(default = "default_n_buckets_1d")]
    /// Number of buckets to use in 1D anon stats computation.
    pub n_buckets_1d: usize,

    #[serde(default = "default_min_1d_job_size")]
    /// Minimum job size for 1D anon stats computation.
    /// If the available job size is smaller than this, the party will wait until enough data is available.
    pub min_1d_job_size: usize,

    #[serde(default = "default_poll_interval_secs")]
    /// Interval, in seconds, between polling attempts.
    pub poll_interval_secs: u64,

    #[serde(default = "default_max_sync_failures_before_reset")]
    /// Number of consecutive sync mismatches before clearing the local queue for an origin.
    pub max_sync_failures_before_reset: usize,

    #[serde(default)]
    /// Database connection URL.
    pub db_url: String,

    #[serde(default = "default_schema_name")]
    /// Database schema name.
    pub db_schema_name: String,
}

fn default_healthcheck_port() -> usize {
    8080
}

fn default_n_buckets_1d() -> usize {
    10
}

fn default_min_1d_job_size() -> usize {
    1000
}

fn default_schema_name() -> String {
    "anon_stats_mpc".to_string()
}

fn default_poll_interval_secs() -> u64 {
    30
}

fn default_max_sync_failures_before_reset() -> usize {
    3
}

impl AnonStatsServerConfig {
    pub fn load_config(prefix: &str) -> eyre::Result<AnonStatsServerConfig> {
        let settings = config::Config::builder();
        let settings = settings
            .add_source(
                config::Environment::with_prefix(prefix)
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;

        let config: AnonStatsServerConfig = settings.try_deserialize::<AnonStatsServerConfig>()?;
        Ok(config)
    }

    pub fn overwrite_defaults_with_cli_args(&mut self, opts: Opt) {
        if let Some(healthcheck_port) = opts.healthcheck_port {
            self.healthcheck_port = healthcheck_port;
        }

        if let Some(party_id) = opts.party_id {
            self.party_id = party_id;
        }

        if let Some(addresses) = opts.addresses {
            self.addresses = addresses;
        }

        if let Some(results_topic_arn) = opts.results_topic_arn {
            self.results_topic_arn = results_topic_arn;
        }
    }
}

fn deserialize_yaml_json_string<'de, D>(deserializer: D) -> eyre::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: String = Deserialize::deserialize(deserializer)?;
    serde_json::from_str(&value).map_err(serde::de::Error::custom)
}
