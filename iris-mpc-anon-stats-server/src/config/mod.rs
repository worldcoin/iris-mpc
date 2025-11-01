use clap::Parser;
use iris_mpc_common::config::ServiceConfig;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Parser)]
pub struct Opt {
    #[clap(long)]
    pub party_id: Option<usize>,

    #[clap(long)]
    pub bind_addr: Option<String>,

    /// The addresses for the networking parties.
    #[clap(long)]
    pub addresses: Option<Vec<String>>,

    #[clap(long)]
    pub healthcheck_port: Option<usize>,
}

/// CLI configuration for the anon stats server.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonStatsServerConfig {
    /// The socket address the HTTP server listens on.
    #[serde(default = "default_bind_addr")]
    pub bind_addr: String,

    /// The addresses for the networking parties.
    #[serde(default)]
    pub addresses: Vec<String>,

    #[serde(default = "default_healthcheck_port")]
    pub healthcheck_port: usize,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub party_id: usize,

    #[serde(default)]
    pub environment: String,

    #[serde(default = "default_n_buckets_1d")]
    /// Number of buckets to use in 1D anon stats computation.
    pub n_buckets_1d: usize,

    #[serde(default, deserialize_with = "deserialize_yaml_json_string")]
    pub node_hostnames: Vec<String>,

    #[serde(default, deserialize_with = "deserialize_yaml_json_string")]
    pub service_ports: Vec<String>,

    #[serde(skip)]
    explicit_bind_addr: bool,
}

fn default_bind_addr() -> String {
    "127.0.0.1:3000".to_string()
}

fn default_healthcheck_port() -> usize {
    8080
}

fn default_n_buckets_1d() -> usize {
    10
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

        let mut config: AnonStatsServerConfig =
            settings.try_deserialize::<AnonStatsServerConfig>()?;
        if config.bind_addr != default_bind_addr() {
            config.explicit_bind_addr = true;
        }

        Ok(config)
    }

    pub fn overwrite_defaults_with_cli_args(&mut self, opts: Opt) {
        if let Some(bind_addr) = opts.bind_addr {
            self.bind_addr = bind_addr;
            self.explicit_bind_addr = true;
        }

        if let Some(healthcheck_port) = opts.healthcheck_port {
            self.healthcheck_port = healthcheck_port;
        }

        if let Some(party_id) = opts.party_id {
            self.party_id = party_id;
        }

        if let Some(addresses) = opts.addresses {
            self.addresses = addresses;
        }
    }

    pub fn apply_party_network_defaults(&mut self) -> eyre::Result<()> {
        if self.explicit_bind_addr {
            return Ok(());
        }

        if self.node_hostnames.is_empty() || self.service_ports.is_empty() {
            return Ok(());
        }

        let hostname = self
            .node_hostnames
            .get(self.party_id)
            .cloned()
            .ok_or_else(|| {
                eyre::eyre!("party id {} out of range for node hostnames", self.party_id)
            })?;

        let port_str = self.service_ports.get(self.party_id).ok_or_else(|| {
            eyre::eyre!("party id {} out of range for service ports", self.party_id)
        })?;

        let port = port_str
            .parse::<u16>()
            .map_err(|err| eyre::eyre!("invalid service port '{}': {}", port_str, err))?;

        self.bind_addr = format!("{}:{}", hostname.trim(), port);
        Ok(())
    }
}

fn deserialize_yaml_json_string<'de, D>(deserializer: D) -> eyre::Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: String = Deserialize::deserialize(deserializer)?;
    serde_json::from_str(&value).map_err(serde::de::Error::custom)
}
