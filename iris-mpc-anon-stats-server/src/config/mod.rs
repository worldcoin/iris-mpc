use clap::Parser;
use iris_mpc_common::config::ServiceConfig;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Debug, Parser)]
pub struct Opt {
    #[structopt(long)]
    party_id: Option<usize>,

    #[structopt(long)]
    bind_addr: Option<SocketAddr>,

    #[structopt(long)]
    healthcheck_port: Option<usize>,
}

/// CLI configuration for the anon stats server.
#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonStatsServerConfig {
    /// The socket address the HTTP server listens on.
    #[serde(default = "default_bind_addr")]
    pub bind_addr: SocketAddr,

    #[serde(default = "default_healthcheck_port")]
    pub healthcheck_port: usize,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub party_id: usize,

    #[serde(default)]
    pub environment: String,
}

fn default_bind_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 3000))
}

fn default_healthcheck_port() -> usize {
    8080
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
        if let Some(bind_addr) = opts.bind_addr {
            self.bind_addr = bind_addr;
        }

        if let Some(healthcheck_port) = opts.healthcheck_port {
            self.healthcheck_port = healthcheck_port;
        }

        if let Some(party_id) = opts.party_id {
            self.party_id = party_id;
        }
    }
}
