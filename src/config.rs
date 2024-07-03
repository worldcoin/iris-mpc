use std::path::Path;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub fn load_config<T>(
    prefix: &str,
    config_path: Option<&Path>,
) -> eyre::Result<T>
where
    T: DeserializeOwned,
{
    let mut settings = config::Config::builder();

    if let Some(path) = config_path {
        settings = settings.add_source(config::File::from(path).required(true));
    }

    let settings = settings
        .add_source(
            config::Environment::with_prefix(prefix)
                .separator("__")
                .try_parsing(true),
        )
        .build()?;

    let config = settings.try_deserialize::<T>()?;

    Ok(config)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub environment: Option<String>,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub database: Option<DbConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DbConfig {
    pub url: String,

    #[serde(default)]
    pub migrate: bool,

    #[serde(default)]
    pub create: bool,
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
    pub service_name: String,
    // Traces
    pub traces_endpoint: Option<String>,
}
