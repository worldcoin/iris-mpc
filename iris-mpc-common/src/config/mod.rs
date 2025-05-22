use crate::{config::json_wrapper::JsonStrWrapper, job::Eye};
use clap::Parser;
use eyre::Result;
use serde::{Deserialize, Deserializer, Serialize};
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

#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_schema_name")]
    pub schema_name: String,

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
    pub cpu_database: Option<DbConfig>,

    #[serde(default)]
    pub aws: Option<AwsConfig>,

    #[serde(default = "default_processing_timeout_secs")]
    pub processing_timeout_secs: u64,

    #[serde(default = "default_startup_sync_timeout_secs")]
    pub startup_sync_timeout_secs: u64,

    #[serde(default)]
    pub public_key_base_url: String,

    #[serde(default = "default_shares_bucket_name")]
    pub shares_bucket_name: String,

    #[serde(default)]
    pub clear_db_before_init: bool,

    #[serde(default)]
    pub init_db_size: usize,

    #[serde(default)]
    pub max_db_size: usize,

    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    #[serde(default = "default_heartbeat_interval_secs")]
    pub heartbeat_interval_secs: u64,

    #[serde(default = "default_heartbeat_initial_retries")]
    pub heartbeat_initial_retries: u64,

    #[serde(default)]
    pub fake_db_size: usize,

    #[serde(default)]
    pub return_partial_results: bool,

    #[serde(default)]
    pub disable_persistence: bool,

    #[serde(default)]
    pub cpu_disable_persistence: bool,

    #[serde(default)]
    pub enable_debug_timing: bool,

    #[serde(default, deserialize_with = "deserialize_yaml_json_string")]
    pub node_hostnames: Vec<String>,

    #[serde(
        default = "default_service_ports",
        deserialize_with = "deserialize_yaml_json_string"
    )]
    pub service_ports: Vec<String>,

    #[serde(
        default = "default_healthcheck_ports",
        deserialize_with = "deserialize_yaml_json_string"
    )]
    pub healthcheck_ports: Vec<String>,

    #[serde(default = "default_http_query_retry_delay_ms")]
    pub http_query_retry_delay_ms: u64,

    #[serde(default = "default_shutdown_last_results_sync_timeout_secs")]
    pub shutdown_last_results_sync_timeout_secs: u64,

    #[serde(default)]
    pub image_name: String,

    #[serde(default)]
    pub enable_s3_importer: bool,

    #[serde(default)]
    pub db_chunks_bucket_name: String,

    #[serde(default = "default_load_chunks_parallelism")]
    pub load_chunks_parallelism: usize,

    /// Defines the safety overlap to load the DB records >last_modified_at in
    /// seconds. This is to ensure we don't miss any records that were
    /// updated during the DB export to S3
    #[serde(default = "default_db_load_safety_overlap_seconds")]
    pub db_load_safety_overlap_seconds: i64,

    #[serde(default)]
    pub db_chunks_folder_name: String,

    #[serde(default)]
    pub load_chunks_buffer_size: usize,

    #[serde(default = "default_load_chunks_max_retries")]
    pub load_chunks_max_retries: usize,

    #[serde(default = "default_load_chunks_initial_backoff_ms")]
    pub load_chunks_initial_backoff_ms: u64,

    #[serde(default)]
    pub fixed_shared_secrets: bool,

    /// LUC is the defense mechanism by which iris computations are performed
    /// using the OR rule for right and left matches.
    #[serde(default)]
    pub luc_enabled: bool,

    /// LUC look back is the time frame in days for which to use OR rule for a
    /// new signup against existing signups in that time period.
    #[serde(default = "default_luc_lookback_records")]
    pub luc_lookback_records: usize,

    /// Alternatively, we can use the serial IDs from the SMPc request to mark
    /// which records are to be processed using the OR rule.
    #[serde(default)]
    pub luc_serial_ids_from_smpc_request: bool,

    /// The size of the match distance buffer collecting matches for anonymized
    /// histogram creation. This gets multiplied by the number of GPU
    /// devices.
    #[serde(default = "default_match_distances_buffer_size")]
    pub match_distances_buffer_size: usize,

    #[serde(default = "default_match_distances_buffer_size_extra_percent")]
    pub match_distances_buffer_size_extra_percent: usize,

    #[serde(default = "default_n_buckets")]
    pub n_buckets: usize,

    #[serde(default)]
    pub enable_sending_anonymized_stats_message: bool,

    #[serde(default)]
    pub enable_sending_mirror_anonymized_stats_message: bool,

    #[serde(default)]
    pub enable_reauth: bool,

    #[serde(default)]
    pub enable_reset: bool,

    #[serde(default)]
    pub hnsw_schema_name_suffix: String,

    #[serde(default = "default_hawk_request_parallelism")]
    pub hawk_request_parallelism: usize,

    #[serde(default = "default_hawk_stream_parallelism")]
    pub hawk_stream_parallelism: usize,

    #[serde(default = "default_hawk_connection_parallelism")]
    pub hawk_connection_parallelism: usize,

    #[serde(default = "default_hawk_server_healthcheck_port")]
    pub hawk_server_healthcheck_port: usize,

    #[serde(default = "default_hnsw_param_ef_constr")]
    pub hnsw_param_ef_constr: usize,

    #[serde(default = "default_hnsw_param_M")]
    pub hnsw_param_M: usize,

    #[serde(default = "default_hnsw_param_ef_search")]
    pub hnsw_param_ef_search: usize,

    #[serde(default)]
    pub hawk_prng_seed: Option<u64>,

    #[serde(default = "default_max_deletions_per_batch")]
    pub max_deletions_per_batch: usize,

    /// Server process behaviour can be adjusted as per compute mode.
    #[serde(default)]
    pub mode_of_compute: ModeOfCompute,

    /// Server process behaviour can be adjusted as per deployment mode.
    #[serde(default)]
    pub mode_of_deployment: ModeOfDeployment,

    #[serde(default)]
    pub enable_modifications_sync: bool,

    #[serde(default)]
    pub enable_modifications_replay: bool,

    #[serde(default = "default_sqs_sync_long_poll_seconds")]
    pub sqs_sync_long_poll_seconds: i32,

    #[serde(default = "default_hawk_server_deletions_enabled")]
    pub hawk_server_deletions_enabled: bool,

    #[serde(default = "default_hawk_server_reauths_enabled")]
    pub hawk_server_reauths_enabled: bool,

    #[serde(default = "default_hawk_server_resets_enabled")]
    pub hawk_server_resets_enabled: bool,

    #[serde(default = "default_full_scan_side")]
    pub full_scan_side: Eye,

    #[serde(default = "default_batch_polling_timeout_secs")]
    pub batch_polling_timeout_secs: i32,

    #[serde(default = "default_sqs_long_poll_wait_time")]
    pub sqs_long_poll_wait_time: usize,
}

fn default_full_scan_side() -> Eye {
    Eye::Left
}

/// Enumeration over set of compute modes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModeOfCompute {
    /// Computation with standard CPUs (see HNSW graph).
    Cpu,
    /// Computation with Cuda GPU(s).
    #[default]
    Gpu,
}

/// Enumeration over set of deployment modes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModeOfDeployment {
    // shadow mode for when HSNW deployment does not read from the Gpu implementation
    // it should create and write its own shares DB
    ShadowIsolation,
    // Shadow mode for when HNSW test deployment reads from the Gpu Implementation
    ShadowReadOnly,
    // Standard mode for all other deployments.
    #[default]
    Standard,
}

fn default_load_chunks_parallelism() -> usize {
    32
}

fn default_processing_timeout_secs() -> u64 {
    60
}

fn default_startup_sync_timeout_secs() -> u64 {
    300
}

fn default_max_batch_size() -> usize {
    64
}

fn default_heartbeat_interval_secs() -> u64 {
    2
}

fn default_heartbeat_initial_retries() -> u64 {
    10
}

fn default_shutdown_last_results_sync_timeout_secs() -> u64 {
    10
}

fn default_shares_bucket_name() -> String {
    "wf-mpc-prod-smpcv2-sns-requests".to_string()
}

fn default_schema_name() -> String {
    "SMPC".to_string()
}

fn default_db_load_safety_overlap_seconds() -> i64 {
    60
}

fn default_luc_lookback_records() -> usize {
    0
}

fn default_load_chunks_max_retries() -> usize {
    5
}

fn default_load_chunks_initial_backoff_ms() -> u64 {
    200
}

// This gets multiplied by the number of GPU devices
fn default_match_distances_buffer_size() -> usize {
    1 << 16
}

fn default_match_distances_buffer_size_extra_percent() -> usize {
    20
}

fn default_n_buckets() -> usize {
    375
}

fn default_hawk_request_parallelism() -> usize {
    1024
}

fn default_hawk_stream_parallelism() -> usize {
    8
}

fn default_hawk_connection_parallelism() -> usize {
    16
}

fn default_hawk_server_healthcheck_port() -> usize {
    300
}

fn default_hnsw_param_ef_constr() -> usize {
    320
}

#[allow(non_snake_case)]
fn default_hnsw_param_M() -> usize {
    256
}

fn default_hnsw_param_ef_search() -> usize {
    256
}

fn default_service_ports() -> Vec<String> {
    vec!["4000".to_string(); 3]
}

fn default_healthcheck_ports() -> Vec<String> {
    vec!["3000".to_string(); 3]
}

fn default_http_query_retry_delay_ms() -> u64 {
    1000
}

fn default_max_deletions_per_batch() -> usize {
    100
}

fn default_sqs_sync_long_poll_seconds() -> i32 {
    10
}

fn default_hawk_server_reauths_enabled() -> bool {
    false
}

fn default_hawk_server_resets_enabled() -> bool {
    false
}

fn default_hawk_server_deletions_enabled() -> bool {
    false
}

fn default_batch_polling_timeout_secs() -> i32 {
    1
}

fn default_sqs_long_poll_wait_time() -> usize {
    10
}

impl Config {
    pub fn load_config(prefix: &str) -> Result<Config> {
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
    pub service_name: String,
    // Traces
    pub traces_endpoint: Option<String>,
    // Metrics
    pub metrics: Option<MetricsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub host: String,
    pub port: u16,
    pub queue_size: usize,
    pub buffer_size: usize,
    pub prefix: String,
}

fn deserialize_yaml_json_string<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: String = Deserialize::deserialize(deserializer)?;
    serde_json::from_str(&value).map_err(serde::de::Error::custom)
}

/// This struct is used to extract the common configuration for all servers from their respective configs.
/// It is later used to to hash the config and check if it is the same across all servers as a basic sanity check during startup.
#[allow(non_snake_case)]
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CommonConfig {
    environment: String,
    results_topic_arn: String,
    processing_timeout_secs: u64,
    startup_sync_timeout_secs: u64,
    public_key_base_url: String,
    shares_bucket_name: String,
    clear_db_before_init: bool,
    init_db_size: usize,
    max_db_size: usize,
    max_batch_size: usize,
    heartbeat_interval_secs: u64,
    heartbeat_initial_retries: u64,
    fake_db_size: usize,
    return_partial_results: bool,
    disable_persistence: bool,
    shutdown_last_results_sync_timeout_secs: u64,
    image_name: String,
    fixed_shared_secrets: bool,
    luc_enabled: bool,
    luc_lookback_records: usize,
    luc_serial_ids_from_smpc_request: bool,
    match_distances_buffer_size: usize,
    match_distances_buffer_size_extra_percent: usize,
    n_buckets: usize,
    enable_sending_anonymized_stats_message: bool,
    enable_sending_mirror_anonymized_stats_message: bool,
    enable_reauth: bool,
    enable_reset: bool,
    hawk_request_parallelism: usize,
    hawk_stream_parallelism: usize,
    hawk_connection_parallelism: usize,
    hnsw_param_ef_constr: usize,
    hnsw_param_M: usize,
    hnsw_param_ef_search: usize,
    hawk_prng_seed: Option<u64>,
    max_deletions_per_batch: usize,
    mode_of_compute: ModeOfCompute,
    mode_of_deployment: ModeOfDeployment,
    enable_modifications_sync: bool,
    enable_modifications_replay: bool,
    sqs_sync_long_poll_seconds: i32,
    hawk_server_deletions_enabled: bool,
    hawk_server_reauths_enabled: bool,
    schema_name: String,
    hnsw_schema_name_suffix: String,
    cpu_disable_persistence: bool,
    hawk_server_resets_enabled: bool,
    full_scan_side: Eye,
    batch_polling_timeout_secs: i32,
    sqs_long_poll_wait_time: usize,
}

impl From<Config> for CommonConfig {
    fn from(value: Config) -> Self {
        // This is destructured here intentionally to cause a compile error if
        // any of the fields are added to the struct without being considered if they should be in the common config hash or not.
        let Config {
            environment,
            party_id: _,           // party id is different for each server
            requests_queue_url: _, // requests queue url is different for each server
            results_topic_arn,
            kms_key_arns: _, // kms key arns are different for each server
            service: _,
            database: _,     // database is different for each server
            cpu_database: _, // cpu database is different for each server
            aws: _,          // aws is different for each server
            processing_timeout_secs,
            startup_sync_timeout_secs,
            public_key_base_url,
            shares_bucket_name,
            clear_db_before_init,
            init_db_size,
            max_db_size,
            max_batch_size,
            heartbeat_interval_secs,
            heartbeat_initial_retries,
            fake_db_size,
            return_partial_results,
            disable_persistence,
            enable_debug_timing: _,
            node_hostnames: _,            // Could be different for each server
            service_ports: _,             // Could be different for each server
            healthcheck_ports: _,         // Could be different for each server
            http_query_retry_delay_ms: _, // Could be different for each server
            shutdown_last_results_sync_timeout_secs,
            image_name,
            enable_s3_importer: _, // it does not matter if this is synced or not between servers
            db_chunks_bucket_name: _, // different for each server
            load_chunks_parallelism: _, // could be different for each server
            db_load_safety_overlap_seconds: _, // could be different for each server
            db_chunks_folder_name: _, // different for each server
            load_chunks_buffer_size: _, // could be different for each server
            load_chunks_max_retries: _, // could be different for each server
            load_chunks_initial_backoff_ms: _, // could be different for each server
            fixed_shared_secrets,
            luc_enabled,
            luc_lookback_records,
            luc_serial_ids_from_smpc_request,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            n_buckets,
            enable_sending_anonymized_stats_message,
            enable_sending_mirror_anonymized_stats_message,
            enable_reauth,
            enable_reset,
            hawk_request_parallelism,
            hawk_stream_parallelism,
            hawk_connection_parallelism,
            hawk_server_healthcheck_port: _, // different for each server
            hnsw_param_ef_constr,
            hnsw_param_M,
            hnsw_param_ef_search,
            hawk_prng_seed,
            max_deletions_per_batch,
            mode_of_compute,
            mode_of_deployment,
            enable_modifications_sync,
            enable_modifications_replay,
            sqs_sync_long_poll_seconds,
            hawk_server_deletions_enabled,
            hawk_server_reauths_enabled,
            schema_name,
            hnsw_schema_name_suffix,
            cpu_disable_persistence,
            hawk_server_resets_enabled,
            full_scan_side,
            batch_polling_timeout_secs,
            sqs_long_poll_wait_time,
        } = value;

        Self {
            environment,
            results_topic_arn,
            processing_timeout_secs,
            startup_sync_timeout_secs,
            public_key_base_url,
            shares_bucket_name,
            clear_db_before_init,
            init_db_size,
            max_db_size,
            max_batch_size,
            heartbeat_interval_secs,
            heartbeat_initial_retries,
            fake_db_size,
            return_partial_results,
            disable_persistence,
            shutdown_last_results_sync_timeout_secs,
            image_name,
            fixed_shared_secrets,
            luc_enabled,
            luc_lookback_records,
            luc_serial_ids_from_smpc_request,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            n_buckets,
            enable_sending_anonymized_stats_message,
            enable_sending_mirror_anonymized_stats_message,
            enable_reauth,
            enable_reset,
            hawk_request_parallelism,
            hawk_stream_parallelism,
            hawk_connection_parallelism,
            hnsw_param_ef_constr,
            hnsw_param_M,
            hnsw_param_ef_search,
            hawk_prng_seed,
            max_deletions_per_batch,
            mode_of_compute,
            mode_of_deployment,
            enable_modifications_sync,
            enable_modifications_replay,
            sqs_sync_long_poll_seconds,
            hawk_server_deletions_enabled,
            hawk_server_reauths_enabled,
            schema_name,
            hnsw_schema_name_suffix,
            cpu_disable_persistence,
            hawk_server_resets_enabled,
            full_scan_side,
            batch_polling_timeout_secs,
            sqs_long_poll_wait_time,
        }
    }
}
