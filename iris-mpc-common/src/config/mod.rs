use crate::config::json_wrapper::JsonStrWrapper;
use ampc_actor_utils::network::config::deserialize_yaml_json_string;
use ampc_actor_utils::network::config::TlsConfig;
use ampc_anon_stats::types::Eye;
use ampc_server_utils::{AwsConfig, ServerCoordinationConfig, ServiceConfig};
use clap::Parser;
use eyre::Result;
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;

pub mod json_wrapper;

pub const ENV_DEV: &str = "dev";
pub const ENV_PROD: &str = "prod";
pub const ENV_STAGE: &str = "stage";

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
    pub tls: Option<TlsConfig>,

    #[serde(default)]
    pub service: Option<ServiceConfig>,

    #[serde(default)]
    pub server_coordination: Option<ServerCoordinationConfig>,

    #[serde(default)]
    pub database: Option<DbConfig>,

    #[serde(default)]
    pub cpu_database: Option<DbConfig>,

    #[serde(default)]
    pub anon_stats_database: Option<DbConfig>,

    #[serde(default = "default_anon_stats_schema_name")]
    pub anon_stats_schema_name: String,

    #[serde(default)]
    pub aws: Option<AwsConfig>,

    #[serde(default = "default_processing_timeout_secs")]
    pub processing_timeout_secs: u64,

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

    // used for testing to recreate batch sequence
    #[serde(
        default = "default_predefined_batch_sizes",
        deserialize_with = "deserialize_usize_vec"
    )]
    pub predefined_batch_sizes: Vec<usize>,

    #[serde(default)]
    pub fake_db_size: usize,

    #[serde(default)]
    pub return_partial_results: bool,

    #[serde(default)]
    pub disable_persistence: bool,

    #[serde(default)]
    pub enable_debug_timing: bool,

    #[serde(
        default = "default_service_ports",
        deserialize_with = "deserialize_yaml_json_string"
    )]
    pub service_ports: Vec<String>,

    // should be set to the same as service_ports if not explicitly set.
    #[serde(default, deserialize_with = "deserialize_yaml_json_string")]
    pub service_outbound_ports: Vec<String>,

    #[serde(default = "default_shutdown_last_results_sync_timeout_secs")]
    pub shutdown_last_results_sync_timeout_secs: u64,

    #[serde(default)]
    pub enable_s3_importer: bool,

    #[serde(default)]
    pub db_chunks_bucket_name: String,

    #[serde(default = "default_db_chunks_bucket_region")]
    pub db_chunks_bucket_region: String,

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

    #[serde(default = "default_match_distances_2d_buffer_size")]
    pub match_distances_2d_buffer_size: usize,

    #[serde(default)]
    pub enable_reauth: bool,

    #[serde(default)]
    pub enable_reset: bool,

    #[serde(default)]
    pub hnsw_schema_name_suffix: String,

    #[serde(default)]
    pub gpu_schema_name_suffix: String,

    #[serde(default = "default_hawk_request_parallelism")]
    pub hawk_request_parallelism: usize,

    #[serde(default = "default_hawk_connection_parallelism")]
    pub hawk_connection_parallelism: usize,

    #[serde(default = "default_hnsw_param_ef_constr")]
    pub hnsw_param_ef_constr: usize,

    #[serde(default = "default_hnsw_param_M")]
    pub hnsw_param_M: usize,

    #[serde(default = "default_hnsw_param_ef_search")]
    pub hnsw_param_ef_search: usize,

    #[serde(default)]
    pub hnsw_layer_density: Option<usize>,

    #[serde(default)]
    pub hawk_prf_key: Option<u64>,

    #[serde(default = "default_hawk_numa")]
    pub hawk_numa: bool,

    #[serde(default = "default_max_deletions_per_batch")]
    pub max_deletions_per_batch: usize,

    #[serde(default = "default_max_modifications_lookback")]
    pub max_modifications_lookback: usize,

    #[serde(default)]
    pub enable_modifications_sync: bool,

    #[serde(default)]
    pub enable_modifications_replay: bool,

    #[serde(default = "default_pprof_s3_bucket")]
    pub pprof_s3_bucket: String,

    #[serde(default = "default_pprof_prefix")]
    pub pprof_prefix: String,

    #[serde(default)]
    pub pprof_run_id: Option<String>,

    #[serde(default = "default_pprof_seconds")]
    pub pprof_seconds: u64,

    #[serde(default = "default_pprof_frequency")]
    pub pprof_frequency: i32,

    #[serde(default = "default_pprof_idle_interval_sec")]
    pub pprof_idle_interval_sec: u64,

    #[serde(default)]
    pub pprof_flame_only: bool,

    #[serde(default)]
    pub pprof_profile_only: bool,

    #[serde(default = "default_pprof_per_batch_enabled")]
    pub enable_pprof_per_batch: bool,

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

    #[serde(default = "default_full_scan_side_switching_enabled")]
    pub full_scan_side_switching_enabled: bool,

    #[serde(default = "default_batch_polling_timeout_secs")]
    pub batch_polling_timeout_secs: i32,

    #[serde(default = "default_sqs_long_poll_wait_time")]
    pub sqs_long_poll_wait_time: usize,

    #[serde(default = "default_batch_sync_polling_timeout_secs")]
    pub batch_sync_polling_timeout_secs: u64,

    #[serde(default = "default_tokio_threads")]
    pub tokio_threads: usize,

    #[serde(default = "default_sns_retry_max_attempts")]
    pub sns_retry_max_attempts: u32,

    #[serde(default = "default_enable_identity_match_check_enabled")]
    pub enable_identity_match_check: bool,
}

fn default_full_scan_side() -> Eye {
    Eye::Left
}

fn default_db_chunks_bucket_region() -> String {
    "eu-north-1".to_string()
}

fn default_load_chunks_parallelism() -> usize {
    32
}

fn default_processing_timeout_secs() -> u64 {
    60
}

fn default_max_batch_size() -> usize {
    64
}

fn default_predefined_batch_sizes() -> Vec<usize> {
    Vec::new()
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

fn default_anon_stats_schema_name() -> String {
    "anon_stats_mpc".to_string()
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

// Default size for the 2D match distances buffer, needs to be a multiple of 64 at least
fn default_match_distances_2d_buffer_size() -> usize {
    1 << 13 // 8192
}

fn default_hawk_request_parallelism() -> usize {
    1024
}

fn default_hawk_connection_parallelism() -> usize {
    16
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

fn default_hawk_numa() -> bool {
    true
}

fn default_service_ports() -> Vec<String> {
    vec!["4000".to_string(); 3]
}

fn default_max_deletions_per_batch() -> usize {
    100
}

fn default_max_modifications_lookback() -> usize {
    (default_max_deletions_per_batch() + default_max_batch_size()) * 2
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

fn default_batch_sync_polling_timeout_secs() -> u64 {
    10
}

fn default_full_scan_side_switching_enabled() -> bool {
    true
}

// ---- pprof collector defaults ----
fn default_pprof_s3_bucket() -> String {
    // Stage default bucket; override in prod via env
    "wf-smpcv2-stage-hnsw-performance-reports".to_string()
}

fn default_pprof_prefix() -> String {
    "hnsw/pprof".to_string()
}

fn default_pprof_seconds() -> u64 {
    30
}

fn default_pprof_frequency() -> i32 {
    99
}

fn default_pprof_idle_interval_sec() -> u64 {
    5
}

fn default_pprof_per_batch_enabled() -> bool {
    false
}

fn default_tokio_threads() -> usize {
    num_cpus::get()
}

fn default_sns_retry_max_attempts() -> u32 {
    5
}

fn default_enable_identity_match_check_enabled() -> bool {
    true
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

        let mut config: Config = settings.try_deserialize::<Config>()?;

        // If service_outbound_ports is not explicitly set,
        // copy service_ports to service_outbound_ports
        if config.service_outbound_ports.is_empty() {
            config.service_outbound_ports = config.service_ports.clone();
        }

        if let Some(service_coordination) = &mut config.server_coordination {
            config.party_id = service_coordination.party_id;
        }
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

impl Config {
    /// Returns the url for connecting to a node's gpu database.
    pub fn get_gpu_db_url(&self) -> Option<String> {
        self.database.as_ref().map(|x| x.url.clone())
    }

    /// Returns the url for connecting to a node's cpu database.
    pub fn get_cpu_db_url(&self) -> Option<String> {
        self.cpu_database.as_ref().map(|x| x.url.clone())
    }

    pub fn get_anon_stats_db_url(&self) -> Option<String> {
        self.anon_stats_database
            .as_ref()
            .map(|x| x.url.clone())
            .or_else(|| self.cpu_database.as_ref().map(|x| x.url.clone()))
    }

    pub fn get_anon_stats_db_schema(&self) -> String {
        self.anon_stats_schema_name.clone()
    }

    /// Returns the name of a database schema for connecting to a node's gpu dB.
    pub fn get_gpu_db_schema(&self) -> String {
        self.format_db_schema(&self.gpu_schema_name_suffix)
    }

    /// Returns the name of a database schema for connecting to a node's cpu dB.
    pub fn get_cpu_db_schema(&self) -> String {
        self.format_db_schema(&self.hnsw_schema_name_suffix)
    }

    /// Returns the name of a database schema for connecting to a node's dB.
    ///
    /// Value of `schema_suffix` should be one of `config.gpu_schema_name_suffix`
    /// or `config.hnsw_schema_name_suffix`.
    fn format_db_schema(&self, schema_suffix: &str) -> String {
        let Self {
            schema_name,
            environment,
            party_id,
            ..
        } = self;

        format!(
            "{}{}_{}_{}",
            schema_name, schema_suffix, environment, party_id
        )
    }
}

/// Encapsulates database configuration settings.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct DbConfig {
    /// Connection string for database.
    pub url: String,

    /// Flag indicating whether to migrate database schema.
    #[serde(default)]
    pub migrate: bool,

    /// Flag indicating whether to create database schema.
    #[serde(default)]
    pub create: bool,

    /// Degree of parallelism for loading data.
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

fn deserialize_usize_vec<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
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
    public_key_base_url: String,
    shares_bucket_name: String,
    clear_db_before_init: bool,
    init_db_size: usize,
    max_db_size: usize,
    max_batch_size: usize,
    #[serde(default)]
    predefined_batch_sizes: Vec<usize>,
    fake_db_size: usize,
    return_partial_results: bool,
    disable_persistence: bool,
    shutdown_last_results_sync_timeout_secs: u64,
    db_chunks_bucket_region: String,
    fixed_shared_secrets: bool,
    luc_enabled: bool,
    luc_lookback_records: usize,
    luc_serial_ids_from_smpc_request: bool,
    match_distances_buffer_size: usize,
    match_distances_buffer_size_extra_percent: usize,
    match_distances_2d_buffer_size: usize,
    enable_reauth: bool,
    enable_reset: bool,
    enable_identity_match_check: bool,
    hawk_request_parallelism: usize,
    hawk_connection_parallelism: usize,
    hnsw_param_ef_constr: usize,
    hnsw_param_M: usize,
    hnsw_param_ef_search: usize,
    hnsw_layer_density: Option<usize>,
    hawk_prf_key: Option<u64>,
    max_deletions_per_batch: usize,
    max_modifications_lookback: usize,
    enable_modifications_sync: bool,
    enable_modifications_replay: bool,
    sqs_sync_long_poll_seconds: i32,
    hawk_server_deletions_enabled: bool,
    hawk_server_reauths_enabled: bool,
    schema_name: String,
    hnsw_schema_name_suffix: String,
    gpu_schema_name_suffix: String,
    hawk_server_resets_enabled: bool,
    full_scan_side: Eye,
    full_scan_side_switching_enabled: bool,
    batch_polling_timeout_secs: i32,
    sqs_long_poll_wait_time: usize,
    batch_sync_polling_timeout_secs: u64,
}

impl CommonConfig {
    pub fn get_max_modifications_lookback(&self) -> usize {
        self.max_modifications_lookback
    }
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
            tls: _,
            service: _,
            server_coordination: _,
            database: _,     // database is different for each server
            cpu_database: _, // cpu database is different for each server
            anon_stats_database: _,
            anon_stats_schema_name: _,
            aws: _, // aws is different for each server
            processing_timeout_secs,
            public_key_base_url,
            shares_bucket_name,
            clear_db_before_init,
            init_db_size,
            max_db_size,
            max_batch_size,
            predefined_batch_sizes,
            fake_db_size,
            return_partial_results,
            disable_persistence,
            enable_debug_timing: _,
            service_ports: _,          // Could be different for each server
            service_outbound_ports: _, // Could be different for each server
            shutdown_last_results_sync_timeout_secs,
            enable_s3_importer: _, // it does not matter if this is synced or not between servers
            db_chunks_bucket_name: _, // different for each server
            db_chunks_bucket_region,
            load_chunks_parallelism: _, // could be different for each server
            db_load_safety_overlap_seconds: _, // could be different for each server
            db_chunks_folder_name: _,   // different for each server
            load_chunks_buffer_size: _, // could be different for each server
            load_chunks_max_retries: _, // could be different for each server
            load_chunks_initial_backoff_ms: _, // could be different for each server
            fixed_shared_secrets,
            luc_enabled,
            luc_lookback_records,
            luc_serial_ids_from_smpc_request,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            match_distances_2d_buffer_size,
            enable_reauth,
            enable_reset,
            hawk_request_parallelism,
            hawk_connection_parallelism,
            hnsw_param_ef_constr,
            hnsw_param_M,
            hnsw_param_ef_search,
            hnsw_layer_density,
            hawk_prf_key,
            hawk_numa: _, // could be different for each server
            max_deletions_per_batch,
            max_modifications_lookback,
            enable_modifications_sync,
            enable_modifications_replay,
            sqs_sync_long_poll_seconds,
            hawk_server_deletions_enabled,
            hawk_server_reauths_enabled,
            schema_name,
            hnsw_schema_name_suffix,
            gpu_schema_name_suffix,
            hawk_server_resets_enabled,
            full_scan_side,
            full_scan_side_switching_enabled,
            batch_polling_timeout_secs,
            sqs_long_poll_wait_time,
            batch_sync_polling_timeout_secs,
            // pprof collector (not part of common hash)
            pprof_s3_bucket: _,
            pprof_prefix: _,
            pprof_run_id: _,
            pprof_seconds: _,
            pprof_frequency: _,
            pprof_idle_interval_sec: _,
            pprof_flame_only: _,
            pprof_profile_only: _,
            enable_pprof_per_batch: _,
            tokio_threads: _,
            sns_retry_max_attempts: _,
            enable_identity_match_check,
        } = value;

        Self {
            environment,
            results_topic_arn,
            processing_timeout_secs,
            public_key_base_url,
            shares_bucket_name,
            clear_db_before_init,
            init_db_size,
            max_db_size,
            predefined_batch_sizes,
            max_batch_size,
            fake_db_size,
            return_partial_results,
            disable_persistence,
            shutdown_last_results_sync_timeout_secs,
            db_chunks_bucket_region,
            fixed_shared_secrets,
            luc_enabled,
            luc_lookback_records,
            luc_serial_ids_from_smpc_request,
            match_distances_buffer_size,
            match_distances_buffer_size_extra_percent,
            match_distances_2d_buffer_size,
            enable_reauth,
            enable_reset,
            hawk_request_parallelism,
            hawk_connection_parallelism,
            hnsw_param_ef_constr,
            hnsw_param_M,
            hnsw_param_ef_search,
            hnsw_layer_density,
            hawk_prf_key,
            max_deletions_per_batch,
            max_modifications_lookback,
            enable_modifications_sync,
            enable_modifications_replay,
            sqs_sync_long_poll_seconds,
            hawk_server_deletions_enabled,
            hawk_server_reauths_enabled,
            schema_name,
            hnsw_schema_name_suffix,
            gpu_schema_name_suffix,
            hawk_server_resets_enabled,
            full_scan_side,
            full_scan_side_switching_enabled,
            batch_polling_timeout_secs,
            sqs_long_poll_wait_time,
            batch_sync_polling_timeout_secs,
            enable_identity_match_check,
        }
    }
}
