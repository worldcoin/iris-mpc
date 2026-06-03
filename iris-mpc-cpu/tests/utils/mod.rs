use iris_mpc_cpu::graph_checkpoint::PruningMode;

pub mod cpu_node;
pub mod runner;
pub mod wait_conditions;
pub mod wal_builder;

/// Number of MPC parties.
pub const COUNT_OF_PARTIES: usize = 3;

/// Per-party configuration array.
pub type CpuConfigs = [CpuNodeConfig; COUNT_OF_PARTIES];

/// Hardcoded loopback ports for hawk_main's MPC network.
/// These must not conflict with SIDECAR_ADDRS or the coordination ports.
pub const HAWK_ADDRS: [&str; COUNT_OF_PARTIES] =
    ["127.0.0.1:16000", "127.0.0.1:16100", "127.0.0.1:16200"];

/// Hardcoded loopback ports for sidecar_main's MPC network.
/// Different from HAWK_ADDRS so both can run in the same test process (wal_103).
pub const SIDECAR_ADDRS: [&str; COUNT_OF_PARTIES] =
    ["127.0.0.1:16010", "127.0.0.1:16110", "127.0.0.1:16210"];

/// Per-party test configuration.
///
/// This is a test-local struct rather than the production `iris_mpc_common::Config`
/// to keep test setup minimal and explicit.
#[derive(Debug, Clone)]
pub struct CpuNodeConfig {
    /// PostgreSQL connection URL for this party's CPU database.
    pub db_url: String,
    /// Schema name for this party (e.g. "cpu_party_0").
    pub db_schema: String,
    /// S3 bucket name for graph checkpoints.
    pub checkpoint_bucket: String,
    /// Party index (0, 1, 2).
    pub party_id: usize,
    /// Port that this party's coordination server listens on.
    /// Used by TC-1 (wait_for_all_ready) to construct a ServerCoordinationConfig.
    /// See open question #5 and #6 in readme.
    pub coordination_port: u16,
    /// Healthcheck port for this party's HTTP health endpoint.
    /// Used to populate `ServerCoordinationConfig::healthcheck_ports`.
    pub healthcheck_port: u16,
    /// Sidecar-specific settings — can be overridden per test.
    pub sidecar: SidecarTestConfig,
}

/// Sidecar settings kept separate so individual tests can override them.
#[derive(Debug, Clone)]
pub struct SidecarTestConfig {
    pub cycle_interval_secs: u64,
    pub retry_interval_secs: u64,
    pub peer_round_timeout_secs: u64,
    /// Guard: sidecar will not checkpoint if fewer than this many new WAL rows exist.
    /// Set to 5 by default; tests must seed at least this many mutations.
    pub min_mutations_per_cycle: u64,
    pub checkpoint_window: usize,
    pub is_archival: bool,
    pub pruning_mode: PruningMode,
}

impl Default for SidecarTestConfig {
    fn default() -> Self {
        Self {
            cycle_interval_secs: 1,
            retry_interval_secs: 1,
            peer_round_timeout_secs: 30,
            min_mutations_per_cycle: 5,
            checkpoint_window: 10,
            is_archival: false,
            pruning_mode: PruningMode::None,
        }
    }
}
