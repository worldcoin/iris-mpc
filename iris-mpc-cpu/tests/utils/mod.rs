pub mod checkpoint_seeder;
pub mod cpu_node;
pub mod runner;
pub mod wait_conditions;
pub mod wal_builder;

/// Number of MPC parties.
pub const COUNT_OF_PARTIES: usize = 3;

/// Per-party configuration array.
// TODO: replace with the real Config type from iris_mpc_common once the config
// structure for cpu-side tests is settled (open question #3 in readme).
pub type CpuConfigs = [CpuNodeConfig; COUNT_OF_PARTIES];

/// Placeholder config type — to be replaced with the real per-party config.
// TODO: derive from iris_mpc_common::config::Config or a dedicated test config struct.
#[derive(Debug, Clone)]
pub struct CpuNodeConfig {
    /// PostgreSQL connection URL for this party's CPU database.
    pub db_url: String,
    /// Schema name for this party (e.g. "party_0", "party_1", "party_2").
    pub db_schema: String,
    /// S3 bucket name for graph checkpoints.
    pub checkpoint_bucket: String,
    /// Party index (0, 1, 2).
    pub party_id: usize,
    /// Address this party's coordination server listens on.
    // TODO: confirm port/path convention for the ready endpoint (open question #1).
    pub coordination_addr: String,
    /// Sidecar-specific settings.
    pub sidecar: SidecarTestConfig,
}

/// Sidecar settings used in tests — kept separate so they can be overridden easily.
#[derive(Debug, Clone)]
pub struct SidecarTestConfig {
    pub cycle_interval_secs: u64,
    pub retry_interval_secs: u64,
    pub peer_round_timeout_secs: u64,
    /// Override to 1 in most tests to avoid needing many seeded WAL rows
    /// (see open question #7 in readme).
    pub min_mutations_per_cycle: u64,
    pub checkpoint_window: usize,
    pub is_archival: bool,
    // TODO: map to the real PruningMode enum from checkpoint_protocol
    pub pruning_mode: Option<String>,
}
