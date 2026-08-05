use iris_mpc_cpu::graph_checkpoint::{PruningMode, TieredPruningConfig};

pub mod configs;
pub mod cpu_node;
pub mod hawk_fleet;
pub mod key_rotation;
pub mod runner;
pub mod wait_conditions;
pub mod wal_builder;

/// Number of MPC parties.
pub const COUNT_OF_PARTIES: usize = 3;

/// Stack size for every thread that may poll a `server_main` future.
///
/// `server_main` is a very large async fn — large enough that both it and this
/// test binary need `#![recursion_limit = "256"]` — and its future lives on the
/// stack of whichever thread polls it. Production polls it with
/// `runtime.block_on` on the **main** thread, which gets the 8 MB Linux default.
/// The harness instead polls it on a tokio **blocking-pool** thread
/// (`spawn_blocking` → inner `rt.block_on`), and those default to 2 MB
/// (`thread_stack_size`, honoured by the blocking pool as well as by workers).
///
/// That 4x gap means the harness overflows on futures production handles
/// comfortably: adding a handful of locals to `server_main` was enough to abort
/// a worker with "has overflowed its stack" partway through converge. Give the
/// harness more headroom than production rather than budgeting to the byte.
pub const TEST_THREAD_STACK_SIZE: usize = 16 * 1024 * 1024;

pub const MIN_MUTATIONS_PER_SIDECAR_CYCLE: usize = 5;

/// Per-party configuration array.
pub type CpuConfigs = [CpuNodeConfig; COUNT_OF_PARTIES];

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
    /// Healthcheck port for this party's HTTP health endpoint.
    /// Used to populate `ServerCoordinationConfig::healthcheck_ports`.
    pub healthcheck_port: u16,
    /// used for the MPC
    pub service_port: u16,
    /// used by the networking for the sidecar
    pub sidecar_port: u16,
    /// Sidecar-specific settings — can be overridden per test.
    pub sidecar: SidecarTestConfig,
}

/// Sidecar settings kept separate so individual tests can override them.
#[derive(Debug, Clone)]
pub struct SidecarTestConfig {
    pub cycle_interval_secs: u64,
    pub retry_interval_secs: u64,
    pub peer_round_timeout_secs: u64,
    pub make_connections_timeout_secs: u64,
    /// Guard: sidecar will not checkpoint if fewer than this many new WAL rows exist.
    /// Set to 5 by default; tests must seed at least this many mutations.
    pub min_mutations_per_cycle: u64,
    pub checkpoint_window: usize,
    pub is_archival: bool,
    pub pruning_mode: PruningMode,
    pub tiered_pruning: TieredPruningConfig,
}

impl Default for SidecarTestConfig {
    fn default() -> Self {
        Self {
            cycle_interval_secs: 1,
            retry_interval_secs: 1,
            peer_round_timeout_secs: 30,
            make_connections_timeout_secs: 300,
            min_mutations_per_cycle: MIN_MUTATIONS_PER_SIDECAR_CYCLE as _,
            checkpoint_window: 10,
            is_archival: false,
            pruning_mode: PruningMode::None,
            tiered_pruning: TieredPruningConfig::default(),
        }
    }
}
