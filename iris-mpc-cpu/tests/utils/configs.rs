//! Hardcoded per-party [`CpuNodeConfig`] values for the WAL workflow tests.
//!
//! Rather than loading TOML files (the genesis approach), all three configs are
//! defined here inline.  The only variable is the DB host, which differs
//! between the `Local` (laptop) and `Docker` (CI) [`TestEnvironment`].
//!
//! # Port allocations (these must not conflict with HAWK_ADDRS / SIDECAR_ADDRS)
//!
//! | Party | coordination_port | healthcheck_port |
//! |-------|------------------|-----------------|
//! |   0   |       17000      |      18000      |
//! |   1   |       17001      |      18001      |
//! |   2   |       17002      |      18002      |

use super::{CpuConfigs, CpuNodeConfig, SidecarTestConfig};
use crate::utils::runner::TestEnvironment;

/// Return the three hardcoded [`CpuNodeConfig`]s for `env`.
pub fn hardcoded_configs(env: &TestEnvironment) -> CpuConfigs {
    let db_host = match env {
        TestEnvironment::Local => "localhost",
        TestEnvironment::Docker => "dev_db",
    };
    [party(0, db_host), party(1, db_host), party(2, db_host)]
}

// ---------------------------------------------------------------------------
// Per-party builder
// ---------------------------------------------------------------------------

fn party(party_id: usize, db_host: &str) -> CpuNodeConfig {
    CpuNodeConfig {
        // All three parties share the same Postgres instance; schemas provide
        // isolation.  The `postgres` database always exists in the hawk-db
        // compose (`docker-compose.hawk-db.yaml`).
        db_url: format!("postgres://postgres:postgres@{db_host}:5432/postgres"),
        db_schema: format!("cpu_party_{party_id}"),

        // S3 bucket created by `scripts/tools/init-localstack.sh`.
        checkpoint_bucket: "wf-smpcv2-dev-hnsw-checkpoint".to_string(),

        party_id,

        // Coordination port: reserved for hawk_main's ServerCoordinationConfig
        // once Q10/Q11 are resolved.  Must not overlap with HAWK_ADDRS or
        // SIDECAR_ADDRS (which use the 16 000 range).
        coordination_port: 17000 + party_id as u16,

        // Healthcheck port: used by TC-1 (`wait_for_all_ready`) and the
        // ServerCoordinationConfig built in `CpuNodes::new`.
        healthcheck_port: 18000 + party_id as u16,

        sidecar: SidecarTestConfig::default(),
    }
}
