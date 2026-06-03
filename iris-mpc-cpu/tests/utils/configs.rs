//! Hardcoded per-party [`CpuNodeConfig`] values for the WAL workflow tests.
//!
//! Rather than loading TOML files (the genesis approach), all three configs are
//! defined here inline.  The only variable is the DB host, which differs
//! between the `Local` (laptop) and `Docker` (CI) [`TestEnvironment`].
//!
//! # Port allocations (must not conflict with service_port 19000–19002 or sidecar_port 20000–20002)
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
    [
        make_config(0, db_host),
        make_config(1, db_host),
        make_config(2, db_host),
    ]
}

// ---------------------------------------------------------------------------
// hawk_main config builder (TC-1)
// ---------------------------------------------------------------------------

/// Build an [`iris_mpc_common::config::Config`] for `server_main` from a
/// `CpuNodeConfig`.  All values are hardcoded inline — no TOML files, no
/// env-var loading.  AWS endpoints point at LocalStack.
///
/// `service_ports` and `service_outbound_ports` should be pre-allocated free
/// ports (see `run_hawk!`).  The schema values are chosen so that
/// `Config::get_cpu_db_schema()` returns the same string as
/// `cpu_cfg.db_schema`, ensuring `server_main` and the test's `DbStores`
/// share the same Postgres schema.
pub fn make_hawk_config(
    cpu_cfg: &CpuNodeConfig,
    all_configs: &CpuConfigs,
    env: &TestEnvironment,
) -> iris_mpc_common::config::Config {
    use ampc_server_utils::{AwsConfig, ServerCoordinationConfig};
    use iris_mpc_common::config::{Config, DbConfig};

    let db = DbConfig {
        url: cpu_cfg.db_url.clone(),
        migrate: true,
        create: true,
        load_parallelism: 8,
    };
    let healthcheck_ports: Vec<String> = all_configs
        .iter()
        .map(|c| c.healthcheck_port.to_string())
        .collect();

    let service_ports: Vec<String> = all_configs
        .iter()
        .map(|c| c.service_port.to_string())
        .collect();

    let service_outbound_ports = service_ports.clone();

    Config {
        party_id: cpu_cfg.party_id,
        environment: "dev".to_string(),

        // CPU database; assign to `database` as well so server_main's
        // iris_store (GPU) and cpu_store share the same schema — same
        // pattern as e2e_hawk.rs (`config.database = config.cpu_database.clone()`).
        database: Some(db.clone()),
        cpu_database: Some(db),

        // Schema names.  prepare_stores() builds the DB schema as:
        //   format!("{}{}_{}_{}", schema_name, hnsw_schema_name_suffix, environment, party_id)
        // hnsw_schema_name_suffix and gpu_schema_name_suffix must be identical
        // across all parties because they flow into CommonConfig and are
        // compared in the cross-party consistency check.  We use empty suffixes
        // and embed the common prefix in schema_name so the formula produces
        // "cpu_party_dev_{N}", which matches CpuNodeConfig::db_schema.
        schema_name: "cpu_party".to_string(),
        hnsw_schema_name_suffix: String::new(),
        gpu_schema_name_suffix: String::new(),

        // Checkpoint bucket (already created by init-localstack.sh).
        graph_checkpoint_bucket_name: cpu_cfg.checkpoint_bucket.clone(),

        // Coordination server — healthcheck ports drive TC-1 wait.
        server_coordination: Some(ServerCoordinationConfig {
            party_id: cpu_cfg.party_id,
            node_hostnames: vec!["127.0.0.1".to_string(); 3],
            healthcheck_ports,
            image_name: String::new(),
            heartbeat_interval_secs: 2,
            heartbeat_initial_retries: 10,
            http_query_retry_delay_ms: 1000,
            startup_sync_timeout_secs: 300,
        }),
        service_ports,
        service_outbound_ports,

        // AWS — LocalStack endpoint + resources from init-localstack.sh.
        aws: Some(AwsConfig {
            endpoint: Some(env.s3_endpoint().to_string()),
            region: Some("us-east-1".to_string()),
        }),
        requests_queue_url: format!(
            "http://sqs.us-east-1.localhost.localstack.cloud:4566/000000000000/smpcv2-{}-dev.fifo",
            cpu_cfg.party_id
        ),
        results_topic_arn: "arn:aws:sns:us-east-1:000000000000:iris-mpc-results.fifo".to_string(),
        shares_bucket_name: "wf-smpcv2-dev-sns-requests".to_string(),
        kms_key_arns: concat!(
            r#"["arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000000","#,
            r#""arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000001","#,
            r#""arn:aws:kms:us-east-1:000000000000:key/00000000-0000-0000-0000-000000000002"]"#
        )
        .parse()
        .expect("kms_key_arns parse"),

        // WAL sync must be enabled so server_main calls sync_graph_mutations.
        enable_modifications_sync: true,
        enable_modifications_replay: false,

        // Minimal settings for a WAL-only test run.
        disable_persistence: true,
        hnsw_disable_memory_persistence: true,
        hawk_numa: false,
        hnsw_param_ef_constr: 320,
        hnsw_param_m: 256,
        hnsw_param_ef_search: 256,
        hnsw_param_ef_supermatch: 4000,

        // Everything else: serde defaults (empty queues, disabled features, etc.)
        ..serde_json::from_str("{}").expect("Config serde defaults")
    }
}

// ---------------------------------------------------------------------------
// Per-party builder
// ---------------------------------------------------------------------------

fn make_config(party_id: usize, db_host: &str) -> CpuNodeConfig {
    CpuNodeConfig {
        // All three parties share the same Postgres instance; schemas provide
        // isolation.  The `postgres` database always exists in the hawk-db
        // compose (`docker-compose.hawk-db.yaml`).
        db_url: format!("postgres://postgres:postgres@{db_host}:5432/postgres"),
        // Must match prepare_stores() formula: schema_name + suffix + "_" + env + "_" + party_id
        // = "cpu_party" + "" + "_dev_" + N  →  "cpu_party_dev_N"
        db_schema: format!("cpu_party_dev_{party_id}"),

        // S3 bucket created by `scripts/tools/init-localstack.sh`.
        checkpoint_bucket: "wf-smpcv2-dev-hnsw-checkpoint".to_string(),

        party_id,

        // Healthcheck port: used by TC-1 (`wait_for_all_ready`) and the
        // ServerCoordinationConfig built in `CpuNodes::new`.
        healthcheck_port: 18000 + party_id as u16,

        service_port: 19000 + party_id as u16,

        sidecar_port: 20000 + party_id as u16,

        sidecar: SidecarTestConfig::default(),
    }
}
