use eyre::Result;
use utils::{defaults, TestRun, TestRunContextInfo};

mod resources;
mod system_state;
mod utils;
mod workflows;

/// HNSW-Genesis-100
///   against:
///     a known set of 100 Iris shares in plaintext format;
///     an empty set of exclusions;
///     an empty set of modifications;
///   asserts:
///     node processes exit normally;
///     graph construction is equivalent for each node;
#[tokio::test]
#[ignore = "requires external setup"]
async fn test_hnsw_genesis_100() -> Result<()> {
    use workflows::genesis::shared::params::TestParamsBuilder;
    use workflows::genesis::wf_100::Test;

    Test::new(
        TestParamsBuilder::new()
            .batch_size(0)
            .batch_size_error_rate(256)
            .max_deletions(0)
            .max_indexation_id(1000)
            .max_modifications(0)
            .node_config_idx(0)
            .use_db_backup_as_source(false)
            .perform_db_snapshot(false)
            .shares_generator_batch_size(defaults::SHARES_GENERATOR_BATCH_SIZE)
            .shares_generator_rng_state(defaults::SHARES_GENERATOR_RNG_STATE)
            .shares_pgres_tx_batch_size(defaults::SHARES_GENERATOR_PGRES_TX_BATCH_SIZE)
            .build(),
    )
    .run(TestRunContextInfo::new(100, 1))
    .await?;

    Ok(())
}
