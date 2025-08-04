use eyre::Result;
use utils::{TestRun, TestRunContextInfo};

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
    use utils::defaults;
    use workflows::genesis::shared::params::TestParams;
    use workflows::genesis::wf_100::Test;

    fn get_params() -> TestParams {
        // Node arguments ... common across all nodes.
        // TODO: consider moving to JSON files ... one per node.
        let batch_size = 0;
        let batch_size_error_rate = 256;
        let max_indexation_id = 100;
        let perform_db_snapshot = false;
        let use_db_backup_as_source = false;

        // Node config.
        // This value indicates the index of the node configuration to load into memory.
        let node_config_idx = 0;

        // System state setup parameters.
        let max_deletions = None;
        let max_modifications = None;
        let shares_generator_batch_size = defaults::SHARES_GENERATOR_BATCH_SIZE;
        let shares_generator_rng_state = defaults::SHARES_GENERATOR_RNG_STATE;
        let shares_pgres_tx_batch_size = defaults::SHARES_GENERATOR_PGRES_TX_BATCH_SIZE;

        TestParams::new(
            batch_size,
            batch_size_error_rate,
            max_deletions,
            max_indexation_id,
            max_modifications,
            perform_db_snapshot,
            use_db_backup_as_source,
            node_config_idx,
            shares_generator_batch_size,
            shares_generator_rng_state,
            shares_pgres_tx_batch_size,
        )
    }

    let ctx = TestRunContextInfo::new(100, 1);
    let params = get_params();
    Test::new(params).run(ctx).await?;

    Ok(())
}
