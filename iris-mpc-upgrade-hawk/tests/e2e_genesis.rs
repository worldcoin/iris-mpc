use eyre::Result;
use serial_test::serial;
use tracing_test::traced_test;
use utils::{TestRun, TestRunContextInfo};

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
#[serial]
#[ignore = "requires external setup"]
async fn test_hnsw_genesis_100() -> Result<()> {
    use workflows::genesis_100::Test;

    let ctx = TestRunContextInfo::new(100, 1);
    Test::new().run(ctx).await?;

    Ok(())
}

#[tokio::test]
#[serial]
#[traced_test]
async fn test_hnsw_genesis_101() -> Result<()> {
    use iris_mpc_cpu::genesis::plaintext::GenesisArgs;
    use workflows::genesis_101::Test;

    let ctx = TestRunContextInfo::new(101, 1);
    Test::new(
        GenesisArgs {
            max_indexation_id: 100,
            batch_size: 10,
            batch_size_error_rate: 256,
        },
        0_u64,
    )
    .run(ctx)
    .await?;

    Ok(())
}
