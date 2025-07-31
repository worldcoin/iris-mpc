use eyre::Result;
use utils::{constants::SECRET_SHARING_PG_TX_SIZE, TestRun, TestRunContextInfo};

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
    use workflows::genesis_100::{Params, Test};

    let ctx = TestRunContextInfo::new(100, 1);
    let params = Params::new(10, 256, 100, false, SECRET_SHARING_PG_TX_SIZE, false);
    let mut test = Test::new(params);

    test.run(ctx).await?;

    Ok(())
}
