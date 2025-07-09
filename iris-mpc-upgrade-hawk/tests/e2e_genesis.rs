use eyre::Result;
use utils::runner::{TestRun, TestRunInfo};

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
async fn test_hnsw_genesis_100() -> Result<()> {
    use workflows::genesis_100::Test;

    Test::new().run(TestRunInfo::new(100)).await?;

    Ok(())
}
