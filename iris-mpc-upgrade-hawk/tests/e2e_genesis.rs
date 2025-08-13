use eyre::Result;
use serial_test::serial;
use utils::{TestRun, TestRunContextInfo};

mod utils;
mod workflows;

// the #[serial] macro doesn't work with #[tokio::test] but it works with #[test].
// rather than implement a static mutex, simply use another macro to launch a tokio runtime
// to run the genesis tests.
//
// as long as there is a struct called Test that can be created by ::new(), and implements TestRun,
// this macro will work.
macro_rules! run_test {
    ($count:expr, $idx:expr) => {{
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let ctx = TestRunContextInfo::new($count, $idx);
            Test::new().run(ctx).await
        })
    }};
}

/// HNSW-Genesis-100
///   against:
///     a known set of 100 Iris shares in plaintext format;
///     an empty set of exclusions;
///     an empty set of modifications;
///   asserts:
///     node processes exit normally;
///     graph construction is equivalent for each node;
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_100() -> Result<()> {
    use workflows::genesis_100::Test;
    run_test!(100, 1)?;
    Ok(())
}

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_101() -> Result<()> {
    use workflows::genesis_101::Test;
    run_test!(101, 1)?;
    Ok(())
}

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_102() -> Result<()> {
    use workflows::genesis_102::Test;
    run_test!(102, 1)?;
    Ok(())
}

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_103() -> Result<()> {
    use workflows::genesis_103::Test;
    run_test!(103, 1)?;
    Ok(())
}

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_104() -> Result<()> {
    use workflows::genesis_104::Test;
    run_test!(104, 1)?;
    Ok(())
}
