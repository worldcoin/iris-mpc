use eyre::{bail, eyre, Result};
use serial_test::serial;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;
use utils::{TestRun, TestRunContextInfo};

mod utils;
mod workflows;

// when a test fails, the unit test hangs forever. this is a attempt to fix that
static TEST_FAILED: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));

const RUST_LOG: &str = "info";

// the #[serial] macro doesn't work with #[tokio::test] but it works with #[test].
// rather than implement a static mutex, simply use another macro to launch a tokio runtime
// to run the genesis tests.
//
// as long as there is a struct called Test that can be created by ::new(), and implements TestRun,
// this macro will work.
macro_rules! run_test {
    ($count:expr, $idx:expr) => {{
        // Initialize tracing to capture debug logs
        tracing_subscriber::fmt()
            .with_env_filter(format!("iris_mpc_cpu={RUST_LOG},iris_mpc_common={RUST_LOG},iris_mpc_upgrade_hawk={RUST_LOG}"))
            .try_init()
            .ok(); // ignore error if already initialized

        if TEST_FAILED.load(Ordering::SeqCst) {
            bail!("A previous test has failed, aborting further tests.");
        }
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let ctx = TestRunContextInfo::new($count, $idx);
            let mut test = Test::new();

            let shutdown = tokio::signal::ctrl_c();
            let r = tokio::select! {
                res = test.run(ctx) => {
                    res
                }
                _ = shutdown => {
                    Err(eyre!("Test aborted by Ctrl+C"))
                }
            };
            if r.is_err() {
                TEST_FAILED.store(true, Ordering::SeqCst);
            }
            r
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

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_105() -> Result<()> {
    use workflows::genesis_105::Test;
    run_test!(105, 1)?;
    Ok(())
}

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_106() -> Result<()> {
    use workflows::genesis_106::Test;
    run_test!(106, 1)?;
    Ok(())
}

/// HNSW-Genesis-200: Chaos test
///   Runs genesis repeatedly with random persistence delays per node
///   to verify sync_peers consensus handles async persistence correctly.
///   asserts:
///     all 3 nodes agree on state after each iteration;
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_hnsw_genesis_200() -> Result<()> {
    use workflows::genesis_200::Test;
    run_test!(200, 1)?;
    Ok(())
}
