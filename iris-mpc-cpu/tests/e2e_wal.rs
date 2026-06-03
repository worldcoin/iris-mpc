// Integration tests for the iris-mpc-cpu WAL pipeline.
//
// See tests/e2e_wal_readme.md for full design documentation.
//
// Run with:
//   cargo test --test e2e_wal -- --nocapture
//
// Requires:
//   - PostgreSQL running (via docker-compose) with per-party schemas
//   - LocalStack at http://localhost:4566 for S3 and Secrets Manager

mod utils;
mod workflows;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    LazyLock,
};

use crate::utils::runner::TestRun;
use eyre::bail;
use serial_test::serial;
use workflows::{
    wal_100::Wal100, wal_101::Wal101, wal_102::Wal102, wal_103::Wal103, wal_104::Wal104,
};

const RUST_LOG: &str = "info";

/// Prevents later tests from running once any single test has failed.
/// Mirrors the pattern in e2e_genesis.rs.
static TEST_FAILED: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));

/// Tracks whether the one-time global setup (localstack wait + key rotation)
/// has already completed.  Reset is not needed — these tests are always run
/// in a fresh process.
static GLOBAL_SETUP_DONE: LazyLock<AtomicBool> = LazyLock::new(|| AtomicBool::new(false));

/// Instantiate a test, build a tokio runtime, run all lifecycle phases.
///
/// Before the first test body executes the macro runs [`global_setup`] once:
///   1. Wait for LocalStack (and its init-script) to become ready.
///   2. Rotate ECDH keys twice for each of the three MPC parties.
///
/// Ctrl+C aborts cleanly.  On failure, sets TEST_FAILED so remaining tests skip.
macro_rules! run_test {
    ($kind:expr, $idx:expr, $test:expr) => {{
        tracing_subscriber::fmt()
            .with_env_filter(format!(
                "iris_mpc_cpu={RUST_LOG},iris_mpc_common={RUST_LOG},warn"
            ))
            .try_init()
            .ok();

        if TEST_FAILED.load(Ordering::SeqCst) {
            bail!("A previous test has failed, aborting further tests.");
        }

        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            let ctx = utils::runner::CpuTestContext::new($kind, $idx);

            // One-time global setup: runs only for the first test in the suite.
            // Tests are serial so there is no concurrent access concern here.
            if !GLOBAL_SETUP_DONE.load(Ordering::SeqCst) {
                match utils::key_rotation::global_setup(ctx.env.s3_endpoint()).await {
                    Ok(()) => GLOBAL_SETUP_DONE.store(true, Ordering::SeqCst),
                    Err(e) => {
                        TEST_FAILED.store(true, Ordering::SeqCst);
                        return Err(e.wrap_err("global setup failed"));
                    }
                }
            }

            let mut test = $test;

            let r = tokio::select! {
                res = test.run(&ctx) => res,
                _ = tokio::signal::ctrl_c() => Err(eyre::eyre!("Test aborted by Ctrl+C")),
            };

            if r.is_err() {
                TEST_FAILED.store(true, Ordering::SeqCst);
            }
            r
        })
    }};
}

// ---------------------------------------------------------------------------
// Test functions — one per scenario, run serially to avoid port/DB conflicts.
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn test_wal_100() -> eyre::Result<()> {
    run_test!(100, 1, Wal100::new())
}

#[test]
#[serial]
fn test_wal_101() -> eyre::Result<()> {
    run_test!(101, 1, Wal101::new())
}

#[test]
#[serial]
fn test_wal_102() -> eyre::Result<()> {
    run_test!(102, 1, Wal102::new())
}

#[test]
#[serial]
fn test_wal_103() -> eyre::Result<()> {
    run_test!(103, 1, Wal103::new())
}

#[test]
#[serial]
fn test_wal_104() -> eyre::Result<()> {
    run_test!(104, 1, Wal104::new())
}
