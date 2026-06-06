#![recursion_limit = "256"]
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
    wal_102::Wal102, wal_103::Wal103, wal_104::Wal104, wal_105::Wal105, wal_106::Wal106,
    wal_107::Wal107, wal_109::Wal109, wal_110::Wal110,
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
                "iris_mpc={RUST_LOG},iris_mpc_cpu={RUST_LOG},iris_mpc_common={RUST_LOG},ampc_actor_utils={RUST_LOG},ampc_server_utils={RUST_LOG},{}={RUST_LOG}",
                env!("CARGO_CRATE_NAME")
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

            // Cancel ctx.abort on Ctrl+C so that run_hawk!/run_sidecar! can
            // shut down their services cleanly rather than being dropped mid-flight.
            {
                let abort = ctx.abort.clone();
                tokio::spawn(async move {
                    if tokio::signal::ctrl_c().await.is_ok() {
                        tracing::warn!("Ctrl+C received — aborting test");
                        abort.cancel();
                    }
                });
            }

            let mut test = $test;
            let r = test.run(&ctx).await;

            if r.is_err() && !ctx.abort.is_cancelled() {
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
#[ignore = "requires external setup"]
fn test_wal_102() -> eyre::Result<()> {
    run_test!(102, 1, Wal102::new())
}

#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_wal_103() -> eyre::Result<()> {
    run_test!(103, 1, Wal103::new())
}

#[test]
#[serial]
fn test_wal_104() -> eyre::Result<()> {
    run_test!(104, 1, Wal104::new())
}

// ---------------------------------------------------------------------------
// wal_105 – wal_107: extended scenarios (hawk + sidecar, require external
// setup identical to wal_103).
// ---------------------------------------------------------------------------

/// V4 graph load: hawk_main selects the sidecar checkpoint as its base and
/// rolls forward only the mutations that arrived after the checkpoint.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_wal_105() -> eyre::Result<()> {
    run_test!(105, 1, Wal105::new())
}

/// Checkpoint desync: sidecar completes a cycle and reaches BLAKE3 consensus
/// even when one party's checkpoint table has fallen behind the others.
#[test]
#[serial]
fn test_wal_106() -> eyre::Result<()> {
    run_test!(106, 1, Wal106::new())
}

/// Nontrivial modification sync: hawk_main reconciles a WAL divergence between
/// parties on startup; the subsequent sidecar checkpoint proves the graphs
/// are identical across parties.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_wal_107() -> eyre::Result<()> {
    run_test!(107, 1, Wal107::new())
}

// /// Modification-driven sync roll-forward: parties start with staggered
// /// `persisted` states and WAL row counts; hawk_main syncs all parties to 10 rows.
// /// Mirrors `test_hawk_init` from iris-mpc-upgrade-hawk/tests/e2e_hawk.rs.
// #[test]
// #[serial]
// #[ignore = "requires external setup"]
// fn test_wal_109() -> eyre::Result<()> {
//     run_test!(109, 1, Wal109::new())
// }

// /// Modification sync conflict: parties 1 and 2 hold different bytes for the
// /// same modification_id; hawk_main must bail with the mismatch error.
// /// Mirrors `test_hawk_sync_mutation_mismatch` from iris-mpc-upgrade-hawk/tests/e2e_hawk.rs.
// #[test]
// #[serial]
// #[ignore = "requires external setup"]
// fn test_wal_110() -> eyre::Result<()> {
//     run_test!(110, 1, Wal110::new())
// }
