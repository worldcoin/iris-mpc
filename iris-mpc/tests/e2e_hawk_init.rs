#![recursion_limit = "256"]
// Integration tests for the iris-mpc-cpu WAL pipeline.
//
// Run with:
//   cargo test --test e2e_hawk_init -- --nocapture
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
    startup_120::Startup120, startup_121::Startup121, startup_122::Startup122, wal_104::Wal104,
    wal_105::Wal105, wal_106::Wal106, wal_109::Wal109, wal_110::Wal110, wal_111::Wal111,
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

        // Run the body on a thread we own. The party futures are `tokio::spawn`ed onto
        // the runtime's workers, but the test body — context construction plus the
        // composed setup/execute/teardown future — is polled by `block_on` on the
        // calling thread, and libtest picks that thread's stack size, not us. That
        // future is large enough to blow it.
        let body = std::thread::Builder::new()
            .name(format!("test-{}-{}", $kind, $idx))
            .stack_size(utils::TEST_THREAD_STACK_SIZE)
            .spawn(move || -> eyre::Result<()> {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .thread_stack_size(utils::TEST_THREAD_STACK_SIZE)
                    .build()?;
                rt.block_on(async {
                    let ctx = utils::runner::CpuTestContext::new($kind, $idx).await;

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
                    // shut down their services cleanly rather than being dropped
                    // mid-flight.
                    //
                    // Keeps listening rather than firing once.
                    {
                        let abort = ctx.abort.clone();
                        tokio::spawn(async move {
                            loop {
                                if tokio::signal::ctrl_c().await.is_err() {
                                    return;
                                }
                                if abort.is_cancelled() {
                                    tracing::error!("Ctrl+C again — exiting immediately");
                                    std::process::exit(130);
                                }
                                tracing::warn!(
                                    "Ctrl+C received — aborting test (press again to force exit)"
                                );
                                abort.cancel();
                            }
                        });
                    }

                    let mut test = $test;
                    let r = test.run(&ctx).await;

                    let aborted = ctx.abort.is_cancelled();
                    if r.is_err() || aborted {
                        TEST_FAILED.store(true, Ordering::SeqCst);
                    }
                    // A real phase error is the more informative one, so it wins; the
                    // abort only has to supply a failure when every phase happened to
                    // return `Ok`.
                    if aborted && r.is_ok() {
                        bail!("aborted by Ctrl+C");
                    }
                    r
                })
            })?;

        // Re-raise a panic as a panic so the harness still prints the assertion
        // message and location instead of an opaque join error.
        match body.join() {
            Ok(r) => r,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }};
}

// ---------------------------------------------------------------------------
// Test functions — one per scenario, run serially to avoid port/DB conflicts.
// ---------------------------------------------------------------------------

// #[test]
// #[serial]
// #[ignore = "requires external setup"]
// fn test_wal_102() -> eyre::Result<()> {
//     // TODO: make sidecar allow no checkpoint, or insert an empty graph as a checkpoint when this happens
//     run_test!(102, 1, Wal102::new())
// }

#[test]
#[serial]
#[ignore = "requires external setup"]
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
#[ignore = "requires external setup"]
fn test_wal_106() -> eyre::Result<()> {
    run_test!(106, 1, Wal106::new())
}

/// Modification-driven sync roll-forward: parties start with staggered
/// `persisted` states and WAL row counts; hawk_main syncs all parties to 10 rows.
/// Mirrors `test_hawk_init` from iris-mpc-upgrade-hawk/tests/e2e_hawk.rs.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_wal_109() -> eyre::Result<()> {
    run_test!(109, 1, Wal109::new())
}

/// Modification sync conflict: parties 1 and 2 hold different bytes for the
/// same modification_id; hawk_main must bail with the mismatch error.
/// Mirrors `test_hawk_sync_mutation_mismatch` from iris-mpc-upgrade-hawk/tests/e2e_hawk.rs.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_wal_110() -> eyre::Result<()> {
    run_test!(110, 1, Wal110::new())
}

/// Tiered pruning: a single sidecar cycle in `PruningMode::Tiered` prunes
/// checkpoints across the recent / sparse / ancient tiers,
/// preserving archival checkpoints and the agreed base.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_wal_111() -> eyre::Result<()> {
    run_test!(111, 1, Wal111::new())
}

// ---------------------------------------------------------------------------
// startup_120 – startup_122: single-party restart around the startup handshake,
// exercising the data-derived startup fleet sync-state digest and the boundary
// past which it no longer permits a rejoin.
// ---------------------------------------------------------------------------

/// Rejoin: one party restarts mid-handshake with unchanged data. It recomputes the
/// same fleet sync-state digest and rejoins; the other two neither restart nor advance past the
/// commit barrier while it is gone.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_startup_120() -> eyre::Result<()> {
    run_test!(120, 1, Startup120::new())
}

/// Mismatch: one party restarts mid-handshake with changed data, so it derives a
/// different fleet sync-state digest than its peers hold. No party may come up.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_startup_121() -> eyre::Result<()> {
    run_test!(121, 1, Startup121::new())
}

/// Too late: one party restarts after the fleet is serving, with unchanged data and
/// so an unchanged sync state. The rejoin is scoped to startup, so it must still be
/// refused and no party may be left serving.
#[test]
#[serial]
#[ignore = "requires external setup"]
fn test_startup_122() -> eyre::Result<()> {
    run_test!(122, 1, Startup122::new())
}
