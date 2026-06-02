pub mod wal_100;
pub mod wal_101;
pub mod wal_102;
pub mod wal_103;
pub mod wal_104;

pub use crate::utils::runner::TestRun;

// ---------------------------------------------------------------------------
// Service runner macros
// ---------------------------------------------------------------------------
//
// These macros mirror `run_genesis!` and `join_runners!` from the genesis tests,
// but target the two daemon services used here.
//
// Unlike genesis::exec() which runs to completion, hawk_main and sidecar_main
// run indefinitely.  Tests start them, wait for a termination condition (TC-1 or
// TC-2), then cancel them via a shared CancellationToken.
//
// Open question #3: can hawk_main / sidecar_main be called inline as async
// functions from test code, or do they require subprocess spawning?
//
// ---------------------------------------------------------------------------

/// Spawn `hawk_main` for all 3 parties concurrently.
///
/// Returns `(CancellationToken, Vec<JoinHandle<eyre::Result<()>>>)`.
///
/// Usage:
/// ```rust
/// let (shutdown, handles) = run_hawk!(ctx.configs);
/// wait_for_all_ready(&ctx.configs, Duration::from_secs(30)).await?;
/// stop_and_join!(shutdown, handles).await?;
/// ```
#[macro_export]
macro_rules! run_hawk {
    ($configs:expr) => {{
        // TODO:
        //   1. create a shared CancellationToken
        //   2. for each party config: tokio::spawn(hawk_main(HawkArgs::from(&config), ct.clone()))
        //      — open question #2: does this need a loopback MPC network?
        //   3. return (ct, handles)
        todo!("spawn hawk_main for all 3 parties")
    }};
}

/// Spawn `sidecar_main` for all 3 parties concurrently.
///
/// Requires a pre-built loopback MPC network handle for the 3-party hash
/// consensus step (see open question #5 in readme).
///
/// Returns `(CancellationToken, Vec<JoinHandle<eyre::Result<()>>>)`.
///
/// Usage:
/// ```rust
/// let (shutdown, handles) = run_sidecar!(ctx.configs, nodes);
/// let new_checkpoints = wait_for_new_checkpoint(&nodes, &ctx.configs, baseline, timeout).await?;
/// stop_and_join!(shutdown, handles).await?;
/// ```
#[macro_export]
macro_rules! run_sidecar {
    ($configs:expr, $nodes:expr) => {{
        // TODO:
        //   1. build loopback networking handles (reuse pattern from tests/e2e.rs)
        //      — open question #5: confirm sidecar networking setup
        //   2. create a shared CancellationToken
        //   3. for each party: tokio::spawn(sidecar_main(SidecarConfig::from(&config), ..., ct.clone()))
        //   4. return (ct, handles)
        todo!("spawn sidecar_main for all 3 parties")
    }};
}

/// Cancel the shared shutdown token and join all handles.
///
/// Propagates any task errors.
#[macro_export]
macro_rules! stop_and_join {
    ($token:expr, $handles:expr) => {{
        $token.cancel();
        for handle in $handles {
            handle.await??;
        }
    }};
}
