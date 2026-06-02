pub mod wal_100;
pub mod wal_101;
pub mod wal_102;
pub mod wal_103;
pub mod wal_104;

pub use crate::utils::runner::TestRun;

use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::utils::{CpuConfigs, HAWK_ADDRS, SIDECAR_ADDRS};

// ---------------------------------------------------------------------------
// Service runner macros
// ---------------------------------------------------------------------------
//
// Both hawk_main and sidecar_main are daemon loops.  Tests start them, wait
// for a termination condition (TC-1 or TC-2), then cancel via a shared
// CancellationToken.
//
// Both services can be called inline (no subprocess needed; no global
// side-effects).  A JoinSet is used so the wait_for_all_ready (TC-1) helper
// can monitor for unexpected early task exit.
//
// hawk_main and sidecar_main each require a loopback MPC network.  They use
// different port sets (HAWK_ADDRS vs SIDECAR_ADDRS) so both can run in the
// same process during wal_103 without bind conflicts.
//
// Open questions still blocking full macro bodies:
//   #1: hawk_main callable signature (function accepting CancellationToken)
//   #2: build_hawk_network_handle exact import path and signature
// ---------------------------------------------------------------------------

/// Spawn `hawk_main` for all 3 parties concurrently.
///
/// Each party gets its own loopback network handle built from `HAWK_ADDRS`.
/// All parties share the provided `CancellationToken` for clean shutdown.
///
/// Returns a `JoinSet<eyre::Result<()>>` that can be polled in TC-1 to detect
/// unexpected early exit.
///
/// Usage:
/// ```rust
/// let shutdown = CancellationToken::new();
/// let mut hawk_set = run_hawk!(configs, shutdown.clone());
/// wait_for_all_ready(&configs, &mut hawk_set, Duration::from_secs(60)).await?;
/// stop_and_join!(shutdown, hawk_set);
/// ```
#[macro_export]
macro_rules! run_hawk {
    ($configs:expr, $shutdown:expr) => {{
        let mut join_set: tokio::task::JoinSet<eyre::Result<()>> =
            tokio::task::JoinSet::new();
        let hawk_addresses: Vec<String> =
            crate::utils::HAWK_ADDRS.iter().map(|s| s.to_string()).collect();

        for config in ($configs).iter() {
            let config = config.clone();
            let shutdown = $shutdown.clone();
            let addresses = hawk_addresses.clone();

            join_set.spawn(async move {
                // TODO (open question #2): build network handle.
                // let networking = build_hawk_network_handle(
                //     config.party_id,
                //     &addresses,
                //     &addresses,   // outbound_addrs same as inbound for loopback
                //     /* connection_parallelism */ 2,
                //     /* request_parallelism */ 2,
                //     &shutdown,
                // ).await?;

                // TODO (open question #1): call hawk_main with shutdown token.
                // let args = HawkArgs {
                //     party_index: config.party_id,
                //     addresses: addresses.clone(),
                //     outbound_addrs: addresses,
                //     ..HawkArgs::default_for_test()
                // };
                // iris_mpc_cpu::execution::hawk_main::exec(args, networking, shutdown).await

                let _ = (config, addresses, shutdown);
                todo!("spawn hawk_main for party — pending open questions #1 and #2")
            });
        }

        join_set
    }};
}

/// Spawn `sidecar_main` for all 3 parties concurrently.
///
/// Uses `SIDECAR_ADDRS` (different ports from `HAWK_ADDRS`) so hawk_main and
/// sidecar_main can co-exist in the same test process.
///
/// Returns a `JoinSet<eyre::Result<()>>`.
///
/// Usage:
/// ```rust
/// let shutdown = CancellationToken::new();
/// let mut sidecar_set = run_sidecar!(configs, shutdown.clone());
/// wait_for_new_checkpoint(&nodes, &configs, baseline, Duration::from_secs(120)).await?;
/// stop_and_join!(shutdown, sidecar_set);
/// ```
#[macro_export]
macro_rules! run_sidecar {
    ($configs:expr, $shutdown:expr) => {{
        let mut join_set: tokio::task::JoinSet<eyre::Result<()>> =
            tokio::task::JoinSet::new();
        let sidecar_addresses: Vec<String> =
            crate::utils::SIDECAR_ADDRS.iter().map(|s| s.to_string()).collect();

        for config in ($configs).iter() {
            let config = config.clone();
            let shutdown = $shutdown.clone();
            let addresses = sidecar_addresses.clone();

            join_set.spawn(async move {
                // TODO (open question #2): build network handle with SIDECAR_ADDRS.
                // let mut networking = build_hawk_network_handle(
                //     config.party_id, &addresses, &addresses, 2, 2, &shutdown,
                // ).await?;

                // TODO: construct GraphPg<Aby3Store> for the live service (not PlaintextStore).
                // The sidecar operates on the real graph store, not the test setup store.

                // Build SidecarConfig from test config.
                // let cfg = SidecarConfig {
                //     bucket: config.checkpoint_bucket.clone(),
                //     party_id: config.party_id,
                //     cycle_interval: Duration::from_secs(config.sidecar.cycle_interval_secs),
                //     retry_interval: Duration::from_secs(config.sidecar.retry_interval_secs),
                //     peer_round_timeout: Duration::from_secs(config.sidecar.peer_round_timeout_secs),
                //     min_mutations_per_cycle: config.sidecar.min_mutations_per_cycle,
                //     checkpoint_window: config.sidecar.checkpoint_window,
                //     is_archival: config.sidecar.is_archival,
                //     pruning_mode: config.sidecar.pruning_mode.into(),
                // };

                // TODO: build s3_client from config.checkpoint_bucket / env endpoint.

                // iris_mpc_cpu::checkpoint_protocol::runner::sidecar_main(
                //     cfg, &graph_store, &s3_client, &mut networking, shutdown,
                // ).await

                let _ = (config, addresses, shutdown);
                todo!("spawn sidecar_main for party — pending open questions #1 and #2")
            });
        }

        join_set
    }};
}

/// Cancel the shared `CancellationToken` and drain the `JoinSet`.
///
/// Propagates the first task error encountered.  Should be called even when
/// the test is about to fail so that background tasks are cleaned up.
#[macro_export]
macro_rules! stop_and_join {
    ($token:expr, $join_set:expr) => {{
        $token.cancel();
        while let Some(result) = $join_set.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::warn!("task error during shutdown: {e:#}"),
                Err(e) if e.is_cancelled() => {}
                Err(e) => tracing::warn!("task join error during shutdown: {e}"),
            }
        }
    }};
}
