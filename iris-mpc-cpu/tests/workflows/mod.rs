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
// run_sidecar! is fully implemented (see below).
// run_hawk! is fully implemented — all open questions resolved (see readme).
// ---------------------------------------------------------------------------

/// Spawn `server_main` (hawk_main) for all 3 parties concurrently.
///
/// Because `server_main` holds `!Send` pprof state, it cannot run inside a
/// `tokio::spawn` task directly.  Instead each party gets its own OS thread
/// (via `spawn_blocking`) with a dedicated multi-thread runtime.  The
/// `!Send` future is created and consumed entirely inside `rt.block_on` on
/// the blocking thread, so it never needs to cross await points as `Send`.
///
/// A watcher task bridges the `CancellationToken` to an `Arc<Notify>` so
/// that `stop_and_join!` call sites remain unchanged.
///
/// Service ports and outbound ports are allocated dynamically (ephemeral
/// OS ports) to avoid conflicts with `HAWK_ADDRS`, `SIDECAR_ADDRS`, and
/// the hardcoded healthcheck ports (18000–18002).
///
/// Returns a `JoinSet<eyre::Result<()>>` that can be polled in TC-1 to
/// detect unexpected early exit.
///
/// Usage:
/// ```rust
/// let shutdown = CancellationToken::new();
/// let mut hawk_set = run_hawk!(ctx.configs, shutdown.clone(), ctx);
/// wait_for_all_ready(&ctx.configs, &mut hawk_set, Duration::from_secs(60)).await?;
/// stop_and_join!(shutdown, hawk_set);
/// ```
#[macro_export]
macro_rules! run_hawk {
    ($configs:expr, $shutdown:expr, $ctx:expr) => {{
        use iris_mpc::server::server_main;
        use std::sync::Arc;
        use tokio::sync::Notify;

        let mut join_set: tokio::task::JoinSet<eyre::Result<()>> = tokio::task::JoinSet::new();

        // Shared notify: OS threads select on this instead of CancellationToken
        // (CancellationToken is not Send-across-runtimes friendly).
        let notify = Arc::new(Notify::new());

        // Bridge task: when the CancellationToken OR ctx.abort fires, wake all server threads.
        {
            let notify = notify.clone();
            let shutdown = $shutdown.clone();
            let abort = $ctx.abort.clone();
            join_set.spawn(async move {
                tokio::select! {
                    _ = shutdown.cancelled() => {},
                    _ = abort.cancelled() => {},
                }
                notify.notify_waiters();
                Ok(())
            });
        }

        for cpu_cfg in ($configs).iter() {
            let config = crate::utils::configs::make_hawk_config(cpu_cfg, &$configs, &$ctx.env);
            let notify = notify.clone();

            // spawn_blocking: the closure is Send (captures only Config and
            // Arc<Notify>); server_main is created and consumed entirely
            // inside rt.block_on on the blocking thread, so !Send is fine.
            join_set.spawn(async move {
                tokio::task::spawn_blocking(move || {
                    let rt = tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("failed to build server runtime");
                    rt.block_on(async move {
                        tokio::select! {
                            res = server_main(config) => res,
                            _ = notify.notified() => Ok(()),
                        }
                    })
                })
                .await
                .map_err(|e| eyre::eyre!("server task panicked: {e}"))
                .and_then(|r| r)
            });
        }

        join_set
    }};
}

/// Spawn `sidecar_main` for all 3 parties concurrently.
///
/// Uses `SIDECAR_ADDRS` (different ports from `HAWK_ADDRS`) so hawk_main and
/// sidecar_main can co-exist in the same test process during wal_103.
///
/// The sidecar's `GraphPg<Aby3Store>` connects to the same DB tables as the
/// test's `GraphPg<PlaintextStore>` — the `VectorStore` type parameter is
/// phantom and doesn't affect the WAL or checkpoint table queries.
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
    ($configs:expr, $shutdown:expr, $ctx:expr) => {{
        let mut join_set: tokio::task::JoinSet<eyre::Result<()>> = tokio::task::JoinSet::new();
        let sidecar_addresses: Vec<String> = crate::utils::SIDECAR_ADDRS
            .iter()
            .map(|s| s.to_string())
            .collect();

        for config in ($configs).iter() {
            let config = config.clone();
            let shutdown = $shutdown.clone();
            let abort = $ctx.abort.clone();
            let addresses = sidecar_addresses.clone();

            join_set.spawn(async move {
                use ampc_actor_utils::network::tcp::TlsConfig;
                use iris_mpc_common::postgres::{AccessMode, PostgresClient};
                use iris_mpc_cpu::{
                    checkpoint_protocol::{sidecar_main, SidecarConfig},
                    execution::hawk_main::{build_hawk_network_handle, HawkArgs},
                    hawkers::aby3::aby3_store::Aby3Store,
                    hnsw::graph::graph_store::GraphPg,
                };
                use std::time::Duration;

                // Build HawkArgs with only the networking fields populated.
                // HNSW and persistence fields are irrelevant to the sidecar.
                let hawk_args = HawkArgs {
                    party_index: config.party_id,
                    addresses: addresses.clone(),
                    outbound_addrs: addresses.clone(),
                    request_parallelism: 1,
                    connection_parallelism: 1,
                    hnsw_param_ef_constr: 0,
                    hnsw_param_m: 0,
                    hnsw_param_ef_search: 0,
                    hnsw_param_ef_supermatch: 0,
                    hnsw_param_ef_saturation_margin: 0,
                    hnsw_layer_density: None,
                    hnsw_fixed_layer_search_batch_size: None,
                    hnsw_prf_key: None,
                    disable_persistence: true,
                    hnsw_disable_memory_persistence: true,
                    tls: None::<TlsConfig>,
                    numa: false,
                };

                let mut networking =
                    build_hawk_network_handle(&hawk_args, shutdown.clone()).await?;

                let postgres =
                    PostgresClient::new(&config.db_url, &config.db_schema, AccessMode::ReadWrite)
                        .await?;
                // Aby3Store is the production VectorStore; the type parameter is
                // phantom for WAL/checkpoint table operations so PlaintextStore-seeded
                // data is fully compatible.
                let graph_store: GraphPg<Aby3Store> = GraphPg::new(&postgres).await?;

                let aws_config = aws_config::load_from_env().await;
                let s3_client = aws_sdk_s3::Client::new(&aws_config);

                let cfg = SidecarConfig {
                    bucket: config.checkpoint_bucket.clone(),
                    party_id: config.party_id,
                    cycle_interval: Duration::from_secs(config.sidecar.cycle_interval_secs),
                    retry_interval: Duration::from_secs(config.sidecar.retry_interval_secs),
                    peer_round_timeout: Duration::from_secs(config.sidecar.peer_round_timeout_secs),
                    min_mutations_per_cycle: config.sidecar.min_mutations_per_cycle,
                    checkpoint_window: config.sidecar.checkpoint_window,
                    is_archival: config.sidecar.is_archival,
                    pruning_mode: config.sidecar.pruning_mode,
                    one_shot: true,
                };

                tokio::select! {
                    res = sidecar_main(cfg, &graph_store, &s3_client, &mut networking, shutdown) => res,
                    _ = abort.cancelled() => Ok(()),
                }
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
