pub mod wal_100;
#[allow(dead_code)]
pub mod wal_102;
pub mod wal_104;
pub mod wal_105;
pub mod wal_106;
pub mod wal_109;
pub mod wal_110;

use ampc_actor_utils::network::tcp::TlsConfig;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::{
    checkpoint_protocol::{sidecar_main, SidecarConfig},
    execution::hawk_main::{build_hawk_network_handle, HawkArgs},
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::graph::graph_store::GraphPg,
};
use std::time::Duration;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{info_span, Instrument};

use crate::utils::{runner::CpuTestContext, CpuNodeConfig};

/// Spawn `server_main` (hawk_main) for all 3 parties concurrently.
///
/// `server_main` holds `!Send` state, so each party runs in its own OS thread
/// via `spawn_blocking` with a dedicated multi-thread runtime.  A shared
/// `Arc<Notify>` bridges the `CancellationToken` to the blocking threads.
pub fn run_hawk(
    configs: &[CpuNodeConfig; 3],
    shutdown: CancellationToken,
    ctx: &CpuTestContext,
) -> JoinSet<eyre::Result<()>> {
    use iris_mpc::server::server_main;
    use std::sync::Arc;
    use tokio::sync::Notify;

    let mut join_set: JoinSet<eyre::Result<()>> = JoinSet::new();

    // Arc<Notify> is used instead of CancellationToken because the latter
    // is not Send across runtimes.
    let notify = Arc::new(Notify::new());

    {
        let notify = notify.clone();
        let abort = ctx.abort.clone();
        join_set.spawn(async move {
            tokio::select! {
                _ = shutdown.cancelled() => {},
                _ = abort.cancelled() => {},
            }
            notify.notify_waiters();
            Ok(())
        });
    }

    for (party_idx, cpu_cfg) in configs.iter().enumerate() {
        let config = crate::utils::configs::make_hawk_config(cpu_cfg, configs, &ctx.env);
        let notify = notify.clone();

        // server_main is !Send; block_on inside spawn_blocking avoids Send requirements.
        join_set.spawn(async move {
            tokio::task::spawn_blocking(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("failed to build server runtime");
                let span = info_span!("mpc_node", idx = party_idx);
                rt.block_on(async move {
                    tokio::select! {
                        res = server_main(config).instrument(span) => res,
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
}

/// Spawn `sidecar_main` for all 3 parties concurrently in one-shot mode.
///
/// Uses `CpuNodeConfig.sidecar_port` so hawk_main and sidecar_main can
/// co-exist in the same test process without port conflicts.
pub fn run_sidecar(
    configs: &[CpuNodeConfig],
    shutdown: CancellationToken,
    ctx: &CpuTestContext,
) -> JoinSet<eyre::Result<()>> {
    let s3_client = ctx.s3_client.clone();
    let abort = ctx.abort.clone();
    let mut join_set: JoinSet<eyre::Result<()>> = JoinSet::new();
    let sidecar_addresses: Vec<String> = configs
        .iter()
        .map(|c| format!("127.0.0.1:{}", c.sidecar_port))
        .collect();

    for (party_idx, config) in configs.iter().enumerate() {
        let config = config.clone();
        let shutdown = shutdown.clone();
        let abort = abort.clone();
        let addresses = sidecar_addresses.clone();
        let s3_client = s3_client.clone();

        join_set.spawn(async move {
            // Only networking fields are relevant; HNSW/persistence fields are unused.
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
            // Aby3Store type param is phantom for WAL/checkpoint ops; store-agnostic.
            let graph_store: GraphPg<Aby3Store> = GraphPg::new(&postgres).await?;

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

            let span = info_span!("mpc_node", idx = party_idx);
            tokio::select! {
                res = sidecar_main(cfg, &graph_store, &s3_client, &mut networking, shutdown).instrument(span) => res,
                _ = abort.cancelled() => Ok(()),
            }
        });
    }

    join_set
}

/// Cancel the shared `CancellationToken` and drain the `JoinSet`.
///
/// Returns the first task error encountered, or `Ok(())` if all tasks shut
/// down cleanly.  Always drains the full set regardless.  Should be called
/// even when the test is about to fail so that background tasks are cleaned up.
pub async fn stop_and_join(
    token: CancellationToken,
    join_set: &mut JoinSet<eyre::Result<()>>,
) -> eyre::Result<()> {
    token.cancel();
    let mut first_err: Option<eyre::Report> = None;
    while let Some(result) = join_set.join_next().await {
        let err = match result {
            Ok(Ok(())) => continue,
            Err(e) if e.is_cancelled() => continue,
            Ok(Err(e)) => e,
            Err(e) => eyre::eyre!("task join error: {e}"),
        };
        tracing::warn!("task error during shutdown: {err:#}");
        first_err.get_or_insert(err);
    }
    first_err.map_or(Ok(()), Err)
}

/// Wait for all sidecar tasks to complete with a 60-second timeout.
/// Cancels and drains remaining tasks on timeout, then returns an error.
pub async fn expect_sidecar_success(
    shutdown: tokio_util::sync::CancellationToken,
    mut join_set: tokio::task::JoinSet<eyre::Result<()>>,
) -> eyre::Result<()> {
    use std::time::Duration;

    let timeout_result = tokio::time::timeout(Duration::from_secs(60), async {
        while let Some(result) = join_set.join_next().await {
            result
                .map_err(|e| eyre::eyre!("sidecar task join error: {e}"))?
                .map_err(|e| eyre::eyre!("sidecar task error: {e:#}"))?;
        }
        tracing::info!("sidecar joined successfully");
        Ok::<(), eyre::Report>(())
    })
    .await;

    if timeout_result.is_err() {
        shutdown.cancel();
        while let Some(result) = join_set.join_next().await {
            match result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!("sidecar task error after timeout: {e:#}")
                }
                Err(e) if e.is_cancelled() => {}
                Err(e) => {
                    tracing::warn!("sidecar task join error after timeout: {e}")
                }
            }
        }
        return Err(eyre::eyre!("sidecar did not complete within 60 seconds"));
    }

    timeout_result?
}
