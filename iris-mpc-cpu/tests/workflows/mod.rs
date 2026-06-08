pub mod wal_100;
pub mod wal_102;
pub mod wal_104;
pub mod wal_105;
pub mod wal_106;
pub mod wal_109;
pub mod wal_110;

/// Spawn `server_main` (hawk_main) for all 3 parties concurrently.
///
/// `server_main` holds `!Send` state, so each party runs in its own OS thread
/// via `spawn_blocking` with a dedicated multi-thread runtime.  A shared
/// `Arc<Notify>` bridges the `CancellationToken` to the blocking threads.
#[macro_export]
macro_rules! run_hawk {
    ($configs:expr, $shutdown:expr, $ctx:expr) => {{
        use iris_mpc::server::server_main;
        use std::sync::Arc;
        use tokio::sync::Notify;
        use tracing::{info_span, Instrument};

        let mut join_set: tokio::task::JoinSet<eyre::Result<()>> = tokio::task::JoinSet::new();

        // Arc<Notify> is used instead of CancellationToken because the latter
        // is not Send across runtimes.
        let notify = Arc::new(Notify::new());

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

        for (party_idx, cpu_cfg) in ($configs).iter().enumerate() {
            let config = $crate::utils::configs::make_hawk_config(cpu_cfg, &$configs, &$ctx.env);
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
    }};
}

/// Spawn `sidecar_main` for all 3 parties concurrently in one-shot mode.
///
/// Uses `CpuNodeConfig.sidecar_port` so hawk_main and sidecar_main can
/// co-exist in the same test process without port conflicts.
#[macro_export]
macro_rules! run_sidecar {
    ($configs:expr, $shutdown:expr, $ctx:expr) => {{
        let mut join_set: tokio::task::JoinSet<eyre::Result<()>> = tokio::task::JoinSet::new();
        let sidecar_addresses: Vec<String> = ($configs)
            .iter()
            .map(|c| format!("127.0.0.1:{}", c.sidecar_port))
            .collect();

        for (party_idx, config) in ($configs).iter().enumerate() {
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
                use tracing::{info_span, Instrument};

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

                let span = info_span!("mpc_node", idx = party_idx);
                tokio::select! {
                    res = sidecar_main(cfg, &graph_store, &s3_client, &mut networking, shutdown).instrument(span) => res,
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
