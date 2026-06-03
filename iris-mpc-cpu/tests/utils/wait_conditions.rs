use std::time::Duration;

use ampc_server_utils::{wait_for_others_ready, ServerCoordinationConfig};
use eyre::bail;
use futures::future::try_join_all;
use tokio::task::JoinSet;
use tokio::time::{sleep, timeout};

use super::{cpu_node::CpuNodes, CpuConfigs, COUNT_OF_PARTIES};

/// TC-1 — Wait for all 3 parties' coordination servers to signal ready.
///
/// Calls `wait_for_others_ready(&server_coord_config)` for each party in parallel
/// (via `try_join_all`), wrapped in a `tokio::select!` that also monitors the
/// `JoinSet` for any unexpected early task exit.
///
/// Pattern taken from `iris-mpc-upgrade-hawk/tests/e2e_hawk.rs`.
pub async fn wait_for_all_ready(
    configs: &CpuConfigs,
    join_set: &mut JoinSet<eyre::Result<()>>,
    dur: Duration,
) -> eyre::Result<()> {
    // Build per-party ServerCoordinationConfig using the shared healthcheck ports
    // from all configs (they form the cross-party view each service needs).
    let healthcheck_ports: Vec<String> = configs
        .iter()
        .map(|c| c.healthcheck_port.to_string())
        .collect();
    let node_hostnames = vec!["127.0.0.1".to_string(); COUNT_OF_PARTIES];

    let ready_futures = configs.iter().map(|config| {
        let coord = ServerCoordinationConfig {
            party_id: config.party_id,
            node_hostnames: node_hostnames.clone(),
            healthcheck_ports: healthcheck_ports.clone(),
            image_name: String::new(),
            heartbeat_interval_secs: 2,
            heartbeat_initial_retries: 10,
            http_query_retry_delay_ms: 1000,
            startup_sync_timeout_secs: 300,
        };
        async move { wait_for_others_ready(&coord).await }
    });

    let ready_all = try_join_all(ready_futures);

    tokio::select! {
        res = timeout(dur, ready_all) => {
            res.map_err(|_| eyre::eyre!("TC-1 timeout: parties did not signal ready within {:?}", dur))??;
            Ok(())
        }
        Some(task_res) = join_set.join_next() => {
            bail!(
                "A hawk_main task exited unexpectedly before ready: {:?}",
                task_res
            )
        }
    }
}

/// TC-2 — Poll the `genesis_graph_checkpoint` table until each party's row count
/// exceeds `baseline_count`, then verify each party's latest checkpoint S3 object exists.
///
/// Polls every 500ms.
pub async fn wait_for_new_checkpoint(
    nodes: &CpuNodes,
    configs: &CpuConfigs,
    baseline_count: usize,
    dur: Duration,
) -> eyre::Result<()> {
    timeout(dur, async {
        loop {
            let counts = nodes.checkpoint_counts().await?;
            if counts.iter().all(|&c| c > baseline_count) {
                break;
            }
            sleep(Duration::from_millis(500)).await;
        }
        Ok::<(), eyre::Error>(())
    })
    .await
    .map_err(|_| {
        eyre::eyre!(
            "TC-2 timeout: sidecar did not produce a new checkpoint within {:?}",
            dur
        )
    })??;

    // Verify S3 objects exist for each party's latest checkpoint.
    for (node, config) in nodes.0.iter().zip(configs.iter()) {
        node.verify_latest_checkpoint_s3_object(&config.checkpoint_bucket)
            .await?;
    }

    Ok(())
}

/// TC-E — Wait for the first hawk_main task to exit with an error.
///
/// Returns the formatted error string (`format!("{err:#}")`) so the caller can
/// assert on the message without inspecting the raw `eyre::Report`.
///
/// Does **not** cancel the remaining tasks — the caller is responsible for
/// calling `stop_and_join!` afterwards.  Tasks that exit cleanly with `Ok(())`
/// (e.g. the bridge watcher task) are skipped; the loop continues until the
/// first erroring task is found.
///
/// Returns `Err` if the timeout fires before any task fails, or if a task
/// panics (i.e. `JoinError`).
pub async fn wait_for_hawk_failure(
    join_set: &mut JoinSet<eyre::Result<()>>,
    dur: Duration,
) -> eyre::Result<String> {
    timeout(dur, async {
        while let Some(result) = join_set.join_next().await {
            match result {
                // Watcher bridge task (or any server that happens to succeed before the
                // failing one) — skip and keep waiting.
                Ok(Ok(())) => {}
                // First task error — return its formatted message.
                Ok(Err(e)) => return Ok(format!("{e:#}")),
                // Task panicked.
                Err(e) => return Err(eyre::eyre!("hawk_main task panicked: {e}")),
            }
        }
        Err(eyre::eyre!(
            "all hawk_main tasks exited without an error (expected a failure)"
        ))
    })
    .await
    .map_err(|_| {
        eyre::eyre!(
            "TC-E timeout: hawk_main did not produce an error within {:?}",
            dur
        )
    })?
}
