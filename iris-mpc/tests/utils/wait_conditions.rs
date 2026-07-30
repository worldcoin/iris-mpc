use std::time::Duration;

use ampc_server_utils::{
    try_get_endpoint_other_nodes, ReadyProbeResponse, ServerCoordinationConfig,
};
use eyre::{bail, WrapErr};
use futures::future::try_join_all;
use tokio::task::JoinSet;
use tokio::time::timeout;

use super::{CpuConfigs, COUNT_OF_PARTIES};

/// Poll every peer's `/health` until all of them report `is_ready`.
///
/// Deliberately does NOT use `ampc_server_utils::wait_for_others_ready`: that
/// function fails fast on any peer UUID outside the caller's startup-verified
/// set, which is correct for a cluster member but wrong for this harness — the
/// harness never runs the startup handshake, so it holds no verified set and
/// every peer would look like a restarted node.
async fn wait_until_peers_ready(coord: &ServerCoordinationConfig) -> eyre::Result<()> {
    let retry_delay = Duration::from_millis(coord.http_query_retry_delay_ms);

    loop {
        if let Ok(responses) = try_get_endpoint_other_nodes(coord, "health").await {
            let mut all_ready = true;
            for (_status, body) in responses {
                let probe: ReadyProbeResponse = serde_json::from_slice(&body)
                    .wrap_err("Failed to deserialize ReadyProbeResponse")?;
                if !probe.is_ready {
                    all_ready = false;
                }
            }

            if all_ready {
                return Ok(());
            }
        }

        tokio::time::sleep(retry_delay).await;
    }
}

/// Wait for all 3 parties' coordination servers to signal ready.
///
/// Polls each party's peer view in parallel (via `try_join_all`), wrapped in a
/// `tokio::select!` that also monitors the `JoinSet` for any unexpected early
/// task exit.
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
            http_query_timeout_ms: 10000,
            startup_sync_timeout_secs: 300,
            startup_visibility_barrier_disabled: false,
        };
        async move { wait_until_peers_ready(&coord).await }
    });

    let ready_all = try_join_all(ready_futures);

    tokio::select! {
        res = timeout(dur, ready_all) => {
            res.map_err(|_| eyre::eyre!("parties did not signal ready within {:?}", dur))??;
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

/// Wait for the first hawk_main task to exit with an error.
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
    .map_err(|_| eyre::eyre!("hawk_main did not produce an error within {:?}", dur))?
}
