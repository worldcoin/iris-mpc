use std::time::Duration;

use eyre::bail;
use futures::future::try_join_all;
use tokio::task::JoinSet;
use tokio::time::{sleep, timeout};

use super::{cpu_node::CpuNodes, CpuConfigs};

// TODO (open question #5): confirm ampc_server_utils is a dev-dependency and that
// these imports are correct.
// use ampc_server_utils::{wait_for_others_ready, ServerCoordinationConfig};

// TODO (open question #9): confirm GraphCheckpointRow import path.
// use iris_mpc_cpu::hnsw::graph::graph_store::GraphCheckpointRow;

/// TC-1 — Wait for all 3 parties' coordination servers to signal ready.
///
/// Calls `wait_for_others_ready(&server_coord_config)` for each party in parallel
/// (via `try_join_all`), wrapped in a `tokio::select!` that also monitors the
/// `JoinSet` for any unexpected early task exit.
///
/// Pattern taken from `iris-mpc-upgrade-hawk/tests/e2e_hawk.rs`.
///
/// # Open questions
/// - #5: ampc_server_utils availability as dev-dependency
/// - #6: how to build a ServerCoordinationConfig from CpuNodeConfig.coordination_port
pub async fn wait_for_all_ready(
    configs: &CpuConfigs,
    join_set: &mut JoinSet<eyre::Result<()>>,
    dur: Duration,
) -> eyre::Result<()> {
    let ready_futures = configs.iter().map(|config| {
        let _port = config.coordination_port;
        async move {
            // TODO (open question #5 and #6): construct ServerCoordinationConfig from
            // config.coordination_port and all parties' coordination_ports, then call:
            //   wait_for_others_ready(&server_coord_config).await
            //
            // Placeholder — replace once open questions #5/#6 are resolved:
            let _ = _port;
            Ok::<(), eyre::Error>(())
        }
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
/// Polls every 500ms.  Returns the new `GraphCheckpointRow` for each party.
///
/// # Open question #9
/// The return type uses a placeholder `()` until `GraphCheckpointRow` import is confirmed.
pub async fn wait_for_new_checkpoint(
    nodes: &CpuNodes,
    configs: &CpuConfigs,
    baseline_count: usize,
    dur: Duration,
) -> eyre::Result<()> /* TODO: -> [GraphCheckpointRow; 3] */ {
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
        node.stores
            .verify_latest_checkpoint_s3_object(&config.checkpoint_bucket)
            .await?;
    }

    // TODO: return [GraphCheckpointRow; 3] once type import is resolved
    Ok(())
}

/// Count checkpoint rows for one party.  Used to establish a TC-2 baseline.
pub async fn count_checkpoints_for(node: &super::cpu_node::CpuNode) -> eyre::Result<usize> {
    node.stores.count_checkpoints().await
}
