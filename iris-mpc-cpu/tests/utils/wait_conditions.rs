use std::time::Duration;

use super::{cpu_node::CpuNodes, CpuConfigs};

/// TC-1 — Poll each party's coordination server ready endpoint until all 3
/// respond with a success status, or until `timeout` is exceeded.
///
/// # Open question #1
///
/// The exact URL path and port for the ready endpoint need to be confirmed.
/// The coordination server exposes readiness via `set_node_ready()` /
/// `wait_for_others_ready()`.  The test needs to either:
///   (a) poll an HTTP endpoint, e.g. GET http://{coordination_addr}/ready → 200
///   (b) use the coordination client type directly from within the test process
///
/// # Open question #2
///
/// Does `hawk_main` require the full 3-party MPC network to be up before it
/// signals ready (even for pure roll-forward startup)?  If yes, TC-1 tests need
/// the same loopback network setup as TC-2 sidecar tests.
pub async fn wait_for_all_ready(
    _configs: &CpuConfigs,
    _timeout: Duration,
) -> eyre::Result<()> {
    // TODO:
    //   for each party config (in parallel):
    //     loop until GET {config.coordination_addr}/ready returns 200
    //     or until timeout elapses (return Err on timeout)
    todo!("poll ready endpoint for all 3 parties")
}

/// TC-2 — Poll the DB checkpoint table until each party's checkpoint row count
/// exceeds `baseline_count`, then verify that each party's latest checkpoint S3
/// object exists.
///
/// Returns the new `GraphCheckpointRow` for each party once all 3 have advanced.
///
/// `baseline_count` should be recorded by calling
/// `CpuNodes::assert_checkpoint_count` before starting the sidecar.
pub async fn wait_for_new_checkpoint(
    _nodes: &CpuNodes,
    _configs: &CpuConfigs,
    _baseline_count: usize,
    _timeout: Duration,
) -> eyre::Result<[(); 3]> /* TODO: -> [GraphCheckpointRow; 3] */ {
    // TODO:
    //   loop (with sleep between iterations) until timeout:
    //     for each party:
    //       rows = graph.recent_checkpoints(large_window)
    //       if rows.len() > baseline_count: record new row
    //     if all 3 parties have a new row: break
    //   verify S3 object exists for each party's new checkpoint:
    //     s3_client.head_object(bucket, row.s3_key) -> Ok(())
    todo!("poll checkpoint table and verify S3 objects for all 3 parties")
}

/// Convenience: count checkpoint rows for one party's graph store.
/// Used to establish a baseline before starting the sidecar.
pub async fn count_checkpoints(_graph: &()) /* TODO: &GraphPg<PlaintextStore> */ -> eyre::Result<usize> {
    // TODO: graph.recent_checkpoints(usize::MAX).await?.len()
    todo!("count checkpoint rows")
}
