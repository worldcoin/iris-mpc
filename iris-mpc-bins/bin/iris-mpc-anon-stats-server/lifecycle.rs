use ampc_anon_stats::CoordinationHandles;
use ampc_server_utils::{shutdown_handler::ShutdownHandler, TaskMonitor};
use eyre::{eyre, Result, WrapErr};
use std::{future::Future, time::Duration};
use tracing::{info, warn};

pub async fn startup<F, T>(
    shutdown: &ShutdownHandler,
    coordination: &CoordinationHandles,
    work: F,
) -> Result<Option<T>>
where
    F: Future<Output = Result<T>>,
{
    tokio::select! {
        biased;
        _ = shutdown.wait_for_shutdown() => {
            coordination.mark_shutting_down();
            info!("Shutdown interrupted anon stats startup");
            Ok(None)
        }
        result = work => result.map(Some),
    }
}

pub fn set_ready(shutdown: &ShutdownHandler, coordination: &CoordinationHandles) {
    if shutdown.is_shutting_down() {
        coordination.mark_shutting_down();
        return;
    }
    coordination.set_ready();
    // The shutdown signal can arrive between the check and the store.
    if shutdown.is_shutting_down() {
        coordination.mark_shutting_down();
    }
}

pub async fn iteration<F>(
    shutdown: &ShutdownHandler,
    coordination: &CoordinationHandles,
    drain_timeout: Duration,
    work: F,
) -> Result<()>
where
    F: Future<Output = Result<()>>,
{
    // Do not start new work if shutdown won a simultaneous ready/timer tick.
    if shutdown.is_shutting_down() {
        coordination.mark_shutting_down();
        return Ok(());
    }
    tokio::pin!(work);
    tokio::select! {
        biased;
        _ = shutdown.wait_for_shutdown() => {
            coordination.mark_shutting_down();
            info!(?drain_timeout, "Draining active anon stats iteration");
            // Publication and DB marking are not atomic. Keep the same future
            // and its network alive until completion or the explicit deadline.
            match tokio::time::timeout(drain_timeout, work.as_mut()).await {
                Ok(result) => result.wrap_err("anon stats iteration failed while draining"),
                Err(_) => {
                    warn!(?drain_timeout, "Anon stats drain timed out; in-flight publication or processed marking may be incomplete");
                    Err(eyre!("anon stats shutdown drain timed out after {drain_timeout:?}; publication or processed marking may be incomplete"))
                }
            }
        }
        result = work.as_mut() => result,
    }
}

pub fn check_background_tasks(tasks: &mut TaskMonitor) -> Result<()> {
    if let Some(result) = tasks.tasks.try_join_next() {
        return match result {
            Ok(Ok(())) => Err(eyre!("anon stats background task unexpectedly finished")),
            Ok(Err(err)) => Err(err.wrap_err("anon stats background task failed")),
            Err(err) => Err(eyre!("anon stats background task failed: {err}")),
        };
    }
    Ok(())
}

pub async fn cleanup(
    shutdown: &ShutdownHandler,
    coordination: &CoordinationHandles,
    tasks: &mut TaskMonitor,
    result: Result<()>,
) -> Result<()> {
    shutdown.trigger_manual_shutdown();
    coordination.mark_shutting_down();
    shutdown.get_network_cancellation_token().cancel();
    // TaskMonitor::abort_and_wait_for_finish panics before aborting if a task
    // already exited. Abort and join its set directly so errors still clean up.
    tasks.tasks.abort_all();
    let mut cleanup_error = None;
    while let Some(result) = tasks.tasks.join_next().await {
        let error = match result {
            Err(err) if err.is_cancelled() => None,
            Err(err) => Some(eyre!(
                "anon stats background task failed during cleanup: {err}"
            )),
            Ok(Err(err)) => Some(err.wrap_err("anon stats background task failed during cleanup")),
            Ok(Ok(())) => Some(eyre!("anon stats background task unexpectedly finished")),
        };
        if let Some(error) = error {
            warn!(?error, "Anon stats background cleanup failed");
            cleanup_error.get_or_insert(error);
        }
    }
    // Keep the original startup/iteration/drain failure after joining tasks.
    result.and(match cleanup_error {
        Some(error) => Err(error),
        None => Ok(()),
    })
}

#[cfg(test)]
mod tests;
