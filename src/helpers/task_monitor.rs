//! Long-running async task monitoring.

use std::panic;

use tokio::task::{JoinError, JoinSet};

/// A long-running async task monitor which cancels all its tasks when dropped.
/// Designed for ongoing tasks which run until the program exits.
///
/// Create with `new()`, and monitor for panics or exits regularly with `check_tasks()`.
///
/// When exiting the program, `abort()`, wait, then check for hangs with `check_tasks_finished()`.
#[derive(Debug, Default)]
pub struct TaskMonitor {
    pub tasks: JoinSet<()>,
}

impl TaskMonitor {
    /// Create a new task monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Panics if any of the monitored tasks have finished normally, were cancelled, or panicked.
    /// This function panics even if a task finishes without an error.
    ///
    /// Call this method after adding each new task, and before starting a new batch or
    /// long-running operation.
    #[track_caller]
    pub fn check_tasks(&mut self) {
        // Any finished task is an error, so we just need to check for the first one.
        if let Some(finished_task) = self.tasks.try_join_next() {
            finished_task.expect("Monitored task was panicked or cancelled");
            panic!("Monitored task unexpectedly finished without an error");
        }
    }

    /// Aborts all tasks, but doesn't wait for them to finish.
    pub fn abort_all(&mut self) {
        self.tasks.abort_all();
    }

    /// Panics if any of the `server_tasks` have finished with a panic or hang.
    /// (Ignores tasks that have finished normally or were cancelled).
    ///
    /// When exiting the program, call `abort_all()`, wait for the tasks to finish,
    /// then call this function.
    #[track_caller]
    pub fn check_tasks_finished(&mut self) {
        // Any hung task is an error, so we need to check they've all finished.
        while let Some(finished_task) = self.tasks.try_join_next() {
            TaskMonitor::resume_panic(finished_task);
        }

        if !self.tasks.is_empty() {
            // If this panics, try waiting for longer between the abort and this function call.
            panic!("{} monitored tasks hung even when aborted", self.tasks.len());
        }
    }

    /// Panics if any of the `server_tasks` have finished with a panic.
    /// (Ignores tasks that have finished normally).
    ///
    /// When exiting the program, call `abort_all()`, wait for the tasks to finish,
    /// then call this function.
    ///
    /// This function can't detect hangs: it hangs if any task does not finish when aborted.
    pub async fn abort_and_wait_for_finish(&mut self) {
        self.abort_all();

        // Any hung task is an error, so we need to check they've all finished.
        while let Some(finished_task) = self.tasks.join_next().await {
            TaskMonitor::resume_panic(finished_task);
        }

        assert!(self.tasks.is_empty());
    }

    /// If `err` is a task panic, resume that panic.
    #[track_caller]
    pub fn resume_panic(err: Result<(), JoinError>) {
        if let Err(err) = err {
            if !err.is_cancelled() {
                panic::resume_unwind(err.into_panic());
            }
        }
    }
}
