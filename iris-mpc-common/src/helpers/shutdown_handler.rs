use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::signal;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
pub struct ShutdownHandler {
    ct: CancellationToken,
    n_batches_pending_completion: Arc<AtomicUsize>,
    last_results_sync_timeout: Duration,
}

impl ShutdownHandler {
    pub fn new(shutdown_last_results_sync_timeout_secs: u64) -> Self {
        Self {
            ct: CancellationToken::new(),
            n_batches_pending_completion: Arc::new(AtomicUsize::new(0)),
            last_results_sync_timeout: Duration::from_secs(shutdown_last_results_sync_timeout_secs),
        }
    }

    pub fn is_shutting_down(&self) -> bool {
        self.ct.is_cancelled()
    }

    pub fn trigger_manual_shutdown(&self) {
        self.ct.cancel()
    }

    pub async fn wait_for_shutdown(&self) {
        self.ct.cancelled().await
    }

    pub fn get_cancellation_token(&self) -> CancellationToken {
        self.ct.clone()
    }

    pub async fn register_signal_handler(&self) {
        let ct = self.ct.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            ct.cancel();
            tracing::info!("Shutdown signal received.");
        });
    }

    pub fn increment_batches_pending_completion(&self) {
        tracing::debug!("Incrementing pending batches count");
        self.n_batches_pending_completion
            .fetch_add(1, Ordering::SeqCst);
    }

    pub fn decrement_batches_pending_completion(&self) {
        tracing::debug!("Decrementing pending batches count");
        self.n_batches_pending_completion
            .fetch_sub(1, Ordering::SeqCst);
    }

    pub async fn wait_for_pending_batches_completion(&self) {
        let check_interval = Duration::from_millis(100);
        let start = Instant::now();

        while self.n_batches_pending_completion.load(Ordering::SeqCst) > 0 {
            // Check if the timeout has been exceeded
            if start.elapsed() >= self.last_results_sync_timeout {
                tracing::error!("Timed out waiting for pending batches to complete.");
                return;
            }

            // Wait before checking again
            tokio::time::sleep(check_interval).await;
        }

        tracing::info!("Pending batches count reached zero.");
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
        tracing::info!("Ctrl+C received.");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
        tracing::info!("SIGTERM received.");
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_shutdown_handler() {
        let mut handler = ShutdownHandler::new(1);
        handler.last_results_sync_timeout /= 10; // Shorten timeout for test

        // Start work.
        assert!(!handler.is_shutting_down());
        handler.increment_batches_pending_completion();

        // Initiate a shutdown.
        handler.trigger_manual_shutdown();
        assert!(handler.is_shutting_down());

        // If batches do not complete, return anyway after timeout.
        handler.wait_for_pending_batches_completion().await;

        // Complete the batch.
        handler.decrement_batches_pending_completion();

        // Should return quickly since no batches are pending
        let quick = timeout(
            Duration::from_millis(10),
            handler.wait_for_pending_batches_completion(),
        );
        assert!(quick.await.is_ok());
    }
}
