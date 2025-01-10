use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::signal;

#[derive(Clone, Debug)]
pub struct ShutdownHandler {
    shutdown_received:            Arc<AtomicBool>,
    n_batches_pending_completion: Arc<AtomicUsize>,
    last_results_sync_timeout:    Duration,
}

impl ShutdownHandler {
    pub fn new(shutdown_last_results_sync_timeout_secs: u64) -> Self {
        Self {
            shutdown_received:            Arc::new(AtomicBool::new(false)),
            n_batches_pending_completion: Arc::new(AtomicUsize::new(0)),
            last_results_sync_timeout:    Duration::from_secs(
                shutdown_last_results_sync_timeout_secs,
            ),
        }
    }

    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_received.load(Ordering::Relaxed)
    }

    pub fn trigger_manual_shutdown(&self) {
        self.shutdown_received.store(true, Ordering::Relaxed);
    }

    pub async fn wait_for_shutdown_signal(&self) {
        let shutdown_flag = self.shutdown_received.clone();
        tokio::spawn(async move {
            shutdown_signal().await;
            shutdown_flag.store(true, Ordering::Relaxed);
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
