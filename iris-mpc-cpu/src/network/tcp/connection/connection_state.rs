use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

// want every connection to be able to do the following:
// log connection errors if it is the first one to occur since all connections were successfully established
// notify that an error occurred via a cancellation token (err_ct)
// terminate in case of shutdown (shutdown_ct) and log a single message upon shutdown (exited)
// allow one to wait for a connection to be re-established
#[derive(Clone)]
pub struct ConnectionState {
    inner: Arc<RwLock<ConnectionStateInner>>,
}

impl ConnectionState {
    pub fn new(shutdown_ct: CancellationToken, err_ct: CancellationToken) -> Self {
        let inner = Arc::new(RwLock::new(ConnectionStateInner::new(shutdown_ct, err_ct)));
        Self { inner }
    }

    pub async fn replace_cancellation_token(&self, err_ct: CancellationToken) {
        self.inner.write().await.replace_cancellation_token(err_ct);
    }

    pub async fn set_exited(&self) -> bool {
        self.inner
            .read()
            .await
            .exited
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub async fn set_cancelled(&self) -> bool {
        self.inner
            .read()
            .await
            .cancelled
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub async fn shutdown_ct(&self) -> CancellationToken {
        self.inner.read().await.shutdown_ct.clone()
    }

    pub async fn err_ct(&self) -> CancellationToken {
        self.inner.read().await.err_ct.clone()
    }
}

struct ConnectionStateInner {
    exited: AtomicBool,
    cancelled: AtomicBool,
    shutdown_ct: CancellationToken,
    err_ct: CancellationToken,
}

impl ConnectionStateInner {
    fn new(shutdown_ct: CancellationToken, err_ct: CancellationToken) -> Self {
        Self {
            exited: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            shutdown_ct,
            err_ct,
        }
    }

    fn replace_cancellation_token(&mut self, err_ct: CancellationToken) {
        self.cancelled = AtomicBool::new(false);
        self.err_ct = err_ct;
    }
}
