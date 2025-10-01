use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::{mpsc, Mutex, RwLock};
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
    pub fn new(
        num_connections: usize,
        shutdown_ct: CancellationToken,
        err_ct: CancellationToken,
    ) -> Self {
        let inner = Arc::new(RwLock::new(ConnectionStateInner::new(
            num_connections,
            shutdown_ct,
            err_ct,
        )));
        Self { inner }
    }

    pub async fn replace_cancellation_token(&self, err_ct: CancellationToken) {
        self.inner.write().await.replace_cancellation_token(err_ct);
    }

    pub async fn wait_for_ready(&self) {
        let mut lock = self.inner.write().await;
        let (tx, mut rx) = mpsc::channel(1);
        if lock.is_ready().await {
            return;
        }
        lock.ready_tx.replace(tx);
        drop(lock);

        let _ = rx.recv().await;
    }

    pub async fn exited(&self) -> bool {
        self.inner
            .read()
            .await
            .exited
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    pub async fn incr_ready(&self) -> bool {
        self.inner.read().await.incr_ready().await
    }

    pub async fn decr_ready(&self) -> bool {
        self.inner.read().await.decr_ready().await
    }

    pub async fn shutdown_ct(&self) -> CancellationToken {
        self.inner.read().await.shutdown_ct.clone()
    }

    pub async fn err_ct(&self) -> CancellationToken {
        self.inner.read().await.err_ct.clone()
    }
}

struct ConnectionStateInner {
    not_ready: Counter,
    exited: AtomicBool,
    shutdown_ct: CancellationToken,
    err_ct: CancellationToken,
    ready_tx: Option<mpsc::Sender<()>>,
    // perhaps add flag for err_logged and reset on ready
}

impl ConnectionStateInner {
    fn new(
        num_connections: usize,
        shutdown_ct: CancellationToken,
        err_ct: CancellationToken,
    ) -> Self {
        Self {
            not_ready: Counter::new(num_connections),
            exited: AtomicBool::new(false),
            shutdown_ct,
            err_ct,
            ready_tx: None,
        }
    }

    fn replace_cancellation_token(&mut self, err_ct: CancellationToken) {
        self.err_ct = err_ct;
    }

    async fn incr_ready(&self) -> bool {
        let r = self.not_ready.decrement().await;
        if r {
            if let Some(tx) = self.ready_tx.as_ref() {
                // don't want to block if ConnectionState isn't waiting for this to be ready.
                let _ = tx.try_send(());
            }
        }
        r
    }

    pub async fn decr_ready(&self) -> bool {
        self.not_ready.increment().await
    }

    pub async fn is_ready(&self) -> bool {
        self.not_ready.is_zero().await
    }
}

pub struct Counter {
    num: Mutex<usize>,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Counter {
    pub fn new(val: usize) -> Self {
        Self {
            num: Mutex::new(val),
        }
    }

    pub async fn is_zero(&self) -> bool {
        *self.num.lock().await == 0
    }

    // returns true if num was zero
    pub async fn increment(&self) -> bool {
        let mut l = self.num.lock().await;
        *l += 1;
        *l == 1
    }

    // returns true if num was one before decrementing
    pub async fn decrement(&self) -> bool {
        let mut l = self.num.lock().await;
        let r = *l == 1;
        *l = l.saturating_sub(1);
        r
    }
}
