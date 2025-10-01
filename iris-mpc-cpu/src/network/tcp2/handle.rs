use std::sync::{atomic::AtomicBool, Arc};

use tokio::sync::{mpsc, oneshot, RwLock};
use tokio_util::sync::CancellationToken;

use crate::{
    execution::{player::Identity, session::SessionId},
    network::{tcp::config::TcpConfig, tcp2::{connection::Connection, NetworkConnection}},
};

pub struct<T: NetworkConnection> TcpNetworkHandle {
    peers: [Identity; 2],
    connections0: Vec<Connection<T>>,
    connections1: Vec<Connection<T>>,
    connection_state: ConnectionState,
    config: TcpConfig,
}

impl TcpNetworkHandle {
    pub fn new<I>(mut identities: I, config: TcpConfig, shutdown_ct: CancellationToken) -> Self
    where
        I: Iterator<Item = Identity>,
    {
        let peers: [Identity; 2] = [
            identities.next().expect("expected at least 2 identities"),
            identities.next().expect("expected at least 2 identities"),
        ];
        let mut connections0 = Vec::new();
        let mut connections1 = Vec::new();

        let connection_state = ConnectionState::new(shutdown_ct, CancellationToken::new());

        for _ in 0..config.num_connections {
            // connect to peers[0]
            // connect to peers[1]
        }

        assert_eq!(connections0.len(), config.num_connections);
        assert_eq!(connections1.len(), config.num_connections);
        TcpNetworkHandle {
            peers,
            connections0,
            connections1,
            connection_state,
            config,
        }
    }
}

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

    pub async fn replace(&self, shutdown_ct: CancellationToken, err_ct: CancellationToken) {
        *self.inner.write().await = ConnectionStateInner::new(shutdown_ct, err_ct);
    }

    pub async fn wait_for_exit(&self) {
        let mut lock = self.inner.write().await;
        let (tx, rx) = mpsc::channel(1);
        if lock.is_ready().await() {
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

    pub async fn incr_reconnect(&self) -> bool {
        self.inner.read().await.reconnecting.increment().await
    }

    pub async fn decr_reconnect(&self) -> bool {
        self.inner.read().await.decr_reconnect().await
    }

    pub fn shutdown_ct(&self) -> CancellationToken {
        self.inner.read().await.shutdown_ct.clone()
    }

    pub fn err_ct(&self) -> CancellationToken {
        self.inner.read().await.err_ct.clone()
    }
}

struct ConnectionStateInner {
    reconnecting: Counter,
    exited: AtomicBool,
    shutdown_ct: CancellationToken,
    err_ct: CancellationToken,
    ready_tx: Option<mpsc::Sender<()>>,
}

impl ConnectionStateInner {
    fn new(shutdown_ct: CancellationToken, err_ct: CancellationToken) -> Self {
        Self {
            reconnecting: Counter::new(),
            exited: AtomicBool::new(false),
            shutdown_ct,
            err_ct,
            ready_tx: None,
        }
    }

    async fn decr_reconnect(&self) -> bool {
        let r = self.reconnecting.decrement().await;
        if r {
            if let Some(tx) = self.ready_tx {
                // don't want to block if ConnectionState isn't waiting for this to be ready.
                let _ = tx.try_send(());
            }
        }
    }

    pub async fn is_ready(&self) -> bool {
        self.reconnecting.is_zero().await
    }
}

pub struct Counter {
    num: Mutex<usize>,
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { num: Mutex::new(0) }
    }

    pub async fn is_zero(&self) -> bool {
        self.num.lock().await == 0
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
