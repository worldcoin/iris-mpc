mod client; // trait for initiating a connection. hides details of TCP vs TLS
mod connection_state;
mod handshake;
mod listener; // accept inbound connections
mod server; // trait for accepting connections. hides details of TCP vs TLS // used to determine the peer id and connection id

pub use connection_state::ConnectionState;
pub use listener::{accept_loop, ConnectionRequest};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    execution::player::Identity,
    network::tcp2::{
        data::{ConnectionId, Peer},
        Client, NetworkConnection,
    },
};
use eyre::Result;
use socket2::{SockRef, TcpKeepalive};
use std::{sync::Arc, time::Duration};
use tokio::{
    net::TcpStream,
    sync::{mpsc, oneshot},
    time::sleep,
};

pub struct Connection {
    cmd_tx: mpsc::Sender<InnerCmd>,
}

impl Connection {
    pub fn new<T: NetworkConnection + 'static, C: Client<Output = T> + 'static>(
        connection_id: ConnectionId,
        own_id: Arc<Identity>,
        peer: Arc<Peer>,
        connection_state: ConnectionState,
        client: C,
        conn_req_tx: mpsc::Sender<ConnectionRequest<T>>,
    ) -> Self {
        let inner = ConnectionInner {
            connection_id,
            own_id,
            peer,
            connection_state,
            client,
            conn_req_tx,
        };
        let (cmd_tx, cmd_rx) = mpsc::channel(1);
        tokio::spawn(manage_connection(inner, cmd_rx));
        Self { cmd_tx }
    }

    pub async fn connect(&self) {
        let _ = self.cmd_tx.send(InnerCmd::Connect).await;
    }

    pub async fn disconect(&self) {
        let _ = self.cmd_tx.send(InnerCmd::Close).await;
    }
}

#[derive(PartialEq, Eq)]
enum InnerState {
    Idle,
    Connecting,
    Ready,
}

enum InnerCmd {
    Connect,
    Close,
}

struct ConnectionInner<T: NetworkConnection, C: Client> {
    connection_id: ConnectionId,
    own_id: Arc<Identity>,
    peer: Arc<Peer>,
    connection_state: ConnectionState,
    // initiates the connection
    client: C,
    // listens for the connection
    conn_req_tx: mpsc::Sender<ConnectionRequest<T>>,
}

impl<T: NetworkConnection, C: Client<Output = T>> ConnectionInner<T, C> {
    async fn connect(&self) -> Result<T> {
        if &*self.own_id > self.peer.id() {
            let mut stream = self.client.connect(self.peer.url().to_string()).await?;
            handshake::outbound(&mut stream, &self.own_id, &self.connection_id).await?;
            Ok(stream)
        } else {
            let (rsp_tx, rsp_rx) = oneshot::channel();
            let req = ConnectionRequest::new(self.peer.id().clone(), self.connection_id, rsp_tx);
            let _ = self.conn_req_tx.send(req).await;
            let r = rsp_rx.await?;
            Ok(r)
        }
    }
    async fn connect_loop(&self) -> T {
        let mut rng: StdRng =
            StdRng::from_rng(&mut rand::thread_rng()).expect("Failed to seed RNG");

        // backoff when retrying
        let mut retry_sec = 2;

        sleep(Duration::from_millis(rng.gen_range(0..=3000))).await;

        loop {
            match self.connect().await {
                Ok(stream) => return stream,
                Err(e) => {
                    tracing::debug!("connect failed: {e:?}");
                }
            }

            sleep(Duration::from_secs(retry_sec)).await;
            if retry_sec < 32 {
                retry_sec *= 2;
            }
        }
    }

    async fn do_idle(&self, cmd_rx: &mut mpsc::Receiver<InnerCmd>) -> Option<()> {
        loop {
            let err_ct = self.connection_state.err_ct().await;
            let shutdown_ct = self.connection_state.shutdown_ct().await;

            let opt = tokio::select! {
                o = cmd_rx.recv() => o,
                _ = err_ct.cancelled() => continue,
                _ = shutdown_ct.cancelled() => {
                    return None;
                }
            };

            match opt {
                Some(cmd) => match cmd {
                    InnerCmd::Connect => {
                        return Some(());
                    }
                    InnerCmd::Close => continue,
                },
                None => return None,
            }
        }
    }

    async fn do_connect(&self, cmd_rx: &mut mpsc::Receiver<InnerCmd>) -> Option<T> {
        let err_ct = self.connection_state.err_ct().await;
        let shutdown_ct = self.connection_state.shutdown_ct().await;

        loop {
            let opt = tokio::select! {
                r = self.connect_loop() => return Some(r),
                _ = err_ct.cancelled() => {
                    return None;
                },
                _ = shutdown_ct.cancelled() => {
                    return None;
                },
                o = cmd_rx.recv() => o,
            };

            match opt {
                Some(InnerCmd::Connect) => continue,
                _ => return None,
            }
        }
    }

    async fn do_run(&self, connection: T) {
        let err_ct = self.connection_state.err_ct().await;
        let shutdown_ct = self.connection_state.shutdown_ct().await;

        todo!();
    }
}

async fn manage_connection<T: NetworkConnection, C: Client<Output = T>>(
    inner: ConnectionInner<T, C>,
    mut cmd_rx: mpsc::Receiver<InnerCmd>,
) {
    let mut inner_state = InnerState::Idle;
    let mut connection: Option<T> = None;

    loop {
        match inner_state {
            InnerState::Idle => match inner.do_idle(&mut cmd_rx).await {
                Some(_) => inner_state = InnerState::Connecting,
                None => break,
            },
            InnerState::Connecting => match inner.do_connect(&mut cmd_rx).await {
                Some(stream) => {
                    connection.replace(stream);
                    inner_state = InnerState::Ready;
                    inner.connection_state.incr_ready().await;
                }
                None => {
                    // if there was a shutdown, do_idle() will return None and this loop will exit.
                    inner_state = InnerState::Idle;
                }
            },
            InnerState::Ready => {
                let c = connection.take().unwrap();
                inner.do_run(c).await;
                inner_state = InnerState::Idle;
                inner.connection_state.decr_ready().await;
            }
        }
    }
}
