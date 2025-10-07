mod client; // trait for initiating a connection. hides details of TCP vs TLS
mod connection_state;
mod handshake;
mod listener; // accept inbound connections
mod server; // trait for accepting connections. hides details of TCP vs TLS // used to determine the peer id and connection id

pub use connection_state::ConnectionState;
pub use listener::{accept_loop, ConnectionRequest};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    execution::{player::Identity, session::SessionId},
    network::{
        tcp2::{
            data::{ConnectionId, OutboundMsg, Peer},
            Client, NetworkConnection,
        },
        value::NetworkValue,
    },
};
use eyre::Result;
use socket2::{SockRef, TcpKeepalive};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf},
    net::TcpStream,
    sync::{mpsc, oneshot},
    time::sleep,
};

struct ConnectCmd {
    num_sessions: usize,
    session_start_idx: usize,
    rsp: oneshot::Sender<Vec<ConnectRsp>>,
}

pub struct ConnectRsp {
    pub session_id: SessionId,
    pub session_tx: mpsc::Sender<OutboundMsg>,
    pub session_rx: mpsc::Receiver<NetworkValue>,
}

enum InnerCmd {
    Connect(ConnectCmd),
    Close,
}

#[derive(PartialEq, Eq)]
enum LoopState {
    Idle,
    Connecting,
    Ready,
}

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

    pub async fn connect(
        &self,
        num_sessions: usize,
        session_start_idx: usize,
    ) -> Result<oneshot::Receiver<Vec<ConnectRsp>>> {
        let (rsp_tx, rsp_rx) = oneshot::channel();
        let cmd = InnerCmd::Connect(ConnectCmd {
            num_sessions,
            session_start_idx,
            rsp: rsp_tx,
        });
        let _ = self.cmd_tx.send(cmd).await?;
        Ok(rsp_rx)
    }

    pub async fn disconect(&self) {
        let _ = self.cmd_tx.send(InnerCmd::Close).await;
    }
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

    async fn do_idle(&self, cmd_rx: &mut mpsc::Receiver<InnerCmd>) -> Option<InnerCmd> {
        loop {
            let err_ct = self.connection_state.err_ct().await;
            let shutdown_ct = self.connection_state.shutdown_ct().await;

            tokio::select! {
                o = cmd_rx.recv() => return o,
                _ = err_ct.cancelled() => continue,
                _ = shutdown_ct.cancelled() => {
                    return None;
                }
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
                Some(InnerCmd::Connect(_)) => continue,
                _ => return None,
            }
        }
    }

    async fn do_run(
        &self,
        connection: T,
        cmd: ConnectCmd,
        conn_state: ConnectionState,
        cmd_rx: &mut mpsc::Receiver<InnerCmd>,
    ) -> Option<ConnectCmd> {
        let err_ct = self.connection_state.err_ct().await;
        let shutdown_ct = self.connection_state.shutdown_ct().await;

        let ConnectCmd {
            num_sessions,
            session_start_idx,
            rsp,
        } = cmd;

        let mut r = vec![];
        let mut inbound_map: HashMap<SessionId, mpsc::Sender<NetworkValue>> = HashMap::new();
        let (outbound_tx, outbound_rx) = mpsc::channel(num_sessions);

        for session_id in session_start_idx..session_start_idx + num_sessions {
            let (inbound_tx, inbound_rx) = mpsc::channel(1);
            inbound_map.insert(SessionId::from(session_id as u32), inbound_tx);

            r.push(ConnectRsp {
                session_id: SessionId::from(session_id as u32),
                session_tx: outbound_tx.clone(),
                session_rx: inbound_rx,
            });
        }

        let (reader, writer) = tokio::io::split(connection);
        let reader = BufReader::new(reader);

        let _ = rsp.send(r);

        tokio::select! {
            opt = cmd_rx.recv() => match opt {
                Some(cmd) => match cmd {
                    InnerCmd::Connect(cmd) => todo!(),
                    _ => todo!(),
                },
                None => todo!(),
            },
            _ = err_ct.cancelled() => todo!(),
            _ = shutdown_ct.cancelled() => todo!(),
            _ = handle_outbound_traffic(writer, outbound_rx, num_sessions) => todo!(),
            _ = handle_inbound_traffic(reader, inbound_map) => todo!(),
        }
    }
}

async fn handle_outbound_traffic<T: NetworkConnection>(
    mut stream: WriteHalf<T>,
    mut outbound_rx: mpsc::Receiver<OutboundMsg>,
    num_sessions: usize,
) {
}

async fn handle_inbound_traffic<T: NetworkConnection>(
    mut reader: BufReader<ReadHalf<T>>,
    inbound_tx: HashMap<SessionId, mpsc::Sender<NetworkValue>>,
) {
}

async fn manage_connection<T: NetworkConnection, C: Client<Output = T>>(
    inner: ConnectionInner<T, C>,
    mut cmd_rx: mpsc::Receiver<InnerCmd>,
) {
    let mut loop_state = LoopState::Idle;
    let mut connection: Option<T> = None;
    let mut conn_cmd: Option<ConnectCmd> = None;

    loop {
        match loop_state {
            LoopState::Idle => match inner.do_idle(&mut cmd_rx).await {
                Some(cmd) => match cmd {
                    InnerCmd::Connect(c) => {
                        conn_cmd.replace(c);
                        loop_state = LoopState::Connecting;
                    }
                    InnerCmd::Close => continue,
                },
                None => break,
            },
            LoopState::Connecting => match inner.do_connect(&mut cmd_rx).await {
                Some(stream) => {
                    connection.replace(stream);
                    loop_state = LoopState::Ready;
                    inner.connection_state.incr_ready().await;
                }
                None => {
                    // if there was a shutdown, do_idle() will return None and this loop will exit.
                    loop_state = LoopState::Idle;
                }
            },
            LoopState::Ready => {
                let c = connection.take().unwrap();
                let cmd = conn_cmd.take().unwrap();
                inner
                    .do_run(c, cmd, inner.connection_state.clone(), &mut cmd_rx)
                    .await;
                loop_state = LoopState::Idle;
                // each connection needs to decr_ready on its own.
            }
        }
    }
}
