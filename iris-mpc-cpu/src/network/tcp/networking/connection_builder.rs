use crate::{
    execution::player::Identity,
    network::tcp::{
        data::{PeerConnections, StreamId, TcpConnection},
        networking::handshake,
        TcpConfig,
    },
};
use eyre::{eyre, Result};
use std::{
    collections::{HashMap, HashSet},
    io,
    net::SocketAddr,
    time::Duration,
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    time::sleep,
};
use tokio_util::sync::CancellationToken;

/// creates a list of peer connections, used to initialize a
/// TcpNetworkHandle
pub struct PeerConnectionBuilder {
    id: Identity,
    config: TcpConfig,
    cmd_tx: UnboundedSender<Cmd>,
}

/// re-uses the Worker task from PeerConnectionBuilder
/// allows waiting for pending connections to complete
#[derive(Clone)]
pub struct Reconnector {
    cmd_tx: UnboundedSender<Cmd>,
}

enum Cmd {
    Connect {
        peer: Identity,
        addr: SocketAddr,
        stream_id: StreamId,
        rsp: oneshot::Sender<Result<()>>,
    },
    WaitForConnections {
        rsp: oneshot::Sender<Result<PeerConnections>>,
    },
    Reconnect {
        peer: Identity,
        stream_id: StreamId,
        rsp: oneshot::Sender<TcpStream>,
    },
    WaitForReconnections {
        rsp: oneshot::Sender<()>,
    },
}

#[derive(Eq, PartialEq, Clone, Copy)]
enum State {
    Connect,
    Reconnect,
}

impl PeerConnectionBuilder {
    pub async fn new(id: Identity, addr: SocketAddr, config: TcpConfig) -> Result<Self> {
        let cmd_tx = Worker::spawn(id.clone(), addr).await?;
        Ok(Self { id, config, cmd_tx })
    }

    // returns when the command is queued. will not block for long.
    pub async fn include_peer(&self, peer: Identity, addr: SocketAddr) -> Result<()> {
        if peer == self.id {
            return Err(eyre!("cannot connect to self"));
        }
        for idx in 0..self.config.num_connections {
            let stream_id = StreamId::from(idx as u32);
            let (tx, rx) = oneshot::channel();
            self.cmd_tx.send(Cmd::Connect {
                peer: peer.clone(),
                addr,
                stream_id,
                rsp: tx,
            })?;
            rx.await??;
        }
        Ok(())
    }

    pub async fn build(self) -> Result<(Reconnector, PeerConnections)> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx.send(Cmd::WaitForConnections { rsp: tx })?;
        let connections = rx.await.unwrap()?;
        Ok((
            Reconnector {
                cmd_tx: self.cmd_tx.clone(),
            },
            connections,
        ))
    }
}

impl Reconnector {
    pub async fn wait_for_reconnections(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Cmd::WaitForReconnections { rsp: tx })
            .map_err(|_| eyre!("worker task dropped"))?;
        rx.await.map_err(|_| eyre!("worker task dropped"))?;
        Ok(())
    }
    pub async fn reconnect(&self, peer: Identity, stream_id: StreamId) -> Result<TcpStream> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Cmd::Reconnect {
                peer,
                stream_id,
                rsp: tx,
            })
            .map_err(|_| eyre!("worker task dropped"))?;
        rx.await.map_err(|_| eyre!("worker task dropped"))
    }
}

struct Worker {
    id: Identity,
    cmd_rx: UnboundedReceiver<Cmd>,

    ct: CancellationToken,

    // used to reconnect
    peer_addrs: HashMap<Identity, SocketAddr>,

    // used for both the initial connection setup and reconnection
    pending_connections: HashMap<Identity, HashMap<StreamId, TcpConnection>>,
    pending_tx: UnboundedSender<TcpConnection>,
    pending_rx: UnboundedReceiver<TcpConnection>,

    // after State::Connect, requested_connections is used
    // to determine which incoming connections are allowed.
    requested_connections: HashMap<Identity, HashSet<StreamId>>,
    connect_rsp: Option<oneshot::Sender<Result<PeerConnections>>>,

    // reconnections are requested by a forwarder. instead of one response with all connections,
    // (which is what happens with the initial connection), there is now one response per
    // connection
    requested_reconnections: HashMap<Identity, HashMap<StreamId, oneshot::Sender<TcpStream>>>,

    // used when waiting for all reconnections to finish
    reconnect_rsp: Option<oneshot::Sender<()>>,

    state: State,
}

impl Worker {
    pub async fn spawn(id: Identity, own_addr: SocketAddr) -> io::Result<UnboundedSender<Cmd>> {
        let listener = TcpListener::bind(own_addr).await?;
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (pending_tx, pending_rx) = mpsc::unbounded_channel::<TcpConnection>();
        let mut worker = Self {
            id: id.clone(),
            cmd_rx,
            ct: CancellationToken::new(),
            peer_addrs: HashMap::new(),
            pending_connections: HashMap::new(),
            pending_tx,
            pending_rx,
            requested_connections: HashMap::new(),
            connect_rsp: None,
            requested_reconnections: HashMap::new(),
            reconnect_rsp: None,
            state: State::Connect,
        };

        tokio::spawn(async move {
            let ct = worker.ct.clone();
            if let Err(e) = worker.run(listener).await {
                tracing::error!(error=%e, "ConnectionManager failed");
            }
            ct.cancel();
        });
        Ok(cmd_tx)
    }

    async fn run(&mut self, listener: TcpListener) -> io::Result<()> {
        let (accept_tx, mut accept_rx) = oneshot::channel();
        let own_id = self.id.clone();
        let pending_tx2 = self.pending_tx.clone();
        let ct2 = self.ct.clone();
        tokio::spawn(async move {
            accept_loop(own_id.clone(), listener, pending_tx2, ct2).await;
            tracing::debug!("{:?}: connection manager's accept_loop exited", own_id);
            let _ = accept_tx.send(());
        });

        loop {
            tokio::select! {
                opt = self.pending_rx.recv() => {
                    let Some(c) = opt else {
                        tracing::error!("{:?} connection manager failed: pending_rx dropped",self.id);
                        break;
                    };
                    self.handle_pending_connection(c);
                }
                opt = self.cmd_rx.recv() => {
                    let Some(cmd) = opt else {
                        tracing::error!("{:?} connection manager failed: cmd_rx dropped",self.id);
                        break;
                    };
                    self.handle_cmd(cmd);
                }
                _ = &mut accept_rx => {
                    break;
                }
            }

            self.handle_reconnections();
            if self.is_reconnect_ready() {
                tracing::debug!("{:?}: ready: no pending re-connections", self.id);
            }
            if self.is_connect_ready() {
                tracing::debug!("{:?}: connections successfully established", self.id);
            }
        }
        Ok(())
    }

    fn handle_pending_connection(&mut self, c: TcpConnection) {
        c.stream.set_nodelay(true).unwrap();

        if self
            .pending_connections
            .get(&c.peer_id())
            .map(|x| x.contains_key(&c.stream_id))
            .unwrap_or_default()
        {
            tracing::warn!(
                "{:?} received duplication connection request {:?}",
                self.id,
                c
            );
        } else {
            let should_insert = match self.state {
                State::Connect => true,
                State::Reconnect => self
                    .requested_reconnections
                    .get(&c.peer_id())
                    .map(|streams| streams.contains_key(&c.stream_id))
                    .unwrap_or(false),
            };
            if should_insert {
                self.pending_connections
                    .entry(c.peer_id())
                    .or_default()
                    .insert(c.stream_id, c);
            } else {
                tracing::warn!("connection received which was not requested");
            }
        }
    }

    fn handle_cmd(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Connect {
                peer,
                addr,
                stream_id,
                rsp,
            } => {
                self.peer_addrs.insert(peer.clone(), addr);
                if !self
                    .requested_connections
                    .entry(peer.clone())
                    .or_default()
                    .insert(stream_id)
                {
                    let _ = rsp.send(Err(eyre!(
                        "connection to peer already requested for {:?}",
                        stream_id
                    )));
                    return;
                }
                if self.id > peer {
                    self.initiate_connection(peer, addr, stream_id);
                }
                rsp.send(Ok(())).unwrap();
            }
            Cmd::WaitForConnections { rsp } => {
                self.connect_rsp.replace(rsp);
            }
            Cmd::Reconnect {
                peer,
                stream_id,
                rsp,
            } => {
                self.requested_reconnections
                    .entry(peer.clone())
                    .or_default()
                    .insert(stream_id, rsp);
                if self.id > peer {
                    match self.peer_addrs.get(&peer) {
                        Some(addr) => {
                            self.initiate_connection(peer.clone(), *addr, stream_id);
                        }
                        None => {
                            tracing::error!(
                                "reconnect for {:?} requested but addr not found",
                                peer
                            );
                        }
                    }
                }
            }
            Cmd::WaitForReconnections { rsp } => {
                self.reconnect_rsp.replace(rsp);
            }
        }
    }

    fn handle_reconnections(&mut self) {
        if self.state != State::Reconnect {
            return;
        }

        // For each requested reconnection, if the connection is ready, send it immediately.
        let mut peers_to_remove = vec![];
        // rc: requested connections
        for (peer, peer_rc) in self.requested_reconnections.iter_mut() {
            let mut ready = HashSet::new();
            // pc: pending connections
            if let Some(peer_pc) = self.pending_connections.get_mut(peer) {
                for stream_id in peer_rc.keys() {
                    if peer_pc.contains_key(stream_id) {
                        ready.insert(*stream_id);
                    }
                }

                for stream_id in ready {
                    let c = peer_pc.remove(&stream_id).unwrap();
                    let rsp = peer_rc.remove(&stream_id).unwrap();
                    rsp.send(c.stream).unwrap();
                }
            }

            // If all streams for this peer are handled, mark for removal
            if peer_rc.is_empty() {
                peers_to_remove.push(peer.clone());
            }
        }
        // Remove peers with no more pending streams
        for peer in peers_to_remove {
            self.requested_reconnections.remove(&peer);
        }
    }

    fn is_connect_ready(&mut self) -> bool {
        if self.state != State::Connect || self.connect_rsp.is_none() {
            return false;
        }

        // check if ready
        for (peer, streams) in &self.requested_connections {
            let pending = self.pending_connections.get(peer);
            if pending.is_none() || !streams.is_subset(&pending.unwrap().keys().cloned().collect())
            {
                // not ready
                return false;
            }
        }

        let mut result = HashMap::new();
        for (peer, streams) in &self.requested_connections {
            let mut conns = HashMap::new();
            if let Some(pending_map) = self.pending_connections.get_mut(peer) {
                for stream_id in streams {
                    if let Some(tcp_conn) = pending_map.remove(stream_id) {
                        conns.insert(*stream_id, tcp_conn);
                    } else {
                        unreachable!();
                    }
                }
            } else {
                unreachable!();
            }
            result.insert(peer.clone(), conns);
        }

        // Remove any connections that were not requested
        self.pending_connections.clear();

        self.connect_rsp.take().unwrap().send(Ok(result)).unwrap();
        self.state = State::Reconnect;
        true
    }

    fn is_reconnect_ready(&mut self) -> bool {
        if self.state != State::Reconnect
            || self.reconnect_rsp.is_none()
            || !self.requested_reconnections.is_empty()
        {
            return false;
        }

        let rsp = self.reconnect_rsp.take().unwrap();
        let _ = rsp.send(());
        true
    }

    fn initiate_connection(&mut self, peer: Identity, addr: SocketAddr, stream_id: StreamId) {
        let own_id = self.id.clone();
        let pending_tx = self.pending_tx.clone();
        let ct = self.ct.clone();
        tokio::spawn(async move {
            tracing::trace!(
                "{:?}: initiating connection to {:?} for {:?}",
                own_id,
                peer,
                stream_id
            );
            let retry = Duration::from_millis(500);
            loop {
                let r = tokio::select! {
                    res = TcpStream::connect(addr) => res,
                    _ = ct.cancelled() => {
                        return;
                    }
                };
                match r {
                    Ok(mut stream) => {
                        if let Err(e) = handshake::outbound(&mut stream, &own_id, &stream_id).await
                        {
                            tracing::error!("{e:?}");
                            continue;
                        }
                        let pending = TcpConnection::new(peer, stream, stream_id);
                        if pending_tx.send(pending).is_err() {
                            tracing::error!("accept loop receiver dropped");
                        }
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(%e, "dial {:?} failed, retrying", addr);
                        sleep(retry).await;
                    }
                }
            }
        });
    }
}

/// Just accepts and forwards new connections
async fn accept_loop(
    id: Identity,
    listener: TcpListener,
    pending_tx: UnboundedSender<TcpConnection>,
    ct: CancellationToken,
) {
    loop {
        let r = tokio::select! {
            res = listener.accept() => res,
            _ = ct.cancelled() => {
                break;
            }
        };
        match r {
            Ok((mut stream, _addr)) => {
                let peer_addr = match stream.peer_addr() {
                    Ok(addr) => addr,
                    Err(e) => {
                        tracing::error!(error=%e, "Failed to get peer_addr");
                        continue;
                    }
                };
                tracing::trace!("{:?} accepted connection from {:?}", id, peer_addr);

                let (peer_id, stream_id) = match handshake::inbound(&mut stream).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("{e:?}");
                        continue;
                    }
                };

                if pending_tx
                    .send(TcpConnection::new(peer_id, stream, stream_id))
                    .is_err()
                {
                    tracing::error!("accept_loop: incoming_rx dropped");
                    return;
                }
            }
            Err(e) => tracing::error!(%e, "accept_loop error"),
        }
    }
}
