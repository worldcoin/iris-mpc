use crate::{
    execution::player::Identity,
    network::tcp::{
        config::TcpConfig,
        data::{Connection, PeerConnections, StreamId},
        networking::handshake,
        Client, NetworkConnection, Server,
    },
};
use eyre::{eyre, Result};
use rand::{thread_rng, Rng};
use std::{
    collections::{HashMap, HashSet},
    io,
    marker::PhantomData,
    time::Duration,
};
use tokio::{
    sync::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    time::sleep,
};
use tokio_util::sync::CancellationToken;

/// creates a list of peer connections, used to initialize a
/// TcpNetworkHandle
pub struct PeerConnectionBuilder<T: NetworkConnection, C: Client<Output = T>, S: Server<Output = T>>
{
    id: Identity,
    tcp_config: TcpConfig,
    cmd_tx: UnboundedSender<Cmd<T>>,
    _marker: PhantomData<(T, C, S)>,
}

/// re-uses the Worker task from PeerConnectionBuilder
/// allows waiting for pending connections to complete
pub struct Reconnector<T: NetworkConnection> {
    cmd_tx: UnboundedSender<Cmd<T>>,
}

// the #[derive(Clone)] macro may require that all generics implement Clone.
// that is not needed for the Reconnector
impl<T: NetworkConnection> Clone for Reconnector<T> {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
        }
    }
}

enum Cmd<T> {
    Connect {
        peer: Identity,
        url: String,
        stream_id: StreamId,
        rsp: oneshot::Sender<Result<()>>,
    },
    WaitForConnections {
        rsp: oneshot::Sender<Result<PeerConnections<T>>>,
    },
    Reconnect {
        peer: Identity,
        stream_id: StreamId,
        rsp: oneshot::Sender<T>,
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

impl<T, C, S> PeerConnectionBuilder<T, C, S>
where
    T: NetworkConnection + 'static,
    C: Client<Output = T> + 'static,
    S: Server<Output = T> + 'static,
{
    pub async fn new(
        id: Identity,
        tcp_config: TcpConfig,
        listener: S,
        connector: C,
        ct: CancellationToken,
    ) -> Result<Self> {
        let cmd_tx = Worker::spawn(id.clone(), listener, connector, ct).await?;
        Ok(Self {
            id,
            tcp_config,
            cmd_tx,
            _marker: std::marker::PhantomData,
        })
    }

    // returns when the command is queued. will not block for long.
    pub async fn include_peer(&self, peer: Identity, url: String) -> Result<()> {
        if peer == self.id {
            return Err(eyre!("cannot connect to self"));
        }
        for idx in 0..self.tcp_config.num_connections {
            let url = url.clone();
            let stream_id = StreamId::from(idx as u32);
            let (tx, rx) = oneshot::channel();
            self.cmd_tx.send(Cmd::Connect {
                peer: peer.clone(),
                url,
                stream_id,
                rsp: tx,
            })?;
            rx.await??;
        }
        Ok(())
    }

    pub async fn build(self) -> Result<(Reconnector<T>, PeerConnections<T>)> {
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

impl<T: NetworkConnection> Reconnector<T> {
    pub async fn wait_for_reconnections(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Cmd::WaitForReconnections { rsp: tx })
            .map_err(|_| eyre!("worker task dropped"))?;
        rx.await.map_err(|_| eyre!("worker task dropped"))?;
        Ok(())
    }
    pub async fn reconnect(&self, peer: Identity, stream_id: StreamId) -> Result<T> {
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

struct Worker<T: NetworkConnection, C: Client<Output = T>, S: Server<Output = T>> {
    id: Identity,
    cmd_rx: UnboundedReceiver<Cmd<T>>,

    ct: CancellationToken,

    // used to reconnect
    peer_addrs: HashMap<Identity, String>,
    connector: C,

    // used for both the initial connection setup and reconnection
    pending_connections: HashMap<Identity, HashMap<StreamId, Connection<T>>>,
    pending_tx: UnboundedSender<Connection<T>>,
    pending_rx: UnboundedReceiver<Connection<T>>,

    // after State::Connect, requested_connections is used
    // to determine which incoming connections are allowed.
    requested_connections: HashMap<Identity, HashSet<StreamId>>,
    connect_rsp: Option<oneshot::Sender<Result<PeerConnections<T>>>>,

    // reconnections are requested by a forwarder. instead of one response with all connections,
    // (which is what happens with the initial connection), there is now one response per
    // connection
    requested_reconnections: HashMap<Identity, HashMap<StreamId, oneshot::Sender<T>>>,

    // used when waiting for all reconnections to finish
    reconnect_rsp: Option<oneshot::Sender<()>>,

    state: State,

    _marker: PhantomData<S>,
}

impl<T, C, S> Worker<T, C, S>
where
    T: NetworkConnection + 'static,
    C: Client<Output = T> + 'static,
    S: Server<Output = T> + 'static,
{
    pub async fn spawn(
        id: Identity,
        listener: S,
        connector: C,
        ct: CancellationToken,
    ) -> io::Result<UnboundedSender<Cmd<T>>> {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (pending_tx, pending_rx) = mpsc::unbounded_channel::<Connection<T>>();
        let mut worker = Self {
            id: id.clone(),
            cmd_rx,
            ct,
            peer_addrs: HashMap::new(),
            connector,
            pending_connections: HashMap::new(),
            pending_tx,
            pending_rx,
            requested_connections: HashMap::new(),
            connect_rsp: None,
            requested_reconnections: HashMap::new(),
            reconnect_rsp: None,
            state: State::Connect,
            _marker: PhantomData,
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

    async fn run(&mut self, listener: S) -> Result<()> {
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

            self.handle_reconnections()
                .map_err(|e| eyre!("reconnect failed: {:?}", e))?;
            if self.is_reconnect_ready() {
                tracing::debug!("{:?}: ready: no pending re-connections", self.id);
            }
            if self.is_connect_ready() {
                tracing::debug!("{:?}: connections successfully established", self.id);
            }
        }
        Ok(())
    }

    fn handle_pending_connection(&mut self, c: Connection<T>) {
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

    fn handle_cmd(&mut self, cmd: Cmd<T>) {
        match cmd {
            Cmd::Connect {
                peer,
                url,
                stream_id,
                rsp,
            } => {
                self.peer_addrs.insert(peer.clone(), url.clone());
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
                    self.initiate_connection(peer, url, stream_id);
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
                        Some(url) => {
                            let url = url.clone();
                            self.initiate_connection(peer.clone(), url, stream_id);
                        }
                        None => {
                            tracing::error!("reconnect for {:?} requested but URL not found", peer);
                        }
                    }
                }
            }
            Cmd::WaitForReconnections { rsp } => {
                self.reconnect_rsp.replace(rsp);
            }
        }
    }

    fn handle_reconnections(&mut self) -> Result<()> {
        if self.state != State::Reconnect {
            return Ok(());
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
                    let c = peer_pc.remove(&stream_id).ok_or(eyre!(
                        "pending connection lookup failed: {:?}, {:?}",
                        peer,
                        stream_id
                    ))?;
                    let rsp = peer_rc.remove(&stream_id).ok_or(eyre!(
                        "requested connection lookup failed: {:?}, {:?}",
                        peer,
                        stream_id
                    ))?;
                    rsp.send(c.stream)
                        .map_err(|_| eyre!("reconnect failed to send response"))?;
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
        Ok(())
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

    fn initiate_connection(&mut self, peer: Identity, url: String, stream_id: StreamId) {
        let own_id = self.id.clone();
        let pending_tx = self.pending_tx.clone();
        let ct = self.ct.clone();
        let connector = self.connector.clone();

        // don't want all the connections to come in at the same time
        let mut rng = thread_rng();
        let delay_ms = rng.gen_range(0..=3000);
        // backoff when retrying
        let mut retry_sec = 2;

        tokio::spawn(async move {
            tracing::trace!(
                "{:?}: initiating connection to {:?} for {:?}",
                own_id,
                peer,
                stream_id
            );

            // putting ThreadRng in here makes the future not Send
            sleep(Duration::from_millis(delay_ms)).await;
            loop {
                let r = tokio::select! {
                    res = connector.connect(url.clone()) => res,
                    _ = ct.cancelled() => {
                        return;
                    }
                };
                match r {
                    Ok(mut stream) => {
                        if let Err(e) = handshake::outbound(&mut stream, &own_id, &stream_id).await
                        {
                            tracing::error!("{e:?}");
                        } else {
                            let pending = Connection::new(peer, stream, stream_id);
                            if pending_tx.send(pending).is_err() {
                                tracing::debug!("accept loop receiver dropped");
                            }
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::debug!(%e, "dial {:?} failed, retrying", url.clone());
                    }
                };
                sleep(Duration::from_secs(retry_sec)).await;
                if retry_sec < 32 {
                    retry_sec *= 2;
                }
            }
        });
    }
}

/// Just accepts and forwards new connections
async fn accept_loop<T: NetworkConnection, S: Server<Output = T>>(
    id: Identity,
    listener: S,
    pending_tx: UnboundedSender<Connection<T>>,
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
            Ok((peer_addr, mut stream)) => {
                tracing::trace!("{:?} accepted connection from {:?}", id, peer_addr);
                let (peer_id, stream_id) = match handshake::inbound(&mut stream).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!(error=%e, "application level handshake failed");
                        continue;
                    }
                };
                if pending_tx
                    .send(Connection::new(peer_id, stream, stream_id))
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
