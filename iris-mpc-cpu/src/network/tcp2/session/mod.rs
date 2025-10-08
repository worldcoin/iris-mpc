pub mod multiplexer;

use crate::{
    execution::{player::Identity, session::SessionId},
    network::{
        tcp::config::TcpConfig,
        tcp2::{
            connection::ConnectionState,
            data::{ConnectionId, InStream, OutStream, OutboundMsg, Peer},
            NetworkConnection,
        },
        value::NetworkValue,
        Networking,
    },
};
use async_trait::async_trait;
use eyre::{eyre, Result};
use itertools::izip;
use std::{collections::HashMap, sync::Arc};
use tokio::{
    sync::mpsc::{self, UnboundedSender},
    time::timeout,
};

#[derive(Debug)]
pub struct TcpSession {
    session_id: SessionId,
    tx: HashMap<Identity, OutStream>,
    rx: HashMap<Identity, InStream>,
    config: TcpConfig,
}

impl TcpSession {
    pub fn new(
        session_id: SessionId,
        tx: HashMap<Identity, OutStream>,
        rx: HashMap<Identity, InStream>,
        config: TcpConfig,
    ) -> Self {
        Self {
            session_id,
            tx,
            rx,
            config,
        }
    }

    pub fn id(&self) -> SessionId {
        self.session_id
    }
}

#[async_trait]
impl Networking for TcpSession {
    async fn send(&mut self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        let outgoing_stream = self.tx.get(receiver).ok_or(eyre!(
            "Outgoing stream for {receiver:?} in session {:?} not found",
            self.session_id
        ))?;
        outgoing_stream
            .send((self.session_id, value))
            .map_err(|e| eyre!(e.to_string()))?;
        Ok(())
    }

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue> {
        let incoming_stream = self.rx.get_mut(sender).ok_or(eyre!(
            "Incoming stream for {sender:?} in session {:?} not found",
            self.session_id
        ))?;
        match timeout(self.config.timeout_duration, incoming_stream.recv()).await {
            Ok(res) => res.ok_or(eyre!("No message received")),
            Err(_) => Err(eyre!(
                "Timeout while waiting for message from {sender:?} in \
                 {:?}",
                self.session_id
            )),
        }
    }
}

#[derive(Default)]
pub struct SessionChannels {
    pub outbound_tx: HashMap<Identity, HashMap<ConnectionId, mpsc::UnboundedSender<OutboundMsg>>>,
    pub outbound_rx: HashMap<Identity, HashMap<ConnectionId, mpsc::UnboundedReceiver<OutboundMsg>>>,
    pub inbound_tx: HashMap<Identity, HashMap<SessionId, mpsc::UnboundedSender<NetworkValue>>>,
    pub inbound_rx: HashMap<Identity, HashMap<SessionId, mpsc::UnboundedReceiver<NetworkValue>>>,
}

pub async fn make_sessions<T: NetworkConnection + 'static>(
    peers: &[Arc<Peer>],
    mut conn0: Vec<T>,
    mut conn1: Vec<T>,
    connection_state: ConnectionState,
    config: &TcpConfig,
    next_session_id: usize,
) -> Vec<TcpSession> {
    let sc = make_channels(peers, config, next_session_id);
    make_sessions_inner(
        peers,
        conn0,
        conn1,
        connection_state,
        config,
        next_session_id,
        sc,
    )
    .await
}

fn make_channels(
    peers: &[Arc<Peer>],
    config: &TcpConfig,
    next_session_id: usize,
) -> SessionChannels {
    let mut sc = SessionChannels::default();
    tracing::info!(
        "creating {} sessions starting from id {}",
        config.num_sessions,
        next_session_id
    );

    for peer_id in peers.iter().map(|x| x.id()) {
        let mut outbound_tx = HashMap::new();
        let mut outbound_rx = HashMap::new();
        let mut inbound_tx = HashMap::new();
        let mut inbound_rx = HashMap::new();

        for connection_id in (0..config.num_connections).map(|x| ConnectionId::from(x as u32)) {
            let (tx, rx) = mpsc::unbounded_channel::<OutboundMsg>();
            outbound_tx.insert(connection_id, tx);
            outbound_rx.insert(connection_id, rx);
        }

        for session_id in (next_session_id..next_session_id + config.num_sessions)
            .map(|x| SessionId::from(x as u32))
        {
            let (tx, rx) = mpsc::unbounded_channel::<NetworkValue>();
            inbound_tx.insert(session_id, tx);
            inbound_rx.insert(session_id, rx);
        }

        sc.outbound_tx.insert(peer_id.clone(), outbound_tx);
        sc.outbound_rx.insert(peer_id.clone(), outbound_rx);
        sc.inbound_tx.insert(peer_id.clone(), inbound_tx);
        sc.inbound_rx.insert(peer_id.clone(), inbound_rx);
    }
    sc
}

async fn make_sessions_inner<T: NetworkConnection + 'static>(
    peers: &[Arc<Peer>],
    mut conn0: Vec<T>,
    mut conn1: Vec<T>,
    connection_state: ConnectionState,
    config: &TcpConfig,
    next_session_id: usize,
    mut sc: SessionChannels,
) -> Vec<TcpSession> {
    let num_connections = config.num_connections;
    let num_sessions = config.num_sessions;

    // spawn the forwarders
    for (peer_id, mut conns) in izip!(peers.iter().map(|x| x.id()), [conn0, conn1]) {
        for (idx, connection) in conns.drain(..).enumerate() {
            let connection_id = ConnectionId::from(idx as u32);
            let outbound_rx = sc
                .outbound_rx
                .get_mut(peer_id)
                .unwrap()
                .remove(&connection_id)
                .unwrap();

            let inbound_forwarder = sc.inbound_tx.get(peer_id).cloned().unwrap();
            let cs = connection_state.clone();

            tokio::spawn(multiplexer::run(
                connection,
                num_sessions,
                cs,
                inbound_forwarder,
                outbound_rx,
            ));
        }
    }

    // create the sessions
    let mut sessions = vec![];
    for (idx, session_id) in (next_session_id..next_session_id + num_sessions)
        .map(|x| SessionId::from(x as u32))
        .enumerate()
    {
        let mut tx_map = HashMap::new();
        let mut rx_map = HashMap::new();
        let connection_id = ConnectionId::from((idx % num_connections) as u32);

        for peer_id in peers.iter().map(|x| x.id()) {
            let outbound_tx = sc
                .outbound_tx
                .get(peer_id)
                .unwrap()
                .get(&connection_id)
                .cloned()
                .unwrap();
            tx_map.insert(peer_id.clone(), outbound_tx.clone());
            let inbound_rx = sc
                .inbound_rx
                .get_mut(peer_id)
                .unwrap()
                .remove(&session_id)
                .unwrap();
            rx_map.insert(peer_id.clone(), inbound_rx);
        }

        // Create the TcpSession for this stream
        let session = TcpSession::new(session_id, tx_map, rx_map, config.clone());
        sessions.push(session);
    }

    sessions
}
