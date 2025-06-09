use super::{
    session::TcpSession, NetworkMsg, OutStream, OutboundMsg, PeerConnections, StreamId,
    TcpConnection,
};
use crate::{
    execution::{player::Identity, session::SessionId},
    network::{
        grpc::GrpcConfig, tcp::networking::connection_builder::Reconnector, value::DescriptorByte,
    },
};
use bytes::BytesMut;
use eyre::Result;
use std::collections::HashMap;
use std::{io, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    sync::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        Mutex,
    },
    task::JoinSet,
    time::{self},
};

const FLUSH_INTERVAL_MS: u64 = 2;
const BUFFER_CAPACITY: usize = 64 * 1024;
const READ_BUF_SIZE: usize = 8 * 1024;

#[derive(Default)]
struct SessionChannels {
    outbound_tx: HashMap<Identity, HashMap<StreamId, UnboundedSender<OutboundMsg>>>,
    outbound_rx: HashMap<Identity, HashMap<StreamId, UnboundedReceiver<OutboundMsg>>>,
    inbound_tx: HashMap<Identity, HashMap<SessionId, UnboundedSender<NetworkMsg>>>,
    inbound_rx: HashMap<Identity, HashMap<SessionId, UnboundedReceiver<NetworkMsg>>>,
}

/// spawns a task for each TCP connection (there are x connections per peer and each of the x
/// connections has y sessions, of the same session id)
pub struct TcpNetworkHandle {
    peers: Vec<Identity>,
    ch_map: HashMap<Identity, HashMap<StreamId, UnboundedSender<NewSessions>>>,
    reconnector: Reconnector,
    config: GrpcConfig,
}

struct NewSessions {
    inbound_forwarder: HashMap<SessionId, OutStream>,
    outbound_rx: UnboundedReceiver<OutboundMsg>,
}

impl TcpNetworkHandle {
    pub fn new(reconnector: Reconnector, connections: PeerConnections, config: GrpcConfig) -> Self {
        let peers = connections.keys().cloned().collect();
        let sessions_per_connection = config.stream_parallelism;
        let mut ch_map = HashMap::new();
        for (peer_id, connections) in connections {
            let mut m = HashMap::new();
            for (stream_id, connection) in connections {
                let rc = reconnector.clone();
                let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
                m.insert(stream_id, cmd_tx);

                tokio::spawn(manage_connection(
                    connection,
                    rc,
                    sessions_per_connection,
                    cmd_rx,
                ));
            }
            ch_map.insert(peer_id.clone(), m);
        }
        Self {
            peers,
            ch_map,
            reconnector,
            config,
        }
    }
    pub async fn make_sessions(&self) -> Result<Vec<TcpSession>> {
        tracing::debug!("make_sessions");
        self.reconnector.wait_for_reconnections().await?;
        let sc = make_channels(&self.peers, &self.config);
        Ok(self.make_sessions_inner(sc))
    }

    fn make_sessions_inner(&self, mut sc: SessionChannels) -> Vec<TcpSession> {
        let sessions_per_connection = self.config.stream_parallelism;
        // spawn the forwarders
        for peer_id in &self.peers {
            for tcp_stream in 0..self.config.connection_parallelism {
                let stream_id = StreamId::from(tcp_stream as u32);
                let outbound_rx = sc
                    .outbound_rx
                    .get_mut(peer_id)
                    .unwrap()
                    .remove(&stream_id)
                    .unwrap();

                // Build inbound forwarder map for this stream
                let mut inbound_forwarder = HashMap::new();
                for session_idx in 0..sessions_per_connection {
                    let session_id = SessionId::from(
                        (tcp_stream * sessions_per_connection + session_idx) as u32,
                    );
                    let inbound_tx = sc
                        .inbound_tx
                        .get(peer_id)
                        .unwrap()
                        .get(&session_id)
                        .cloned()
                        .unwrap();
                    inbound_forwarder.insert(session_id, inbound_tx);
                }

                let ch = self.ch_map.get(peer_id).unwrap().get(&stream_id).unwrap();
                ch.send(NewSessions {
                    inbound_forwarder,
                    outbound_rx,
                })
                .unwrap();
            }
        }

        // create the sessions
        let mut sessions = vec![];
        for tcp_stream in 0..self.config.connection_parallelism {
            let stream_id = StreamId::from(tcp_stream as u32);
            for session_idx in 0..sessions_per_connection {
                let mut tx_map = HashMap::new();
                let mut rx_map = HashMap::new();

                let session_id =
                    SessionId::from((tcp_stream * sessions_per_connection + session_idx) as u32);

                for peer_id in &self.peers {
                    let outbound_tx = sc
                        .outbound_tx
                        .get(peer_id)
                        .unwrap()
                        .get(&stream_id)
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
                let session = TcpSession::new(session_id, tx_map, rx_map, self.config.clone());
                sessions.push(session);
            }
        }
        sessions
    }
}

fn make_channels(peers: &[Identity], config: &GrpcConfig) -> SessionChannels {
    let mut sc = SessionChannels::default();
    for peer_id in peers {
        let mut outbound_tx = HashMap::new();
        let mut outbound_rx = HashMap::new();
        let mut inbound_tx = HashMap::new();
        let mut inbound_rx = HashMap::new();

        for tcp_stream in 0..config.connection_parallelism {
            // insert one pair of outbound channels per TcpStream
            let stream_id = StreamId::from(tcp_stream as u32);
            let (tx, rx) = mpsc::unbounded_channel::<OutboundMsg>();
            outbound_tx.insert(stream_id, tx);
            outbound_rx.insert(stream_id, rx);

            // insert one pair of inbound channels per stream_parallelism
            for session_idx in 0..config.stream_parallelism {
                let session_id =
                    SessionId::from((tcp_stream * config.stream_parallelism + session_idx) as u32);
                let (tx, rx) = mpsc::unbounded_channel::<NetworkMsg>();
                inbound_tx.insert(session_id, tx);
                inbound_rx.insert(session_id, rx);
            }
        }

        sc.outbound_tx.insert(peer_id.clone(), outbound_tx);
        sc.outbound_rx.insert(peer_id.clone(), outbound_rx);
        sc.inbound_tx.insert(peer_id.clone(), inbound_tx);
        sc.inbound_rx.insert(peer_id.clone(), inbound_rx);
    }
    sc
}

async fn manage_connection(
    connection: TcpConnection,
    reconnector: Reconnector,
    num_sessions: usize,
    mut cmd_ch: UnboundedReceiver<NewSessions>,
) {
    let TcpConnection {
        peer,
        stream,
        stream_id,
    } = connection;

    let (reader, writer) = stream.into_split();
    let reader = Arc::new(Mutex::new(BufReader::new(reader)));
    let writer = Arc::new(Mutex::new(writer));

    // We enter the loop only after receiving the first NewSessions
    let NewSessions {
        inbound_forwarder,
        outbound_rx,
    } = match cmd_ch.recv().await {
        Some(cmd) => cmd,
        None => {
            tracing::debug!("cmd channel closed before first session assignment");
            return;
        }
    };
    let inbound_forwarder = Arc::new(Mutex::new(inbound_forwarder));
    let outbound_rx = Arc::new(Mutex::new(outbound_rx));

    // on disconnect, reconnect automatically. tear down and stand up the forwarders.
    // when new sessions are requested, also tear down and stand up the forwarders.
    loop {
        // clone the Arcs to pass to the forwarders
        let reader2 = reader.clone();
        let writer2 = writer.clone();
        let inbound2 = inbound_forwarder.clone();
        let outbound2 = outbound_rx.clone();

        // spawn the forwarders
        let mut set = JoinSet::new();
        set.spawn(async move {
            let mut writer = writer2.lock().await;
            let mut outbound_rx = outbound2.lock().await;
            let r = handle_outbound_traffic(&mut writer, &mut outbound_rx, num_sessions).await;
            tracing::debug!("handle_outbound_traffic exited: {r:?}");
        });
        set.spawn(async move {
            let mut reader = reader2.lock().await;
            let inbound_forwarder = inbound2.lock().await;
            let r = handle_inbound_traffic(&mut reader, &inbound_forwarder).await;
            tracing::debug!("handle_inbound_traffic exited: {r:?}");
        });

        // wait for an event
        enum ConnectionEvent {
            NewSessions(NewSessions),
            Disconnected,
        }
        let event = tokio::select! {
            maybe_cmd = cmd_ch.recv() => {
                match maybe_cmd {
                    Some(cmd) => ConnectionEvent::NewSessions(cmd),
                    None => {
                        tracing::debug!("cmd channel closed");
                        return;
                    }
                }
            }
            _ = set.join_next() => ConnectionEvent::Disconnected,
        };

        // shut down the forwarders
        set.shutdown().await;

        // update the Arcs depending on the event. wait for reconnect if needed.
        match event {
            ConnectionEvent::NewSessions(cmd) => {
                tracing::debug!("updating sessions for {:?}: {:?}", peer, session_id);
                *inbound_forwarder.lock().await = cmd.inbound_forwarder;
                *outbound_rx.lock().await = cmd.outbound_rx;
                continue;
            }
            ConnectionEvent::Disconnected => {
                tracing::debug!("reconnecting to {:?}: {:?}", peer, stream_id);
                let stream = match reconnector.reconnect(peer.clone(), stream_id).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::error!("reconnection failed: {e:?}");
                        return;
                    }
                };
                let (r, w) = stream.into_split();
                *reader.lock().await = BufReader::new(r);
                *writer.lock().await = w;
            }
        }
    }
}

/// Outbound: send messages from rx to the socket.
/// the sender needs to prepend the session id to the message.
async fn handle_outbound_traffic(
    stream: &mut OwnedWriteHalf,
    outbound_rx: &mut UnboundedReceiver<OutboundMsg>,
    num_sessions: usize,
) -> io::Result<()> {
    let mut buffered_msgs = 0;
    let mut buf = BytesMut::with_capacity(BUFFER_CAPACITY);
    let mut ticker = time::interval(Duration::from_millis(FLUSH_INTERVAL_MS));
    ticker.reset();

    loop {
        tokio::select! {
            maybe_msg = outbound_rx.recv() => {
                match maybe_msg {
                    Some((session_id, msg)) => {
                        buffered_msgs += 1;
                        buf.extend_from_slice(&session_id.0.to_le_bytes());
                        buf.extend_from_slice(&msg);
                        if buf.len() >= BUFFER_CAPACITY || buffered_msgs == num_sessions {
                            if buf.len() >= BUFFER_CAPACITY {
                                metrics::counter!("network::flush_reason::buf_len").increment(1);
                            } else if buffered_msgs == num_sessions {
                                metrics::counter!("network::flush_reason::msg_count").increment(1);
                            }
                            ticker.reset();
                            if let Err(e) = flush_buf(stream, &mut buf, &mut buffered_msgs).await {
                                tracing::error!(error=%e, "Failed to flush buffer on outbound_rx");
                                return Err(e);
                            }
                        }
                    }
                    None => {
                        if !buf.is_empty() {
                            if let Err(e) = flush_buf(stream, &mut buf, &mut buffered_msgs).await {
                                tracing::error!(error=%e, "Failed to flush buffer when outbound_rx closed");
                                return Err(e);
                            }
                        }
                        // the channel will not receive any more commands
                       tracing::debug!("outbound_rx closed");
                       return Ok(());
                    }
                }
            }
            _ = ticker.tick() => {
                if !buf.is_empty() {
                    metrics::counter!("network::flush_reason::timeout").increment(1);
                    if let Err(e) = flush_buf(stream, &mut buf, &mut buffered_msgs).await {
                        tracing::error!(error=%e, "Failed to flush buffer on tick()");
                       return Err(e);
                    }
                }
            }
        }
    }
}

/// Inbound: read from the socket and send to tx.
async fn handle_inbound_traffic(
    reader: &mut BufReader<OwnedReadHalf>,
    inbound_tx: &HashMap<SessionId, UnboundedSender<NetworkMsg>>,
) -> io::Result<()> {
    let mut buf = vec![0u8; READ_BUF_SIZE];

    loop {
        let mut buf_offset = 0;

        // first read the session id. this does not get passes to the next layer.
        let mut session_id_buf = [0u8; 4];
        reader.read_exact(&mut session_id_buf).await?;
        let session_id = u32::from_le_bytes(session_id_buf);

        // then read the descriptor byte
        reader.read_exact(&mut buf[..1]).await?;
        buf_offset += 1;

        // depending on the descriptor, read the length field too
        let nd: DescriptorByte = buf[0]
            .try_into()
            .map_err(|_e| io::Error::new(io::ErrorKind::Other, "invalid descriptor byte"))?;
        // base_len includes the descriptor byte
        let base_len = nd.base_len();
        let total_len: usize = if matches!(
            nd,
            DescriptorByte::VecRing16
                | DescriptorByte::VecRing32
                | DescriptorByte::VecRing64
                | DescriptorByte::NetworkVec
        ) {
            reader.read_exact(&mut buf[1..5]).await?;
            buf_offset += 4;
            let payload_len = u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
            base_len + payload_len
        } else {
            base_len
        };

        // then read the rest of the message
        if buf.len() < total_len {
            buf.resize(total_len, 0);
        }
        reader.read_exact(&mut buf[buf_offset..total_len]).await?;

        // forward the message to the correct session.
        if let Some(ch) = inbound_tx.get(&SessionId::from(session_id)) {
            if ch.send(buf[..total_len].to_vec()).is_err() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    "failed to forward message",
                ));
            }
        } else {
            tracing::warn!(
                "failed to forward message for {:?} - channel not found",
                session_id
            );
        }
    }
}

/// Helper to write & flush, then clear the buffer
async fn flush_buf(
    writer: &mut OwnedWriteHalf,
    buf: &mut BytesMut,
    buffered_msgs: &mut usize,
) -> io::Result<()> {
    metrics::histogram!("network::buffered_msgs").record(*buffered_msgs as f64);
    metrics::histogram!("network::bytes_flushed").record(buf.len() as f64);
    *buffered_msgs = 0;
    writer.write_all(buf).await?;
    writer.flush().await?;
    buf.clear();
    Ok(())
}
