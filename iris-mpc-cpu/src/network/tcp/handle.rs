use super::{
    session::TcpSession, OutStream, OutboundMsg, PeerConnections, StreamId, TcpConnection,
};
use crate::{
    execution::{player::Identity, session::SessionId},
    network::{
        tcp::{networking::connection_builder::Reconnector, TcpConfig},
        value::{DescriptorByte, NetworkValue},
    },
};
use bytes::BytesMut;
use eyre::Result;
use itertools::Itertools;
use std::{collections::HashMap, time::Instant};
use std::{io, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    net::tcp::{OwnedReadHalf, OwnedWriteHalf},
    sync::{
        mpsc::{self, error::TryRecvError, UnboundedReceiver, UnboundedSender},
        oneshot, Mutex,
    },
    task::JoinSet,
};

const FLUSH_INTERVAL_US: u64 = 500;
const BUFFER_CAPACITY: usize = 2 * 1024 * 1024;
const READ_BUF_SIZE: usize = BUFFER_CAPACITY;

#[derive(Default)]
struct SessionChannels {
    outbound_tx: HashMap<Identity, HashMap<StreamId, UnboundedSender<OutboundMsg>>>,
    outbound_rx: HashMap<Identity, HashMap<StreamId, UnboundedReceiver<OutboundMsg>>>,
    inbound_tx: HashMap<Identity, HashMap<SessionId, UnboundedSender<NetworkValue>>>,
    inbound_rx: HashMap<Identity, HashMap<SessionId, UnboundedReceiver<NetworkValue>>>,
}

/// spawns a task for each TCP connection (there are x connections per peer and each of the x
/// connections has y sessions, of the same session id)
pub struct TcpNetworkHandle {
    peers: Vec<Identity>,
    ch_map: HashMap<Identity, HashMap<StreamId, UnboundedSender<Cmd>>>,
    reconnector: Reconnector,
    config: TcpConfig,
    next_session_id: usize,
}

enum Cmd {
    NewSessions {
        inbound_forwarder: HashMap<SessionId, OutStream>,
        outbound_rx: UnboundedReceiver<OutboundMsg>,
        rsp: oneshot::Sender<()>,
    },
    #[cfg(test)]
    TestReconnect { rsp: oneshot::Sender<Result<()>> },
}

impl TcpNetworkHandle {
    pub fn new(reconnector: Reconnector, connections: PeerConnections, config: TcpConfig) -> Self {
        tracing::info!("TcpNetworkHandle with config: {:?}", config);
        let peers = connections.keys().cloned().collect();
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
                    config.get_sessions_for_stream(&stream_id),
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
            next_session_id: 0,
        }
    }

    pub async fn make_sessions(&mut self) -> Result<Vec<TcpSession>> {
        tracing::debug!("make_sessions");
        self.reconnector.wait_for_reconnections().await?;
        let sc = make_channels(&self.peers, &self.config, self.next_session_id);
        Ok(self.make_sessions_inner(sc).await)
    }

    #[cfg(test)]
    pub async fn test_reconnect(&self, peer: Identity, stream_id: StreamId) -> Result<()> {
        let ch = self
            .ch_map
            .get(&peer)
            .and_then(|m| m.get(&stream_id))
            .unwrap();
        let (rsp_tx, rsp_rx) = oneshot::channel();
        ch.send(Cmd::TestReconnect { rsp: rsp_tx }).unwrap();
        rsp_rx.await?
    }

    async fn make_sessions_inner(&mut self, mut sc: SessionChannels) -> Vec<TcpSession> {
        let num_connections = self.config.num_connections;
        let max_sessions_per_connection = self.config.max_sessions_per_connection;
        let request_parallelism = self.config.request_parallelism;

        // spawn the forwarders
        for peer_id in &self.peers {
            for stream_id in (0..num_connections).map(|x| StreamId::from(x as u32)) {
                let outbound_rx = sc
                    .outbound_rx
                    .get_mut(peer_id)
                    .unwrap()
                    .remove(&stream_id)
                    .unwrap();

                let inbound_forwarder = sc.inbound_tx.get(peer_id).cloned().unwrap();
                let ch = self.ch_map.get(peer_id).unwrap().get(&stream_id).unwrap();
                let (rsp_tx, rsp_rx) = oneshot::channel();
                ch.send(Cmd::NewSessions {
                    inbound_forwarder,
                    outbound_rx,
                    rsp: rsp_tx,
                })
                .unwrap();
                rsp_rx.await.unwrap();
            }
        }

        // create the sessions
        let mut sessions = vec![];
        for (idx, session_id) in (self.next_session_id..self.next_session_id + request_parallelism)
            .map(|x| SessionId::from(x as u32))
            .enumerate()
        {
            let mut tx_map = HashMap::new();
            let mut rx_map = HashMap::new();
            let stream_id = StreamId::from((idx / max_sessions_per_connection) as u32);

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

        // ensures that if new sessions are created, the setup process (which requires communication with peers) isn't
        // interfered with by communications from old sessions.
        self.next_session_id = self.next_session_id.saturating_add(request_parallelism);
        if self.next_session_id > 1 << 30 {
            self.next_session_id = 0;
        }
        assert_eq!(self.config.request_parallelism, sessions.len());
        sessions
    }
}

fn make_channels(
    peers: &[Identity],
    config: &TcpConfig,
    next_session_id: usize,
) -> SessionChannels {
    let mut sc = SessionChannels::default();

    for peer_id in peers {
        let mut outbound_tx = HashMap::new();
        let mut outbound_rx = HashMap::new();
        let mut inbound_tx = HashMap::new();
        let mut inbound_rx = HashMap::new();

        let mut session_ids = (next_session_id..next_session_id + config.request_parallelism)
            .map(|x| SessionId::from(x as u32))
            .chunks(config.max_sessions_per_connection)
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for (stream_id, sessions) in (0..config.num_connections)
            .map(|x| StreamId::from(x as u32))
            .zip(session_ids.drain(..))
        {
            // insert one pair of outbound channels per TcpStream
            let (tx, rx) = mpsc::unbounded_channel::<OutboundMsg>();
            outbound_tx.insert(stream_id, tx);
            outbound_rx.insert(stream_id, rx);

            // insert one pair of inbound channels per stream_parallelism
            for session_id in sessions {
                let (tx, rx) = mpsc::unbounded_channel::<NetworkValue>();
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
    mut cmd_ch: UnboundedReceiver<Cmd>,
) {
    let TcpConnection {
        peer,
        stream,
        stream_id,
    } = connection;

    let (reader, writer) = stream.into_split();
    // these are made so that the stream can be dropped for testing purposes.
    let reader = Arc::new(Mutex::new(Some(BufReader::new(reader))));
    let writer = Arc::new(Mutex::new(Some(writer)));

    // We enter the loop only after receiving the first NewSessions
    let (inbound_forwarder, outbound_rx) = match cmd_ch.recv().await {
        Some(Cmd::NewSessions {
            inbound_forwarder,
            outbound_rx,
            rsp,
        }) => {
            let _ = rsp.send(());
            (inbound_forwarder, outbound_rx)
        }
        #[cfg(test)]
        Some(_) => {
            tracing::error!("invalid command received. expected NewSessions");
            return;
        }
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
        let mut set = spawn_forwarders(
            reader.clone(),
            writer.clone(),
            inbound_forwarder.clone(),
            outbound_rx.clone(),
            num_sessions,
        );

        enum Evt {
            Cmd(Cmd),
            Disconnected,
        }
        let event = tokio::select! {
            maybe_cmd = cmd_ch.recv() => {
                match maybe_cmd {
                    Some(cmd) => Evt::Cmd(cmd),
                    None => {
                        tracing::debug!("cmd channel closed");
                        return;
                    }
                }
            }
            _ = set.join_next() => Evt::Disconnected,
        };

        // shut down the forwarders
        tracing::info!("shutting down tasks for {:?}: {:?}", peer, stream_id);
        set.shutdown().await;

        // update the Arcs depending on the event. wait for reconnect if needed.
        match event {
            Evt::Cmd(cmd) => match cmd {
                Cmd::NewSessions {
                    inbound_forwarder: tx,
                    outbound_rx: rx,
                    rsp,
                } => {
                    tracing::debug!("updating sessions for {:?}: {:?}", peer, stream_id);
                    *inbound_forwarder.lock().await = tx;
                    *outbound_rx.lock().await = rx;
                    let _ = rsp.send(());
                    continue;
                }
                #[cfg(test)]
                Cmd::TestReconnect { rsp } => {
                    if let Err(e) =
                        reconnect_and_replace(&reconnector, &peer, stream_id, &reader, &writer)
                            .await
                    {
                        tracing::error!("reconnect failed: {e:?}");
                        return;
                    }
                    rsp.send(Ok(())).unwrap();
                }
            },
            Evt::Disconnected => {
                tracing::info!("reconnecting to {:?}: {:?}", peer, stream_id);
                if let Err(e) =
                    reconnect_and_replace(&reconnector, &peer, stream_id, &reader, &writer).await
                {
                    tracing::error!("reconnect failed: {e:?}");
                    return;
                }
            }
        }
    }
}

#[allow(clippy::type_complexity)]
fn spawn_forwarders(
    reader: Arc<Mutex<Option<BufReader<OwnedReadHalf>>>>,
    writer: Arc<Mutex<Option<OwnedWriteHalf>>>,
    inbound_forwarder: Arc<Mutex<HashMap<SessionId, UnboundedSender<NetworkValue>>>>,
    outbound_rx: Arc<Mutex<UnboundedReceiver<(SessionId, NetworkValue)>>>,
    num_sessions: usize,
) -> JoinSet<()> {
    let mut join_set = JoinSet::new();

    let writer = writer.clone();
    let outbound_rx = outbound_rx.clone();
    join_set.spawn(async move {
        let mut writer = writer.lock().await;
        let mut outbound_rx = outbound_rx.lock().await;
        let r =
            handle_outbound_traffic(writer.as_mut().unwrap(), &mut outbound_rx, num_sessions).await;
        tracing::debug!("handle_outbound_traffic exited: {r:?}");
    });

    let reader = reader.clone();
    let inbound_forwarder = inbound_forwarder.clone();
    join_set.spawn(async move {
        let mut reader = reader.lock().await;
        let inbound = inbound_forwarder.lock().await;
        let r = handle_inbound_traffic(reader.as_mut().unwrap(), &inbound).await;
        tracing::debug!("handle_inbound_traffic exited: {r:?}");
    });

    join_set
}

async fn reconnect_and_replace(
    reconnector: &Reconnector,
    peer: &Identity,
    stream_id: StreamId,
    reader: &Arc<Mutex<Option<BufReader<OwnedReadHalf>>>>,
    writer: &Arc<Mutex<Option<OwnedWriteHalf>>>,
) -> Result<(), eyre::Report> {
    let old_writer = writer.lock().await.take();
    let old_reader = reader.lock().await.take().map(|br| br.into_inner());
    drop(old_writer);
    drop(old_reader);

    let stream = reconnector.reconnect(peer.clone(), stream_id).await?;
    let (r, w) = stream.into_split();

    reader.lock().await.replace(BufReader::new(r));
    writer.lock().await.replace(w);

    Ok(())
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
    while let Some((session_id, msg)) = outbound_rx.recv().await {
        let _wakeup_time = Instant::now();
        buffered_msgs += 1;
        buf.extend_from_slice(&session_id.0.to_le_bytes());
        msg.serialize(&mut buf);

        let loop_start_time = Instant::now();
        while buffered_msgs < num_sessions {
            match outbound_rx.try_recv() {
                Ok((session_id, msg)) => {
                    buffered_msgs += 1;
                    buf.extend_from_slice(&session_id.0.to_le_bytes());
                    msg.serialize(&mut buf);
                    if buf.len() >= BUFFER_CAPACITY {
                        break;
                    }
                }
                Err(TryRecvError::Empty) => {
                    if loop_start_time.elapsed() >= Duration::from_micros(FLUSH_INTERVAL_US) {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                Err(_) => break,
            }
        }

        #[cfg(feature = "networking_metrics")]
        {
            if buf.len() >= BUFFER_CAPACITY {
                metrics::counter!("network::flush_reason::buf_len").increment(1);
            } else if buffered_msgs >= num_sessions {
                metrics::counter!("network::flush_reason::msg_count").increment(1);
            } else {
                metrics::counter!("network::flush_reason::timeout").increment(1);
            }
        }

        if let Err(e) = write_buf(stream, &mut buf, &mut buffered_msgs).await {
            tracing::error!(error=%e, "Failed to flush buffer on outbound_rx");
            return Err(e);
        }

        #[cfg(feature = "networking_metrics")]
        {
            let elapsed = _wakeup_time.elapsed().as_micros();
            metrics::histogram!("network::outbound::tx_time_us").record(elapsed as f64);
        }
    }

    if !buf.is_empty() {
        if let Err(e) = write_buf(stream, &mut buf, &mut buffered_msgs).await {
            tracing::error!(error=%e, "Failed to flush buffer when outbound_rx closed");
            return Err(e);
        }
    }
    // the channel will not receive any more commands
    tracing::debug!("outbound_rx closed");
    Ok(())
}

/// Inbound: read from the socket and send to tx.
async fn handle_inbound_traffic(
    reader: &mut BufReader<OwnedReadHalf>,
    inbound_tx: &HashMap<SessionId, UnboundedSender<NetworkValue>>,
) -> io::Result<()> {
    let mut buf = vec![0u8; READ_BUF_SIZE];

    loop {
        let mut buf_offset = 0;

        // first read the session id. this does not get passed to the next layer.
        let mut session_id_buf = [0u8; 4];
        reader.read_exact(&mut session_id_buf).await?;
        let _rx_start = Instant::now();
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

        #[cfg(feature = "networking_metrics")]
        {
            let elapsed = _rx_start.elapsed().as_micros();
            metrics::histogram!("network::inbound::rx_time_us").record(elapsed as f64);
        }
        // forward the message to the correct session.
        if let Some(ch) = inbound_tx.get(&SessionId::from(session_id)) {
            let nv = match NetworkValue::deserialize(&buf[..total_len]) {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!("failed to deserialize message: {e}");
                    continue;
                }
            };
            if ch.send(nv).is_err() {
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
async fn write_buf(
    writer: &mut OwnedWriteHalf,
    buf: &mut BytesMut,
    buffered_msgs: &mut usize,
) -> io::Result<()> {
    #[cfg(feature = "networking_metrics")]
    {
        metrics::histogram!("network::buffered_msgs").record(*buffered_msgs as f64);
        metrics::histogram!("network::bytes_flushed").record(buf.len() as f64);
    }
    *buffered_msgs = 0;
    writer.write_all(buf).await?;
    buf.clear();
    Ok(())
}
