//! multiplexes multiple sessions over a single connection

use std::{collections::HashMap, io, time::Instant};

use bytes::BytesMut;
use iris_mpc_common::fast_metrics::FastHistogram;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf},
    sync::mpsc::{error::TryRecvError, UnboundedReceiver, UnboundedSender},
};

use crate::{
    execution::session::SessionId,
    network::{
        tcp::{connection::ConnectionState, data::OutboundMsg, NetworkConnection},
        value::{DescriptorByte, NetworkValue},
    },
};

const BUFFER_CAPACITY: usize = 32 * 1024;
const READ_BUF_SIZE: usize = 2 * 1024 * 1024;

pub async fn run<T: NetworkConnection>(
    stream: T,
    num_sessions: u32,
    connection_state: ConnectionState,
    inbound_forwarder: HashMap<SessionId, UnboundedSender<NetworkValue>>,
    outbound_rx: UnboundedReceiver<OutboundMsg>,
) {
    let shutdown_ct = connection_state.shutdown_ct().await;
    let err_ct = connection_state.err_ct().await;

    let (reader, writer) = tokio::io::split(stream);
    let reader = BufReader::new(reader);

    enum Event {
        Shutdown,
        Error,
        // the sessions may have been dropped
        // don't set the error logging flags in response to this.
        ClosedOk,
    }

    let evt = tokio::select! {
        _ = shutdown_ct.cancelled() => Event::Shutdown,
        _ = err_ct.cancelled() => Event::Error,
        r = handle_outbound_traffic(writer, outbound_rx, num_sessions) => {
            tracing::debug!("handle_outbound_traffic: {:?}", r);
            if r.is_err() {
                err_ct.cancel();
                Event::Error
            } else {
                Event::ClosedOk
            }
        },
        r = handle_inbound_traffic(reader, inbound_forwarder) => {
            tracing::debug!("handle_inbound_traffic: {:?}",  r);
            if r.is_err() {
                err_ct.cancel();
                Event::Error
            } else {
                Event::ClosedOk
            }
        },
    };

    match evt {
        Event::Shutdown => {
            if connection_state.set_exited().await {
                tracing::info!("shutting down TCP/TLS networking stack");
            }
        }
        Event::Error => {
            if connection_state.set_cancelled().await {
                tracing::info!("closing TCP/TLS connections");
            }
        }
        Event::ClosedOk => {
            // do nothing
        }
    }
}

/// Outbound: send messages from rx to the socket.
/// the sender needs to prepend the session id to the message.
async fn handle_outbound_traffic<T: NetworkConnection>(
    mut stream: WriteHalf<T>,
    mut outbound_rx: UnboundedReceiver<OutboundMsg>,
    num_sessions: u32,
) -> io::Result<()> {
    // Time spent buffering between the first and last messages of a packet.
    let mut metrics_buffer_latency = FastHistogram::new("outbound_buffer_latency");
    // Number of messages per packet.
    let mut metrics_messages = FastHistogram::new("outbound_packet_messages");
    // Number of bytes per packet.
    let mut metrics_packet_bytes = FastHistogram::new("outbound_packet_bytes");

    let mut buf = BytesMut::with_capacity(BUFFER_CAPACITY);

    while let Some((session_id, msg)) = outbound_rx.recv().await {
        // First message of the next packet.
        let mut buffered_msgs = 1;
        buf.extend_from_slice(&session_id.0.to_le_bytes());
        msg.serialize(&mut buf);

        // Try to fill the buffer with more messages, up to num_sessions, BUFFER_CAPACITY or two attempts.
        let loop_start_time = Instant::now();
        let mut retried = false;
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
                Err(TryRecvError::Empty) if !retried => {
                    retried = true;
                    tokio::task::yield_now().await;
                }
                _ => break,
            }
        }

        metrics_buffer_latency.record(loop_start_time.elapsed().as_secs_f64());
        metrics_messages.record(buffered_msgs as f64);
        metrics_packet_bytes.record(buf.len() as f64);

        #[cfg(feature = "networking_metrics")]
        {
            if buf.len() >= BUFFER_CAPACITY {
                metrics::counter!("network.flush_reason.buf_len").increment(1);
            } else if buffered_msgs >= num_sessions {
                metrics::counter!("network.flush_reason.msg_count").increment(1);
            } else {
                metrics::counter!("network.flush_reason.timeout").increment(1);
            }
        }

        write_buf(&mut stream, &mut buf).await?
    }

    if !buf.is_empty() {
        write_buf(&mut stream, &mut buf).await?
    }
    // the channel will not receive any more commands
    tracing::debug!("outbound_rx closed");
    Ok(())
}

/// Inbound: read from the socket and send to tx.
async fn handle_inbound_traffic<T: NetworkConnection>(
    mut reader: BufReader<ReadHalf<T>>,
    inbound_tx: HashMap<SessionId, UnboundedSender<NetworkValue>>,
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
            metrics::histogram!("network.inbound.rx_time_us").record(elapsed as f64);
        }
        // forward the message to the correct session.
        if let Some(ch) = inbound_tx.get(&SessionId::from(session_id)) {
            match NetworkValue::deserialize(&buf[..total_len]) {
                Ok(nv) => {
                    if ch.send(nv).is_err() {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "failed to forward message",
                        ));
                    }
                }
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("failed to deserialize message: {e}"),
                    ));
                }
            };
        } else {
            tracing::debug!(
                "failed to forward message for {:?} - channel not found",
                session_id
            );
        }
    }
}

/// Helper to write & flush, then clear the buffer
async fn write_buf<T: NetworkConnection>(
    stream: &mut WriteHalf<T>,
    buf: &mut BytesMut,
) -> io::Result<()> {
    stream.write_all(buf).await?;
    stream.flush().await?;
    buf.clear();
    Ok(())
}
