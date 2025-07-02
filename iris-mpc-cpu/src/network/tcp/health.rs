use eyre::{eyre, Result};
use libc::{c_int, socklen_t};
use std::mem::{self, size_of};
use std::os::fd::RawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Once;
use std::sync::{Arc, LazyLock};
use std::time::Instant;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio::time::{interval, Duration};

static SOCKET_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

static LOG_CH: LazyLock<LogCh> = LazyLock::new(|| {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    LogCh {
        tx,
        rx: Arc::new(Mutex::new(rx)),
    }
});

struct LogCh {
    tx: UnboundedSender<LogData>,
    rx: Arc<Mutex<UnboundedReceiver<LogData>>>,
}

struct LogData {
    time_ms: u128,
    socket_id: u32,
    snd_buf: i32,
    rcv_buf: i32,
    info: tcp_info,
}

pub async fn log_from_channel_to_file(filename: String) -> Result<()> {
    let mut file = File::create(filename.clone())
        .await
        .map_err(|e| eyre!("Failed to create log file {}: {}", filename, e))?;
    file.write_all(
        b"time_ms,socket_id,snd_buf,rcv_buf,unacked,bytes_acked,bytes_sent,bytes_received,busy_time,rwnd_limited,sndbuf_limited,snd_wnd,rcv_wnd,notsent_bytes,delivery_rate\n",
    )
    .await?;

    let mut rx = LOG_CH.rx.lock().await;
    while let Some(log) = rx.recv().await {
        let line = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            log.time_ms,
            log.socket_id,
            log.snd_buf,
            log.rcv_buf,
            log.info.tcpi_unacked,
            log.info.tcpi_bytes_acked,
            log.info.tcpi_bytes_sent,
            log.info.tcpi_bytes_received,
            log.info.tcpi_busy_time,
            log.info.tcpi_rwnd_limited,
            log.info.tcpi_sndbuf_limited,
            log.info.tcpi_snd_wnd,
            log.info.tcpi_rcv_wnd,
            log.info.tcpi_notsent_bytes,
            log.info.tcpi_delivery_rate,
        );
        file.write_all(line.as_bytes()).await?;
        file.flush().await?;
    }
    Ok(())
}

pub async fn watch_socket(fd: RawFd) -> Result<()> {
    static LOG_ONCE: Once = Once::new();
    LOG_ONCE.call_once(|| {
        tokio::spawn(async {
            let mut idx = 0;
            let candidate = loop {
                let candidate = format!("tcp_stats{}.csv", idx);
                if !Path::new(&candidate).exists() {
                    break candidate;
                }
                idx += 1;
            };
            if let Err(e) = log_from_channel_to_file(candidate).await {
                tracing::error!("failed to log tcp stats: {e}");
            }
        });
    });
    let tx_ch = LOG_CH.tx.clone();
    let socket_id = SOCKET_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tick_ms = 100;

    // these are not expected to change
    let snd = get_snd_buf(fd)?;
    let rcv = get_rcv_buf(fd)?;

    let mut ticker = interval(Duration::from_millis(tick_ms));
    let start_time = Instant::now();

    loop {
        ticker.tick().await;
        let ti = get_tcp_info(fd)?;

        let _ = tx_ch.send(LogData {
            time_ms: start_time.elapsed().as_millis(),
            snd_buf: snd,
            rcv_buf: rcv,
            socket_id,
            info: ti,
        });
    }
}

fn get_tcp_info(fd: RawFd) -> Result<tcp_info> {
    let (r, ti) = getsockopt(fd, libc::IPPROTO_TCP, libc::TCP_INFO);
    match r {
        0 => Ok(ti),
        _ => Err(eyre!("getsockopt TCP_INFO failed")),
    }
}

fn get_snd_buf(fd: RawFd) -> Result<c_int> {
    let (r, sndbuf) = getsockopt(fd, libc::SOL_SOCKET, libc::SO_SNDBUF);
    match r {
        0 => Ok(sndbuf),
        _ => Err(eyre::eyre!("getsockopt SO_SNDBUF failed")),
    }
}

fn get_rcv_buf(fd: RawFd) -> Result<c_int> {
    let (r, rcvbuf) = getsockopt(fd, libc::SOL_SOCKET, libc::SO_RCVBUF);
    match r {
        0 => Ok(rcvbuf),
        _ => Err(eyre::eyre!("getsockopt SO_RCVBUF failed")),
    }
}

fn getsockopt<T>(fd: RawFd, level: c_int, opt: c_int) -> (c_int, T) {
    let mut val: T = unsafe { mem::zeroed() };
    let mut len = size_of::<T>() as socklen_t;
    let p = &mut val as *mut T;
    let ret = unsafe { libc::getsockopt(fd, level, opt, p as *mut _, &mut len) };
    (ret, val)
}

// libc::tcp_info doesn't have all the metrics. making my own version of it here.
// reflects the tcp_info struct found in /usr/include/linux/tcp.h
// the size is 248 bytes. seems like there is little to gain from removing a few unnecessary fields.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct tcp_info {
    /// Current state of the TCP connection (e.g., ESTABLISHED, SYN_SENT).
    pub tcpi_state: u8,
    /// Current congestion avoidance state.
    pub tcpi_ca_state: u8,
    /// Number of retransmissions.
    pub tcpi_retransmits: u8,
    /// Number of probe packets sent.
    pub tcpi_probes: u8,
    /// Backoff value for retransmissions.
    pub tcpi_backoff: u8,
    /// Enabled TCP options (bitmask).
    pub tcpi_options: u8,
    /// Send and receive window scale (4 bits each, packed).
    /// multiply the tcp window by 1 << scale_factor
    pub tcpi_snd_wscale_rcv_wscale: u8,
    /// Delivery rate app-limited and fastopen client fail (1 and 2 bits, packed).
    pub tcpi_delivery_rate_app_limited_fastopen_client_fail: u8,

    /// Retransmission timeout (usec).
    pub tcpi_rto: u32,
    /// Delayed acknowledgment timeout (usec).
    pub tcpi_ato: u32,
    /// Maximum segment size for sending.
    pub tcpi_snd_mss: u32,
    /// Maximum segment size for receiving.
    pub tcpi_rcv_mss: u32,

    /// Number of unacknowledged packets.
    pub tcpi_unacked: u32,
    /// Number of selectively acknowledged packets.
    pub tcpi_sacked: u32,
    /// Number of lost packets.
    pub tcpi_lost: u32,
    /// Number of retransmitted packets.
    pub tcpi_retrans: u32,
    /// Number of forward acknowledgment packets.
    pub tcpi_fackets: u32,

    /// Time since last data was sent (ms).
    pub tcpi_last_data_sent: u32,
    /// Time since last acknowledgment was sent (ms).
    pub tcpi_last_ack_sent: u32,
    /// Time since last data was received (ms).
    pub tcpi_last_data_recv: u32,
    /// Time since last acknowledgment was received (ms).
    pub tcpi_last_ack_recv: u32,

    /// Path maximum transmission unit.
    pub tcpi_pmtu: u32,
    /// Receive slow start threshold.
    pub tcpi_rcv_ssthresh: u32,
    /// Smoothed round-trip time (usec).
    pub tcpi_rtt: u32,
    /// RTT variance (usec).
    pub tcpi_rttvar: u32,
    /// Send slow start threshold.
    pub tcpi_snd_ssthresh: u32,
    /// Size of the send congestion window.
    pub tcpi_snd_cwnd: u32,
    /// Advertised maximum segment size.
    pub tcpi_advmss: u32,
    /// Packet reordering metric.
    pub tcpi_reordering: u32,

    /// Receiver RTT estimate (usec).
    pub tcpi_rcv_rtt: u32,
    /// Receiver buffer space.
    pub tcpi_rcv_space: u32,

    /// Total number of retransmissions.
    pub tcpi_total_retrans: u32,

    /// Current pacing rate (bytes/sec).
    pub tcpi_pacing_rate: u64,
    /// Maximum pacing rate (bytes/sec).
    pub tcpi_max_pacing_rate: u64,
    /// Total bytes acknowledged.
    pub tcpi_bytes_acked: u64,
    /// Total bytes received.
    pub tcpi_bytes_received: u64,
    /// Total outgoing TCP segments.
    pub tcpi_segs_out: u32,
    /// Total incoming TCP segments.
    pub tcpi_segs_in: u32,

    /// Number of bytes not yet sent.
    pub tcpi_notsent_bytes: u32,
    /// Minimum observed RTT (usec).
    pub tcpi_min_rtt: u32,
    /// Number of incoming data segments.
    pub tcpi_data_segs_in: u32,
    /// Number of outgoing data segments.
    pub tcpi_data_segs_out: u32,

    /// Current delivery rate (bytes/sec).
    pub tcpi_delivery_rate: u64,

    /// Time spent busy sending data (usec).
    pub tcpi_busy_time: u64,
    /// Time limited by receiver window (usec).
    pub tcpi_rwnd_limited: u64,
    /// Time limited by sender buffer (usec).
    pub tcpi_sndbuf_limited: u64,

    /// Number of packets delivered.
    pub tcpi_delivered: u32,
    /// Number of packets delivered with CE (Congestion Experienced) mark.
    pub tcpi_delivered_ce: u32,

    /// Total bytes sent.
    pub tcpi_bytes_sent: u64,
    /// Total bytes retransmitted.
    pub tcpi_bytes_retrans: u64,
    /// Number of duplicate SACKs received.
    pub tcpi_dsack_dups: u32,
    /// Number of times reordering was detected.
    pub tcpi_reord_seen: u32,

    /// Number of out-of-order packets received.
    pub tcpi_rcv_ooopack: u32,

    /// Current send window size.
    pub tcpi_snd_wnd: u32,
    /// Current receive window size.
    pub tcpi_rcv_wnd: u32,

    /// Rehash value for the connection.
    pub tcpi_rehash: u32,

    /// Total number of RTO (retransmission timeout) events.
    pub tcpi_total_rto: u16,
    /// Total number of RTO recoveries.
    pub tcpi_total_rto_recoveries: u16,
    /// Total time spent in RTO (ms).
    pub tcpi_total_rto_time: u32,
}
