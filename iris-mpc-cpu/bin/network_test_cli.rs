//! network-test-cli
//! This tool is intended to test the maximum throughput for one or more TCP connections.
//! The client logs TCP_INFO and SO_MEMINFO while the server only logs SO_MEMINFO. This information
//! should make clear when a link is overloaded and provide information needed to configure the TCP buffer sizes
//! used by the Linux kernel.

#![allow(dead_code)]
#[path = "../src/network/tcp/health.rs"]
mod health;

use clap::Parser;
use eyre::{eyre, Result};
use futures::future::try_join_all;
use health::get_tcp_info;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::{net::SocketAddr, os::fd::AsRawFd, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{self, unbounded_channel, UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, Interval};

use crate::health::{get_meminfo, sk_meminfo, tcp_info};
use std::cmp::max;

/// start the program in one of two modes: client or server
#[derive(Parser)]
#[command(name = "network_test_cli")]
#[command(about = "test X connections with Y throughput between two peers", long_about = None)]
enum Cmd {
    /// connect to the specified peer (the server) and run a test to determine the maximum throughput supported by X connections
    Client(ClientArgs),
    /// Listen at the specified port and accept connections
    Server { listen: String },
}

#[derive(clap::Args)]
struct ClientArgs {
    /// <ip address>:<socket>
    #[arg(long)]
    server: String,

    /// number of connections to establish
    #[arg(short, long)]
    connections: usize,

    /// duration in seconds for each step of the test
    #[arg(short, long)]
    step_sec: Option<u64>,

    /// the maximum number of tests to run
    #[arg(short, long)]
    num_steps: Option<usize>,

    /// MB/s to start the test at
    #[arg(short, long)]
    throughput_start: Option<u64>,

    /// if set, the throughput will be incremented by X MB every step. otherwise it doubles.
    #[arg(short, long)]
    increment: Option<u64>,

    /// send (<throughput> / sends_per_sec) every (1000ms / sends_per_sec)
    /// defaults to 10
    #[arg(long)]
    sends_per_sec: Option<u64>,
}

#[derive(Clone)]
struct ClientCmd {
    duration_sec: u64,
    throughput: u64,
    sends_per_sec: u64,
}

#[derive(Default)]
struct MetricsRsp {
    // in MB/s
    avg_delivery_rate: f64,
    // in MB
    bytes_sent: f64,
    // ms
    busy_time: f64,
    // ms
    rwnd_limited: f64,
    // ms
    sndbuf_limited: f64,
    // us
    rtt: u32,
    // us
    rtt_var: u32,
    // mb
    snd_wnd: f64,
    // packets
    snd_cwnd: u32,
    // packets
    delivered: u32,
    // packets
    delivered_ce: u32,
    app_limited: bool,
    mem_info: sk_meminfo,
}

#[tokio::main]
async fn main() -> Result<()> {
    let start_cmd = Cmd::parse();
    match start_cmd {
        Cmd::Server { listen } => {
            let listen_addr: SocketAddr = listen.parse::<SocketAddr>()?;
            run_server(listen_addr).await?;
        }
        Cmd::Client(args) => {
            run_client(args).await?;
        }
    }
    Ok(())
}

async fn run_server(listen_addr: SocketAddr) -> Result<()> {
    let listener = TcpListener::bind(listen_addr).await?;

    let num_ports = Arc::new(AtomicU64::new(0));
    let num_ports1 = num_ports.clone();
    let (tx, mut rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        let mut ticker = interval(Duration::from_millis(1000));
        let start = Instant::now();
        loop {
            ticker.tick().await;
            let num = num_ports1.load(Ordering::Relaxed) as usize;
            let mut responses: Vec<sk_meminfo> = Vec::with_capacity(num);
            for _ in 0..num {
                if let Ok(rsp) = rx.try_recv() {
                    responses.push(rsp);
                }
            }

            if responses.is_empty() {
                continue;
            }
            println!("=================================");
            println!("elapsed: {}s", start.elapsed().as_secs());
            print_meminfo(responses.iter());
            println!();
            println!("=================================");
        }
    });

    loop {
        let r = tokio::select! {
            res = listener.accept() => res,
            _ = tokio::signal::ctrl_c() => {
                println!("Ctrl+C received, shutting down server.");
                break;
            }
        };
        match r {
            Ok((tcp_stream, _)) => {
                num_ports.fetch_add(1, Ordering::Relaxed);
                tcp_stream.set_nodelay(true)?;
                let num_ports2 = num_ports.clone();
                let m_tx = tx.clone();
                tokio::spawn(async move {
                    server_task(tcp_stream, m_tx).await;
                    num_ports2.fetch_sub(1, Ordering::Relaxed);
                });
            }
            Err(e) => {
                eprintln!("accept_loop error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

async fn server_task(stream: TcpStream, metrics_tx: UnboundedSender<sk_meminfo>) {
    let fd = stream.as_raw_fd();
    let mut buf = vec![0u8; 2048];
    let mut stream = BufReader::new(stream);
    let run = async move {
        loop {
            let len = match stream.read_u32_le().await {
                Ok(n) => n as usize,
                Err(_e) => {
                    break;
                }
            };

            if buf.len() < len {
                buf.resize(len, 0);
            }

            if let Err(e) = stream.read_exact(&mut buf).await {
                eprintln!("Failed to read {} bytes: {}", len, e);
                break;
            }
        }
    };

    let log = async move {
        let mut ticker = interval(Duration::from_millis(5000));
        loop {
            ticker.tick().await;
            match get_meminfo(fd) {
                Ok(x) => {
                    let _ = metrics_tx.send(x);
                }
                Err(_) => break,
            }
        }
    };

    let tasks = vec![tokio::spawn(run), tokio::spawn(log)];
    let _ = try_join_all(tasks).await;
}

async fn run_client(args: ClientArgs) -> Result<()> {
    let server_addr: SocketAddr = args.server.parse::<SocketAddr>()?;
    let connect_futures = (0..args.connections).map(|_| async {
        let stream = TcpStream::connect(server_addr).await?;
        stream.set_nodelay(true)?;
        Ok::<_, eyre::Report>(stream)
    });
    let streams = try_join_all(connect_futures).await?;

    let mut cmd_ch = vec![];
    let (metrics_tx, mut metrics_rx) = unbounded_channel();

    for stream in streams {
        let (cmd_tx, cmd_rx) = unbounded_channel();
        let m_tx = metrics_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = client_task(stream, cmd_rx, m_tx.clone()).await {
                let _ = m_tx.send(Err(e));
            }
        });
        cmd_ch.push(cmd_tx);
    }

    let mut mb_sec = args.throughput_start.unwrap_or(10);
    let step_sec = args.step_sec.unwrap_or(3);
    let num_steps = args.num_steps.unwrap_or(10);
    let sends_per_sec = args.sends_per_sec.unwrap_or(10);

    for idx in 0..num_steps {
        let cmd = ClientCmd {
            duration_sec: step_sec,
            throughput: mb_sec,
            sends_per_sec,
        };
        for ch in &cmd_ch {
            ch.send(cmd.clone())?;
        }

        let mut rsp: Vec<MetricsRsp> = vec![];
        for _ in 0..args.connections {
            match metrics_rx.recv().await.ok_or(eyre!("channel closed"))? {
                Ok(m) => rsp.push(m),
                Err(e) => {
                    eprintln!("connection failed: {}", e);
                    break;
                }
            };
        }

        client_log(idx, num_steps, &cmd, &rsp);

        mb_sec = match args.increment {
            Some(mb) => mb_sec + mb,
            None => mb_sec * 2,
        };
    }

    println!("test finished");
    Ok(())
}

async fn client_task(
    mut stream: TcpStream,
    mut cmd_rx: UnboundedReceiver<ClientCmd>,
    metrics_tx: UnboundedSender<Result<MetricsRsp>>,
) -> Result<()> {
    let fd = stream.as_raw_fd();
    while let Some(cmd) = cmd_rx.recv().await {
        let tick_ms: u64 = 1_000 / cmd.sends_per_sec;
        let mut data =
            vec![0_u8; (cmd.throughput as usize * 1_000_000 / cmd.sends_per_sec as usize) + 4]; // +4 for the length
        let len = data.len() - 4;
        data[..4].copy_from_slice(&(len as u32).to_le_bytes());

        // want to only show metrics per step, not since the socket was open
        let s_ti = get_tcp_info(fd)?;

        let mut snd_ticker = interval(Duration::from_millis(tick_ms));

        // warm up - want to sample socket statistics when sending is already in progress
        for _ in 0..6 {
            snd_ticker.tick().await;
            stream.write_all(&data).await?;
        }

        let metrics = get_metrics(
            fd,
            &mut snd_ticker,
            ((cmd.duration_sec * 1000) / tick_ms) as usize,
            s_ti,
            Some((&mut stream, &data)),
        )
        .await?;

        metrics_tx.send(Ok(metrics))?;
    }
    stream.shutdown().await?;

    Ok(())
}

// s_ti = starting tcp_info
async fn get_metrics(
    fd: RawFd,
    interval: &mut Interval,
    steps: usize,
    s_ti: tcp_info,
    mut send_args: Option<(&mut TcpStream, &[u8])>,
) -> Result<MetricsRsp> {
    let mut samples = 0;
    let mut delivery_rate_sum = 0;
    let mut app_limited = false;
    let mut max_rmem_alloc = 0;
    let mut max_rcvbuf = 0;
    let mut max_wmem_alloc = 0;
    let mut max_sndbuf = 0;
    for _ in 0..steps {
        interval.tick().await;

        let tcp_info = get_tcp_info(fd)?;
        if tcp_info.tcpi_delivery_rate_app_limited_fastopen_client_fail & 1 == 1 {
            app_limited = true;
        }
        delivery_rate_sum += tcp_info.tcpi_delivery_rate;
        samples += 1;

        let mem_info = get_meminfo(fd)?;
        max_rmem_alloc = max(max_rmem_alloc, mem_info.rmem_alloc);
        max_rcvbuf = max(max_rcvbuf, mem_info.rcvbuf);
        max_wmem_alloc = max(max_wmem_alloc, mem_info.wmem_alloc);
        max_sndbuf = max(max_sndbuf, mem_info.sndbuf);

        if let Some((stream, data)) = send_args.as_mut() {
            stream.write_all(&(data.len() as u32).to_le_bytes()).await?;
            stream.write_all(data).await?;
        }
    }

    // wait for final send to finish
    interval.tick().await;

    let ti = get_tcp_info(fd)?;
    let mem_info = get_meminfo(fd)?;
    delivery_rate_sum += ti.tcpi_delivery_rate;
    samples += 1;

    let rsp = MetricsRsp {
        // calculate avg bytes/sec and convert to MB/s
        avg_delivery_rate: (delivery_rate_sum / samples) as f64 / 1_000_000.0,
        bytes_sent: (ti.tcpi_bytes_sent - s_ti.tcpi_bytes_sent) as f64 / 1_000_000.0,
        busy_time: (ti.tcpi_busy_time - s_ti.tcpi_busy_time) as f64 / 1000.0,
        rwnd_limited: (ti.tcpi_rwnd_limited - s_ti.tcpi_rwnd_limited) as f64 / 1000.0,
        sndbuf_limited: (ti.tcpi_sndbuf_limited - s_ti.tcpi_sndbuf_limited) as f64 / 1000.0,
        rtt: ti.tcpi_rtt,
        rtt_var: ti.tcpi_rttvar,
        snd_wnd: (ti.tcpi_snd_wnd * (1 << (s_ti.tcpi_snd_wscale_rcv_wscale & 0x11))) as f64
            / 1_000_000.0,
        snd_cwnd: ti.tcpi_snd_cwnd,
        delivered: ti.tcpi_delivered - s_ti.tcpi_delivered,
        delivered_ce: ti.tcpi_delivered_ce - s_ti.tcpi_delivered_ce,
        app_limited,
        mem_info,
    };

    Ok(rsp)
}

fn client_log(idx: usize, num_steps: usize, cmd: &ClientCmd, rsp: &[MetricsRsp]) {
    println!(
        "stats for step {}/{}, duration(s) {}, throughput(MB/s) {}",
        idx + 1,
        num_steps,
        cmd.duration_sec,
        cmd.throughput
    );
    println!("=================================");
    print_metrics(rsp);
    println!();
    println!("=================================");
    println!();
    println!();
}

macro_rules! print_list {
    ($label:expr, $slice:expr, $fmt:expr) => {{
        println!("{}", $label);
        print!("    ");
        for v in $slice {
            print!($fmt, v);
            print!(" ");
        }
        println!();
    }};
}

#[rustfmt::skip]
fn print_meminfo<'a, I>(rsp: I)
where
    I: IntoIterator<Item = &'a sk_meminfo> + Clone,
{
    print_list!("rmem_alloc",rsp.clone().into_iter().map(|m| m.rmem_alloc),"{}");
    print_list!("rcvbuf", rsp.clone().into_iter().map(|m| m.rcvbuf), "{}");
    print_list!("wmem_alloc", rsp.clone().into_iter().map(|m| m.wmem_alloc), "{}");
    print_list!("sndbuf", rsp.clone().into_iter().map(|m| m.sndbuf), "{}");
    print_list!("fwd_alloc", rsp.clone().into_iter().map(|m| m.fwd_alloc), "{}");
    print_list!("wmem_queued", rsp.clone().into_iter().map(|m| m.wmem_queued), "{}");
    print_list!("backlog", rsp.clone().into_iter().map(|m| m.backlog), "{}");
    print_list!("drops", rsp.clone().into_iter().map(|m| m.drops), "{}");
}

#[rustfmt::skip]
fn print_metrics(rsp: &[MetricsRsp]) {
    print_list!("avg_delivery_rate (MB/s):", rsp.iter().map(|m| m.avg_delivery_rate), "{:.2}");
    print_list!("bytes_sent (MB):", rsp.iter().map(|m| m.bytes_sent), "{:.2}");
    println!("---------------------------------");

    print_list!("busy_time (ms)", rsp.iter().map(|m| m.busy_time), "{:.3}");
    print_list!("rwnd_limited (ms)", rsp.iter().map(|m| m.rwnd_limited), "{:.3}");
    print_list!("sndbuf_limited (ms)", rsp.iter().map(|m| m.sndbuf_limited), "{:.3}");
    println!("---------------------------------");

    print_list!("rtt (us)", rsp.iter().map(|m| m.rtt), "{}");
    print_list!("rtt_var (us)", rsp.iter().map(|m| m.rtt_var), "{}");
    println!("---------------------------------");

    print_list!("snd_wnd (MB)", rsp.iter().map(|m| m.snd_wnd), "{:.2}");
    print_list!("snd_cwnd (packets)", rsp.iter().map(|m| m.snd_cwnd), "{}");
    println!("---------------------------------");

    print_list!("delivered", rsp.iter().map(|m| m.delivered), "{}");
    print_list!("delivered_ce", rsp.iter().map(|m| m.delivered_ce), "{}");
    print_list!("app_limited", rsp.iter().map(|m| m.app_limited), "{}");
    println!("---------------------------------");

    print_meminfo(rsp.iter().map(|x| &x.mem_info))
}
