#![allow(dead_code)]
#[path = "../src/network/tcp/health.rs"]
mod health;

use clap::Parser;
use eyre::{eyre, Result};
use futures::future::try_join_all;
use health::{get_tcp_info, tcp_info};
use std::{
    net::SocketAddr,
    os::fd::AsRawFd,
    time::{Duration, Instant},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};
use tokio::time::{interval, sleep};

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
    server: String,
    /// number of connections to establish
    connections: usize,

    /// duration in seconds for each step of the test
    step_sec: Option<u64>,
    /// the maximum number of tests to run
    num_steps: Option<usize>,

    /// MB/s to start the test at
    throughput_start: Option<usize>,
    /// if set, the throughput will be incremented by X MB every step. otherwise it doubles.
    increment: Option<usize>,
}

#[derive(Clone)]
struct ClientCmd {
    duration_sec: u64,
    throughput: usize,
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
    // packets
    delivered: u32,
    // packets
    delivered_ce: u32,
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
                tcp_stream.set_nodelay(true)?;
                tokio::spawn(server_task(tcp_stream));
            }
            Err(e) => {
                eprintln!("accept_loop error: {}", e);
                break;
            }
        }
    }

    Ok(())
}

async fn server_task(stream: TcpStream) {
    let mut buf = vec![0u8; 2048];
    let mut stream = BufReader::new(stream);
    loop {
        let len = match stream.read_u32().await {
            Ok(n) => n as usize,
            Err(e) => {
                eprintln!("Failed to read length: {}", e);
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
    let (metrics_tx, mut metrics_rx) = mpsc::unbounded_channel();

    for stream in streams {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
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

    for idx in 0..num_steps {
        let cmd = ClientCmd {
            duration_sec: step_sec,
            throughput: mb_sec,
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

        mb_sec = match args.increment {
            Some(mb) => mb_sec + mb,
            None => mb_sec * 2,
        };

        // todo: add a way to log to csv instead of the terminal

        println!(
            "stats for step {}, duration {}, throughput {}",
            idx, step_sec, mb_sec
        );
        println!("---------------------------------");

        println!("avg_delivery_rate (MB/s):");
        for m in &rsp {
            print!("{:.2} ", m.avg_delivery_rate);
        }
        println!();

        println!("bytes_sent (MB):");
        for m in &rsp {
            print!("{:.2} ", m.bytes_sent);
        }
        println!();

        println!("busy_time (ms)");
        for m in &rsp {
            print!("{:.3} ", m.busy_time);
        }
        println!();

        println!("rwnd_limited (ms)");
        for m in &rsp {
            print!("{:.3} ", m.rwnd_limited);
        }
        println!();

        println!("sndbuf_limited (ms)");
        for m in &rsp {
            print!("{:.3} ", m.sndbuf_limited);
        }
        println!();

        println!("delivered");
        for m in &rsp {
            print!("{} ", m.delivered);
        }
        println!();

        println!("delivered_ce");
        for m in &rsp {
            print!("{} ", m.delivered_ce);
        }
        println!();

        println!("");
        println!("---------------------------------");
        println!("");
        println!("");
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
    let tick_ms = 500;
    while let Some(cmd) = cmd_rx.recv().await {
        let mut last_info: Option<tcp_info> = None;
        let mut samples = 0;
        let mut delivery_rate_sum = 0;

        let start = Instant::now();
        let mut snd_ticker = interval(Duration::from_millis(tick_ms));

        let data = vec![0_u8; cmd.throughput * 1_000_000 / 100];
        for _ in 0..6 {
            snd_ticker.tick().await;
            stream.write_all(&(data.len() as u32).to_le_bytes()).await?;
            stream.write_all(&data).await?;
        }

        // want the stats to be offset from the sending
        sleep(Duration::from_millis(tick_ms / 2)).await;
        let mut stats_ticker = interval(Duration::from_millis(tick_ms));

        while start.elapsed().as_secs() < cmd.duration_sec {
            tokio::select! {
                _ = snd_ticker.tick() => {
                    stream.write_all(&(data.len() as u32).to_le_bytes()).await?;
                    stream.write_all(&data).await?;
                },
                _ = stats_ticker.tick() => {
                    let mut tcp_info = get_tcp_info(fd)?;
                    delivery_rate_sum += tcp_info.tcpi_delivery_rate;
                    samples += 1;

                    // for ease of use, only log the diff.
                    if let Some(prev) = last_info.as_ref() {
                        tcp_info.tcpi_bytes_sent -= prev.tcpi_bytes_sent;
                        tcp_info.tcpi_busy_time -= prev.tcpi_busy_time;
                        tcp_info.tcpi_rwnd_limited -= prev.tcpi_rwnd_limited;
                        tcp_info.tcpi_sndbuf_limited -= prev.tcpi_sndbuf_limited;
                        tcp_info.tcpi_delivered -= prev.tcpi_delivered;
                        tcp_info.tcpi_delivered_ce -= prev.tcpi_delivered_ce;
                    }
                    last_info.replace(tcp_info);
                }
            }
        }

        let ti = last_info.unwrap();
        let rsp = MetricsRsp {
            avg_delivery_rate: (delivery_rate_sum / samples) as f64 / 1_000_000.0,
            bytes_sent: ti.tcpi_bytes_sent as f64 / 1_000_000.0,
            busy_time: ti.tcpi_busy_time as f64 / 1000.0,
            rwnd_limited: ti.tcpi_rwnd_limited as f64 / 1000.0,
            sndbuf_limited: ti.tcpi_sndbuf_limited as f64 / 1000.0,
            delivered: ti.tcpi_delivered,
            delivered_ce: ti.tcpi_delivered_ce,
        };

        metrics_tx.send(Ok(rsp))?;
    }

    Ok(())
}
