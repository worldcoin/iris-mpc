use clap::{Parser, ValueEnum};
use eyre::{eyre, Result};
use futures::future::try_join_all;
use std::net::SocketAddr;
use tokio::io::{stdin, AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

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
    /// the type of experiment to run
    strategy: Strategy,
    /// duration in seconds for each step of the test
    step_sec: Option<usize>,
    /// MB/s to start the test at
    throughput_start: Option<usize>,
    /// the maximum number of tests to run
    num_steps: Option<usize>,
}

#[derive(Clone, Debug, ValueEnum)]
enum Strategy {
    /// increment by X MB
    Increment,
    /// double the throughput each step
    Double,
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

async fn server_task(mut stream: TcpStream) {
    let mut buf = vec![0u8; 2048];
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

        println!(
            "stats for step {}, duration {}, throughput {}",
            idx, step_sec, mb_sec
        );
        println!("---------------------------------");

        println!("avg_delivery_rate:");
        for m in &rsp {
            print!("{} ", m.avg_delivery_rate);
        }
        println!();

        println!("busy_time");
        for m in &rsp {
            print!("{} ", m.busy_time);
        }
        println!();

        println!("rwnd_limited");
        for m in &rsp {
            print!("{} ", m.rwnd_limited);
        }
        println!();

        println!("sndbuf_limited");
        for m in &rsp {
            print!("{} ", m.sndbuf_limited);
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

#[derive(Clone)]
struct ClientCmd {
    duration_sec: usize,
    throughput: usize,
}

#[derive(Default)]
struct MetricsRsp {
    duration_sec: usize,
    desired_throughput: usize,
    // in MB/s
    avg_delivery_rate: u64,
    // usec
    busy_time: u64,
    // usec
    rwnd_limited: u64,
    // usec
    sndbuf_limited: u64,
    // packets
    delivered: u32,
    // packets
    delivered_ce: u32,
}

async fn client_task(
    stream: TcpStream,
    mut cmd_rx: UnboundedReceiver<ClientCmd>,
    metrics_tx: UnboundedSender<Result<MetricsRsp>>,
) -> Result<()> {
    while let Some(cmd) = cmd_rx.recv().await {}
}
