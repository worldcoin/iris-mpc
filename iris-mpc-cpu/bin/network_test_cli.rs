use clap::Parser;
use eyre::Result;
use std::net::SocketAddr;
use tokio::io::{stdin, AsyncBufReadExt, BufReader};

/// start the program in one of two modes: client or server
#[derive(Parser)]
#[command(name = "network_test_cli")]
#[command(about = "test X connections with Y throughput between two peers", long_about = None)]
enum StartCmd {
    /// connect to the specified peer (the server) and start a read-eval-print loop
    Client { server: String },
    /// Listen at the specified port and accept connections
    Server { listen: String },
}

#[derive(Parser)]
enum ClientReplCmd {
    /// establish X connections to the server.
    Connect {
        connections: usize,
    },
    /// execute the test for secs seconds, using up to mb bandwidth per connection
    Test {
        secs: usize,
        mb: usize,
    },
    Quit,
}

#[tokio::main]
async fn main() -> Result<()> {
    let start_cmd = StartCmd::parse();
    match start_cmd {
        StartCmd::Server { listen } => {
            let listen_addr: SocketAddr = listen.parse::<SocketAddr>()?;
            run_server(listen_addr).await?;
        }
        StartCmd::Client { server } => {
            let server_addr: SocketAddr = server.parse::<SocketAddr>()?;
            run_client(server_addr).await?;
        }
    }
    Ok(())
}

async fn run_server(listen_addr: SocketAddr) -> Result<()> {
    todo!()
}

async fn run_client(server_addr: SocketAddr) -> Result<()> {
    let stdin = BufReader::new(stdin());
    let mut lines = stdin.lines();
    while let Ok(Some(input)) = lines.next_line().await {
        match ClientReplCmd::try_parse_from(input.split_whitespace()) {
            Ok(cmd) => {
                todo!()
            }
            Err(e) => {
                println!("Parse error: {}", e);
            }
        }
    }

    todo!()
}
