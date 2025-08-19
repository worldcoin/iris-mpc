use eyre::Result;
use socket2::{SockRef, TcpKeepalive};
use std::time::Duration;
use tokio::net::TcpStream;

pub mod client;
pub mod connection_builder;
mod handshake;
pub mod server;

/// set no_delay and keepalive
fn configure_tcp_stream(stream: &TcpStream) -> Result<()> {
    let params = TcpKeepalive::new()
        // idle time before keepalives get sent. NGINX default is 60 seconds. want to be less than that.
        .with_time(Duration::from_secs(30))
        // how often to send keepalives
        .with_interval(Duration::from_secs(30))
        // how many unanswered probes before the connection is closed
        .with_retries(4);
    let socket_ref = SockRef::from(&stream);
    socket_ref.set_tcp_nodelay(true)?;
    socket_ref.set_tcp_keepalive(&params)?;
    Ok(())
}
