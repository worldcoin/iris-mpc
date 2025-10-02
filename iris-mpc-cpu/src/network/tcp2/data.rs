use crate::{
    execution::{player::Identity, session::SessionId},
    network::{
        tcp2::{Client, NetworkConnection},
        value::NetworkValue,
    },
};
use eyre::Result;
use socket2::{SockRef, TcpKeepalive};
use std::{sync::Arc, time::Duration};
use tokio::{
    net::TcpStream,
    sync::{mpsc, oneshot},
    time::sleep,
};

pub type OutboundMsg = (SessionId, NetworkValue);

#[derive(Clone, Copy, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
pub struct ConnectionId(pub u32);

impl ConnectionId {
    pub fn new(val: u32) -> Self {
        Self(val)
    }
}

impl From<u32> for ConnectionId {
    fn from(val: u32) -> Self {
        ConnectionId::new(val)
    }
}

/// set no_delay and keepalive
pub fn configure_tcp_stream(stream: &TcpStream) -> Result<()> {
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

pub struct Peer {
    id: Identity,
    url: String,
}

impl Peer {
    pub fn new(id: Identity, url: String) -> Self {
        Peer { id, url }
    }

    pub fn id(&self) -> &Identity {
        &self.id
    }

    pub fn url(&self) -> &str {
        &self.url
    }
}

impl From<(Identity, String)> for Peer {
    fn from((id, url): (Identity, String)) -> Self {
        Peer::new(id, url)
    }
}
