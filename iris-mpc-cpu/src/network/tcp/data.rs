use crate::{
    execution::{player::Identity, session::SessionId},
    network::{tcp::NetworkConnection, value::NetworkValue},
};
use eyre::{bail, Result};
use socket2::{SockRef, TcpKeepalive};
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::{net::TcpStream, sync::mpsc};

// session multiplexing over a socket requires a SessionId
pub type OutboundMsg = (SessionId, NetworkValue);
pub type OutStream = mpsc::UnboundedSender<OutboundMsg>;
pub type InStream = mpsc::UnboundedReceiver<NetworkValue>;

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

pub struct PeerConnections<T: NetworkConnection + 'static> {
    peers: [Arc<Peer>; 2],
    c0: Vec<T>,
    c1: Vec<T>,
}

impl<T: NetworkConnection + 'static> PeerConnections<T> {
    pub fn new(peers: [Arc<Peer>; 2], c0: Vec<T>, c1: Vec<T>) -> Self {
        Self { peers, c0, c1 }
    }

    pub async fn sync(&mut self) -> Result<()> {
        let all_conns = self.c0.iter_mut().chain(self.c1.iter_mut());
        let _replies = futures::future::join_all(all_conns.map(send_and_receive))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    pub fn peer_ids(&self) -> Vec<Identity> {
        self.peers.iter().map(|peer| peer.id.clone()).collect()
    }
}

impl<T: NetworkConnection + 'static> IntoIterator for PeerConnections<T> {
    type Item = (Identity, Vec<T>);
    type IntoIter = std::vec::IntoIter<(Identity, Vec<T>)>;

    fn into_iter(self) -> Self::IntoIter {
        vec![
            (self.peers[0].id.clone(), self.c0),
            (self.peers[1].id.clone(), self.c1),
        ]
        .into_iter()
    }
}

// ensure all peers are connected to each other.
async fn send_and_receive<T: NetworkConnection>(conn: &mut T) -> Result<()> {
    let snd_buf: [u8; 3] = [2, b'o', b'k'];
    let mut rcv_buf = [0_u8; 3];
    conn.write_all(&snd_buf).await?;
    conn.flush().await?;
    conn.read_exact(&mut rcv_buf).await?;
    if rcv_buf != snd_buf {
        bail!("ok failed");
    }
    Ok(())
}
