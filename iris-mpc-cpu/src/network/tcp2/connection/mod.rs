mod client; // trait for initiating a connection. hides details of TCP vs TLS
mod connection_state;
mod handshake;
mod listener; // accept inbound connections
mod server; // trait for accepting connections. hides details of TCP vs TLS // used to determine the peer id and connection id

pub use connection_state::ConnectionState;
pub use listener::{accept_loop, ConnectionRequest};

use crate::{
    execution::player::Identity,
    network::tcp2::{data::ConnectionId, Client, NetworkConnection},
};
use eyre::Result;
use socket2::{SockRef, TcpKeepalive};
use std::{sync::Arc, time::Duration};
use tokio::{net::TcpStream, sync::mpsc};

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

pub struct Connection {
    cmd_tx: mpsc::Sender<InnerCmd>,
}

impl Connection {
    pub fn new<T: NetworkConnection + 'static, C: Client + 'static>(
        connection_id: ConnectionId,
        own_id: Arc<Identity>,
        peer: Arc<Peer>,
        connection_state: ConnectionState,
        client: C,
        conn_req_tx: mpsc::Sender<ConnectionRequest<T>>,
    ) -> Self {
        let inner = ConnectionInner {
            connection_id,
            own_id,
            peer,
            connection_state,
            client,
            conn_req_tx,
        };
        let (cmd_tx, cmd_rx) = mpsc::channel(1);
        tokio::spawn(manage_connection(inner, cmd_rx));
        Self { cmd_tx }
    }

    async fn connect(&self) {
        let _ = self.cmd_tx.send(InnerCmd::Connect).await;
    }

    async fn disconect(&self) {
        let _ = self.cmd_tx.send(InnerCmd::Close).await;
    }
}

enum InnerState {
    Idle,
    Connecting,
    Ready,
}

enum InnerCmd {
    Connect,
    Close,
}

struct ConnectionInner<T: NetworkConnection, C: Client> {
    connection_id: ConnectionId,
    own_id: Arc<Identity>,
    peer: Arc<Peer>,
    connection_state: ConnectionState,
    // initiates the connection
    client: C,
    // listens for the connection
    conn_req_tx: mpsc::Sender<ConnectionRequest<T>>,
}

impl<T: NetworkConnection, C: Client> ConnectionInner<T, C> {
    async fn connect(&self) -> T {
        todo!()
    }
}

async fn manage_connection<T: NetworkConnection, C: Client>(
    inner: ConnectionInner<T, C>,
    cmd_rx: mpsc::Receiver<InnerCmd>,
) {
    let mut inner_state = InnerState::Connecting;
}
