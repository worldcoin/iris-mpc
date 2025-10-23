pub mod client; // trait for initiating a connection. hides details of TCP vs TLS
mod connection_state;
mod handshake;
mod listener; // accept inbound connections
pub mod server; // trait for accepting connections. hides details of TCP vs TLS // used to determine the peer id and connection id

pub use connection_state::ConnectionState;
pub use listener::{accept_loop, ConnectionRequest};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    execution::player::Identity,
    network::tcp::{
        data::{ConnectionId, Peer},
        Client, NetworkConnection,
    },
};
use eyre::Result;
use std::{sync::Arc, time::Duration};
use tokio::{
    io::AsyncReadExt,
    sync::{mpsc::UnboundedSender, oneshot},
    time::sleep,
};

// connect and perform handshake
pub async fn connect<T: NetworkConnection + 'static, C: Client<Output = T> + 'static>(
    connection_id: ConnectionId,
    own_id: Arc<Identity>,
    peer: Arc<Peer>,
    connection_state: ConnectionState,
    client: C,
    conn_cmd_tx: UnboundedSender<ConnectionRequest<T>>,
) -> Result<T> {
    let peer_id = peer.id().clone();
    let connector = Connector {
        connection_id,
        own_id: own_id.clone(),
        peer,
        connection_state,
        client,
        conn_req_tx: conn_cmd_tx,
    };
    let (rsp_tx, rsp_rx) = oneshot::channel();
    tokio::spawn(async move {
        if let Some(c) = connector.run().await {
            let _ = rsp_tx.send(c);
        }
    });
    let r = rsp_rx.await?;
    tracing::debug!(
        "connection succeeded for {:?} -> {:?}, {:?}",
        own_id,
        peer_id,
        connection_id
    );
    Ok(r)
}

struct Connector<T: NetworkConnection, C: Client> {
    connection_id: ConnectionId,
    own_id: Arc<Identity>,
    peer: Arc<Peer>,
    connection_state: ConnectionState,
    // initiates the connection
    client: C,
    // listens for the connection
    conn_req_tx: UnboundedSender<ConnectionRequest<T>>,
}

impl<T: NetworkConnection, C: Client<Output = T>> Connector<T, C> {
    async fn connect(&self) -> Result<T> {
        if &*self.own_id > self.peer.id() {
            let mut stream = self.client.connect(self.peer.url().to_string()).await?;
            handshake::outbound(&mut stream, &self.own_id, &self.connection_id).await?;
            let mut rsp = [0; 3];
            let n = stream.read(&mut rsp[..]).await?;
            if n != rsp.len() || &rsp != b"2ok" {
                Err(eyre::eyre!("handshake not accepted: rsp={:?}", rsp))
            } else {
                Ok(stream)
            }
        } else {
            let (rsp_tx, rsp_rx) = oneshot::channel();
            let req = ConnectionRequest::new(self.peer.id().clone(), self.connection_id, rsp_tx);
            let _ = self.conn_req_tx.send(req);
            let r = rsp_rx.await?;
            Ok(r)
        }
    }

    async fn connect_loop(&self) -> T {
        let mut rng: StdRng =
            StdRng::from_rng(&mut rand::thread_rng()).expect("Failed to seed RNG");

        let retry_sec = 2;

        sleep(Duration::from_millis(rng.gen_range(0..=3000))).await;

        loop {
            match self.connect().await {
                Ok(stream) => return stream,
                Err(_e) => {
                    //tracing::debug!("connect failed: {e:?}");
                }
            }

            sleep(Duration::from_secs(retry_sec)).await;
        }
    }

    async fn run(&self) -> Option<T> {
        let err_ct = self.connection_state.err_ct().await;
        let shutdown_ct = self.connection_state.shutdown_ct().await;

        tokio::select! {
            r = self.connect_loop() => {
                Some(r)
            },
            _ = err_ct.cancelled() => {
                None
            },
            _ = shutdown_ct.cancelled() => {
                 None
            }
        }
    }
}
