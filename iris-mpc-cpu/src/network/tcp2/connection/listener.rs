use std::{collections::HashMap, marker::PhantomData};

use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{
    execution::player::Identity,
    network::tcp2::{connection::handshake, data::ConnectionId, NetworkConnection, Server},
};

pub struct ConnectionRequest<T: NetworkConnection> {
    peer_id: Identity,
    connection_id: ConnectionId,
    rsp: oneshot::Sender<T>,
}

impl<T: NetworkConnection> ConnectionRequest<T> {
    pub fn new(peer_id: Identity, connection_id: ConnectionId, rsp: oneshot::Sender<T>) -> Self {
        Self {
            peer_id,
            connection_id,
            rsp,
        }
    }
}

pub async fn accept_loop<T: NetworkConnection, S: Server<Output = T>>(
    id: Identity,
    listener: S,
    mut cmd_ch: mpsc::UnboundedReceiver<ConnectionRequest<T>>,
    shutdown_ct: CancellationToken,
) {
    let mut connection_requests: HashMap<Identity, HashMap<ConnectionId, oneshot::Sender<T>>> =
        HashMap::new();

    loop {
        let r = tokio::select! {
            res = listener.accept() => res,
            cmd = cmd_ch.recv() => {
                if let Some(cmd) = cmd {
                    let peer_map = connection_requests.entry(cmd.peer_id).or_insert_with(HashMap::new);
                    peer_map.insert(cmd.connection_id, cmd.rsp);
                    continue;
                } else {
                    tracing::warn!("shutting down accept loop: cmd_ch closed");
                    break;
                }
            },
            _ = shutdown_ct.cancelled() => {
                break;
            }
        };
        match r {
            Ok((peer_addr, mut stream)) => {
                tracing::trace!("{:?} accepted connection from {:?}", id, peer_addr);
                let (peer_id, connection_id) = match handshake::inbound(&mut stream).await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::debug!("application level handshake failed: {e:?}");
                        continue;
                    }
                };

                if let Some(peer_map) = connection_requests.get_mut(&peer_id) {
                    if let Some(rsp) = peer_map.remove(&connection_id) {
                        let _ = rsp.send(stream);
                    } else {
                        tracing::debug!(
                            "no pending request for connection_id {connection_id:?} from peer {peer_id:?}"
                        );
                    }
                } else {
                    tracing::debug!("no pending requests from peer {peer_id:?}");
                }
            }
            Err(e) => tracing::error!(%e, "accept_loop error"),
        }
    }
}
