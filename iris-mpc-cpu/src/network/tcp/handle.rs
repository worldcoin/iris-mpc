use std::sync::Arc;

use crate::{
    execution::{
        hawk_main::scheduler::parallelize,
        player::{Identity, Role, RoleAssignment},
        session::{NetworkSession, Session},
    },
    network::{
        tcp::{
            config::TcpConfig,
            connection::{accept_loop, ConnectionRequest, ConnectionState},
            data::{ConnectionId, Peer, PeerConnections},
            session::TcpSession,
            Client, NetworkConnection, NetworkHandle, Server,
        },
        value::NetworkValue,
        Networking,
    },
    protocol::ops::setup_replicated_prf,
};
use async_trait::async_trait;
use eyre::{bail, eyre, Result};
use futures::future::join_all;
use itertools::Itertools;
use rand::{thread_rng, Rng};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> {
    peers: [Arc<Peer>; 2],
    my_id: Arc<Identity>,
    connector: C,
    conn_cmd_tx: UnboundedSender<ConnectionRequest<T>>,
    connection_state: ConnectionState,
    config: TcpConfig,
    next_session_id: u32,
    shutdown_ct: CancellationToken,
    party_index: usize,
    role_assignments: Arc<RoleAssignment>,
}

impl<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> Drop
    for TcpNetworkHandle<T, C>
{
    fn drop(&mut self) {
        self.shutdown_ct.cancel();
        tracing::debug!("TcpNetworkHandle dropped");
    }
}

#[async_trait]
impl<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> NetworkHandle
    for TcpNetworkHandle<T, C>
{
    async fn make_network_sessions(&mut self) -> Result<(Vec<NetworkSession>, CancellationToken)> {
        let err_ct = CancellationToken::new();
        // cancel the old token just in case
        self.connection_state.err_ct().await.cancel();
        self.connection_state
            .replace_cancellation_token(err_ct.clone())
            .await;

        // wait for all peers to establish all connections
        let mut connections = self
            .make_peer_connections(self.config.num_connections)
            .await?;

        connections.sync().await?;

        // calls multiplexer::run() on each TCP/TLS stream
        let mut tcp_sessions = super::session::make_sessions(
            connections,
            self.connection_state.clone(),
            &self.config,
            self.next_session_id,
        )
        .await;

        self.validate_sessions(&mut tcp_sessions).await?;

        let network_sessions: Vec<_> = tcp_sessions
            .into_iter()
            .map(|tcp_session| NetworkSession {
                session_id: tcp_session.id(),
                role_assignments: self.role_assignments.clone(),
                networking: Box::new(tcp_session),
                own_role: Role::new(self.party_index),
            })
            .collect();

        let sessions_per_conn = (0..self.config.num_connections)
            .map(|idx| self.config.get_sessions_for_connection(idx))
            .collect_vec();
        tracing::info!(
            "make_sessions succeeded. starting id: {} sessions per connection: {:?}",
            self.next_session_id,
            sessions_per_conn
        );

        self.next_session_id = self
            .next_session_id
            .saturating_add(self.config.num_sessions);
        if self.next_session_id >= u32::MAX - self.config.num_sessions {
            self.next_session_id = 0;
        }

        Ok((network_sessions, err_ct))
    }

    async fn make_sessions(&mut self) -> Result<(Vec<Session>, CancellationToken)> {
        let (network_sessions, ct) = self.make_network_sessions().await?;

        let mut session_futures = vec![];
        for mut network_session in network_sessions.into_iter() {
            let fut = async move {
                let my_session_seed = thread_rng().gen();
                let prf = setup_replicated_prf(&mut network_session, my_session_seed).await?;
                Ok::<Session, eyre::Report>(Session {
                    network_session,
                    prf,
                })
            };
            session_futures.push(fut);
        }

        let r = parallelize(session_futures.into_iter()).await?;
        Ok((r, ct))
    }

    async fn sync_peers(&mut self) -> Result<()> {
        let mut connections = self
            .make_peer_connections(1)
            .await
            .map_err(|e| eyre!("make_peer_connections failed: {}", e))?;
        connections
            .sync()
            .await
            .map_err(|_| eyre!("sync connections failed"))?;
        Ok(())
    }
}

impl<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> TcpNetworkHandle<T, C> {
    #[allow(clippy::too_many_arguments)]
    pub async fn new<I, S>(
        my_id: Identity,
        mut peers: I,
        connector: C,
        listener: S,
        config: TcpConfig,
        shutdown_ct: CancellationToken,
        party_index: usize,
        role_assignments: Arc<RoleAssignment>,
    ) -> Result<Self>
    where
        I: Iterator<Item = (Identity, String)>,
        S: Server<Output = T> + 'static,
    {
        let my_id = Arc::new(my_id);
        let peers: [Arc<Peer>; 2] = [
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
        ];

        // use the shutdown_ct to cancel anything spawned by the NetworkHandle. But don't want this to affect the calling code.
        // Hence the child_token()
        let shutdown_ct = shutdown_ct.child_token();
        let connection_state = ConnectionState::new(shutdown_ct.clone(), CancellationToken::new());

        let (conn_cmd_tx, conn_cmd_rx) = mpsc::unbounded_channel::<ConnectionRequest<T>>();

        // be sure not to make more than one network handle...
        tokio::spawn(accept_loop(listener, conn_cmd_rx, shutdown_ct.clone()));

        Ok(Self {
            my_id,
            peers,
            connector,
            config,
            conn_cmd_tx,
            connection_state,
            next_session_id: 0,
            shutdown_ct,
            party_index,
            role_assignments,
        })
    }

    // associates the connections with an Identity
    async fn make_peer_connections(&self, conns_per_peer: u32) -> Result<PeerConnections<T>> {
        let (c0, c1) = self.make_connections(conns_per_peer).await?;
        Ok(PeerConnections::new(self.peers.clone(), c0, c1))
    }

    // returns the connections for each peer
    // when returned, the handshakes have successfully completed
    async fn make_connections(&self, conns_per_peer: u32) -> Result<(Vec<T>, Vec<T>)> {
        assert_eq!(self.peers.len(), 2);
        let mut connect_futures = Vec::with_capacity(conns_per_peer as usize * self.peers.len());

        // peers[0] will be associated with connections c0
        // peers[1] will be associated with connections c1
        for peer in self.peers.iter() {
            for idx in 0..conns_per_peer {
                let connection_id = ConnectionId::new(idx);
                let fut = super::connection::connect(
                    connection_id,
                    self.my_id.clone(),
                    peer.clone(),
                    self.connection_state.clone(),
                    self.connector.clone(),
                    self.conn_cmd_tx.clone(),
                );
                connect_futures.push(fut);
            }
        }
        let results: Vec<T> = join_all(connect_futures)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        let mut c1 = results;
        let mut c0 = vec![];
        c0.extend(c1.drain(0..c1.len() / 2));
        assert_eq!(c1.len(), c0.len());

        Ok((c0, c1))
    }

    async fn validate_sessions(&self, sessions: &mut [TcpSession]) -> Result<()> {
        // make sure all the sessions are working
        for (idx, session) in sessions.iter_mut().enumerate() {
            // turn the session id into a byte array.
            let mut id_arr = [0u8; 16];
            let idx_bytes = idx.to_le_bytes();
            let len = std::cmp::min(id_arr.len(), idx_bytes.len());
            id_arr[..len].copy_from_slice(&idx_bytes[..len]);

            let prf = NetworkValue::PrfKey(id_arr);
            for peer in &self.peers {
                session.send(prf.clone(), peer.id()).await?;
            }
            for peer in &self.peers {
                let r = session.receive(peer.id()).await?;
                match r {
                    NetworkValue::PrfKey(arr) => {
                        assert_eq!(id_arr, arr);
                    }
                    _ => bail!("invalid msg received in validate_sessions()"),
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;

    use eyre::Result;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::task::JoinSet;

    use tracing_test::traced_test;

    use crate::execution::local::generate_local_identities;

    use crate::network::tcp::TcpStreamConn;

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_tcp_network_handle() -> Result<()> {
        const CONNECTIONS_PER_PEER: u32 = 2;

        let identities = generate_local_identities();
        let handles = get_local_tcp_handles(identities, CONNECTIONS_PER_PEER as usize, 1).await?;

        // for each peer, a vec of the connections to other peers
        let connections: Vec<(Vec<TcpStreamConn>, Vec<TcpStreamConn>)> = futures::future::join_all(
            handles
                .iter()
                .map(|h| h.make_connections(CONNECTIONS_PER_PEER)),
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

        let mut jobs = JoinSet::new();
        let peer_ids: [u8; 3] = [0, 1, 2];

        tracing::debug!("connections created. sending data");

        for (peer_idx, (p0, p1)) in connections.into_iter().enumerate() {
            let my_peer_ids = peer_ids
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != peer_idx)
                .map(|(_, id)| *id)
                .collect::<Vec<_>>();

            for (conn_idx, (mut c0, mut c1)) in p0.into_iter().zip(p1.into_iter()).enumerate() {
                let p0_data = [my_peer_ids[0], conn_idx as u8];
                let p1_data = [my_peer_ids[1], conn_idx as u8];

                jobs.spawn(async move {
                    c0.write_all(&p0_data).await.unwrap();
                    c0.flush().await.unwrap();

                    let mut recv_data = [0u8; 2];
                    c0.read_exact(&mut recv_data).await.unwrap();
                    assert_eq!(&recv_data, &[peer_idx as u8, conn_idx as u8]);
                });

                jobs.spawn(async move {
                    c1.write_all(&p1_data).await.unwrap();
                    c1.flush().await.unwrap();

                    let mut recv_data = [0u8; 2];
                    c1.read_exact(&mut recv_data).await.unwrap();
                    assert_eq!(&recv_data, &[peer_idx as u8, conn_idx as u8]);
                });
            }
        }

        jobs.join_all().await;
        Ok(())
    }
}
