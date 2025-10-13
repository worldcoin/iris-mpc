use std::sync::Arc;

use crate::{
    execution::player::Identity,
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
};
use async_trait::async_trait;
use eyre::{bail, Result};
use futures::future::join_all;
use itertools::Itertools;
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> {
    peers: [Arc<Peer>; 2],
    my_id: Arc<Identity>,
    connector: C,
    conn_cmd_tx: UnboundedSender<ConnectionRequest<T>>,
    connection_state: ConnectionState,
    config: TcpConfig,
    next_session_id: usize,
}

impl<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> Drop
    for TcpNetworkHandle<T, C>
{
    fn drop(&mut self) {
        tracing::debug!("TcpNetworkHandle dropped");
    }
}

#[async_trait]
impl<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> NetworkHandle
    for TcpNetworkHandle<T, C>
{
    async fn make_sessions(&mut self) -> Result<(Vec<TcpSession>, CancellationToken)> {
        let err_ct = CancellationToken::new();
        // cancel the old token just in case
        self.connection_state.err_ct().await.cancel();
        self.connection_state
            .replace_cancellation_token(err_ct.clone())
            .await;

        // wait for all peers to establish all connections
        let mut connections = self.make_connections(self.config.num_connections).await?;
        connections.sync().await?;

        // calls multiplexer::run() on each TCP/TLS stream
        let mut sessions = super::session::make_sessions(
            connections,
            self.connection_state.clone(),
            &self.config,
            self.next_session_id,
        )
        .await;

        self.validate_sessions(&mut sessions).await?;

        let sessions_per_conn = (0..self.config.num_connections)
            .map(|idx| self.config.get_sessions_for_connection(idx))
            .collect_vec();
        tracing::info!(
            "make_sessions succeeded. starting id: {} sessions per connection: {:?}",
            self.next_session_id,
            sessions_per_conn
        );

        self.next_session_id = self.next_session_id.wrapping_add(self.config.num_sessions);
        if self.next_session_id >= usize::MAX - self.config.num_sessions {
            self.next_session_id = 0;
        }

        Ok((sessions, err_ct))
    }

    async fn sync_peers(&mut self) -> Result<()> {
        let mut connections = self.make_connections(1).await?;
        connections.sync().await?;
        Ok(())
    }
}

impl<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> TcpNetworkHandle<T, C> {
    pub async fn new<I, S>(
        my_id: Identity,
        mut peers: I,
        connector: C,
        listener: S,
        config: TcpConfig,
        shutdown_ct: CancellationToken,
    ) -> Self
    where
        I: Iterator<Item = (Identity, String)>,
        S: Server<Output = T> + 'static,
    {
        let my_id = Arc::new(my_id);
        let peers: [Arc<Peer>; 2] = [
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
        ];

        let connection_state = ConnectionState::new(shutdown_ct, CancellationToken::new());

        let (conn_cmd_tx, conn_cmd_rx) = mpsc::unbounded_channel::<ConnectionRequest<T>>();

        // be sure not to make more than one network handle...
        tokio::spawn(accept_loop(
            listener,
            conn_cmd_rx,
            connection_state.shutdown_ct().await,
        ));

        Self {
            my_id,
            peers,
            connector,
            config,
            conn_cmd_tx,
            connection_state,
            next_session_id: 0,
        }
    }

    // returns the connections for each peer
    // when returned, the handshakes have successfully completed
    pub async fn make_connections(&self, conns_per_peer: usize) -> Result<PeerConnections<T>> {
        assert_eq!(self.peers.len(), 2);
        let mut connect_futures = Vec::with_capacity(conns_per_peer * self.peers.len());

        // peers[0] will be associated with connections c0
        // peers[1] will be associated with connections c1
        for peer in self.peers.iter() {
            for idx in 0..conns_per_peer {
                let connection_id = ConnectionId::new(idx as u32);
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

        Ok(PeerConnections::new(self.peers.clone(), c0, c1))
    }

    async fn validate_sessions(&self, sessions: &mut [TcpSession]) -> Result<()> {
        // make sure all the sessions are working
        for (idx, session) in sessions.iter_mut().enumerate() {
            let prf = NetworkValue::PrfKey([idx as u8; 16]);
            for peer in &self.peers {
                session.send(prf.clone(), peer.id()).await?;
            }
            for peer in &self.peers {
                let r = session.receive(peer.id()).await?;
                match r {
                    NetworkValue::PrfKey(arr) => {
                        assert_eq!([idx as u8; 16], arr);
                    }
                    _ => bail!("invalid msg received in validate_sessions()"),
                }
            }
        }
        Ok(())
    }
}
