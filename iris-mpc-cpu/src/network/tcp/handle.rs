use std::sync::Arc;

use crate::{
    execution::player::Identity,
    network::{
        tcp::config::TcpConfig,
        tcp::{
            connection::{accept_loop, ConnectionRequest, ConnectionState},
            data::{ConnectionId, Peer},
            session::TcpSession,
            Client, NetworkConnection, NetworkHandle, Server,
        },
    },
};
use async_trait::async_trait;
use eyre::Result;
use futures::future::join_all;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> {
    peers: [Arc<Peer>; 2],
    my_id: Arc<Identity>,
    connector: C,
    conn_cmd_tx: mpsc::Sender<ConnectionRequest<T>>,
    connection_state: ConnectionState,
    config: TcpConfig,
    next_session_id: usize,
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

        let (c0, c1) = self.make_connections(self.config.num_connections).await?;

        // calls multiplexer::run() on each TCP/TLS stream
        let sessions = tcp::session::make_sessions(
            &self.peers,
            c0,
            c1,
            self.connection_state.clone(),
            &self.config,
            self.next_session_id,
        )
        .await;

        self.next_session_id = self.next_session_id.wrapping_add(self.config.num_sessions);
        if self.next_session_id >= usize::MAX - self.config.num_sessions {
            self.next_session_id = 0;
        }

        Ok((sessions, err_ct))
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

        let (conn_cmd_tx, conn_cmd_rx) =
            mpsc::channel::<ConnectionRequest<T>>(config.num_connections);

        // be sure not to make more than one network handle...
        tokio::spawn(accept_loop(
            my_id.clone(),
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
    pub async fn make_connections(&self, conns_per_peer: usize) -> Result<(Vec<T>, Vec<T>)> {
        assert_eq!(self.peers.len(), 2);
        let mut connect_futures = Vec::with_capacity(conns_per_peer * self.peers.len());

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

        Ok((c0, c1))
    }
}
