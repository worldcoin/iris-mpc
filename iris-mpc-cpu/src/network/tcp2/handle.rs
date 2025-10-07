use std::{marker::PhantomData, sync::Arc};

use crate::{
    execution::player::Identity,
    network::{
        tcp::config::TcpConfig,
        tcp2::{
            connection::{accept_loop, Connection, ConnectionRequest, ConnectionState},
            data::{ConnectionId, Peer},
            Client, NetworkConnection, Server,
        },
    },
};
use futures::future::join_all;
use itertools::izip;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection + 'static, C: Client<Output = T> + 'static> {
    peers: [Arc<Peer>; 2],
    my_id: Arc<Identity>,
    connector: C,
    conn_cmd_tx: mpsc::Sender<ConnectionRequest<T>>,
    connection_state: ConnectionState,
    config: TcpConfig,
}

#[async_trait]
impl<T: NetworkConnection + 'static> NetworkHandle for TcpNetworkHandle<T> {
    async fn make_sessions(&mut self) -> Result<(Vec<TcpSession>, CancellationToken)> {
        // set up the connection state
        self.connection_state.err_ct().await.cancel();
        self.connection_state
            .replace_cancellation_token(CancellationToken::new());

        // make the connections
        let connections = self.make_connections().await?;

        // pass them off to the session managers
        // return the tcp sessions
        todo!()
    }
}

impl<T: NetworkConnection + 'static> TcpNetworkHandle<T> {
    pub async fn new<I, C, S>(
        my_id: Identity,
        mut peers: I,
        connector: C,
        listener: S,
        config: TcpConfig,
        shutdown_ct: CancellationToken,
    ) -> Self
    where
        I: Iterator<Item = (Identity, String)>,
        C: Client<Output = T> + 'static,
        S: Server<Output = T> + 'static,
    {
        let my_id = Arc::new(my_id);
        let peers: [Arc<Peer>; 2] = [
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
        ];

        let connection_state = ConnectionState::new(
            config.num_connections * 2, // 2 peers
            shutdown_ct,
            CancellationToken::new(),
        );

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
            conn_cmd_tx,
            connection_state,
        }
    }

    pub async fn make_connections(&self) -> Result<Vec<T>> {
        let mut connect_futures = Vec::with_capacity(self.config.num_connections * 2);
        for peer in self.peers.iter().cloned() {
            for idx in 0..self.config.num_connections {
                let conn_id = ConnectionId::new(idx as u32);
                let fut = super::connection::connect(
                    connection_id,
                    self.my_id.clone(),
                    peer,
                    self.connection_state.clone(),
                    self.connector.clone(),
                    self.conn_cmd_tx.clone(),
                )?;
                connect_futures.push(fut);
            }
        }
        join_all(connect_futures).await
    }

    // returns None if cancelled
    pub async fn wait_for_ready(&mut self) -> Option<()> {
        let mut session_id = self.session_start_id;

        self.session_start_id = self.session_start_id.wrapping_add(self.config.num_sessions);
        if self.session_start_id >= usize::MAX - self.config.num_sessions {
            self.session_start_id = 0;
        }

        let mut responses0 = vec![];
        let mut responses1 = vec![];
        for (idx, (conn0, conn1)) in izip!(self.connections0, self.connections1)
            .enumerate()
            .take(self.config.num_connections)
        {
            let num_sessions = self.config.get_sessions_for_connection(idx);
            responses0.push(conn0.connect(num_sessions, session_id).await);
            responses1.push(conn1.connect(num_sessions, session_id).await);
            session_id += num_sessions;
        }

        let shutdown_ct = self.connection_state.shutdown_ct().await;
        tokio::select! {
            _ = self.connection_state.wait_for_ready() => Some(()),
            _ = shutdown_ct.cancelled() => None,
        }
    }

    pub async fn get_err_ct(&self) -> CancellationToken {
        self.connection_state.err_ct().await
    }
}
