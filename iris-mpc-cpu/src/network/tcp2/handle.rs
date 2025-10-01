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
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection + 'static> {
    connections0: Vec<Connection>,
    connections1: Vec<Connection>,
    connection_state: ConnectionState,
    config: TcpConfig,
    _marker: PhantomData<T>,
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
        let mut connections0 = Vec::new();
        let mut connections1 = Vec::new();

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

        for idx in 0..config.num_connections {
            let connection_id = ConnectionId(idx as u32);
            connections0.push(Connection::new(
                connection_id.clone(),
                my_id.clone(),
                peers[0].clone(),
                connection_state.clone(),
                connector.clone(),
                conn_cmd_tx.clone(),
            ));
            connections1.push(Connection::new(
                connection_id.clone(),
                my_id.clone(),
                peers[0].clone(),
                connection_state.clone(),
                connector.clone(),
                conn_cmd_tx.clone(),
            ));
        }

        assert_eq!(connections0.len(), config.num_connections);
        assert_eq!(connections1.len(), config.num_connections);

        Self {
            connections0,
            connections1,
            connection_state,
            config,
            _marker: PhantomData,
        }
    }

    // returns None if cancelled
    pub async fn wait_for_ready(&self) -> Option<()> {
        for conn in self.connections0.iter().chain(self.connections1.iter()) {
            conn.connect().await;
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
