use std::{marker::PhantomData, sync::Arc};

use crate::{
    execution::player::Identity,
    network::{
        tcp::config::TcpConfig,
        tcp2::{
            connection::{Connection, ConnectionRequest, ConnectionState, Peer},
            NetworkConnection,
        },
    },
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection> {
    connections0: Vec<Connection>,
    connections1: Vec<Connection>,
    connection_state: ConnectionState,
    config: TcpConfig,
    _marker: PhantomData<T>,
}

impl<T: NetworkConnection> TcpNetworkHandle<T> {
    pub fn new<I>(
        my_id: Identity,
        mut peers: I,
        config: TcpConfig,
        shutdown_ct: CancellationToken,
    ) -> Self
    where
        I: Iterator<Item = (Identity, String)>,
    {
        let my_id = Arc::new(Identity);
        let peers: [Arc<Peer>; 2] = [
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
            Arc::new(peers.next().expect("expected at least 2 identities").into()),
        ];
        let mut connections0 = Vec::new();
        let mut connections1 = Vec::new();

        let connection_state = ConnectionState::new(
            config.num_connections,
            shutdown_ct,
            CancellationToken::new(),
        );

        let (conn_cmd_tx, conn_cmd_rx) =
            mpsc::channel::<ConnectionRequest<T>>(config.num_connections);

        for _ in 0..config.num_connections {
            // connect to peers[0]
            // connect to peers[1]
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
}
