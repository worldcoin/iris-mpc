use crate::{
    execution::player::Identity,
    network::{
        tcp::config::TcpConfig,
        tcp2::{
            connection::{Connection, ConnectionState},
            NetworkConnection,
        },
    },
};
use tokio_util::sync::CancellationToken;

pub struct TcpNetworkHandle<T: NetworkConnection> {
    peers: [Identity; 2],
    connections0: Vec<Connection<T>>,
    connections1: Vec<Connection<T>>,
    connection_state: ConnectionState,
    config: TcpConfig,
}

impl<T: NetworkConnection> TcpNetworkHandle<T> {
    pub fn new<I>(mut identities: I, config: TcpConfig, shutdown_ct: CancellationToken) -> Self
    where
        I: Iterator<Item = Identity>,
    {
        let peers: [Identity; 2] = [
            identities.next().expect("expected at least 2 identities"),
            identities.next().expect("expected at least 2 identities"),
        ];
        let mut connections0 = Vec::new();
        let mut connections1 = Vec::new();

        let connection_state = ConnectionState::new(
            config.num_connections,
            shutdown_ct,
            CancellationToken::new(),
        );

        for _ in 0..config.num_connections {
            // connect to peers[0]
            // connect to peers[1]
        }

        assert_eq!(connections0.len(), config.num_connections);
        assert_eq!(connections1.len(), config.num_connections);
        TcpNetworkHandle {
            peers,
            connections0,
            connections1,
            connection_state,
            config,
        }
    }
}
