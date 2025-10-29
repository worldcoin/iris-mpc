use crate::{
    execution::{
        hawk_main::HawkArgs,
        local::generate_local_identities,
        player::{Role, RoleAssignment},
        session::{NetworkSession, Session},
    },
    network::tcp::{
        config::TcpConfig,
        handle::TcpNetworkHandle,
        networking::{
            client::{TcpClient, TlsClient},
            connection_builder::PeerConnectionBuilder,
            server::TcpServer,
        },
    },
};
use async_trait::async_trait;
use eyre::Result;
use iris_mpc_common::config::TlsConfig;
use itertools::izip;
use std::sync::{Arc, Once};
use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    time::Duration,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_util::sync::CancellationToken;

pub mod config;
mod data;
pub mod handle;
pub mod networking;
pub mod session;

use crate::network::tcp::networking::client::BoxTcpClient;
use crate::network::tcp::networking::server::{BoxTcpServer, TlsServer};
use data::*;

#[async_trait]
pub trait NetworkHandle: Send + Sync {
    async fn make_network_sessions(&mut self) -> Result<Vec<NetworkSession>>;
    async fn make_sessions(&mut self) -> Result<Vec<Session>>;
}

pub trait NetworkConnection: AsyncRead + AsyncWrite + Send + Sync + Unpin {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send + ?Sized + Sync> NetworkConnection for T {}

// used to establish an outbound connection
#[async_trait]
pub trait Client: Send + Sync + Clone {
    type Output: NetworkConnection;
    async fn connect(&self, url: String) -> Result<Self::Output>;
}

// used for a server to accept an incoming connection
#[async_trait]
pub trait Server: Send {
    type Output: NetworkConnection;
    async fn accept(&self) -> Result<(SocketAddr, Self::Output)>;
}

pub struct NetworkHandleArgs {
    pub party_index: usize,
    pub addresses: Vec<String>,
    pub connection_parallelism: usize,
    pub request_parallelism: usize,
    pub sessions_per_request: usize,
    pub tls: Option<TlsConfig>,
}

impl NetworkHandleArgs {
    pub fn from_hawk(args: &HawkArgs, sessions_per_request: usize) -> Self {
        Self {
            party_index: args.party_index,
            addresses: args.addresses.clone(),
            connection_parallelism: args.connection_parallelism,
            request_parallelism: args.request_parallelism,
            sessions_per_request,
            tls: args.tls.clone(),
        }
    }
}

pub async fn build_network_handle(
    args: NetworkHandleArgs,
    ct: CancellationToken,
) -> Result<Box<dyn NetworkHandle>> {
    static INSTALL_CRYPTO_PROVIDER: Once = Once::new();
    INSTALL_CRYPTO_PROVIDER.call_once(|| {
        if tokio_rustls::rustls::crypto::aws_lc_rs::default_provider()
            .install_default()
            .is_err()
        {
            tracing::error!("failed to install CryptoProvider for rustls");
        }
    });

    let identities = generate_local_identities();
    let role_assignments: RoleAssignment = identities
        .iter()
        .enumerate()
        .map(|(index, id)| (Role::new(index), id.clone()))
        .collect();
    let role_assignments = Arc::new(role_assignments);

    let my_index = args.party_index;
    let my_identity = identities[my_index].clone();
    let my_address = &args.addresses[my_index];
    let my_addr = to_inaddr_any(my_address.parse::<SocketAddr>()?);

    let tcp_config = TcpConfig::new(
        Duration::from_secs(10),
        args.connection_parallelism,
        args.request_parallelism * args.sessions_per_request,
    );

    // PeerConnectionBuilder is generic over listener and connector and don't want to use a boxed trait to
    // reduce code duplication. instead, use a macro which makes use of local variables.
    macro_rules! build_network_handle {
        ($listener:expr, $connector:expr) => {{
            let connection_builder = PeerConnectionBuilder::new(
                my_identity,
                tcp_config.clone(),
                $listener,
                $connector,
                ct.clone(),
            )
            .await?;

            for (identity, url) in
                izip!(identities, &args.addresses).filter(|(_, address)| address != &my_address)
            {
                connection_builder
                    .include_peer(identity.clone(), url.clone())
                    .await?;
            }

            let (reconnector, connections) = connection_builder.build().await?;
            let networking = TcpNetworkHandle::new(
                reconnector,
                connections,
                tcp_config,
                ct,
                my_index,
                role_assignments,
            );
            Ok(Box::new(networking))
        }};
    }

    if let Some(tls) = args.tls.as_ref() {
        tracing::info!(
            "Building NetworkHandle, with TLS, from configs: {:?} {:?}",
            tcp_config,
            tls,
        );

        let root_certs = tls.clone().root_certs;

        tracing::info!("Running in full app TLS mode.");
        if tls.private_key.is_none() || tls.leaf_cert.is_none() {
            return Err(eyre::eyre!(
                "TLS configuration is required for this operation"
            ));
        }
        let private_key = tls
            .private_key
            .as_ref()
            .ok_or(eyre::eyre!("Private key is required for TLS"))?;

        let leaf_cert = tls
            .leaf_cert
            .as_ref()
            .ok_or(eyre::eyre!("Leaf certificate is required for TLS"))?;

        let listener = TlsServer::new(my_addr, private_key, leaf_cert, &root_certs).await?;
        let connector = TlsClient::new_with_ca_certs(&root_certs).await?;
        build_network_handle!(listener, connector)
    } else {
        tracing::info!(
            "Building NetworkHandle, without TLS, from config: {:?}",
            tcp_config
        );
        let listener = BoxTcpServer(TcpServer::new(my_addr).await?);
        let connector = BoxTcpClient(TcpClient::default());
        build_network_handle!(listener, connector)
    }
}

fn to_inaddr_any(mut socket: SocketAddr) -> SocketAddr {
    if socket.is_ipv4() {
        socket.set_ip(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    } else {
        socket.set_ip(IpAddr::V6(Ipv6Addr::UNSPECIFIED));
    }
    socket
}

pub mod testing {
    use super::*;
    use eyre::Result;

    use itertools::izip;
    use std::{collections::HashSet, net::SocketAddr, sync::LazyLock, time::Duration};
    use tokio::{net::TcpStream, sync::Mutex, time::sleep};
    use tokio_util::sync::CancellationToken;

    use crate::{
        execution::player::Identity,
        network::tcp::{
            config::TcpConfig,
            handle::{self, TcpNetworkHandle},
            networking::{
                client::TcpClient, connection_builder::PeerConnectionBuilder, server::TcpServer,
            },
            NetworkHandle,
        },
    };

    static USED_PORTS: LazyLock<Mutex<HashSet<SocketAddr>>> =
        LazyLock::new(|| Mutex::new(HashSet::new()));

    async fn get_free_local_addresses(num_ports: usize) -> Result<Vec<SocketAddr>> {
        let mut addresses = vec![];
        while addresses.len() < num_ports {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let addr = listener.local_addr()?;
            if USED_PORTS.lock().await.insert(addr) {
                addresses.push(addr);
            } else {
                tracing::warn!("SocketAddr {addr} already in use, retrying");
            }
        }
        tracing::info!("Found free addresses: {addresses:?}");
        Ok(addresses)
    }

    pub async fn setup_local_tcp_networking(
        parties: Vec<Identity>,
        connection_parallelism: usize,
        request_parallelism: usize,
    ) -> Result<(
        Vec<handle::TcpNetworkHandle<TcpStream>>,
        Vec<Vec<NetworkSession>>,
    )> {
        assert_eq!(parties.len(), 3);

        let config = TcpConfig::new(
            Duration::from_secs(5),
            connection_parallelism,
            request_parallelism,
        );

        let addresses = get_free_local_addresses(parties.len()).await?;
        // Create NetworkHandles for each party
        let mut builders = Vec::with_capacity(parties.len());
        let connector = TcpClient::default();
        let ct = CancellationToken::new();
        for (party, addr) in izip!(parties.iter(), addresses.iter()) {
            let listener = TcpServer::new(*addr).await?;
            builders.push(
                PeerConnectionBuilder::new(
                    party.clone(),
                    config.clone(),
                    listener,
                    connector.clone(),
                    ct.clone(),
                )
                .await?,
            );
        }

        sleep(Duration::from_secs(1)).await;

        tracing::debug!("initiating connections");
        // Connect each handle to every other handle
        for i in 0..builders.len() {
            for j in 0..builders.len() {
                if i != j {
                    builders[i]
                        .include_peer(parties[j].clone(), addresses[j].to_string())
                        .await?;
                }
            }
        }

        tracing::debug!("waiting for connections to complete");
        let mut connections = vec![];
        for b in builders {
            let x = b.build().await?;
            tracing::debug!("connections completed for player");
            connections.push(x);
        }
        tracing::debug!("Players connected to each other");

        let identities = generate_local_identities();
        let role_assignments: RoleAssignment = identities
            .iter()
            .enumerate()
            .map(|(index, id)| (Role::new(index), id.clone()))
            .collect();
        let role_assignments = Arc::new(role_assignments);
        let ct = CancellationToken::new();
        let mut handles = vec![];
        for (idx, (r, c)) in connections.into_iter().enumerate() {
            handles.push(TcpNetworkHandle::new(
                r,
                c,
                config.clone(),
                ct.clone(),
                idx,
                role_assignments.clone(),
            ));
        }

        tracing::debug!("waiting for make_sessions to complete");
        let mut sessions = vec![];
        for h in handles.iter_mut() {
            sessions.push(h.make_network_sessions().await?);
        }

        Ok((handles, sessions))
    }

    /// Interleaves a Vec of Vecs into a single Vec by taking one element from each inner Vec in turn.
    /// For example, interleaving `[[1,2,3],[4,5,6],[7,8,9]]` yields `[1,4,7,2,5,8,3,6,9]`.
    pub fn interleave_vecs<T>(vecs: Vec<Vec<T>>) -> Vec<T> {
        let mut result = Vec::new();
        let mut iters: Vec<_> = vecs.into_iter().map(|v| v.into_iter()).collect();
        loop {
            let mut did_push = false;
            for iter in iters.iter_mut() {
                if let Some(item) = iter.next() {
                    result.push(item);
                    did_push = true;
                }
            }
            if !did_push {
                break;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use eyre::Result;

    use std::time::Duration;
    use tokio::task::JoinSet;
    use tokio::time::sleep;
    use tracing_test::traced_test;

    use crate::execution::local::generate_local_identities;
    use crate::execution::player::{Identity, Role};
    use crate::execution::session::NetworkSession;
    use crate::network::tcp::data::StreamId;
    use crate::network::value::NetworkValue;
    use rand::Rng;

    use super::testing::*;

    // can only send NetworkValue over the network. PrfKey is easy to make so this is used here.
    fn get_prf() -> NetworkValue {
        let mut rng = rand::thread_rng();
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        NetworkValue::PrfKey(key)
    }

    async fn all_parties_talk(identities: Vec<Identity>, sessions: Vec<NetworkSession>) {
        let mut tasks = JoinSet::new();
        let message_to_next = get_prf();
        let message_to_prev = get_prf();

        for (player_id, session) in sessions.into_iter().enumerate() {
            let role = Role::new(player_id);
            let next = role.next(3).index();
            let prev = role.prev(3).index();

            let next_id = identities[next].clone();
            let prev_id = identities[prev].clone();
            let message_to_next = message_to_next.clone();
            let message_to_prev = message_to_prev.clone();

            let mut session = session;
            tasks.spawn(async move {
                // Sending
                session
                    .networking
                    .send(message_to_next.clone(), &next_id)
                    .await
                    .unwrap();
                session
                    .networking
                    .send(message_to_prev.clone(), &prev_id)
                    .await
                    .unwrap();

                // Receiving
                let received_message_from_prev =
                    session.networking.receive(&prev_id).await.unwrap();
                assert_eq!(received_message_from_prev, message_to_next);
                let received_message_from_next =
                    session.networking.receive(&next_id).await.unwrap();
                assert_eq!(received_message_from_next, message_to_prev);
            });
        }
        tasks.join_all().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_tcp_comms_correct() -> Result<()> {
        let identities = generate_local_identities();
        let (_managers, mut sessions) =
            setup_local_tcp_networking(identities.clone(), 1, 4).await?;
        sleep(Duration::from_millis(500)).await;

        assert_eq!(sessions.len(), 3);
        assert_eq!(sessions[0].len(), 4);

        let mut iters = vec![];
        for session in sessions.iter_mut() {
            iters.push(session.drain(..));
        }

        let mut session_list = vec![];
        for _ in 0..3 {
            let mut s = vec![];
            for x in iters.iter_mut() {
                s.push(x.next().unwrap());
            }
            session_list.push(s);
        }
        let mut session_list = session_list.drain(..);

        let mut jobs = JoinSet::new();

        // Simple session with one message sent from one party to another
        let mut players = session_list.next().unwrap();
        {
            jobs.spawn(async move {
                // we don't need the last player here
                players.pop();

                let mut bob = players.pop().unwrap();
                let mut alice = players.pop().unwrap();

                // Send a message from the first party to the second party
                let alice_prf = get_prf();
                let alice_msg = alice_prf.clone();

                let task1 = tokio::spawn(async move {
                    alice
                        .networking
                        .send(alice_msg, &"bob".into())
                        .await
                        .unwrap();
                });
                let task2 = tokio::spawn(async move {
                    let rx_msg = bob.networking.receive(&"alice".into()).await.unwrap();
                    assert_eq!(alice_prf, rx_msg);
                });
                let _ = tokio::try_join!(task1, task2).unwrap();
            });
        }

        // Multiple parties sending messages to each other
        let players = session_list.next().unwrap();
        // Each party sending and receiving messages to each other
        {
            let identities = identities.clone();
            jobs.spawn(async move {
                // Test that parties can send and receive messages
                all_parties_talk(identities, players).await;
            });
        }

        let players = session_list.next().unwrap();
        // Parties create a session asynchronously
        {
            // Test that parties can send and receive messages
            all_parties_talk(identities, players).await;
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_tcp_comms_reconnect() -> Result<()> {
        let identities = generate_local_identities();
        let (managers, mut sessions) = setup_local_tcp_networking(identities.clone(), 1, 2).await?;
        sleep(Duration::from_millis(500)).await;

        assert_eq!(sessions.len(), 3);
        assert_eq!(sessions[0].len(), 2);

        let mut iters = vec![];
        for session in sessions.iter_mut() {
            iters.push(session.drain(..));
        }

        let mut session_list = vec![];
        for _ in 0..2 {
            let mut s = vec![];
            for x in iters.iter_mut() {
                s.push(x.next().unwrap());
            }
            session_list.push(s);
        }
        let mut session_list = session_list.drain(..);

        all_parties_talk(identities.clone(), session_list.next().unwrap()).await;
        tracing::debug!("all_parties_talk works. testing reconnect");

        // this will disconnect from the other party. the other party will reconnect without
        // exercising any test code.
        managers[0]
            .test_reconnect(identities[1].clone(), StreamId::from(0))
            .await
            .unwrap();

        tracing::debug!("reconnect successful. testing all_parties_talk again");
        all_parties_talk(identities, session_list.next().unwrap()).await;
        Ok(())
    }
}
