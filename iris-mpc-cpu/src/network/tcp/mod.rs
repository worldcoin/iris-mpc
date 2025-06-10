use std::time::Duration;

use crate::execution::session::SessionId;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

mod data;
pub mod handle;
pub mod networking;
pub mod session;

use data::*;

type NetworkMsg = Vec<u8>;
// session multiplexing over a socket requires a SessionId
type OutboundMsg = (SessionId, NetworkMsg);
type OutStream = UnboundedSender<NetworkMsg>;
type InStream = UnboundedReceiver<NetworkMsg>;

#[derive(Default, Clone, Debug)]
pub struct TcpConfig {
    pub timeout_duration: Duration,
    // number of tcp connections to create
    pub connection_parallelism: usize,
    // number of sessions per connection
    pub stream_parallelism: usize,
}

impl TcpConfig {
    pub fn new(
        timeout_duration: Duration,
        connection_parallelism: usize,
        request_parallelism: usize,
    ) -> Self {
        Self {
            timeout_duration,
            connection_parallelism,
            stream_parallelism: std::cmp::max(request_parallelism / connection_parallelism, 1),
        }
    }
}

pub mod testing {
    use eyre::Result;

    use itertools::izip;
    use std::{collections::HashSet, net::SocketAddr, sync::LazyLock, time::Duration};
    use tokio::{sync::Mutex, time::sleep};

    use crate::{
        execution::player::Identity,
        network::tcp::{
            handle::{self, TcpNetworkHandle},
            networking::connection_builder::PeerConnectionBuilder,
            session::TcpSession,
            TcpConfig,
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
    ) -> Result<(Vec<handle::TcpNetworkHandle>, Vec<Vec<TcpSession>>)> {
        assert_eq!(parties.len(), 3);

        let config = TcpConfig::new(
            Duration::from_secs(5),
            connection_parallelism,
            request_parallelism,
        );

        let addresses = get_free_local_addresses(parties.len()).await?;
        // Create NetworkHandles for each party
        let mut builders = Vec::with_capacity(parties.len());
        for (party, addr) in izip!(parties.iter(), addresses.iter()) {
            builders.push(PeerConnectionBuilder::new(party.clone(), *addr, config.clone()).await?);
        }

        sleep(Duration::from_secs(1)).await;

        tracing::debug!("initiating connections");
        // Connect each handle to every other handle
        for i in 0..builders.len() {
            for j in 0..builders.len() {
                if i != j {
                    builders[i]
                        .include_peer(parties[j].clone(), addresses[j])
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

        let mut handles = vec![];
        for (r, c) in connections {
            handles.push(TcpNetworkHandle::new(r, c, config.clone()));
        }

        tracing::debug!("waiting for make_sessions to complete");
        let mut sessions = vec![];
        for h in &handles {
            sessions.push(h.make_sessions().await?);
        }

        Ok((handles, sessions))
    }

    /// Interleaves a Vec of Vecs into a single Vec by taking one element from each inner Vec in turn.
    /// For example, interleaving [[1,2,3],[4,5,6],[7,8,9]] yields [1,4,7,2,5,8,3,6,9].
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
    use crate::network::tcp::data::StreamId;
    use crate::network::value::NetworkValue;
    use crate::network::{tcp::session::TcpSession, NetworkType, Networking};
    use rand::Rng;

    use super::testing::*;

    // can only send NetworkValue over the network. PrfKey is easy to make so this is used here.
    fn get_prf() -> NetworkValue {
        let mut rng = rand::thread_rng();
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        NetworkValue::PrfKey(key)
    }

    async fn all_parties_talk(identities: Vec<Identity>, sessions: Vec<TcpSession>) {
        let mut tasks = JoinSet::new();
        let message_to_next = get_prf().to_network();
        let message_to_prev = get_prf().to_network();

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
                    .send(message_to_next.clone(), &next_id)
                    .await
                    .unwrap();
                session
                    .send(message_to_prev.clone(), &prev_id)
                    .await
                    .unwrap();

                // Receiving
                let received_message_from_prev = session.receive(&prev_id).await.unwrap();
                assert_eq!(received_message_from_prev, message_to_next);
                let received_message_from_next = session.receive(&next_id).await.unwrap();
                assert_eq!(received_message_from_next, message_to_prev);
            });
        }
        tasks.join_all().await;
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_tcp_comms_correct() -> Result<()> {
        let identities = generate_local_identities();
        let (_managers, mut sessions) = setup_local_tcp_networking(
            identities.clone(),
            3,
            NetworkType::default_request_parallelism(),
        )
        .await?;
        sleep(Duration::from_millis(500)).await;

        let num_sessions = sessions[0].len();
        assert_eq!(num_sessions, 3);

        let mut iters = vec![];
        for session in sessions.iter_mut() {
            iters.push(session.drain(..));
        }

        let mut session_list = vec![];
        for _ in 0..num_sessions {
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
                let alice = players.pop().unwrap();

                // Send a message from the first party to the second party
                let alice_prf = get_prf();
                let alice_msg = alice_prf.to_network();

                let task1 = tokio::spawn(async move {
                    alice.send(alice_msg, &"bob".into()).await.unwrap();
                });
                let task2 = tokio::spawn(async move {
                    let received_message = bob.receive(&"alice".into()).await;
                    let rx_msg = NetworkValue::from_network(received_message).unwrap();
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
        let (managers, mut sessions) = setup_local_tcp_networking(
            identities.clone(),
            3,
            NetworkType::default_request_parallelism(),
        )
        .await?;
        sleep(Duration::from_millis(500)).await;

        let num_sessions = sessions[0].len();
        assert_eq!(num_sessions, 3);

        let mut iters = vec![];
        for session in sessions.iter_mut() {
            iters.push(session.drain(..));
        }

        let mut session_list = vec![];
        for _ in 0..num_sessions {
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
