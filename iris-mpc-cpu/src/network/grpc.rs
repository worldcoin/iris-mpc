use super::Networking;
use crate::{
    execution::{local::get_free_local_addresses, player::Identity},
    network::SessionId,
};
use backoff::{future::retry, ExponentialBackoff};
use dashmap::DashMap;
use eyre::{eyre, OptionExt};
use party_node::{
    party_node_client::PartyNodeClient,
    party_node_server::{PartyNode, PartyNodeServer},
    SendRequest, SendResponse,
};
use std::sync::Arc;
use tonic::{
    async_trait,
    transport::{Channel, Server},
    Request, Response, Status,
};

mod party_node {
    tonic::include_proto!("party_node");
}

type TonicResult<T> = Result<T, Status>;

fn err_to_status(e: eyre::Error) -> Status {
    Status::internal(e.to_string())
}

#[derive(Clone)]
struct QueueChannel {
    pub sender:   Arc<async_channel::Sender<Vec<u8>>>,
    pub receiver: Arc<async_channel::Receiver<Vec<u8>>>,
}

#[derive(Clone)]
struct MessageQueueStore {
    queues: DashMap<Identity, QueueChannel>,
}

impl MessageQueueStore {
    fn new() -> Self {
        MessageQueueStore {
            queues: DashMap::new(),
        }
    }

    pub fn add_channel(&self, party_id: &Identity) -> QueueChannel {
        // check that the party_id is not already in the queues
        if self.queues.contains_key(party_id) {
            return self.queues.get(party_id).unwrap().clone();
        }
        let (sender, receiver) = async_channel::unbounded();
        let channel = QueueChannel {
            sender:   Arc::new(sender),
            receiver: Arc::new(receiver),
        };
        self.queues.insert(party_id.clone(), channel.clone());
        channel
    }

    fn get_channel(&self, party_id: &Identity) -> eyre::Result<QueueChannel> {
        let channel = self.queues.get(party_id).ok_or_eyre(format!(
            "Channel not found for party {:?}, existing channels: alice {}, bob {}, charlie {}",
            party_id,
            self.queues.contains_key(&Identity("alice".into())),
            self.queues.contains_key(&Identity("bob".into())),
            self.queues.contains_key(&Identity("charlie".into()))
        ))?;
        Ok((*channel).clone())
    }

    pub async fn push_back(&self, party_id: &Identity, value: Vec<u8>) -> eyre::Result<()> {
        let channel = self.get_channel(party_id)?;
        // sends the value via the channel sender; if failed, returns an error
        channel.sender.send(value).await.map_err(|e| e.into())
    }

    pub async fn pop_front(&self, party_id: &Identity) -> eyre::Result<Vec<u8>> {
        let channel = self.get_channel(party_id)?;
        channel.receiver.recv().await.map_err(|e| e.into())
    }
}

#[derive(Clone)]
pub struct GrpcNetworking {
    party_id:       Identity,
    // other party id -> client to call that party
    clients:        Arc<DashMap<Identity, PartyNodeClient<Channel>>>,
    message_queues: Arc<DashMap<SessionId, MessageQueueStore>>,
}

impl GrpcNetworking {
    pub fn new(party_id: Identity) -> Self {
        GrpcNetworking {
            party_id,
            clients: Arc::new(DashMap::new()),
            message_queues: Arc::new(DashMap::new()),
        }
    }

    pub async fn connect_to_party(&self, party_id: Identity, address: &str) -> eyre::Result<()> {
        let client = PartyNodeClient::connect(address.to_string()).await?;
        self.clients.insert(party_id, client);
        Ok(())
    }

    pub async fn create_session(&self, session_id: SessionId) -> eyre::Result<()> {
        if self.message_queues.contains_key(&session_id) {
            return Err(eyre!("Session already exists"));
        }

        let queue = MessageQueueStore::new();
        for client in self.clients.iter() {
            queue.add_channel(client.key());
        }
        self.message_queues.insert(session_id, queue);
        Ok(())
    }
}

// Server implementation
#[async_trait]
impl PartyNode for GrpcNetworking {
    async fn send_message(
        &self,
        request: Request<SendRequest>,
    ) -> TonicResult<Response<SendResponse>> {
        let request = request.into_inner();
        let sender_id = Identity::from(request.sender_id);
        if sender_id == self.party_id {
            return Err(Status::unauthenticated(format!(
                "Sender ID coincides with receiver ID: {:?}",
                sender_id
            )));
        }
        let session_id = SessionId::from(request.session_id);
        let message_queue = self
            .message_queues
            .get(&session_id)
            .ok_or(Status::not_found(format!(
                "Session {:?} hasn't been created",
                session_id
            )))?;
        message_queue
            .push_back(&sender_id, request.data)
            .await
            .map_err(err_to_status)?;
        Ok(Response::new(SendResponse {}))
    }
}

// Client implementation
#[async_trait]
impl Networking for GrpcNetworking {
    async fn send(
        &self,
        value: Vec<u8>,
        receiver: &Identity,
        session_id: &SessionId,
    ) -> eyre::Result<()> {
        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(std::time::Duration::from_secs(2)),
            max_interval: std::time::Duration::from_secs(1),
            multiplier: 1.1,
            ..Default::default()
        };
        retry(backoff, || async {
            // Send message via gRPC client
            let mut client = self
                .clients
                .get(receiver)
                .ok_or_eyre(format!("Client not found {:?}", receiver))?
                .clone();
            let request = Request::new(SendRequest {
                sender_id:  self.party_id.0.clone(),
                session_id: session_id.0,
                data:       value.clone(),
            });
            println!(
                "Sending message {:?} from {:?} to {:?}",
                value, self.party_id, receiver
            );
            let _response = client
                .send_message(request)
                .await
                .map_err(|err| eyre!(err.to_string()))?;
            println!(
                "SUCCESS: Sending message {:?} from {:?} to {:?}",
                value, self.party_id, receiver
            );
            Ok(())
        })
        .await
    }

    async fn receive(&self, sender: &Identity, session_id: &SessionId) -> eyre::Result<Vec<u8>> {
        // Just retrieve the first message from the corresponding queue
        self.message_queues
            .get(session_id)
            .ok_or_eyre(format!(
                "Session {session_id:?} hasn't been added to message queues"
            ))?
            .pop_front(sender)
            .await
    }
}

pub async fn setup_local_grpc_networking(
    parties: Vec<Identity>,
) -> eyre::Result<Vec<GrpcNetworking>> {
    let players = parties
        .iter()
        .map(|party| GrpcNetworking::new(party.clone()))
        .collect::<Vec<GrpcNetworking>>();

    let addresses = get_free_local_addresses(players.len())?;

    let players_addresses = players
        .iter()
        .cloned()
        .zip(addresses.iter().cloned())
        .collect::<Vec<_>>();

    // Initialize servers
    for (player, addr) in &players_addresses {
        let player = player.clone();
        let socket = addr.parse().unwrap();
        tokio::spawn(async move {
            Server::builder()
                .add_service(PartyNodeServer::new(player))
                .serve(socket)
                .await
                .unwrap();
        });
    }

    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Connect to each other
    for (player, addr) in &players_addresses {
        for (other_player, other_addr) in &players_addresses.clone() {
            if addr != other_addr {
                let other_addr = format!("http://{}", other_addr);
                player
                    .connect_to_party(other_player.party_id.clone(), &other_addr)
                    .await
                    .unwrap();
            }
        }
    }

    Ok(players)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::{local::generate_local_identities, player::Role},
        hawkers::galois_store::LocalNetAby3NgStoreProtocol,
    };
    use aes_prng::AesRng;
    use hawk_pack::hnsw_db::HawkSearcher;
    use rand::SeedableRng;
    use std::time::Duration;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_correct() -> eyre::Result<()> {
        let identities = generate_local_identities();
        let players = setup_local_grpc_networking(identities.clone()).await?;

        let mut jobs = JoinSet::new();

        // Simple session with one message sent from one party to another
        {
            let alice = players[0].clone();
            let bob = players[1].clone();

            let session_id = SessionId::from(0);

            jobs.spawn(async move {
                // Send a message from the first party to the second party
                let message = b"Hey, Bob. I'm Alice. Do you copy?".to_vec();
                let message_copy = message.clone();

                let task1 = tokio::spawn(async move {
                    alice.create_session(session_id).await.unwrap();
                    // Add a delay to ensure that the session is created before sending
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    alice
                        .send(message.clone(), &"bob".into(), &session_id)
                        .await
                        .unwrap();
                });
                let task2 = tokio::spawn(async move {
                    bob.create_session(session_id).await.unwrap();
                    // Add a delay to ensure that the session is created before receiving
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    let received_message = bob.receive(&"alice".into(), &session_id).await.unwrap();
                    assert_eq!(message_copy, received_message);
                });
                let _ = tokio::try_join!(task1, task2).unwrap();
            });
        }

        // Each party sending and receiving messages to each other
        {
            jobs.spawn(async move {
                let session_id = SessionId::from(1);

                let mut tasks = JoinSet::new();
                // Send messages
                for (player_id, player) in players.iter().enumerate() {
                    let role = Role::new(player_id);
                    let next = role.next(3).index();
                    let prev = role.prev(3).index();

                    let player = player.clone();
                    let next_id = identities[next].clone();
                    let prev_id = identities[prev].clone();

                    tasks.spawn(async move {
                        player.create_session(session_id).await.unwrap();
                        // Add a delay to ensure that the session is created before
                        // sending/receiving
                        tokio::time::sleep(Duration::from_millis(100)).await;

                        // Sending
                        let msg_to_next =
                            format!("From player {} to player {} with love", player_id, next)
                                .into_bytes();
                        let msg_to_prev =
                            format!("From player {} to player {} with love", player_id, prev)
                                .into_bytes();
                        player
                            .send(msg_to_next.clone(), &next_id, &session_id)
                            .await
                            .unwrap();
                        player
                            .send(msg_to_prev.clone(), &prev_id, &session_id)
                            .await
                            .unwrap();

                        // Receiving
                        let received_msg_from_prev =
                            player.receive(&prev_id, &session_id).await.unwrap();
                        let expected_msg_from_prev =
                            format!("From player {} to player {} with love", prev, player_id)
                                .into_bytes();
                        assert_eq!(received_msg_from_prev, expected_msg_from_prev);
                        let received_msg_from_next =
                            player.receive(&next_id, &session_id).await.unwrap();
                        let expected_msg_from_next =
                            format!("From player {} to player {} with love", next, player_id)
                                .into_bytes();
                        assert_eq!(received_msg_from_next, expected_msg_from_next);
                    });
                }
                tasks.join_all().await;
            });
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_fail() -> eyre::Result<()> {
        let parties = generate_local_identities();

        let players = setup_local_grpc_networking(parties.clone()).await?;

        let mut jobs = JoinSet::new();

        // Send to a non-existing party
        {
            let alice = players[0].clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);
                alice.create_session(session_id).await.unwrap();

                let message = b"Hey, Eve. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("eve"), &session_id)
                    .await;
                assert!(res.is_err());
            });
        }

        // Receive from a wrong party
        {
            let alice = players[0].clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(3);
                alice.create_session(session_id).await.unwrap();

                let res = alice.receive(&Identity::from("eve"), &session_id).await;
                assert!(res.is_err());
            });
        }

        // Send to itself
        {
            let alice = players[0].clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);
                alice.create_session(session_id).await.unwrap();

                let message = b"Hey, Alice. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("alice"), &session_id)
                    .await;
                assert!(res.is_err());
            });
        }

        // Add the same session
        {
            let alice = players[0].clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);

                // Delay to ensure that the session is created in the previous example
                tokio::time::sleep(Duration::from_millis(100)).await;
                let res = alice.create_session(session_id).await;

                assert!(res.is_err());
            });
        }

        // Retrieve from a non-existing session
        {
            let alice = players[0].clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(50);

                let message = b"Hey, Bob. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("bob"), &session_id)
                    .await;
                assert!(res.is_err());
                let res = alice.receive(&Identity::from("bob"), &session_id).await;
                assert!(res.is_err());
            });
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test]
    async fn test_hnsw_local() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HawkSearcher::default();
        let mut vectors_and_graphs = LocalNetAby3NgStoreProtocol::shared_random_setup(
            &mut rng,
            database_size,
            crate::network::NetworkType::GrpcChannel,
        )
        .await
        .unwrap();

        for i in 0..database_size {
            let mut jobs = JoinSet::new();
            for (store, graph) in vectors_and_graphs.iter_mut() {
                let mut store = store.clone();
                let mut graph = graph.clone();
                let searcher = searcher.clone();
                jobs.spawn(async move {
                    let secret_neighbors = searcher
                        .search_to_insert(&mut store, &mut graph, &i.into())
                        .await;
                    searcher.is_match(&mut store, &secret_neighbors).await
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }
}
