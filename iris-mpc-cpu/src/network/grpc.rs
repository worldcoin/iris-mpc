use super::Networking;
use crate::{
    execution::{local::get_free_local_addresses, player::Identity},
    network::SessionId,
    proto_generated::party_node::{
        party_node_client::PartyNodeClient,
        party_node_server::{PartyNode, PartyNodeServer},
        SendRequest, SendResponse,
    },
};

use backon::{ExponentialBuilder, Retryable};
use eyre::eyre;
use std::{collections::HashMap, str::FromStr, sync::Arc, time::Duration};
use tokio::{
    sync::{
        mpsc::{self, UnboundedSender},
        Mutex, RwLock,
    },
    time::{sleep, timeout},
};
use tokio_stream::StreamExt;
use tonic::{
    async_trait,
    metadata::AsciiMetadataValue,
    transport::{Channel, Server},
    Request, Response, Status, Streaming,
};

type TonicResult<T> = Result<T, Status>;

type RwMap<K, V> = RwLock<HashMap<K, V>>;

fn err_to_status(e: eyre::Error) -> Status {
    Status::internal(e.to_string())
}

#[derive(Default)]
struct MessageQueueStore {
    queues: RwMap<Identity, Mutex<Streaming<SendRequest>>>,
}

impl MessageQueueStore {
    async fn insert(
        &self,
        sender_id: Identity,
        stream: Streaming<SendRequest>,
    ) -> eyre::Result<()> {
        let mut queues = self.queues.write().await;
        if queues.contains_key(&sender_id) {
            return Err(eyre!("Player {:?} already has a message queue", sender_id));
        }
        queues.insert(sender_id, Mutex::new(stream));
        Ok(())
    }

    async fn count_senders(&self) -> usize {
        self.queues.read().await.len()
    }

    async fn pop(&self, sender_id: &Identity) -> eyre::Result<Vec<u8>> {
        let queues = self.queues.read().await;
        let queue = queues.get(sender_id).ok_or(eyre!(format!(
            "RECEIVE: Sender {sender_id:?} hasn't been found in the message queues"
        )))?;

        let mut queue = queue.lock().await;

        let msg = queue.next().await.ok_or(eyre!("No message received"))??;

        Ok(msg.data)
    }
}

type Sender = UnboundedSender<SendRequest>;

#[derive(Default)]
struct OutgoingStreams {
    streams: RwMap<(SessionId, Identity), Arc<Sender>>,
}

impl OutgoingStreams {
    async fn add_session_stream(
        &self,
        session_id: SessionId,
        receiver_id: Identity,
        stream: Sender,
    ) {
        self.streams
            .write()
            .await
            .insert((session_id, receiver_id), Arc::new(stream));
    }

    async fn get_stream(
        &self,
        session_id: SessionId,
        receiver_id: Identity,
    ) -> eyre::Result<Arc<Sender>> {
        self.streams
            .read()
            .await
            .get(&(session_id, receiver_id.clone()))
            .ok_or(eyre!(
                "Streams for session {session_id:?} and receiver {receiver_id:?} not found"
            ))
            .map(Arc::clone)
    }

    async fn count_receivers(&self, session_id: SessionId) -> usize {
        self.streams
            .read()
            .await
            .iter()
            .filter(|((sid, _), _)| *sid == session_id)
            .count()
    }
}

#[derive(Default, Clone)]
pub struct GrpcConfig {
    pub timeout_duration: Duration,
}

// WARNING: this implementation assumes that messages for a specific player
// within one session are sent in order and consecutively. Don't send messages
// to the same player in parallel within the same session. Use batching instead.
#[derive(Clone)]
pub struct GrpcNetworking {
    party_id: Identity,
    // other party id -> client to call that party
    clients: Arc<RwMap<Identity, PartyNodeClient<Channel>>>,
    // other party id -> outgoing streams to send messages to that party in different sessions
    outgoing_streams: Arc<OutgoingStreams>,
    // session id -> incoming message streams
    message_queues: Arc<RwMap<SessionId, MessageQueueStore>>,

    pub config: GrpcConfig,
}

impl GrpcNetworking {
    pub fn new(party_id: Identity, config: GrpcConfig) -> Self {
        GrpcNetworking {
            party_id,
            clients: Arc::new(RwLock::new(HashMap::new())),
            outgoing_streams: Arc::new(OutgoingStreams::default()),
            message_queues: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    // TODO: from config?
    fn backoff(&self) -> ExponentialBuilder {
        ExponentialBuilder::new()
            .with_min_delay(std::time::Duration::from_millis(500))
            .with_factor(1.1)
            .with_max_delay(std::time::Duration::from_secs(5))
            .with_max_times(27) // about 60 seconds overall delay
    }

    pub async fn connect_to_party(&self, party_id: Identity, address: &str) -> eyre::Result<()> {
        let client = (|| async { PartyNodeClient::connect(address.to_string()).await })
            .retry(self.backoff())
            .sleep(tokio::time::sleep)
            .await?;
        tracing::trace!(
            "Player {:?} connected to player {:?} at address {:?}",
            self.party_id,
            party_id,
            address
        );
        let mut clients = self.clients.write().await;
        clients.insert(party_id.clone(), client);
        Ok(())
    }

    pub async fn create_session(&self, session_id: SessionId) -> eyre::Result<()> {
        if self.outgoing_streams.count_receivers(session_id).await > 0 {
            return Err(eyre!(
                "Player {:?} has already created session {session_id:?}",
                self.party_id
            ));
        }

        let mut clients = self.clients.write().await;
        for (client_key, client) in clients.iter_mut() {
            let (tx, rx) = mpsc::unbounded_channel();
            tracing::trace!(
                "Player {:?} is adding outgoing stream of session {:?} for player {:?}",
                self.party_id,
                session_id,
                client_key
            );
            self.outgoing_streams
                .add_session_stream(session_id, client_key.clone(), tx)
                .await;
            let receiving_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
            let mut request = Request::new(receiving_stream);
            request.metadata_mut().insert(
                "sender_id",
                AsciiMetadataValue::from_str(&self.party_id.0).unwrap(),
            );
            request.metadata_mut().insert(
                "session_id",
                AsciiMetadataValue::from_str(&session_id.0.to_string()).unwrap(),
            );
            tracing::trace!(
                "Player {:?} is sending incoming stream of session {:?} for player {:?}",
                self.party_id,
                session_id,
                client_key
            );
            let _response = client.start_message_stream(request).await?;
        }
        Ok(())
    }

    pub async fn is_session_ready(&self, session_id: SessionId) -> bool {
        let n_senders = match self.message_queues.read().await.get(&session_id) {
            None => 0,
            Some(q) => q.count_senders().await,
        };

        let clients = self.clients.read().await;
        if n_senders != clients.len() {
            return false;
        }

        self.outgoing_streams.count_receivers(session_id).await == clients.len()
    }

    pub async fn wait_for_session(&self, session_id: SessionId) {
        while !self.is_session_ready(session_id).await {
            sleep(Duration::from_millis(100)).await;
        }
    }
}

// Server implementation
#[async_trait]
impl PartyNode for GrpcNetworking {
    async fn start_message_stream(
        &self,
        request: Request<Streaming<SendRequest>>,
    ) -> TonicResult<Response<SendResponse>> {
        let sender_id: Identity = request
            .metadata()
            .get("sender_id")
            .ok_or(Status::unauthenticated("Sender ID not found"))?
            .to_str()
            .map_err(|_| Status::unauthenticated("Sender ID is not a string"))?
            .to_string()
            .into();
        if sender_id == self.party_id {
            return Err(Status::unauthenticated(format!(
                "Sender ID coincides with receiver ID: {:?}",
                sender_id
            )));
        }
        let session_id: u64 = request
            .metadata()
            .get("session_id")
            .ok_or(Status::not_found("Session ID not found"))?
            .to_str()
            .map_err(|_| Status::not_found("Session ID malformed"))?
            .parse()
            .map_err(|_| Status::invalid_argument("Session ID is not a u64 number"))?;
        let session_id = SessionId::from(session_id);

        tracing::trace!(
            "Player {:?} received incoming stream from {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        let incoming_stream = request.into_inner();

        tracing::trace!(
            "Player {:?}. Creating session {:?} for player {:?}",
            self.party_id,
            session_id,
            sender_id
        );

        let mut message_queues = self.message_queues.write().await;
        let message_queue = message_queues
            .entry(session_id)
            .or_insert(MessageQueueStore::default());

        message_queue
            .insert(sender_id, incoming_stream)
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
        tracing::trace!(target: "searcher::network", action = "send", party = ?receiver, bytes = value.len(), rounds = 1);
        let outgoing_stream = self
            .outgoing_streams
            .get_stream(*session_id, receiver.clone())
            .await?;

        // Send message via the outgoing stream
        let request = SendRequest { data: value };
        (|| async {
            tracing::trace!(
                "INIT: Sending message {:?} from {:?} to {:?} in session {:?}",
                request.data,
                self.party_id,
                receiver,
                session_id
            );
            outgoing_stream
                .send(request.clone())
                .map_err(|e| eyre!(e.to_string()))?;
            tracing::trace!(
                "SUCCESS: Sending message {:?} from {:?} to {:?} in session {:?}",
                request.data,
                self.party_id,
                receiver,
                session_id
            );
            Ok(())
        })
        .retry(self.backoff())
        .sleep(tokio::time::sleep)
        .await
    }

    async fn receive(&self, sender: &Identity, session_id: &SessionId) -> eyre::Result<Vec<u8>> {
        // Just retrieve the first message from the corresponding queue
        let messages_queues = self.message_queues.read().await;
        let queue = messages_queues.get(session_id).ok_or(eyre!(format!(
            "Session {session_id:?} hasn't been added to message queues"
        )))?;

        tracing::trace!(
            "Player {:?} is receiving message from {:?} in session {:?}",
            self.party_id,
            sender,
            session_id
        );

        match timeout(self.config.timeout_duration, queue.pop(sender)).await {
            Ok(res) => res,
            Err(_) => Err(eyre!(
                "Timeout while waiting for message from {sender:?} in session {session_id:?}"
            )),
        }
    }
}

pub async fn setup_local_grpc_networking(
    parties: Vec<Identity>,
) -> eyre::Result<Vec<GrpcNetworking>> {
    let config = GrpcConfig {
        timeout_duration: Duration::from_secs(1),
    };

    let players = parties
        .iter()
        .map(|party| GrpcNetworking::new(party.clone(), config.clone()))
        .collect::<Vec<GrpcNetworking>>();

    let addresses = get_free_local_addresses(players.len()).await?;

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
        hawkers::aby3::{aby3_store::prepare_query, test_utils::shared_random_setup},
        hnsw::HnswSearcher,
    };
    use aes_prng::AesRng;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    async fn create_session_helper(
        session_id: SessionId,
        players: &[GrpcNetworking],
    ) -> eyre::Result<()> {
        let mut jobs = JoinSet::new();
        for player in players.iter() {
            let player = player.clone();
            jobs.spawn(async move {
                tracing::trace!(
                    "Player {:?} is creating session {:?}",
                    player.party_id,
                    session_id
                );
                player.create_session(session_id).await.unwrap();
            });
        }
        jobs.join_all().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_correct() -> eyre::Result<()> {
        let identities = generate_local_identities();
        let players = setup_local_grpc_networking(identities.clone()).await?;

        let mut jobs = JoinSet::new();

        // Simple session with one message sent from one party to another
        {
            let players = players.clone();

            let session_id = SessionId::from(0);

            jobs.spawn(async move {
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();
                let bob = players[1].clone();

                // Send a message from the first party to the second party
                let message = b"Hey, Bob. I'm Alice. Do you copy?".to_vec();
                let message_copy = message.clone();

                let task1 = tokio::spawn(async move {
                    alice
                        .send(message.clone(), &"bob".into(), &session_id)
                        .await
                        .unwrap();
                });
                let task2 = tokio::spawn(async move {
                    let received_message = bob.receive(&"alice".into(), &session_id).await.unwrap();
                    assert_eq!(message_copy, received_message);
                });
                let _ = tokio::try_join!(task1, task2).unwrap();
            });
        }

        // Each party sending and receiving messages to each other
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);

                create_session_helper(session_id, &players).await.unwrap();

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

        // Parties create a session consecutively
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);

                for player in players.iter() {
                    player.create_session(session_id).await.unwrap();
                }
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
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(0);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();
                let message = b"Hey, Eve. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("eve"), &session_id)
                    .await;
                assert!(res.is_err());
            });
        }

        // Receive from a wrong party
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let res = alice.receive(&Identity::from("eve"), &session_id).await;
                assert!(res.is_err());
            });
        }

        // Send to itself
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let message = b"Hey, Alice. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("alice"), &session_id)
                    .await;
                assert!(res.is_err());
            });
        }

        // Add the same session
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(3);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let res = alice.create_session(session_id).await;

                assert!(res.is_err());
            });
        }

        // Send and retrieve from a non-existing session
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

        // Receive from a party that didn't send a message
        {
            let alice = players[0].clone();
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);
                create_session_helper(session_id, &players).await.unwrap();

                let res = alice.receive(&Identity::from("bob"), &session_id).await;
                assert!(res.is_err());
            });
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_hnsw_local() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HnswSearcher::default();
        let mut vectors_and_graphs = shared_random_setup(
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
                let q = store.storage.get_vector(&i.into()).await;
                let q = prepare_query((*q).clone());
                jobs.spawn(async move {
                    let secret_neighbors = searcher.search(&mut store, &mut graph, &q, 1).await;
                    searcher.is_match(&mut store, &[secret_neighbors]).await
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }
}
