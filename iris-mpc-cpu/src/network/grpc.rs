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
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot, Mutex,
    },
    time::{sleep, timeout},
};
use tonic::{
    async_trait,
    metadata::AsciiMetadataValue,
    transport::{Channel, Server},
    Request, Response, Status, Streaming,
};

type TonicResult<T> = Result<T, Status>;

fn err_to_status(e: eyre::Error) -> Status {
    Status::internal(e.to_string())
}

type Receiver = Arc<Mutex<UnboundedReceiver<SendRequest>>>;

#[derive(Default, Clone)]
struct MessageQueueStore {
    queues: HashMap<Identity, Receiver>,
}

impl MessageQueueStore {
    fn insert(
        &mut self,
        sender_id: Identity,
        stream: UnboundedReceiver<SendRequest>,
    ) -> eyre::Result<()> {
        if self.queues.contains_key(&sender_id) {
            return Err(eyre!("Player {:?} already has a message queue", sender_id));
        }
        self.queues.insert(sender_id, Arc::new(Mutex::new(stream)));
        Ok(())
    }

    fn count_senders(&self) -> usize {
        self.queues.len()
    }

    fn get_sender_queue(&self, sender_id: &Identity) -> eyre::Result<Receiver> {
        self.queues
            .get(sender_id)
            .ok_or(eyre!(format!(
                "Sender {sender_id:?} hasn't been found in the message queues"
            )))
            .cloned()
    }
}

type Sender = UnboundedSender<SendRequest>;

#[derive(Default, Clone)]
struct OutgoingStreams {
    streams: HashMap<(SessionId, Identity), Sender>,
}

impl OutgoingStreams {
    fn add_session_stream(&mut self, session_id: SessionId, receiver_id: Identity, stream: Sender) {
        self.streams.insert((session_id, receiver_id), stream);
    }

    fn get_stream(&self, session_id: SessionId, receiver_id: Identity) -> eyre::Result<Sender> {
        self.streams
            .get(&(session_id, receiver_id.clone()))
            .ok_or(eyre!(
                "Streams for session {session_id:?} and receiver {receiver_id:?} not found"
            ))
            .cloned()
    }

    fn count_receivers(&self, session_id: SessionId) -> usize {
        self.streams
            .iter()
            .filter(|((sid, _), _)| *sid == session_id)
            .count()
    }
}

struct SendTask {
    value: Vec<u8>,
    receiver: Identity,
    session_id: SessionId,
}

struct ReceiveTask {
    sender: Identity,
    session_id: SessionId,
}

struct StartMessageStreamTask {
    sender: Identity,
    session_id: SessionId,
    stream: UnboundedReceiver<SendRequest>,
}

struct ConnectToPartyTask {
    party_id: Identity,
    address: String,
}

enum GrpcTask {
    ConnectToParty(ConnectToPartyTask),
    CreateSession(SessionId),
    StartMessageStream(StartMessageStreamTask),
    IsSessionReady(SessionId),
    Send(SendTask),
    Receive(ReceiveTask),
}

enum MessageResult {
    Empty,
    IsSessionReady(bool),
    ReceivedBytes(Vec<u8>),
}

struct MessageJob {
    task: GrpcTask,
    return_channel: oneshot::Sender<eyre::Result<MessageResult>>,
}

// Concurrency handler for networking operations
#[derive(Clone)]
pub struct GrpcHandle {
    job_queue: mpsc::Sender<MessageJob>,
    party_id: Identity,
}

impl GrpcHandle {
    pub async fn new(mut grpc: GrpcNetworking) -> eyre::Result<Self> {
        let party_id = grpc.party_id.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<MessageJob>(1);

        // Loop to handle incoming tasks from job queue
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                match job.task {
                    GrpcTask::Send(task) => {
                        let job_result = grpc
                            .send(task.value, &task.receiver, &task.session_id)
                            .map(|_| MessageResult::Empty);
                        let _ = job.return_channel.send(job_result);
                    }
                    GrpcTask::Receive(task) => {
                        let sender = task.sender.clone();
                        let session_id = task.session_id;
                        let queue = grpc.get_message_queue(&sender, &session_id);
                        let party_id = grpc.party_id.clone();
                        match queue {
                            Ok(q) => {
                                tokio::spawn(async move {
                                    let mut q = q.lock().await;
                                    let job_result = match timeout(grpc.config.timeout_duration, q.recv()).await {
                                        Ok(res) => res.ok_or(eyre!("No message received")).map(|msg| MessageResult::ReceivedBytes(msg.data)),
                                        Err(_) => Err(eyre!(
                                            "Party {party_id:?}: Timeout while waiting for message from {sender:?} in session \
                                             {session_id:?}"
                                        )),
                                    };
                                    let _ = job.return_channel.send(job_result);
                                });
                            }
                            Err(e) => {
                                let _ = job.return_channel.send(Err(e));
                            }
                        }
                    }
                    GrpcTask::CreateSession(session_id) => {
                        let job_result = grpc
                            .create_session(session_id)
                            .await
                            .map(|_| MessageResult::Empty);
                        let _ = job.return_channel.send(job_result);
                    }
                    GrpcTask::IsSessionReady(session_id) => {
                        let job_result = Ok(MessageResult::IsSessionReady(
                            grpc.is_session_ready(session_id),
                        ));
                        let _ = job.return_channel.send(job_result);
                    }
                    GrpcTask::ConnectToParty(task) => {
                        let job_result = grpc
                            .connect_to_party(task.party_id, &task.address)
                            .await
                            .map(|_| MessageResult::Empty);
                        let _ = job.return_channel.send(job_result);
                    }
                    GrpcTask::StartMessageStream(task) => {
                        let job_result = grpc
                            .start_message_stream(task.sender, task.session_id, task.stream)
                            .await
                            .map(|_| MessageResult::Empty);
                        let _ = job.return_channel.send(job_result);
                    }
                }
            }
        });

        Ok(GrpcHandle {
            party_id,
            job_queue: tx,
        })
    }

    // Send a task to the job queue and wait for the result
    async fn submit(&self, task: GrpcTask) -> eyre::Result<MessageResult> {
        let (tx, rx) = oneshot::channel();
        let job = MessageJob {
            task,
            return_channel: tx,
        };
        self.job_queue.send(job).await?;
        rx.await?
    }
}

// Networking I/O operations
#[async_trait]
impl Networking for GrpcHandle {
    async fn send(
        &self,
        value: Vec<u8>,
        receiver: &Identity,
        session_id: &SessionId,
    ) -> eyre::Result<()> {
        let task = GrpcTask::Send(SendTask {
            value,
            receiver: receiver.clone(),
            session_id: *session_id,
        });
        let _ = self.submit(task).await?;
        Ok(())
    }

    async fn receive(&self, sender: &Identity, session_id: &SessionId) -> eyre::Result<Vec<u8>> {
        let task = GrpcTask::Receive(ReceiveTask {
            sender: sender.clone(),
            session_id: *session_id,
        });
        let res = self.submit(task).await?;
        match res {
            MessageResult::ReceivedBytes(bytes) => Ok(bytes),
            _ => Err(eyre!("No message received")),
        }
    }
}

// Server implementation
#[async_trait]
impl PartyNode for GrpcHandle {
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
        let session_id: u64 = request
            .metadata()
            .get("session_id")
            .ok_or(Status::not_found("Session ID not found"))?
            .to_str()
            .map_err(|_| Status::not_found("Session ID malformed"))?
            .parse()
            .map_err(|_| Status::invalid_argument("Session ID is not a u64 number"))?;
        let session_id = SessionId::from(session_id);

        let mut incoming_stream = request.into_inner();

        tracing::debug!(
            "Player {:?} is starting message stream with player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        let (tx, rx) = mpsc::unbounded_channel();
        // Spawn a task to receive messages from the incoming stream
        tokio::spawn(async move {
            while let Some(req) = incoming_stream.message().await.unwrap_or(None) {
                let _ = tx.send(req);
            }
        });

        tracing::debug!(
            "Player {:?} has spawned message loop with player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        let task = StartMessageStreamTask {
            sender: sender_id.clone(),
            session_id,
            stream: rx,
        };
        let task = GrpcTask::StartMessageStream(task);
        let _ = self.submit(task).await.map_err(err_to_status)?;

        tracing::debug!(
            "Player {:?} has started message stream with player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        Ok(Response::new(SendResponse {}))
    }
}

// Connection and session management
impl GrpcHandle {
    pub async fn connect_to_party(&self, party_id: Identity, address: &str) -> eyre::Result<()> {
        let task = ConnectToPartyTask {
            party_id,
            address: address.to_string(),
        };
        let task = GrpcTask::ConnectToParty(task);
        let _ = self.submit(task).await?;
        Ok(())
    }

    pub async fn create_session(&self, session_id: SessionId) -> eyre::Result<()> {
        let task = GrpcTask::CreateSession(session_id);
        let _ = self.submit(task).await?;
        Ok(())
    }

    // This function should be called after all parties have called `create_session`
    pub async fn wait_for_session(&self, session_id: SessionId) -> eyre::Result<()> {
        while matches!(
            self.submit(GrpcTask::IsSessionReady(session_id)).await?,
            MessageResult::IsSessionReady(false)
        ) {
            tracing::debug!(
                "Player {:?} is waiting for session {:?} to be ready",
                self.party_id,
                session_id
            );
            sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct GrpcConfig {
    pub timeout_duration: Duration,
}

// WARNING: this implementation assumes that messages for a specific player
// within one session are sent in order and consecutively. Don't send messages
// to the same player in parallel within the same session. Use batching instead.
pub struct GrpcNetworking {
    party_id: Identity,
    // other party id -> client to call that party
    clients: HashMap<Identity, PartyNodeClient<Channel>>,
    // other party id -> outgoing streams to send messages to that party in different sessions
    outgoing_streams: OutgoingStreams,
    // session id -> incoming message streams
    message_queues: HashMap<SessionId, MessageQueueStore>,

    pub config: GrpcConfig,
}

impl GrpcNetworking {
    pub fn new(party_id: Identity, config: GrpcConfig) -> Self {
        GrpcNetworking {
            party_id,
            clients: HashMap::new(),
            outgoing_streams: OutgoingStreams::default(),
            message_queues: HashMap::new(),
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

    pub async fn connect_to_party(
        &mut self,
        party_id: Identity,
        address: &str,
    ) -> eyre::Result<()> {
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
        self.clients.insert(party_id.clone(), client);
        Ok(())
    }

    pub async fn create_session(&mut self, session_id: SessionId) -> eyre::Result<()> {
        if self.outgoing_streams.count_receivers(session_id) > 0 {
            return Err(eyre!(
                "Player {:?} has already created session {session_id:?}",
                self.party_id
            ));
        }

        for (client_id, client) in self.clients.iter() {
            let (tx, rx) = mpsc::unbounded_channel();
            tracing::trace!(
                "Player {:?} is adding outgoing stream of session {:?} for player {:?}",
                self.party_id,
                session_id,
                client_id
            );
            self.outgoing_streams
                .add_session_stream(session_id, client_id.clone(), tx);
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
            let mut client = client.clone();
            tokio::spawn(async move {
                let _response = client.start_message_stream(request).await.unwrap();
            });
            tracing::debug!(
                "Player {:?} has created session {:?} with player {:?}",
                self.party_id,
                session_id,
                client_id
            );
        }
        Ok(())
    }

    pub fn is_session_ready(&self, session_id: SessionId) -> bool {
        let n_senders = match self.message_queues.get(&session_id) {
            None => 0,
            Some(q) => q.count_senders(),
        };

        if n_senders != self.clients.len() {
            return false;
        }

        self.outgoing_streams.count_receivers(session_id) == self.clients.len()
    }
}

// Server implementation
impl GrpcNetworking {
    async fn start_message_stream(
        &mut self,
        sender_id: Identity,
        session_id: SessionId,
        stream: UnboundedReceiver<SendRequest>,
    ) -> eyre::Result<()> {
        if sender_id == self.party_id {
            return Err(eyre!(
                "Sender ID coincides with receiver ID: {:?}",
                sender_id
            ));
        }

        tracing::debug!(
            "Player {:?} is adding message queue from player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        let message_queue = self.message_queues.entry(session_id).or_default();

        message_queue
            .insert(sender_id.clone(), stream)
            .map_err(err_to_status)?;

        tracing::debug!(
            "Player {:?} has added message queue from player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        Ok(())
    }
}

// I/O operations
impl GrpcNetworking {
    fn send(
        &self,
        value: Vec<u8>,
        receiver: &Identity,
        session_id: &SessionId,
    ) -> eyre::Result<()> {
        tracing::trace!(target: "searcher::network", action = "send", party = ?receiver, bytes = value.len(), rounds = 1);
        let outgoing_stream = self
            .outgoing_streams
            .get_stream(*session_id, receiver.clone())?;

        // Send message via the outgoing stream
        let request = SendRequest { data: value };
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
    }

    fn get_message_queue(
        &mut self,
        sender: &Identity,
        session_id: &SessionId,
    ) -> eyre::Result<Receiver> {
        let session_queue = self
            .message_queues
            .get_mut(session_id)
            .ok_or(eyre!(format!(
                "Session {session_id:?} hasn't been added to message queues"
            )))?;

        session_queue.get_sender_queue(sender)
    }
}

pub async fn setup_local_grpc_networking(parties: Vec<Identity>) -> eyre::Result<Vec<GrpcHandle>> {
    let config = GrpcConfig {
        timeout_duration: Duration::from_secs(5),
    };

    let nets = parties
        .iter()
        .map(|party| GrpcNetworking::new(party.clone(), config.clone()))
        .collect::<Vec<GrpcNetworking>>();

    // Create handles consecutively to preserve the order of players
    let mut players = Vec::with_capacity(nets.len());
    for net in nets {
        players.push(GrpcHandle::new(net).await?);
    }

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

    tracing::debug!("Players connected to each other");

    Ok(players)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::{local::generate_local_identities, player::Role},
        hawkers::aby3_store::Aby3Store,
        hnsw::HnswSearcher,
    };
    use aes_prng::AesRng;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    async fn create_session_helper(
        session_id: SessionId,
        players: &[GrpcHandle],
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
                player.wait_for_session(session_id).await.unwrap();
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

        // Multiple parties sending messages to each other
        let all_parties_talk = |identities: Vec<Identity>,
                                players: Vec<GrpcHandle>,
                                session_id: SessionId| async move {
            let mut tasks = JoinSet::new();
            for (player_id, player) in players.iter().enumerate() {
                let role = Role::new(player_id);
                let next = role.next(3).index();
                let prev = role.prev(3).index();

                let player = player.clone();
                let next_id = identities[next].clone();
                let prev_id = identities[prev].clone();

                let message_to_next =
                    format!("From player {} to player {} with love", player_id, next).into_bytes();
                let message_to_prev =
                    format!("From player {} to player {} with love", player_id, prev).into_bytes();

                tasks.spawn(async move {
                    // Sending
                    player
                        .send(message_to_next.clone(), &next_id, &session_id)
                        .await
                        .unwrap();
                    player
                        .send(message_to_prev.clone(), &prev_id, &session_id)
                        .await
                        .unwrap();

                    // Receiving
                    let received_message_from_prev =
                        player.receive(&prev_id, &session_id).await.unwrap();
                    let expected_message_from_prev =
                        format!("From player {} to player {} with love", prev, player_id)
                            .into_bytes();
                    assert_eq!(received_message_from_prev, expected_message_from_prev);
                    let received_message_from_next =
                        player.receive(&next_id, &session_id).await.unwrap();
                    let expected_message_from_next =
                        format!("From player {} to player {} with love", next, player_id)
                            .into_bytes();
                    assert_eq!(received_message_from_next, expected_message_from_next);
                });
            }
            tasks.join_all().await;
        };

        // Each party sending and receiving messages to each other
        {
            let players = players.clone();
            let identities = identities.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);

                create_session_helper(session_id, &players).await.unwrap();

                // Test that parties can send and receive messages
                all_parties_talk(identities, players, session_id).await;
            });
        }

        // Parties create a session consecutively
        {
            let players = players.clone();
            let session_id = SessionId::from(2);
            // Session is consecutively created
            {
                let players = players.clone();
                let mut create_session_jobs = JoinSet::new();
                create_session_jobs.spawn(async move {
                    let session_id = SessionId::from(2);

                    for player in players.iter() {
                        player.create_session(session_id).await.unwrap();
                    }

                    // Wait for all parties to create the session
                    for player in players.iter() {
                        player.wait_for_session(session_id).await.unwrap();
                    }
                });
                create_session_jobs.join_all().await;
            }

            // Test that parties can send and receive messages
            all_parties_talk(identities, players, session_id).await;
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

        {
            // Send to a non-existing party
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(0);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();
                let message = b"Hey, Eve. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("eve"), &session_id)
                    .await;
                assert_eq!(
                    "Streams for session SessionId(0) and receiver Identity(\"eve\") not found",
                    res.unwrap_err().to_string()
                );
            });
        }

        {
            // Receive from a wrong party
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let res = alice.receive(&Identity::from("eve"), &session_id).await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Sender Identity(\"eve\") hasn't been found in the message queues"
                );
            });
        }

        {
            // Send to itself
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let message = b"Hey, Alice. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("alice"), &session_id)
                    .await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Streams for session SessionId(2) and receiver Identity(\"alice\") not found"
                );
            });
        }

        {
            // Add the same session
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(3);
                create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let res = alice.create_session(session_id).await;

                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Player Identity(\"alice\") has already created session SessionId(3)"
                );
            });
        }

        {
            // Send and retrieve from a non-existing session
            let alice = players[0].clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(50);

                let message = b"Hey, Bob. I'm Alice. Do you copy?".to_vec();
                let res = alice
                    .send(message.clone(), &Identity::from("bob"), &session_id)
                    .await;
                assert!(res.is_err());
                let res = alice.receive(&Identity::from("bob"), &session_id).await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Session SessionId(50) hasn't been added to message queues"
                );
            });
        }

        {
            // Receive from a party that didn't send a message (timeout error)

            let alice = players[0].clone();
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);
                create_session_helper(session_id, &players).await.unwrap();

                let res = alice.receive(&Identity::from("bob"), &session_id).await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Party Identity(\"alice\"): Timeout while waiting for message from \
                     Identity(\"bob\") in session SessionId(4)"
                );
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
        let mut vectors_and_graphs = Aby3Store::shared_random_setup(
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
                let q = store.prepare_query((*q).clone());
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
