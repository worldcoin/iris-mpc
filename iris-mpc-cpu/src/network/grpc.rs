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
use eyre::{eyre, Result};
use futures::future::JoinAll;
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
    time::Duration,
};
use tokio::{
    sync::{
        mpsc::{self, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    time::{sleep, timeout},
};
use tonic::{
    async_trait,
    metadata::AsciiMetadataValue,
    transport::{Channel, Server},
    Request, Response, Status, Streaming,
};
use tracing::trace;

type TonicResult<T> = Result<T, Status>;

fn err_to_status(e: eyre::Error) -> Status {
    Status::internal(e.to_string())
}

type OutStream = UnboundedSender<SendRequest>;
type OutStreams = HashMap<Identity, OutStream>;
type InStream = UnboundedReceiver<SendRequest>;
type InStreams = HashMap<Identity, InStream>;

#[derive(Debug)]
pub struct GrpcSession {
    pub session_id: SessionId,
    pub own_identity: Identity,
    pub out_streams: HashMap<Identity, OutStream>,
    pub in_streams: HashMap<Identity, InStream>,
    pub config: GrpcConfig,
}

#[async_trait]
impl Networking for GrpcSession {
    async fn send(&self, value: Vec<u8>, receiver: &Identity) -> Result<()> {
        let outgoing_stream = self.out_streams.get(receiver).ok_or(eyre!(
            "Outgoing stream for {receiver:?} in session {:?} not found",
            self.session_id
        ))?;
        trace!(target: "searcher::network", action = "send", party = ?receiver, bytes = value.len(), rounds = 1);
        metrics::counter!(
            "smpc.rounds",
            "session_id" => self.session_id.0.to_string(),
        )
        .increment(1);
        metrics::counter!(
            "smpc.bytes",
            "session_id" => self.session_id.0.to_string(),
        )
        .increment(value.len() as u64);
        let request = SendRequest { data: value };
        outgoing_stream
            .send(request)
            .map_err(|e| eyre!(e.to_string()))?;
        Ok(())
    }

    async fn receive(&mut self, sender: &Identity) -> Result<Vec<u8>> {
        let incoming_stream = self.in_streams.get_mut(sender).ok_or(eyre!(
            "Incoming stream for {sender:?} in session {:?} not found",
            self.session_id
        ))?;
        match timeout(self.config.timeout_duration, incoming_stream.recv()).await {
            Ok(res) => res.ok_or(eyre!("No message received")).map(|msg| msg.data),
            Err(_) => Err(eyre!(
                "Party {:?}: Timeout while waiting for message from {sender:?} in session \
                 {:?}",
                self.own_identity,
                self.session_id
            )),
        }
    }
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
    CreateOutgoingStreams(SessionId),
    CreateIncomingStreams(SessionId),
    StartMessageStream(StartMessageStreamTask),
    IsSessionReady(SessionId),
}

enum MessageResult {
    Empty,
    IsSessionReady(bool),
    OutgoingStreams(OutStreams),
    IncomingStreams(InStreams),
}

struct MessageJob {
    task: GrpcTask,
    return_channel: oneshot::Sender<Result<MessageResult>>,
}

// Concurrency handler for networking operations
#[derive(Clone)]
pub struct GrpcHandle {
    job_queue: mpsc::Sender<MessageJob>,
    party_id: Identity,
    config: GrpcConfig,
}

impl GrpcHandle {
    pub async fn new(mut grpc: GrpcNetworking) -> Result<Self> {
        let party_id = grpc.party_id.clone();
        let config = grpc.config.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<MessageJob>(1);

        // Loop to handle incoming tasks from job queue
        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                match job.task {
                    GrpcTask::CreateOutgoingStreams(session_id) => {
                        let job_result = grpc
                            .create_outgoing_streams(session_id)
                            .await
                            .map(MessageResult::OutgoingStreams);
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
                    GrpcTask::CreateIncomingStreams(session_id) => {
                        let job_result = grpc
                            .create_incoming_streams(session_id)
                            .await
                            .map(MessageResult::IncomingStreams);
                        let _ = job.return_channel.send(job_result);
                    }
                }
            }
        });

        Ok(GrpcHandle {
            party_id,
            job_queue: tx,
            config,
        })
    }

    // Send a task to the job queue and wait for the result
    async fn submit(&self, task: GrpcTask) -> Result<MessageResult> {
        let (tx, rx) = oneshot::channel();
        let job = MessageJob {
            task,
            return_channel: tx,
        };
        self.job_queue.send(job).await?;
        rx.await?
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
    pub async fn connect_to_party(&self, party_id: Identity, address: &str) -> Result<()> {
        let task = ConnectToPartyTask {
            party_id,
            address: address.to_string(),
        };
        let task = GrpcTask::ConnectToParty(task);
        let _ = self.submit(task).await?;
        Ok(())
    }

    pub async fn create_session(&self, session_id: SessionId) -> Result<GrpcSession> {
        // Create outgoing streams and ask other parties to send incoming streams
        let task = GrpcTask::CreateOutgoingStreams(session_id);
        let res = self.submit(task).await?;
        let outstreams = match res {
            MessageResult::OutgoingStreams(streams) => Ok(streams),
            _ => Err(eyre!("Wrong result type while creating outgoing streams")),
        }?;

        // Wait for incoming streams to be created and sent by other parties
        self.wait_for_session(session_id).await?;

        // Fetch incoming streams from GrpcNetworking
        let task = GrpcTask::CreateIncomingStreams(session_id);
        let res = self.submit(task).await?;
        let instreams = match res {
            MessageResult::IncomingStreams(streams) => Ok(streams),
            _ => Err(eyre!("Wrong result type while creating incoming streams")),
        }?;

        Ok(GrpcSession {
            session_id,
            own_identity: self.party_id.clone(),
            out_streams: outstreams,
            in_streams: instreams,
            config: self.config.clone(),
        })
    }

    // This function should be called after all parties have called `create_session`
    pub async fn wait_for_session(&self, session_id: SessionId) -> Result<()> {
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

#[derive(Default, Clone, Debug)]
pub struct GrpcConfig {
    pub timeout_duration: Duration,
    pub connection_parallelism: usize,
}

// WARNING: this implementation assumes that messages for a specific player
// within one session are sent in order and consecutively. Don't send messages
// to the same player in parallel within the same session. Use batching instead.
pub struct GrpcNetworking {
    party_id: Identity,
    // other party id -> client to call that party
    clients: HashMap<Identity, Vec<PartyNodeClient<Channel>>>,
    // session_id -> incoming streams
    instreams: HashMap<SessionId, InStreams>,
    // sessions in use
    // TODO: deletion logic
    active_sessions: HashSet<SessionId>,

    pub config: GrpcConfig,
}

impl GrpcNetworking {
    pub fn new(party_id: Identity, config: GrpcConfig) -> Self {
        GrpcNetworking {
            party_id,
            clients: HashMap::new(),
            instreams: HashMap::new(),
            active_sessions: HashSet::new(),
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

    pub async fn connect_to_party(&mut self, party_id: Identity, address: &str) -> Result<()> {
        if self.clients.contains_key(&party_id) {
            bail!(
                "Player {:?} has already connected to player {:?}",
                self.party_id,
                party_id
            );
        }
        let clients = (0..self.config.connection_parallelism.max(1))
            .map(|_| {
                let address = address.to_string();
                (move || PartyNodeClient::connect(address.clone()))
                    .retry(self.backoff())
                    .sleep(tokio::time::sleep)
            })
            .map(tokio::spawn)
            .collect::<JoinAll<_>>()
            .await
            .into_iter()
            .collect::<Result<Result<Vec<PartyNodeClient<_>>, _>, _>>()??;
        tracing::trace!(
            "Player {:?} connected to player {:?} at address {:?}",
            self.party_id,
            party_id,
            address
        );
        self.clients.insert(party_id.clone(), clients);
        Ok(())
    }

    pub async fn create_outgoing_streams(&self, session_id: SessionId) -> Result<OutStreams> {
        if self.active_sessions.contains(&session_id) {
            bail!(
                "Session {:?} has already been created by player {:?}",
                session_id,
                self.party_id
            );
        }
        let mut out_streams = HashMap::new();
        for (client_id, clients) in self.clients.iter() {
            let (tx, rx) = mpsc::unbounded_channel();
            tracing::trace!(
                "Player {:?} is adding outgoing stream of session {:?} for player {:?}",
                self.party_id,
                session_id,
                client_id
            );
            out_streams.insert(client_id.clone(), tx);
            let receiving_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);
            let mut request = Request::new(receiving_stream);
            request.metadata_mut().insert(
                "sender_id",
                AsciiMetadataValue::from_str(&self.party_id.0)
                    .map_err(|e| eyre!("Failed to convert Sender ID to ASCII: {e}"))?,
            );
            request.metadata_mut().insert(
                "session_id",
                AsciiMetadataValue::from_str(&session_id.0.to_string())
                    .map_err(|e| eyre!("Failed to convert Session ID to ASCII: {e}"))?,
            );

            let round_robin = (session_id.0 as usize) % clients.len();
            let mut client = clients[round_robin].clone();
            tokio::spawn(async move {
                let _response = client.start_message_stream(request).await?;
                Ok::<_, Status>(())
            });
            tracing::debug!(
                "Player {:?} has created session {:?} with player {:?}",
                self.party_id,
                session_id,
                client_id
            );
        }
        Ok(out_streams)
    }

    pub fn is_session_ready(&self, session_id: SessionId) -> bool {
        let n_senders = match self.instreams.get(&session_id) {
            None => 0,
            Some(q) => q.len(),
        };

        n_senders == self.clients.len()
    }

    pub async fn create_incoming_streams(&mut self, session_id: SessionId) -> Result<InStreams> {
        self.active_sessions.insert(session_id);
        self.instreams.remove(&session_id).ok_or(eyre!(format!(
            "Session {session_id:?} hasn't been added to message queues"
        )))
    }
}

// Server implementation
impl GrpcNetworking {
    async fn start_message_stream(
        &mut self,
        sender_id: Identity,
        session_id: SessionId,
        stream: UnboundedReceiver<SendRequest>,
    ) -> Result<()> {
        if sender_id == self.party_id {
            bail!("Sender ID coincides with receiver ID: {:?}", sender_id);
        }

        tracing::debug!(
            "Player {:?} is adding incoming stream from player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        let instreams = self.instreams.entry(session_id).or_default();

        if instreams.contains_key(&sender_id) {
            bail!(
                "Incoming stream for player {:?} has been already created",
                sender_id
            );
        }
        instreams.insert(sender_id.clone(), stream);

        tracing::debug!(
            "Player {:?} has added incoming stream from player {:?} in session {:?}",
            self.party_id,
            sender_id,
            session_id
        );

        Ok(())
    }
}

pub async fn setup_local_grpc_networking(parties: Vec<Identity>) -> Result<Vec<GrpcHandle>> {
    let config = GrpcConfig {
        timeout_duration: Duration::from_secs(5),
        connection_parallelism: 1,
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
        hawkers::aby3::{aby3_store::prepare_query, test_utils::shared_random_setup},
        hnsw::HnswSearcher,
    };
    use aes_prng::AesRng;
    use futures::future::join_all;
    use iris_mpc_common::vector_id::VectorId;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    async fn create_session_helper(
        session_id: SessionId,
        players: &[GrpcHandle],
    ) -> Result<Vec<GrpcSession>> {
        let mut jobs = vec![];
        for player in players.iter() {
            let player = player.clone();
            let task = tokio::spawn(async move {
                tracing::trace!(
                    "Player {:?} is creating session {:?}",
                    player.party_id,
                    session_id
                );
                player.create_session(session_id).await.unwrap()
            });
            jobs.push(task);
        }
        join_all(jobs)
            .await
            .into_iter()
            .map(|r| r.map_err(eyre::Report::new))
            .collect::<Result<Vec<_>>>()
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_correct() -> Result<()> {
        let identities = generate_local_identities();
        let players = setup_local_grpc_networking(identities.clone()).await?;

        let mut jobs = JoinSet::new();

        // Simple session with one message sent from one party to another
        {
            let players = players.clone();

            let session_id = SessionId::from(0);

            jobs.spawn(async move {
                let mut players = create_session_helper(session_id, &players).await.unwrap();

                // we don't need the last player here
                players.pop();

                let mut bob = players.pop().unwrap();
                let alice = players.pop().unwrap();

                // Send a message from the first party to the second party
                let message = b"Hey, Bob. I'm Alice. Do you copy?".to_vec();
                let message_copy = message.clone();

                let task1 = tokio::spawn(async move {
                    alice.send(message.clone(), &"bob".into()).await.unwrap();
                });
                let task2 = tokio::spawn(async move {
                    let received_message = bob.receive(&"alice".into()).await.unwrap();
                    assert_eq!(message_copy, received_message);
                });
                let _ = tokio::try_join!(task1, task2).unwrap();
            });
        }

        // Multiple parties sending messages to each other
        let all_parties_talk = |identities: Vec<Identity>, sessions: Vec<GrpcSession>| async move {
            let mut tasks = JoinSet::new();
            for (player_id, session) in sessions.into_iter().enumerate() {
                let role = Role::new(player_id);
                let next = role.next(3).index();
                let prev = role.prev(3).index();

                let next_id = identities[next].clone();
                let prev_id = identities[prev].clone();

                let message_to_next =
                    format!("From player {} to player {} with love", player_id, next).into_bytes();
                let message_to_prev =
                    format!("From player {} to player {} with love", player_id, prev).into_bytes();

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
                    let expected_message_from_prev =
                        format!("From player {} to player {} with love", prev, player_id)
                            .into_bytes();
                    assert_eq!(received_message_from_prev, expected_message_from_prev);
                    let received_message_from_next = session.receive(&next_id).await.unwrap();
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

                let players = create_session_helper(session_id, &players).await.unwrap();

                // Test that parties can send and receive messages
                all_parties_talk(identities, players).await;
            });
        }

        // Parties create a session asynchronously
        {
            let players = players.clone();
            let session_id = SessionId::from(2);
            // Session is consecutively created
            let sessions = {
                let mut jobs = vec![];
                for (i, player) in players.iter().enumerate() {
                    let player = player.clone();
                    let task = tokio::spawn(async move {
                        tracing::trace!(
                            "Player {:?} is creating session {:?}",
                            player.party_id,
                            session_id
                        );
                        sleep(Duration::from_millis(200 * i as u64)).await;
                        player.create_session(session_id).await.unwrap()
                    });
                    jobs.push(task);
                }
                join_all(jobs)
                    .await
                    .into_iter()
                    .map(|r| r.map_err(eyre::Report::new))
                    .collect::<Result<Vec<_>>>()?
            };

            // Test that parties can send and receive messages
            all_parties_talk(identities, sessions).await;
        }

        jobs.join_all().await;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_grpc_comms_fail() -> Result<()> {
        let parties = generate_local_identities();

        let players = setup_local_grpc_networking(parties.clone()).await?;

        let mut jobs = JoinSet::new();

        {
            // Send to a non-existing party
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(0);
                let sessions = create_session_helper(session_id, &players).await.unwrap();

                let message = b"Hey, Eve. I'm Alice. Do you copy?".to_vec();
                let res = sessions[0]
                    .send(message.clone(), &Identity::from("eve"))
                    .await;
                assert_eq!(
                    "Outgoing stream for Identity(\"eve\") in session SessionId(0) not found",
                    res.unwrap_err().to_string()
                );
            });
        }

        {
            // Receive from a wrong party
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);
                let mut sessions = create_session_helper(session_id, &players).await.unwrap();

                let res = sessions[0].receive(&Identity::from("eve")).await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Incoming stream for Identity(\"eve\") in session SessionId(1) not found"
                );
            });
        }

        {
            // Send to itself
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);
                let sessions = create_session_helper(session_id, &players).await.unwrap();

                let message = b"Hey, Alice. I'm Alice. Do you copy?".to_vec();
                let res = sessions[0]
                    .send(message.clone(), &Identity::from("alice"))
                    .await;
                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Outgoing stream for Identity(\"alice\") in session SessionId(2) not found",
                );
            });
        }

        {
            // Add the same session
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(3);
                let _ = create_session_helper(session_id, &players).await.unwrap();

                let alice = players[0].clone();

                let res = alice.create_session(session_id).await;

                assert_eq!(
                    res.unwrap_err().to_string(),
                    "Session SessionId(3) has already been created by player Identity(\"alice\")"
                );
            });
        }

        {
            // Receive from a party that didn't send a message (timeout error)
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);
                let mut sessions = create_session_helper(session_id, &players).await.unwrap();

                let res = sessions[0].receive(&Identity::from("bob")).await;
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

    #[tokio::test(flavor = "multi_thread")]
    #[traced_test]
    async fn test_hnsw_local() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 2;
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut vectors_and_graphs = shared_random_setup(
            &mut rng,
            database_size,
            crate::network::NetworkType::GrpcChannel,
        )
        .await
        .unwrap();

        for i in 0..database_size {
            let vector_id = VectorId::from_0_index(i as u32);
            let mut jobs = JoinSet::new();

            for (store, graph) in vectors_and_graphs.iter_mut() {
                let searcher = searcher.clone();
                let q = store.lock().await.storage.get_vector(&vector_id).await;
                let q = prepare_query((*q).clone());
                let store = store.clone();
                let mut graph = graph.clone();
                jobs.spawn(async move {
                    let mut store_lock = store.lock().await;
                    let secret_neighbors = searcher
                        .search(&mut *store_lock, &mut graph, &q, 1)
                        .await
                        .unwrap();
                    searcher
                        .is_match(&mut *store_lock, &[secret_neighbors])
                        .await
                        .unwrap()
                });
            }
            let res = jobs.join_all().await;
            for (party_index, r) in res.iter().enumerate() {
                assert!(r, "Failed at index {:?} by party {:?}", i, party_index);
            }
        }
    }
}
