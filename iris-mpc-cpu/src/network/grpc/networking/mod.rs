use crate::{
    execution::{local::get_free_local_addresses, player::Identity},
    network::SessionId,
    proto_generated::party_node::{
        party_node_client::PartyNodeClient, party_node_server::PartyNodeServer, SendRequest,
    },
};
use backon::{ExponentialBuilder, Retryable};
use eyre::{bail, eyre, Result};
use futures::future::JoinAll;
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
    time::Duration,
};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tonic::{
    metadata::AsciiMetadataValue,
    transport::{Channel, Server},
    Request, Status,
};

use super::handle::GrpcHandle;
use super::{GrpcConfig, InStreams, OutStreams};

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

    pub fn party_id(&self) -> Identity {
        self.party_id.clone()
    }

    pub fn config(&self) -> GrpcConfig {
        self.config.clone()
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
    pub async fn start_message_stream(
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
                    .connect_to_party(other_player.party_id(), &other_addr)
                    .await
                    .unwrap();
            }
        }
    }

    tracing::debug!("Players connected to each other");

    Ok(players)
}
