use crate::{
    execution::{local::get_free_local_addresses, player::Identity},
    network::{SessionId, StreamId},
    proto_generated::party_node::{
        party_node_client::PartyNodeClient, party_node_server::PartyNodeServer, SendRequests,
    },
};
use backon::{ExponentialBuilder, Retryable};
use eyre::{bail, eyre, Result};
use futures::future::JoinAll;
use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};
use tokio::sync::mpsc::UnboundedReceiver;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Server, Uri};

use super::handle::GrpcHandle;
use super::{GrpcConfig, InStream, InStreams, OutStream, OutStreams};

mod stream_manager;
use stream_manager::StreamManager;

// WARNING: this implementation assumes that messages for a specific player
// within one session are sent in order and consecutively. Don't send messages
// to the same player in parallel within the same session. Use batching instead.
pub struct GrpcNetworking {
    party_id: Identity,
    // other party id -> client to call that party
    clients: HashMap<Identity, Vec<PartyNodeClient<Channel>>>,
    // session_id -> incoming streams
    inbound_sessions: HashMap<SessionId, InStreams>,
    // sessions in use
    // TODO: deletion logic
    active_sessions: HashSet<SessionId>,
    // creates outbound gRPC streams and multiplexes sessions over them
    sm: StreamManager,

    config: GrpcConfig,
}

impl GrpcNetworking {
    pub fn new(party_id: Identity, config: GrpcConfig) -> Self {
        GrpcNetworking {
            party_id,
            clients: HashMap::new(),
            inbound_sessions: HashMap::new(),
            active_sessions: HashSet::new(),
            sm: StreamManager::new(config.clone()),
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

    pub async fn connect_to_party(
        &mut self,
        party_id: Identity,
        address: &str,
        root_cert: Option<String>,
    ) -> Result<()> {
        if self.clients.contains_key(&party_id) {
            bail!(
                "{:?} has already connected to {:?}",
                self.party_id,
                party_id
            );
        }

        // hacks to build a PartyNodeClient with backoff, retry, and TLS
        // adding TLS requires access to the underlying Channel
        async fn get_party_node(
            endpoint: Endpoint,
        ) -> std::result::Result<PartyNodeClient<Channel>, tonic::transport::Error> {
            let channel = endpoint.connect().await?;
            Ok(PartyNodeClient::new(channel))
        }

        // Use https for TLS
        let uri = if root_cert.is_some() {
            Uri::from_maybe_shared(format!("https://{}", address))?
        } else {
            Uri::from_maybe_shared(format!("http://{}", address))?
        };

        let domain_name = address
            .split(':')
            .next()
            .ok_or(eyre!("failed to get domain_name"))?;

        let endpoint = match root_cert {
            Some(cert) => {
                let cert = std::fs::read_to_string(cert)?;
                let server_ca = Certificate::from_pem(cert);
                let tls_config = ClientTlsConfig::new()
                    .ca_certificate(server_ca)
                    .domain_name(domain_name);

                Channel::builder(uri).tls_config(tls_config)?
            }
            None => Channel::builder(uri),
        };

        let clients = (0..self.config.connection_parallelism.max(1))
            .map(|_| {
                // hacks to make the closure be FnMut, as needed for retry()
                let endpoint = endpoint.clone();
                (move || get_party_node(endpoint.clone()))
                    .retry(self.backoff())
                    .sleep(tokio::time::sleep)
            })
            .map(tokio::spawn)
            .collect::<JoinAll<_>>()
            .await
            .into_iter()
            .collect::<Result<Result<Vec<PartyNodeClient<_>>, _>, _>>()??;
        tracing::trace!(
            "{:?} connected to {:?} at address {:?}",
            self.party_id,
            party_id,
            address
        );
        self.clients.insert(party_id.clone(), clients);
        Ok(())
    }

    // adds a session to a stream, and creates a new stream if needed
    pub async fn create_outgoing_streams(&mut self, session_id: SessionId) -> Result<OutStreams> {
        self.sm
            .add_session(&self.party_id, &self.clients, session_id)
    }

    pub fn is_session_ready(&self, session_id: SessionId) -> bool {
        let n_senders = match self.inbound_sessions.get(&session_id) {
            None => 0,
            Some(q) => q.len(),
        };

        n_senders == self.clients.len()
    }

    pub async fn obtain_incoming_streams(&mut self, session_id: SessionId) -> Result<InStreams> {
        self.active_sessions.insert(session_id);
        self.inbound_sessions
            .remove(&session_id)
            .ok_or(eyre!(format!(
                "{session_id:?} hasn't been added to message queues"
            )))
    }
}

// Server implementation
impl GrpcNetworking {
    pub async fn start_message_stream(
        &mut self,
        sender_id: Identity,
        stream_id: StreamId,
        mut stream: UnboundedReceiver<SendRequests>,
        session_forwarder: HashMap<u32, OutStream>,
        mut inbound_sessions: HashMap<SessionId, InStream>,
    ) -> Result<()> {
        if sender_id == self.party_id {
            bail!("Sender ID coincides with receiver ID: {:?}", sender_id);
        }

        for (session_id, stream) in inbound_sessions.drain() {
            if self
                .inbound_sessions
                .entry(session_id)
                .or_default()
                .insert(sender_id.clone(), stream)
                .is_some()
            {
                tracing::error!(
                    "duplicate session id {} on stream {} from sender {:?}",
                    session_id.0,
                    stream_id.0,
                    sender_id
                );
            }
        }

        // logging here to avoid a clone.
        tracing::debug!(
            "{:?} has added incoming stream  {:?} from {:?}",
            self.party_id,
            stream_id,
            sender_id
        );

        tokio::spawn(async move {
            while let Some(msg) = stream.recv().await {
                for request in msg.requests {
                    let session_id = request.session_id;
                    if let Some(tx) = session_forwarder.get(&session_id) {
                        if let Err(e) = tx.send(request) {
                            tracing::error!(
                                "Failed to forward message for session {:?}: {:?}",
                                session_id,
                                e
                            );
                        }
                    } else {
                        tracing::error!(
                            "{:?} sent message with invalid session id {:?} on stream {:?}",
                            sender_id,
                            session_id,
                            stream_id
                        );
                    }
                }
            }
        });

        Ok(())
    }
}

pub async fn setup_local_grpc_networking(
    parties: Vec<Identity>,
    connection_parallelism: usize,
    stream_parallelism: usize,
) -> Result<Vec<GrpcHandle>> {
    let config = GrpcConfig {
        timeout_duration: Duration::from_secs(5),
        connection_parallelism,
        stream_parallelism,
        request_parallelism: connection_parallelism * stream_parallelism,
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
                    .connect_to_party(other_player.party_id(), &other_addr, None)
                    .await
                    .unwrap();
            }
        }
    }

    tracing::debug!("Players connected to each other");

    Ok(players)
}
