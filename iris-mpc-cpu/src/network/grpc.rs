use super::Networking;
use crate::{execution::player::Identity, network::SessionId};
use dashmap::{DashMap, Entry};
use eyre::OptionExt;
use party_node::{
    party_node_client::PartyNodeClient, party_node_server::PartyNode, SendRequest, SendResponse,
};
use std::sync::Arc;
use tonic::{async_trait, transport::Channel, Request, Response, Status};

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
        let channel = self
            .queues
            .get(party_id)
            .ok_or_eyre(format!("Channel not found for party {:?}", party_id))?;
        Ok((*channel).clone())
    }

    pub async fn push_back(&self, party_id: &Identity, value: Vec<u8>) -> eyre::Result<()> {
        let channel = self.get_channel(party_id)?;
        // sends the value via the channel sender; if failed, returns an error
        channel.sender.send(value).await.map_err(|e| e.into())
    }

    pub fn pop_front(&self, party_id: &Identity) -> eyre::Result<Vec<u8>> {
        let channel = self.get_channel(party_id)?;
        channel.receiver.try_recv().map_err(|e| e.into())
    }
}

#[derive(Clone)]
struct GrpcNetworking {
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
        let maybe_queue = self.message_queues.entry(session_id);
        let message_queue = match maybe_queue {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => {
                let new_queue = MessageQueueStore::new();
                new_queue.add_channel(&sender_id);
                entry.insert(new_queue.clone());
                new_queue
            }
        };
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
        // Send message via gRPC client
        let mut client = self
            .clients
            .get(receiver)
            .ok_or_eyre(format!("Client not found {:?}", receiver))?
            .clone();
        let request = Request::new(SendRequest {
            sender_id:  self.party_id.0.clone(),
            session_id: session_id.0,
            data:       value,
        });
        let _response = client.send_message(request).await?;
        Ok(())
    }

    async fn receive(&self, sender: &Identity, session_id: &SessionId) -> eyre::Result<Vec<u8>> {
        // Just retrieve the first message from the corresponding queue
        self.message_queues
            .get(session_id)
            .ok_or_eyre(format!(
                "No messages are found for session {}",
                session_id.0
            ))?
            .pop_front(sender)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{local::generate_local_identities, player::Role};
    use party_node::party_node_server::PartyNodeServer;
    use tokio::task::JoinSet;
    use tonic::transport::Server;

    #[tokio::test]
    async fn test_three_parties() {
        let parties = generate_local_identities();

        let players = parties
            .iter()
            .map(|party| GrpcNetworking::new(party.clone()))
            .collect::<Vec<GrpcNetworking>>();

        let addresses = ["[::1]:50051", "[::1]:50052", "[::1]:50053"];

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

        let mut jobs = JoinSet::new();

        // Simple session with one message sent from one party to another
        {
            let players = players.clone();
            let parties = parties.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(0);
                // Send a message from the first party to the second party
                let message = b"Hey, player 1. I'm player 0. Do you copy?".to_vec();
                players[0]
                    .send(message.clone(), &parties[1], &session_id)
                    .await
                    .unwrap();
                let received_message = players[1].receive(&parties[0], &session_id).await.unwrap();
                assert_eq!(message, received_message);
                // Check that there are no messages left
                let received_message = players[1].receive(&parties[0], &session_id).await;
                assert!(received_message.is_err());
            });
        }

        // Each party sending and receiving a message
        {
            let players = players.clone();
            let parties = parties.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(1);
                // Send messages
                for (player_id, player) in players.iter().enumerate() {
                    let role = Role::new(player_id);
                    let next = role.next(3).zero_based();

                    let sent_msg =
                        format!("From player {} to player {} with love", player_id, next)
                            .into_bytes();
                    player
                        .send(sent_msg.clone(), &parties[next], &session_id)
                        .await
                        .unwrap();
                }
                // Receive and check messages
                for (player_id, player) in players.iter().enumerate() {
                    let role = Role::new(player_id);
                    let prev = role.prev(3).zero_based();

                    let received_msg = player
                        .receive(&parties[prev], &session_id)
                        .await
                        .unwrap();
                    let expected_msg =
                        format!("From player {} to player {} with love", prev, player_id)
                            .into_bytes();
                    assert_eq!(received_msg, expected_msg);
                }
            });
        }

        // Send to a wrong party
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(2);
                let message = b"Hey, player Eve. I'm player 0. Do you copy?".to_vec();
                let res = players[0]
                    .send(message.clone(), &Identity::from("Eve"), &session_id)
                    .await;
                assert!(res.is_err());
            });
        }

        // Receive from a wrong party
        {
            let players = players.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(3);
                let res = players[0]
                    .receive(&Identity::from("Eve"), &session_id)
                    .await;
                assert!(res.is_err());
            });
        }

        // Receive from a party that hasn't sent anything
        {
            let players = players.clone();
            let parties = parties.clone();
            jobs.spawn(async move {
                let session_id = SessionId::from(4);
                let res = players[0].receive(&parties[1], &session_id).await;
                assert!(res.is_err());
            });
        }

        jobs.join_all().await;
    }
}
