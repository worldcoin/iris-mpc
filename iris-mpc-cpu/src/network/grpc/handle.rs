use crate::{
    execution::player::Identity,
    network::SessionId,
    proto_generated::party_node::{party_node_server::PartyNode, SendRequest, SendResponse},
};
use eyre::{eyre, Result};
use std::time::Duration;
use tokio::{
    sync::{
        mpsc::{self, UnboundedReceiver},
        oneshot,
    },
    time::sleep,
};
use tonic::{async_trait, Request, Response, Status, Streaming};

use super::networking::GrpcNetworking;
use super::session::GrpcSession;
use super::{err_to_status, GrpcConfig, InStreams, OutStreams, TonicResult};

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
        let party_id = grpc.party_id();
        let config = grpc.config();
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

    pub fn party_id(&self) -> Identity {
        self.party_id.clone()
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
