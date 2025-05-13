use crate::{
    execution::player::Identity,
    network::{SessionId, StreamId},
    proto_generated::party_node::{party_node_client::PartyNodeClient, SendRequest, SendRequests},
};
use eyre::{eyre, Result};
use std::{
    collections::{HashMap, HashSet},
    str::FromStr,
};
use tokio::{
    sync::mpsc::{self, error::TryRecvError},
    time::{Duration, Instant},
};
use tonic::{metadata::AsciiMetadataValue, transport::Channel, Request, Status};
use tracing::error;

use super::super::{GrpcConfig, OutStream, OutStreams};

#[derive(Default)]
pub struct StreamManager {
    established_sessions: HashSet<SessionId>,
    established_streams: HashSet<StreamId>,
    stream_channels: HashMap<Identity, HashMap<StreamId, OutStream>>,
    config: GrpcConfig,
}

impl StreamManager {
    pub fn new(config: GrpcConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    // many tasks may try to create sessions at the same time.
    // assume that the tasks will be given the correct session ids (no duplicates with range from [0..n_sessions))
    pub fn add_session(
        &mut self,
        party_id: &Identity,
        clients: &HashMap<Identity, Vec<PartyNodeClient<Channel>>>,
        session_id: SessionId,
    ) -> Result<OutStreams> {
        if !self.established_sessions.insert(session_id) {
            return Err(eyre!(
                "{:?} has already been created by {:?}",
                session_id,
                party_id
            ));
        }

        let stream_id = StreamId::from(session_id.0 / self.config.stream_parallelism as u32);
        if self.established_streams.insert(stream_id) {
            self.add_stream(party_id.clone(), clients, stream_id)?;
        }

        let mut out_streams = HashMap::new();
        for (client_id, stream_map) in self.stream_channels.iter() {
            let tx = stream_map
                .get(&stream_id)
                .ok_or(eyre!(
                    "failed to get stream id {} for {:?}",
                    stream_id.0,
                    client_id
                ))?
                .clone();
            out_streams.insert(client_id.clone(), tx);
        }

        Ok(out_streams)
    }

    fn add_stream(
        &mut self,
        party_id: Identity,
        clients: &HashMap<Identity, Vec<PartyNodeClient<Channel>>>,
        stream_id: StreamId,
    ) -> Result<()> {
        tracing::debug!(
            "{:?} is adding a stream to {} clients",
            party_id,
            clients.len()
        );

        let stream_parallelism = self.config.stream_parallelism;
        for (client_id, clients) in clients.iter() {
            let round_robin = (stream_id.0 as usize) % clients.len();
            let mut client = clients[round_robin].clone();

            let (hawk_tx, mut hawk_rx) = mpsc::unbounded_channel::<SendRequest>();
            let (tonic_tx, tonic_rx) = mpsc::unbounded_channel::<SendRequests>();
            let receiving_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(tonic_rx);
            let mut request = Request::new(receiving_stream);
            request.metadata_mut().insert(
                "sender_id",
                AsciiMetadataValue::from_str(&party_id.0)
                    .map_err(|e| eyre!("Failed to convert Sender ID to ASCII: {e}"))?,
            );
            request.metadata_mut().insert(
                "stream_id",
                AsciiMetadataValue::from_str(&stream_id.0.to_string())
                    .map_err(|e| eyre!("Failed to convert Stream ID to ASCII: {e}"))?,
            );

            tokio::spawn(async move {
                let _response = client.start_message_stream(request).await?;
                Ok::<_, Status>(())
            });

            tokio::spawn(async move {
                while let Some(message) = hawk_rx.recv().await {
                    let mut payload_len = message.data.len();
                    let mut requests = vec![message];
                    let start_time = Instant::now();
                    while requests.len() != stream_parallelism {
                        match hawk_rx.try_recv() {
                            Ok(msg) => {
                                payload_len += msg.data.len();
                                requests.push(msg);
                                // maximum gRPC payload size is 4MB.
                                if payload_len >= 1 << 21 {
                                    break;
                                }
                            }
                            Err(TryRecvError::Empty) => {
                                if start_time.elapsed() >= Duration::from_micros(500) {
                                    break;
                                }
                                tokio::task::yield_now().await;
                            }
                            Err(_) => break,
                        }
                    }
                    let requests = SendRequests { requests };
                    if let Err(e) = tonic_tx.send(requests) {
                        error!(
                            "failed to send message on outbound stream {}: {:?}",
                            stream_id.0, e
                        );
                        break;
                    }
                }
            });

            self.stream_channels
                .entry(client_id.clone())
                .or_default()
                .insert(stream_id, hawk_tx);
        }
        Ok(())
    }
}
