use crate::{
    execution::player::Identity,
    network::{value::NetworkValue, Networking, SessionId},
    proto_generated::party_node::SendRequest,
};
use eyre::{eyre, Result};
use std::collections::HashMap;
use std::time::Duration;
use tokio::{
    sync::Mutex,
    time::{sleep, timeout},
};
use tonic::async_trait;

// Hard-coded artificial one-way link latency used in tests.
const ARTIFICIAL_LINK_DELAY: Duration = Duration::from_millis(1);
use super::{GrpcConfig, InStream, OutStream};

#[derive(Debug)]
pub struct GrpcSession {
    pub session_id: SessionId,
    pub own_identity: Identity,
    pub out_streams: HashMap<Identity, OutStream>,
    pub in_streams: HashMap<Identity, Mutex<InStream>>,
    pub config: GrpcConfig,
}

#[async_trait]
impl Networking for GrpcSession {
    async fn send(&self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        // sleep(ARTIFICIAL_LINK_DELAY).await;
        let value = value.to_network();

        let outgoing_stream = self.out_streams.get(receiver).ok_or(eyre!(
            "Outgoing stream for {receiver:?} in {:?} not found",
            self.session_id
        ))?;
        let request = SendRequest {
            session_id: self.session_id.0,
            data: value,
        };
        outgoing_stream
            .send(request)
            .map_err(|e| eyre!(e.to_string()))?;
        Ok(())
    }

    async fn receive(&self, sender: &Identity) -> Result<NetworkValue> {
        let incoming_stream = self.in_streams.get(sender).ok_or(eyre!(
            "Incoming stream for {sender:?} in {:?} not found",
            self.session_id
        ))?;
        let mut guard = incoming_stream.lock().await;
        match timeout(self.config.timeout_duration, guard.recv()).await {
            Ok(res) => res
                .ok_or(eyre!("No message received"))
                .and_then(|msg| NetworkValue::deserialize(&msg.data)),
            Err(_) => Err(eyre!(
                "{:?}: Timeout while waiting for message from {sender:?} in \
                 {:?}",
                self.own_identity,
                self.session_id
            )),
        }
    }
}
