use crate::{
    execution::player::Identity,
    network::{value::NetworkValue, Networking, SessionId},
    proto_generated::party_node::SendRequest,
};
use eyre::{eyre, Result};
use iris_mpc_common::fast_metrics::FastHistogram;
use std::collections::HashMap;
use tokio::time::timeout;
use tonic::async_trait;

use super::{GrpcConfig, InStream, OutStream};

#[derive(Debug)]
pub struct GrpcSession {
    pub session_id: SessionId,
    pub own_identity: Identity,
    pub out_streams: HashMap<Identity, OutStream>,
    pub in_streams: HashMap<Identity, InStream>,
    pub config: GrpcConfig,
    pub metric: FastHistogram,
}

#[async_trait]
impl Networking for GrpcSession {
    async fn send(&mut self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        let value = value.to_network();
        self.metric.record(value.len() as f64);

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

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue> {
        let incoming_stream = self.in_streams.get_mut(sender).ok_or(eyre!(
            "Incoming stream for {sender:?} in {:?} not found",
            self.session_id
        ))?;
        match timeout(self.config.timeout_duration, incoming_stream.recv()).await {
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
