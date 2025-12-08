use std::collections::HashMap;

use super::{InStream, OutboundMsg};
use crate::{
    execution::{player::Identity, session::SessionId},
    network::{tcp::config::TcpConfig, value::NetworkValue, Networking},
};
use async_trait::async_trait;
use eyre::{eyre, Result};
use tokio::{
    sync::{mpsc::UnboundedSender, Mutex},
    time::timeout,
};

#[derive(Debug)]
pub struct TcpSession {
    session_id: SessionId,
    tx: HashMap<Identity, Vec<UnboundedSender<OutboundMsg>>>,
    rx: HashMap<Identity, Vec<Mutex<InStream>>>,
    config: TcpConfig,
}

impl TcpSession {
    pub fn new(
        session_id: SessionId,
        tx: HashMap<Identity, Vec<UnboundedSender<OutboundMsg>>>,
        rx: HashMap<Identity, Vec<InStream>>,
        config: TcpConfig,
    ) -> Self {
        Self {
            session_id,
            tx,
            rx: rx
                .into_iter()
                .map(|(id, streams)| {
                    (
                        id,
                        streams
                            .into_iter()
                            .map(|stream| Mutex::new(stream))
                            .collect(),
                    )
                })
                .collect(),
            config,
        }
    }

    pub fn id(&self) -> SessionId {
        self.session_id
    }
}

#[async_trait]
impl Networking for TcpSession {
    async fn send(&self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        self.send_on_lane(value, receiver, 0).await
    }

    async fn receive(&self, sender: &Identity) -> Result<NetworkValue> {
        self.receive_on_lane(sender, 0).await
    }

    fn num_lanes(&self) -> usize {
        self.config.num_connections.max(1)
    }

    async fn send_on_lane(
        &self,
        value: NetworkValue,
        receiver: &Identity,
        lane_idx: usize,
    ) -> Result<()> {
        let outgoing_streams = self.tx.get(receiver).ok_or(eyre!(
            "Outgoing stream set for {receiver:?} in session {:?} not found",
            self.session_id
        ))?;
        if outgoing_streams.is_empty() {
            return Err(eyre!(
                "No outgoing lanes configured for receiver {receiver:?} in session {:?}",
                self.session_id
            ));
        }
        let lane = lane_idx % outgoing_streams.len();
        outgoing_streams[lane]
            .send((self.session_id, value))
            .map_err(|e| eyre!(e.to_string()))?;
        Ok(())
    }
    async fn receive_on_lane(&self, sender: &Identity, lane_idx: usize) -> Result<NetworkValue> {
        let incoming_streams = self.rx.get(sender).ok_or(eyre!(
            "Incoming stream set for {sender:?} in session {:?} not found",
            self.session_id
        ))?;
        if incoming_streams.is_empty() {
            return Err(eyre!(
                "No inbound lanes configured for sender {sender:?} in session {:?}",
                self.session_id
            ));
        }
        let lane = lane_idx % incoming_streams.len();
        let mut guard = incoming_streams[lane].lock().await;
        match timeout(self.config.timeout_duration, guard.recv()).await {
            Ok(res) => res.ok_or(eyre!("No message received")),
            Err(_) => Err(eyre!(
                "Timeout while waiting for message from {sender:?} (lane {lane}) in {:?}",
                self.session_id
            )),
        }
    }
}
