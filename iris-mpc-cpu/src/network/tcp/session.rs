use std::collections::HashMap;

use super::{InStream, OutboundMsg};
use crate::{
    execution::{player::Identity, session::SessionId},
    network::{tcp::config::TcpConfig, value::NetworkValue, Networking},
};
use async_trait::async_trait;
use eyre::{eyre, Result};
use tokio::{sync::mpsc::UnboundedSender, time::timeout};

#[derive(Debug)]
pub struct TcpSession {
    session_id: SessionId,
    tx: HashMap<Identity, UnboundedSender<OutboundMsg>>,
    rx: HashMap<Identity, InStream>,
    config: TcpConfig,
}

impl TcpSession {
    pub fn new(
        session_id: SessionId,
        tx: HashMap<Identity, UnboundedSender<OutboundMsg>>,
        rx: HashMap<Identity, InStream>,
        config: TcpConfig,
    ) -> Self {
        Self {
            session_id,
            tx,
            rx,
            config,
        }
    }

    pub fn id(&self) -> SessionId {
        self.session_id
    }
}

#[async_trait]
impl Networking for TcpSession {
    async fn send(&mut self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        let outgoing_stream = self.tx.get(receiver).ok_or(eyre!(
            "Outgoing stream for {receiver:?} in session {:?} not found",
            self.session_id
        ))?;
        outgoing_stream
            .send((self.session_id, value))
            .map_err(|e| eyre!(e.to_string()))?;
        Ok(())
    }

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue> {
        let incoming_stream = self.rx.get_mut(sender).ok_or(eyre!(
            "Incoming stream for {sender:?} in session {:?} not found",
            self.session_id
        ))?;
        match timeout(self.config.timeout_duration, incoming_stream.recv()).await {
            Ok(res) => res.ok_or(eyre!("No message received")),
            Err(_) => Err(eyre!(
                "Timeout while waiting for message from {sender:?} in \
                 {:?}",
                self.session_id
            )),
        }
    }
}
