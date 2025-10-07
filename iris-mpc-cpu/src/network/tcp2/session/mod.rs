pub mod multiplexer;

use crate::{
    execution::{player::Identity, session::SessionId},
    network::{tcp::config::TcpConfig, value::NetworkValue, Networking},
};
use async_trait::async_trait;
use eyre::Result;

#[derive(Debug)]
pub struct TcpSession {
    session_id: SessionId,
    config: TcpConfig,
}

impl TcpSession {
    pub fn new(session_id: SessionId, config: TcpConfig) -> Self {
        Self { session_id, config }
    }

    pub fn id(&self) -> SessionId {
        self.session_id
    }
}

#[async_trait]
impl Networking for TcpSession {
    async fn send(&mut self, value: NetworkValue, receiver: &Identity) -> Result<()> {
        todo!()
    }

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue> {
        todo!()
    }
}
