use crate::execution::{player::Identity, session::SessionId};
use async_trait::async_trait;
use eyre::Result;

/// Requirements for networking.
#[async_trait]
pub trait Networking {
    async fn send(&self, value: Vec<u8>, receiver: &Identity) -> Result<()>;

    async fn receive(&mut self, sender: &Identity) -> Result<Vec<u8>>;
}

#[derive(Clone)]
pub enum NetworkType {
    LocalChannel,
    GrpcChannel,
}

pub mod grpc;
pub mod local;
pub mod value;
