use crate::execution::{
    player::Identity,
    session::{SessionId, StreamId},
};
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
    GrpcChannel {
        connection_parallelism: usize,
        stream_parallelism: usize,
        request_parallelism: usize,
    },
}

impl NetworkType {
    pub fn default_connection_parallelism() -> usize {
        1
    }

    pub fn default_stream_parallelism() -> usize {
        1
    }

    pub fn default_request_parallelism() -> usize {
        1
    }

    pub fn default_grpc() -> Self {
        Self::GrpcChannel {
            connection_parallelism: Self::default_connection_parallelism(),
            stream_parallelism: Self::default_stream_parallelism(),
            request_parallelism: Self::default_request_parallelism(),
        }
    }
}

pub mod grpc;
pub mod local;
pub mod value;
