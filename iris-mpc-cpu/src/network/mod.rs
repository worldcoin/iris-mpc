use crate::{
    execution::{
        player::Identity,
        session::{SessionId, StreamId},
    },
    network::value::NetworkValue,
};
use async_trait::async_trait;
use eyre::Result;

/// Requirements for networking.
#[async_trait]
pub trait Networking {
    async fn send(&mut self, value: NetworkValue, receiver: &Identity) -> Result<()>;

    async fn receive(&mut self, sender: &Identity) -> Result<NetworkValue>;
}

#[derive(Clone)]
pub enum NetworkType {
    Local,
    #[cfg(test)]
    Grpc {
        connection_parallelism: usize,
        stream_parallelism: usize,
        request_parallelism: usize,
    },
    Tcp {
        connection_parallelism: usize,
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

    #[cfg(test)]
    pub fn default_grpc() -> Self {
        Self::Grpc {
            connection_parallelism: Self::default_connection_parallelism(),
            stream_parallelism: Self::default_stream_parallelism(),
            request_parallelism: Self::default_request_parallelism(),
        }
    }

    pub fn default_tcp() -> Self {
        Self::Tcp {
            connection_parallelism: Self::default_connection_parallelism(),
            request_parallelism: Self::default_request_parallelism(),
        }
    }
}

#[cfg(test)]
pub mod grpc;
pub mod local;
pub mod tcp;
pub mod value;
