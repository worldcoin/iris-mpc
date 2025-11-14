use crate::{execution::player::Identity, network::value::NetworkValue};
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
    Tcp {
        connection_parallelism: usize,
        request_parallelism: usize,
    },
}

impl NetworkType {
    pub fn default_connection_parallelism() -> usize {
        1
    }

    pub fn default_request_parallelism() -> usize {
        1
    }

    pub fn default_tcp() -> Self {
        Self::Tcp {
            connection_parallelism: Self::default_connection_parallelism(),
            request_parallelism: Self::default_request_parallelism(),
        }
    }
}

pub mod local;
pub mod tcp;
pub mod value;
