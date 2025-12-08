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
pub trait Networking: Send + Sync {
    async fn send(&self, value: NetworkValue, receiver: &Identity) -> Result<()>;

    async fn receive(&self, sender: &Identity) -> Result<NetworkValue>;

    /// Number of parallel lanes available per peer.
    fn num_lanes(&self) -> usize {
        1
    }

    /// Send using a specific lane. Default implementation falls back to single-lane send.
    async fn send_on_lane(
        &self,
        value: NetworkValue,
        receiver: &Identity,
        lane_idx: usize,
    ) -> Result<()> {
        let _ = lane_idx;
        self.send(value, receiver).await
    }

    async fn receive_on_lane(&self, sender: &Identity, lane_idx: usize) -> Result<NetworkValue> {
        let _ = lane_idx;
        self.receive(sender).await
    }
}

#[derive(Clone)]
pub enum NetworkType {
    Local,
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

    pub fn default_grpc() -> Self {
        Self::Grpc {
            connection_parallelism: Self::default_connection_parallelism(),
            stream_parallelism: Self::default_stream_parallelism(),
            request_parallelism: Self::default_request_parallelism(),
        }
    }
}

pub mod grpc;
pub mod local;
pub mod tcp;
pub mod value;
