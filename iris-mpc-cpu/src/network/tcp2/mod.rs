mod connection;
mod data;
mod handle;
mod session;

use std::net::SocketAddr;

use crate::network::tcp2::session::TcpSession;
use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait NetworkHandle: Send + Sync {
    async fn make_sessions(&mut self) -> Result<(Vec<TcpSession>, CancellationToken)>;
}

pub trait NetworkConnection: AsyncRead + AsyncWrite + Send + Sync + Unpin {}

impl<T: AsyncRead + AsyncWrite + Unpin + Send + ?Sized + Sync> NetworkConnection for T {}

// used to establish an outbound connection
#[async_trait]
pub trait Client: Send + Sync + Clone {
    type Output: NetworkConnection;
    async fn connect(&self, url: String) -> Result<Self::Output>;
}

// used for a server to accept an incoming connection
#[async_trait]
pub trait Server: Send {
    type Output: NetworkConnection;
    async fn accept(&self) -> Result<(SocketAddr, Self::Output)>;
}
