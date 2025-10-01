mod connection;
mod data;
mod handle;
mod session;

use std::net::SocketAddr;

use crate::network::tcp2::session::TcpSession;
use async_trait::async_trait;
use eyre::Result;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsStream;
use tokio_util::sync::CancellationToken;

#[async_trait]
pub trait NetworkHandle: Send + Sync {
    async fn make_sessions(&mut self) -> Result<(Vec<TcpSession>, CancellationToken)>;
}

pub trait NetworkConnection: AsyncRead + AsyncWrite + Send + Sync + Unpin {
    fn close(&mut self);
}

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

//
// below are mostly AI generated trait implementations allowing the connections to be closable.
//

pub struct TcpStreamConn(pub TcpStream);
pub struct TlsStreamConn(pub TlsStream<TcpStream>);

// allow mixing TLS client and TCP server by boxing connections
/// Dynamic stream type for mixed connectors and listeners
pub type DynStreamConn = Box<dyn NetworkConnection>;

impl NetworkConnection for DynStreamConn {
    fn close(&mut self) {
        (**self).close();
    }
}

impl AsyncRead for TcpStreamConn {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_read(cx, buf)
    }
}

impl AsyncWrite for TcpStreamConn {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_shutdown(cx)
    }
}

impl NetworkConnection for TcpStreamConn {
    fn close(&mut self) {
        let _ = self.0.shutdown();
    }
}

impl AsyncRead for TlsStreamConn {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_read(cx, buf)
    }
}

impl AsyncWrite for TlsStreamConn {
    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_write(cx, buf)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll_shutdown(cx)
    }
}

impl NetworkConnection for TlsStreamConn {
    fn close(&mut self) {
        let _ = self.0.shutdown();
    }
}
