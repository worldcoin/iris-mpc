use std::{net::SocketAddr, sync::Arc};

use async_trait::async_trait;
use eyre::Result;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::rustls::{
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer},
    ServerConfig,
};
use tokio_rustls::{TlsAcceptor, TlsStream};

use crate::network::tcp::Server;

pub struct TlsServer {
    listener: TcpListener,
    tls_acceptor: TlsAcceptor,
}

pub struct TcpServer {
    listener: TcpListener,
}

impl TlsServer {
    pub async fn new(own_addr: SocketAddr, key_file: &str, cert_file: &str) -> Result<Self> {
        let listener = TcpListener::bind(own_addr).await?;
        let certs = CertificateDer::pem_file_iter(cert_file)?.collect::<Result<Vec<_>, _>>()?;
        let key = PrivateKeyDer::from_pem_file(key_file)?;
        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(certs, key)?;
        let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));
        Ok(Self {
            listener,
            tls_acceptor,
        })
    }
}

impl TcpServer {
    pub async fn new(own_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(own_addr).await?;
        Ok(Self { listener })
    }
}

#[async_trait]
impl Server for TlsServer {
    type Connection = TlsStream<TcpStream>;
    async fn accept(&self) -> Result<(SocketAddr, Self::Connection)> {
        let (tcp_stream, peer_addr) = self.listener.accept().await?;
        tcp_stream.set_nodelay(true)?;
        let tls_stream = self.tls_acceptor.accept(tcp_stream).await?;
        Ok((peer_addr, TlsStream::Server(tls_stream)))
    }
}

#[async_trait]
impl Server for TcpServer {
    type Connection = TcpStream;
    async fn accept(&self) -> Result<(SocketAddr, Self::Connection)> {
        let (tcp_stream, peer_addr) = self.listener.accept().await?;
        tcp_stream.set_nodelay(true)?;
        Ok((peer_addr, tcp_stream))
    }
}
