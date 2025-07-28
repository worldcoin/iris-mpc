use std::{net::SocketAddr, sync::Arc};

use async_trait::async_trait;
use eyre::Result;
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::rustls::{
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer},
    server::WebPkiClientVerifier,
    RootCertStore, ServerConfig,
};
use tokio_rustls::{TlsAcceptor, TlsStream};

use crate::network::tcp::{networking::configure_tcp_stream, Server};

pub struct TlsServer {
    listener: TcpListener,
    tls_acceptor: TlsAcceptor,
}

pub struct TcpServer {
    listener: TcpListener,
}

impl TlsServer {
    pub async fn new(
        own_addr: SocketAddr,
        key_file: &str,
        cert_file: &str,
        root_cert: &str,
    ) -> Result<Self> {
        let mut root_cert_store = RootCertStore::empty();
        for cert in CertificateDer::pem_file_iter(root_cert)? {
            root_cert_store.add(cert?)?;
        }
        let client_verifier = WebPkiClientVerifier::builder(Arc::new(root_cert_store)).build()?;

        let certs = CertificateDer::pem_file_iter(cert_file)?.collect::<Result<Vec<_>, _>>()?;
        let key = PrivateKeyDer::from_pem_file(key_file)?;
        let server_config = ServerConfig::builder()
            .with_client_cert_verifier(client_verifier)
            .with_single_cert(certs, key)?;

        let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));
        let listener = TcpListener::bind(own_addr).await?;
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
    type Output = TlsStream<TcpStream>;
    async fn accept(&self) -> Result<(SocketAddr, Self::Output)> {
        let (tcp_stream, peer_addr) = self.listener.accept().await?;
        configure_tcp_stream(&tcp_stream)?;
        let tls_stream = self.tls_acceptor.accept(tcp_stream).await?;
        Ok((peer_addr, TlsStream::Server(tls_stream)))
    }
}

#[async_trait]
impl Server for TcpServer {
    type Output = TcpStream;
    async fn accept(&self) -> Result<(SocketAddr, Self::Output)> {
        let (tcp_stream, peer_addr) = self.listener.accept().await?;
        configure_tcp_stream(&tcp_stream)?;
        Ok((peer_addr, tcp_stream))
    }
}
