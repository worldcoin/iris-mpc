use async_trait::async_trait;
use eyre::Result;
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpStream;
use tokio_rustls::rustls::{
    pki_types::{pem::PemObject, CertificateDer, ServerName},
    ClientConfig, RootCertStore,
};
use tokio_rustls::{TlsConnector, TlsStream};

use crate::network::tcp::Client;

#[derive(Clone)]
pub struct TlsClient {
    tls_connector: TlsConnector,
}

#[derive(Clone)]
pub struct TcpClient {}

impl TlsClient {
    pub async fn new(cert_file: &str) -> Result<Self> {
        let mut root_cert_store = RootCertStore::empty();
        for cert in CertificateDer::pem_file_iter(cert_file)? {
            root_cert_store.add(cert?)?;
        }
        let client_config = ClientConfig::builder()
            .with_root_certificates(root_cert_store)
            .with_no_client_auth();
        let tls_connector = TlsConnector::from(Arc::new(client_config));
        Ok(Self { tls_connector })
    }
}

impl TcpClient {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Client for TlsClient {
    type Output = TlsStream<TcpStream>;
    async fn connect(&self, addr: SocketAddr) -> Result<Self::Output> {
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        let domain = ServerName::IpAddress(addr.ip().into());
        let tls_stream = self.tls_connector.connect(domain, stream).await?;
        Ok(TlsStream::Client(tls_stream))
    }
}

#[async_trait]
impl Client for TcpClient {
    type Output = TcpStream;
    async fn connect(&self, addr: SocketAddr) -> Result<Self::Output> {
        let stream = TcpStream::connect(addr).await?;
        stream.set_nodelay(true)?;
        Ok(stream)
    }
}
