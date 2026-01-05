use crate::network::tcp::{networking::configure_tcp_stream, Client, NetworkConnection};
use async_trait::async_trait;
use eyre::{eyre, Result};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio_rustls::rustls::{
    pki_types::{pem::PemObject, CertificateDer, ServerName},
    ClientConfig, RootCertStore,
};
use tokio_rustls::{TlsConnector, TlsStream};

#[derive(Clone)]
pub struct TlsClient {
    tls_connector: TlsConnector,
}

#[derive(Clone, Default)]
pub struct TcpClient {}

impl TlsClient {
    /// Create a client that trusts the given CAs
    pub async fn new_with_ca_certs(root_certs: &[String]) -> Result<Self> {
        let mut roots = RootCertStore::empty();
        for root_cert in root_certs {
            for cert in CertificateDer::pem_file_iter(root_cert)? {
                roots.add(cert?)?;
            }
        }

        let client_config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();

        let tls_connector = TlsConnector::from(Arc::new(client_config));
        Ok(Self { tls_connector })
    }
}

#[async_trait]
impl Client for TlsClient {
    type Output = TlsStream<TcpStream>;
    async fn connect(&self, url: String) -> Result<Self::Output> {
        let hostname = url
            .split(':')
            .next()
            .ok_or_else(|| eyre!("Invalid URL: missing hostname"))?
            .to_string();

        let domain = ServerName::try_from(hostname)?;
        let stream = TcpStream::connect(url).await?;
        configure_tcp_stream(&stream)?;

        let tls_stream = self.tls_connector.connect(domain, stream).await?;
        Ok(TlsStream::Client(tls_stream))
    }
}

#[async_trait]
impl Client for TcpClient {
    type Output = TcpStream;
    async fn connect(&self, url: String) -> Result<Self::Output> {
        let stream = TcpStream::connect(url).await?;
        configure_tcp_stream(&stream)?;
        Ok(stream)
    }
}

// allow mixing TLS client and TCP server by boxing connections
/// Dynamic stream type for mixed connectors and listeners
pub type DynStream = Box<dyn NetworkConnection>;

#[derive(Clone)]
pub struct BoxTcpClient(pub TcpClient);
#[async_trait]
impl Client for BoxTcpClient {
    type Output = DynStream;
    async fn connect(&self, url: String) -> Result<Self::Output> {
        let stream = self.0.connect(url).await?;
        Ok(Box::new(stream))
    }
}
