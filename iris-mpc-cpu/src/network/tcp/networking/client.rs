use crate::network::tcp::{Client, NetworkConnection};
use async_trait::async_trait;
use eyre::Result;
use std::fmt::{Debug, Formatter};
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpStream;
use tokio_rustls::rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
use tokio_rustls::rustls::pki_types::UnixTime;
use tokio_rustls::rustls::{
    pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, ServerName},
    ClientConfig, DigitallySignedStruct, Error, RootCertStore, SignatureScheme,
};
use tokio_rustls::{TlsConnector, TlsStream};

#[derive(Clone)]
pub struct TlsClient {
    tls_connector: TlsConnector,
}

#[derive(Clone)]
pub struct TcpClient {}

/// A `ServerCertVerifier` that blindly accepts **any** certificate.
pub struct NoCertificateVerification;

impl Debug for NoCertificateVerification {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("NoCertificateVerification")
    }
}

impl ServerCertVerifier for NoCertificateVerification {
    #[allow(clippy::too_many_arguments)]
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        // Simply say “everything is fine”.
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, Error> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::RSA_PKCS1_SHA1,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::ECDSA_NISTP521_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::ECDSA_NISTP521_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::ED25519,
            SignatureScheme::ED448,
        ]
    }
}

impl TlsClient {
    pub async fn new_with_skip_verification() -> Result<Self> {
        let client_config = ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertificateVerification {}))
            .with_no_client_auth();

        let tls_connector = TlsConnector::from(Arc::new(client_config));
        Ok(Self { tls_connector })
    }

    pub async fn new(key_file: &str, cert_file: &str, root_cert: &str) -> Result<Self> {
        let mut root_cert_store = RootCertStore::empty();
        for cert in CertificateDer::pem_file_iter(root_cert)? {
            root_cert_store.add(cert?)?;
        }

        let certs = CertificateDer::pem_file_iter(cert_file)?
            .map(|res| res.map_err(eyre::Report::from))
            .collect::<Result<Vec<_>>>()?;
        let key = PrivateKeyDer::from_pem_file(key_file)?;
        let client_config = ClientConfig::builder()
            .with_root_certificates(root_cert_store)
            .with_client_auth_cert(certs, key)?;

        let tls_connector = TlsConnector::from(Arc::new(client_config));
        Ok(Self { tls_connector })
    }

    /// Create a client that trusts the system CAs
    pub async fn new_with_root_certs() -> Result<Self> {
        let mut roots = RootCertStore::empty();
        for cert in rustls_native_certs::load_native_certs().expect("could not load platform certs")
        {
            roots.add(cert)?;
        }
    
        let client_config = ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
    
        let tls_connector = TlsConnector::from(Arc::new(client_config));
        Ok(Self { tls_connector })
    }
    
    /// Create a client that trusts only the given CA certificate file (PEM)
    pub async fn new_with_ca(ca_cert_path: &str) -> Result<Self> {
        let mut root_store = RootCertStore::empty();
        for cert in CertificateDer::pem_file_iter(ca_cert_path)? {
            root_store.add(cert?)?;
        }
        let config = ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        let tls_connector = TlsConnector::from(Arc::new(config));
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

// allow mixing TLS client and TCP server by boxing connections
/// Dynamic stream type for mixed connectors and listeners
pub type DynStream = Box<dyn NetworkConnection>;

#[derive(Clone)]
pub struct BoxTcpClient(pub TcpClient);
#[async_trait]
impl Client for BoxTcpClient {
    type Output = DynStream;
    async fn connect(&self, addr: SocketAddr) -> Result<Self::Output> {
        let stream = self.0.connect(addr).await?;
        Ok(Box::new(stream))
    }
}

#[derive(Clone)]
pub struct BoxTlsClient(pub TlsClient);
#[async_trait]
impl Client for BoxTlsClient {
    type Output = DynStream;
    async fn connect(&self, addr: SocketAddr) -> Result<Self::Output> {
        let stream = self.0.connect(addr).await?;
        Ok(Box::new(stream))
    }
}
