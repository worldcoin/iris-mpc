use clap::Parser;
use iris_mpc_upgrade::{
    config::PingClientConfig,
    proto::iris_mpc_reshare::{ping_pong_client::PingPongClient, Ping},
    utils::{extract_domain, install_tracing},
};
use tonic::transport::{Certificate, Channel, ClientTlsConfig};

#[tokio::main]
async fn main() -> eyre::Result<()> {
    install_tracing();
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");

    let config = PingClientConfig::parse();

    let pem = tokio::fs::read(config.client_tls_cert_path)
        .await
        .expect("oh no, the cert file wasn't loaded");

    let cert = Certificate::from_pem(pem.clone());

    let domain = extract_domain(&config.server_url.clone(), true)?;

    println!(
        "TLS connecting to address {} using domain {}",
        config.server_url.clone(),
        domain
    );
    let tls = ClientTlsConfig::new()
        .domain_name(domain)
        .ca_certificate(cert);

    // build a tonic transport channel ourselves, since we want to add a tls config
    let channel = Channel::from_shared(config.server_url.clone())?
        .tls_config(tls)?
        .connect()
        .await?;

    let mut grpc_client = PingPongClient::new(channel);
    let req = Ping {
        message:       config.message.to_string(),
        delay_seconds: 0f32,
    };
    let resp = grpc_client.send_ping(req).await?;
    let resp = resp.into_inner();
    println!("Received response: {:?}", resp);
    Ok(())
}
