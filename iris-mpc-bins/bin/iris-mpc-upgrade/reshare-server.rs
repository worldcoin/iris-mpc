use ampc_server_utils::TaskMonitor;
use clap::Parser;
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::ReShareServerConfig,
    proto::{
        get_size_of_reshare_iris_code_share_batch,
        iris_mpc_reshare::iris_code_re_share_service_server::IrisCodeReShareServiceServer,
    },
    reshare::{GrpcReshareServer, IrisCodeReshareReceiverHelper},
    utils::{install_tracing, spawn_healthcheck_server},
};
use tonic::transport::Server;

const APP_NAME: &str = "SMPC";

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let config = ReShareServerConfig::parse();

    tracing::info!("Starting healthcheck server.");

    let mut background_tasks = TaskMonitor::new();
    let _health_check_abort = background_tasks
        .spawn(async move { spawn_healthcheck_server(config.healthcheck_port).await });
    background_tasks.check_tasks();
    tracing::info!(
        "Healthcheck server running on port {}.",
        config.healthcheck_port.clone()
    );

    tracing::info!(
        "Healthcheck server running on port {}.",
        config.healthcheck_port
    );

    let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
    let postgres_client =
        PostgresClient::new(&config.db_url, &schema_name, AccessMode::ReadWrite).await?;
    let store = Store::new(&postgres_client).await?;

    let receiver_helper = IrisCodeReshareReceiverHelper::new(
        config.party_id as usize,
        config.sender1_party_id as usize,
        config.sender2_party_id as usize,
        config.max_buffer_size,
    );

    let encoded_message_size =
        get_size_of_reshare_iris_code_share_batch(config.batch_size as usize);
    if encoded_message_size > 100 * 1024 * 1024 {
        tracing::warn!(
            "encoded batch message size is large: {}MB",
            encoded_message_size as f64 / 1024.0 / 1024.0
        );
    }
    let encoded_message_size_with_buf = (encoded_message_size as f64 * 1.1) as usize;
    let iris_reshare_service =
        IrisCodeReShareServiceServer::new(GrpcReshareServer::new(store, receiver_helper))
            .max_decoding_message_size(encoded_message_size_with_buf)
            .max_encoding_message_size(encoded_message_size_with_buf);

    Server::builder()
        .add_service(iris_reshare_service)
        .serve_with_shutdown(config.bind_addr, shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}
