use clap::Parser;
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::ReShareServerConfig,
    proto::{
        self, iris_mpc_reshare::iris_code_re_share_service_server::IrisCodeReShareServiceServer,
    },
    reshare::{GrpcReshareServer, IrisCodeReshareReceiverHelper},
    utils::install_tracing,
};
use tonic::transport::Server;

const APP_NAME: &str = "SMPC";

#[tokio::main]
async fn main() -> eyre::Result<()> {
    install_tracing();
    let config = ReShareServerConfig::parse();

    let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
    let store = Store::new(&config.db_url, &schema_name).await?;

    let receiver_helper = IrisCodeReshareReceiverHelper::new(
        config.party_id as usize,
        config.sender1_party_id as usize,
        config.sender2_party_id as usize,
        config.max_buffer_size,
    );

    let encoded_message_size =
        proto::get_size_of_reshare_iris_code_share_batch(config.batch_size as usize);
    if encoded_message_size > 100 * 1024 * 1024 {
        tracing::warn!(
            "encoded batch message size is large: {}MB",
            encoded_message_size as f64 / 1024.0 / 1024.0
        );
    }
    let encoded_message_size_with_buf = (encoded_message_size as f64 * 1.1) as usize;
    let grpc_server =
        IrisCodeReShareServiceServer::new(GrpcReshareServer::new(store, receiver_helper))
            .max_decoding_message_size(encoded_message_size_with_buf)
            .max_encoding_message_size(encoded_message_size_with_buf);

    Server::builder()
        .add_service(grpc_server)
        .serve_with_shutdown(config.bind_addr, shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}
