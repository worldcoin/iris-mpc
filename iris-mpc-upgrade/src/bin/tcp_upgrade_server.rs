use axum::{routing::get, Router};
use clap::Parser;
use eyre::{bail, Context};
use futures_concurrency::future::Join;
use iris_mpc_common::helpers::task_monitor::TaskMonitor;
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::{Eye, UpgradeServerConfig, BATCH_SUCCESSFUL_ACK, FINAL_BATCH_SUCCESSFUL_ACK},
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    IrisCodeUpgrader, NewIrisShareSink,
};
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};

const APP_NAME: &str = "SMPC";

fn install_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

struct UpgradeTask {
    msg1:  TwoToThreeIrisCodeMessage,
    msg2:  TwoToThreeIrisCodeMessage,
    masks: MaskShareMessage,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    install_tracing();
    let args = UpgradeServerConfig::parse();

    println!("Client bind address: {}", args.bind_addr);

    let schema_name = format!("{}_{}_{}", APP_NAME, args.environment, args.party_id);
    let sink = IrisShareDbSink::new(Store::new(&args.db_url, &schema_name).await?, args.eye);

    tracing::info!("Starting healthcheck server.");

    let mut background_tasks = TaskMonitor::new();
    let _health_check_abort = background_tasks.spawn(async move {
        let app = Router::new().route("/health", get(|| async {})); // implicit 200 return
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .wrap_err("healthcheck listener bind error")?;
        axum::serve(listener, app)
            .await
            .wrap_err("healthcheck listener server launch error")?;
        Ok(())
    });

    background_tasks.check_tasks();
    tracing::info!("Healthcheck server running on port 3000.");

    let upgrader = IrisCodeUpgrader::new(args.party_id, sink.clone());

    // listen for incoming connections from clients
    let client_listener = tokio::net::TcpListener::bind(args.bind_addr).await?;

    let mut client_stream1 = BufReader::new(client_listener.accept().await?.0);
    let mut client_stream2 = BufReader::new(client_listener.accept().await?.0);
    tracing::info!("Both Clients connected");
    let id1 = client_stream1.read_u8().await?;
    let id2 = client_stream2.read_u8().await?;

    let (mut client_stream1, mut client_stream2) = if id1 == 0 && id2 == 1 {
        (client_stream1, client_stream2)
    } else if id1 == 1 && id2 == 0 {
        (client_stream2, client_stream1)
    } else {
        bail!("Invalid client ids: {}, {}", id1, id2);
    };

    let eye1 = client_stream1.read_u8().await?;
    let eye2 = client_stream2.read_u8().await?;
    if eye1 != args.eye as u8 || eye2 != args.eye as u8 {
        bail!(
            "Invalid eye: client1: {}, client2: {}, we want: {:?}={}",
            eye1,
            eye2,
            args.eye,
            args.eye as u8
        );
    }

    // exclusive ranges
    let start1 = client_stream1.read_u64().await?;
    let end1 = client_stream1.read_u64().await?;
    let start2 = client_stream2.read_u64().await?;
    let end2 = client_stream2.read_u64().await?;

    if start1 != start2 || end1 != end2 {
        bail!(
            "Invalid batch ids: {}-{}, {}-{}",
            start1,
            end1,
            start2,
            end2
        );
    }
    let num_elements = end1.checked_sub(start1).unwrap();
    let num_batches = num_elements / u64::from(args.batch_size);
    tracing::info!(
        "Batch size: {}, num batches: {}",
        args.batch_size,
        num_batches
    );

    let mut batch = Vec::new();

    for batch_num in 0..num_batches + 1 {
        tracing::info!(
            "Processing batch {} of size: {}",
            batch_num,
            args.batch_size
        );
        let start_time = Instant::now();
        let batch_size_1_message = client_stream1.read_u8().await?;
        let batch_size_2_message = client_stream2.read_u8().await?;

        if batch_size_1_message != batch_size_2_message {
            bail!(
                "Invalid batch size: client1: {}, client2: {}",
                batch_size_1_message,
                batch_size_2_message,
            );
        }

        for _ in 0..batch_size_1_message {
            let mut message1 = TwoToThreeIrisCodeMessage::default();
            let mut message2 = TwoToThreeIrisCodeMessage::default();
            let mut masks = MaskShareMessage::default();

            let (result1, result2) = (
                message1.recv(&mut client_stream1),
                message2.recv(&mut client_stream2),
            )
                .join()
                .await;

            if let Err(e) = result1 {
                tracing::error!("Failed to receive message1: {:?}", e);
                break;
            }
            if let Err(e) = result2 {
                tracing::error!("Failed to receive message2: {:?}", e);
                break;
            }

            masks.recv(&mut client_stream1).await?;
            if message1.id != message2.id || message1.id != masks.id {
                tracing::error!(
                    "Message IDs out of sync: {} != {} != {}",
                    message1.id,
                    message2.id,
                    masks.id
                );
                return Err(eyre::eyre!("Message ID mismatch"));
            }

            batch.push(UpgradeTask {
                msg1: message1,
                msg2: message2,
                masks,
            });
        }

        for (i, task) in batch.drain(..).enumerate() {
            tracing::debug!("Task: {:?}", i);
            upgrader
                .finalize(task.msg1.clone(), task.msg2.clone(), task.masks.clone())
                .await?;
        }
        // Send an ACK to the client
        client_stream1.write_u8(BATCH_SUCCESSFUL_ACK).await?;
        client_stream2.write_u8(BATCH_SUCCESSFUL_ACK).await?;
        client_stream1.flush().await?;
        client_stream2.flush().await?;

        let duration = start_time.elapsed();
        tracing::info!("Processed batch in {:.2?}", duration);
    }
    client_stream2.write_u8(FINAL_BATCH_SUCCESSFUL_ACK).await?;
    client_stream1.write_u8(FINAL_BATCH_SUCCESSFUL_ACK).await?;

    sink.update_iris_id_sequence().await?;

    Ok(())
}

#[derive(Clone)]
struct IrisShareDbSink {
    store: Store,
    eye:   Eye,
}

impl IrisShareDbSink {
    pub fn new(store: Store, eye: Eye) -> Self {
        Self { store, eye }
    }
}

impl NewIrisShareSink for IrisShareDbSink {
    async fn store_code_mask_share(
        &self,
        share_id: u64,
        code_share: &[u16; iris_mpc_common::IRIS_CODE_LENGTH],
        mask_share: &[u16; iris_mpc_common::MASK_CODE_LENGTH],
    ) -> eyre::Result<()> {
        let id = i64::try_from(share_id).expect("id fits into i64");
        match self.eye {
            Eye::Left => {
                self.store
                    .insert_or_update_left_iris(id, code_share, mask_share)
                    .await
            }
            Eye::Right => {
                self.store
                    .insert_or_update_right_iris(id, code_share, mask_share)
                    .await
            }
        }
    }

    async fn update_iris_id_sequence(&self) -> eyre::Result<()> {
        self.store.update_iris_id_sequence().await
    }
}
