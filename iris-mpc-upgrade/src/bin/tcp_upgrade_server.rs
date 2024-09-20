use async_stream::try_stream;
use async_trait::async_trait;
use axum::{routing::get, Router};
use clap::Parser;
use eyre::{bail, Context, Result};
use futures::{Stream, StreamExt};
use iris_mpc_common::{helpers::task_monitor::TaskMonitor, id::PartyID};
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::{Eye, UpgradeServerConfig},
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    IrisCodeUpgrader, NewIrisShareSink,
};
use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{atomic::AtomicUsize, Arc},
};
use tokio::{
    io::{AsyncReadExt, BufReader},
    net::TcpListener,
    sync::mpsc,
    task::JoinSet,
};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

const APP_NAME: &str = "SMPC";

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let args = UpgradeServerConfig::parse();

    println!("Client bind address: {}", args.bind_addr);

    // Create dependencies
    let schema_name = format!("{}_{}_{}", APP_NAME, args.environment, args.party_id);
    let store = Store::new(&args.db_url, &schema_name).await?;
    let sink = IrisShareDbSink::new(store, args.eye);
    let upgrader = IrisCodeUpgrader::new(args.party_id, sink.clone());

    // Start healthcheck server
    start_healthcheck_server().await?;

    // Create task source and application
    let task_source = TcpUpgradeTaskSource::new(args.bind_addr.clone(), args.eye).await?;
    let application = Application::new(upgrader, task_source, args.threads);

    // Run the application
    application.run().await?;

    Ok(())
}

fn install_tracing() {
    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

async fn start_healthcheck_server() -> Result<()> {
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
    Ok(())
}

struct Application<Sink, Source>
where
    Sink: IrisShareSink,
    Source: UpgradeTaskSource,
{
    upgrader: IrisCodeUpgrader<Sink>,
    task_source: Source,
    num_threads: usize,
}

impl<Sink, Source> Application<Sink, Source>
where
    Sink: IrisShareSink + Clone + 'static,
    Source: UpgradeTaskSource + 'static,
{
    fn new(upgrader: IrisCodeUpgrader<Sink>, task_source: Source, num_threads: usize) -> Self {
        Self {
            upgrader,
            task_source,
            num_threads,
        }
    }

    async fn run(self) -> Result<()> {
        let finished_counter = Arc::new(AtomicUsize::new(0));

        self.task_source
            .task_stream()
            .await?
            .for_each_concurrent(self.num_threads, |task_result| {
                let upgrader = self.upgrader.clone();
                let finished_counter = finished_counter.clone();
                async move {
                    match task_result {
                        Ok(task) => {
                            if let Err(e) = upgrader
                                .finalize(task.msg1, task.msg2, task.masks)
                                .await
                            {
                                tracing::error!("Error processing task: {:?}", e);
                            } else {
                                finished_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            }
                        }
                        Err(e) => {
                            tracing::error!("Error receiving task: {:?}", e);
                        }
                    }
                }
            })
            .await;

        Ok(())
    }
}

struct UpgradeTask {
    msg1:  TwoToThreeIrisCodeMessage,
    msg2:  TwoToThreeIrisCodeMessage,
    masks: MaskShareMessage,
}

#[async_trait]
trait IrisShareSink: Send + Sync + Clone {
    async fn store_code_mask_share(
        &self,
        share_id: u64,
        code_share: &[u16; iris_mpc_common::IRIS_CODE_LENGTH],
        mask_share: &[u16; iris_mpc_common::MASK_CODE_LENGTH],
    ) -> Result<()>;
}

#[async_trait]
trait UpgradeTaskSource: Send + Sync {
    async fn task_stream(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<UpgradeTask>> + Send + '_>>>;
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

#[async_trait]
impl IrisShareSink for IrisShareDbSink {
    async fn store_code_mask_share(
        &self,
        share_id: u64,
        code_share: &[u16; iris_mpc_common::IRIS_CODE_LENGTH],
        mask_share: &[u16; iris_mpc_common::MASK_CODE_LENGTH],
    ) -> Result<()> {
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
}

struct TcpUpgradeTaskSource {
    bind_addr: SocketAddr,
    eye:       Eye,
}

impl TcpUpgradeTaskSource {
    pub async fn new(bind_addr: SocketAddr, eye: Eye) -> Result<Self> {
        Ok(Self { bind_addr, eye })
    }
}

#[async_trait]
impl UpgradeTaskSource for TcpUpgradeTaskSource {
    async fn task_stream(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<UpgradeTask>> + Send + '_>>> {
        let eye = self.eye;
        let bind_addr = self.bind_addr;

        let listener = TcpListener::bind(bind_addr).await?;
        let stream = try_stream! {
            let mut client_stream1 = BufReader::new(listener.accept().await?.0);
            let mut client_stream2 = BufReader::new(listener.accept().await?.0);
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
            if eye1 != eye as u8 || eye2 != eye as u8 {
                bail!(
                    "Invalid eye: client1: {}, client2: {}, we want: {:?}={}",
                    eye1,
                    eye2,
                    eye,
                    eye as u8
                );
            }

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
            tracing::info!("Doing a batch of {} elements", num_elements);

            for i in start1..end1 {
                let mut message1 = TwoToThreeIrisCodeMessage::default();
                let mut message2 = TwoToThreeIrisCodeMessage::default();
                let mut masks = MaskShareMessage::default();

                let (a, b) = (
                    message1.recv(&mut client_stream1),
                    message2.recv(&mut client_stream2),
                )
                    .await;
                a?;
                b?;
                masks.recv(&mut client_stream1).await?;

                if message1.id != i {
                    tracing::error!(
                        "Client messages out of order: got {}, expected {}",
                        message1.id,
                        i
                    );
                    break;
                }
                if message1.id != message2.id {
                    tracing::error!(
                        "Client messages out of order: {} != {}",
                        message1.id,
                        message2.id
                    );
                    break;
                }
                if masks.id != message1.id {
                    tracing::error!(
                        "Client messages out of order: {} != {}",
                        masks.id,
                        message1.id
                    );
                    break;
                }

                yield UpgradeTask {
                    msg1: message1,
                    msg2: message2,
                    masks,
                };
            }
        };

        Ok(Box::pin(stream))
    }
}
