use clap::Parser;
use eyre::{bail, Context};
use futures_concurrency::future::Join;
use iris_mpc_common::id::PartyID;
use iris_mpc_store::Store;
use iris_mpc_upgrade::{
    config::{Eye, UpgradeServerConfig},
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    IrisCodeUpgrader, NewIrisShareSink,
};
use std::{
    sync::{atomic::AtomicUsize, Arc},
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncReadExt, BufReader},
    sync::mpsc,
    task::JoinSet,
};

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

    let mut processing_tasks = JoinSet::new();
    let finished_counter = Arc::new(AtomicUsize::new(0));
    let mut senders = Vec::with_capacity(args.threads);

    let sink = IrisShareDbSink::new(Store::new(&args.db_url, "upgrade").await?, args.eye);

    tracing::info!("Starting healthcheck server.");

    let mut health_task = JoinSet::new();
    let _health_check_abort = health_task.spawn(async move {
        let app = Router::new().route("/health", get(|| async {})); // implicit 200 return
        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
            .await
            .wrap_err("healthcheck listener bind error")?;
        axum::serve(listener, app)
            .await
            .wrap_err("healthcheck listener server launch error")?;

        Ok(())
    });

    for _ in 0..args.threads {
        let (sender, receiver) = mpsc::channel(32);
        let finished_counter = Arc::clone(&finished_counter);

        let sink = sink.clone();
        senders.push(sender);
        processing_tasks.spawn(async move {
            match main_task_loop(args.party_id, receiver, finished_counter, sink).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    tracing::error!("Error in processing task: {:?}", e);
                    Err(e)
                }
            }
        });
    }

    tracing::info!("Spawned processing tasks");

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
    tracing::info!("Doing a batch of {} elements", num_elements);

    let mut sending = Duration::default();
    let mut receiving = Duration::default();
    for i in start1..end1 {
        let start = Instant::now();
        let mut message1 = TwoToThreeIrisCodeMessage::default();
        let mut message2 = TwoToThreeIrisCodeMessage::default();
        let mut masks = MaskShareMessage::default();
        let (a, b) = (
            message1.recv(&mut client_stream1),
            message2.recv(&mut client_stream2),
        )
            .join()
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
        receiving += start.elapsed();
        let start = Instant::now();

        senders[i as usize % args.threads]
            .send(UpgradeTask {
                msg1: message1,
                msg2: message2,
                masks,
            })
            .await?;
        sending += start.elapsed();
    }

    tracing::debug!("Receiving took: {}s", receiving.as_secs_f64());
    tracing::debug!("Sending took: {}s", sending.as_secs_f64());
    // close all senders
    drop(senders);

    tracing::info!("Waiting for remaining tasks to finish...");
    // wait for all tasks should be done, cleanup
    while let Some(r) = processing_tasks.join_next().await {
        r??;
    }
    tracing::info!("All tasks finished");

    Ok(())
}

async fn main_task_loop(
    party_id: PartyID,
    mut receiver: mpsc::Receiver<UpgradeTask>,
    finished_counter: Arc<AtomicUsize>,
    sink: IrisShareDbSink,
) -> eyre::Result<()> {
    let upgrader = IrisCodeUpgrader::new(party_id, sink);
    loop {
        let UpgradeTask { msg1, msg2, masks } = match receiver.recv().await {
            Some(x) => x,
            None => break,
        };
        upgrader.finalize(msg1, msg2, masks).await?;
        finished_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
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
}
