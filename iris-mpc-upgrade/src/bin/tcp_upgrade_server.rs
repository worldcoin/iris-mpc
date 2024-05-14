use std::{
    net::SocketAddr,
    sync::{atomic::AtomicUsize, Arc},
    time::{Duration, Instant},
};

use clap::Parser;
use color_eyre::eyre::{bail, Context};
use futures_concurrency::future::Join;
use iris_mpc_upgrade::{
    packets::{ShamirSharesMessage, TwoToThreeIrisCodeMessage},
    upgrade::IrisCodeUpgrader,
    IrisShareTestFileSink, PartyID, Seed,
};
use rand::{thread_rng, Rng};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader, BufStream},
    net::TcpStream,
    sync::mpsc,
    task::JoinSet,
};

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(long)]
    pub client_bind_addr: SocketAddr,

    #[clap(long)]
    pub mpc_bind_addr: SocketAddr,

    #[clap(long)]
    pub next_server: SocketAddr,

    #[clap(long)]
    pub prev_server: SocketAddr,

    #[clap(long)]
    pub party_id: PartyID,

    #[clap(long, default_value = "8")]
    pub threads: usize,
}

fn install_tracing() {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

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
    msg1: TwoToThreeIrisCodeMessage,
    msg2: TwoToThreeIrisCodeMessage,
    masks: ShamirSharesMessage,
}

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    install_tracing();
    let args = Args::parse();

    println!("Client bind address: {}", args.client_bind_addr);
    println!("MPC bind address: {}", args.mpc_bind_addr);
    println!("Next server address: {}", args.next_server);
    println!("Previous server address: {}", args.prev_server);

    // listen for incoming connections
    let mpc_listener = tokio::net::TcpListener::bind(args.mpc_bind_addr).await?;

    let mut processing_tasks = JoinSet::new();
    let finished_counter = Arc::new(AtomicUsize::new(0));
    let mut senders = Vec::with_capacity(args.threads);

    for tid in 0..args.threads {
        let (sender, receiver) = mpsc::channel(32);
        let finished_counter = Arc::clone(&finished_counter);
        let stream_next = tokio::task::spawn(async move {
            match tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    match TcpStream::connect(args.next_server).await {
                        Err(ref e) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                            // retry after a bit, this is just for convienience so we dont need to start server first
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                        rest => break rest,
                    }
                }
            })
            .await
            {
                Ok(res) => res,
                // err means the timout has elapsed, map this to io::Error
                Err(e) => {
                    tracing::error!("Failed to connect to next server: {}", e);
                    Err(e.into())
                }
            }
        });

        let (stream_prev, _) = mpc_listener.accept().await?;
        let stream_next = stream_next.await??;

        senders.push(sender);
        processing_tasks.spawn(main_task_loop(
            stream_next,
            stream_prev,
            args.party_id,
            receiver,
            tid,
            finished_counter,
        ));
    }

    tracing::info!("Both Servers connected");

    // listen for incoming connections from clients
    let client_listener = tokio::net::TcpListener::bind(args.client_bind_addr).await?;

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

    // wrap into framed
    // let mut client_stream1 = Framed::new(client_stream1, BincodeCodec::<ClientMessages>::new());
    // let mut client_stream2 = Framed::new(client_stream2, BincodeCodec::<ClientMessages>::new());
    let mut sending = Duration::default();
    let mut receiving = Duration::default();
    for i in start1..end1 {
        let start = Instant::now();
        let mut message1 = TwoToThreeIrisCodeMessage::default();
        let mut message2 = TwoToThreeIrisCodeMessage::default();
        let mut masks = ShamirSharesMessage::default();
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

    tracing::info!("Receiving took: {}s", receiving.as_secs_f64());
    tracing::info!("Sending took: {}s", sending.as_secs_f64());
    // close all senders
    drop(senders);

    // wait for all tasks should be done, cleanup
    while let Some(r) = processing_tasks.join_next().await {
        r??;
    }

    Ok(())
}

async fn main_task_loop(
    mut stream_next: TcpStream,
    mut stream_prev: TcpStream,
    party_id: PartyID,
    mut receiver: mpsc::Receiver<UpgradeTask>,
    thread_id: usize,
    finished_counter: Arc<AtomicUsize>,
) -> color_eyre::Result<()> {
    let local_seed = {
        let seed: [u8; 16] = thread_rng().gen();
        stream_next.write_all(&seed[..]).await?;
        stream_next.flush().await?;
        Seed::from(seed)
    };
    let remote_seed = {
        let mut buf = [0u8; 16];
        stream_prev.read_exact(buf.as_mut()).await?;
        Seed::from(buf)
    };
    stream_next.write_u64(thread_id as u64).await?;
    stream_next.flush().await?;
    let tid = stream_prev.read_u64().await?;
    assert_eq!(tid, thread_id as u64);
    let mut stream_next = BufStream::with_capacity(1024 * 1024, 1024 * 1024, stream_next);
    let mut stream_prev = BufStream::with_capacity(1024 * 1024, 1024 * 1024, stream_prev);

    // let mut stream_next = Framed::new(stream_next, BincodeCodec::<MpcMessages>::new());
    // let mut stream_prev = Framed::new(stream_prev, BincodeCodec::<MpcMessages>::new());

    tracing::info!("Server setup complete");
    let upgrader = IrisCodeUpgrader::new(local_seed, remote_seed, party_id);
    let sink = IrisShareTestFileSink::new(format!("./out{}", party_id as u8).into())?;
    loop {
        let UpgradeTask { msg1, msg2, masks } = match receiver.recv().await {
            Some(x) => x,
            None => break,
        };
        let up = upgrader.clone();
        let (local, mut remote) =
            tokio::task::spawn_blocking(move || up.stage1(msg1, msg2)).await??;
        remote
            .send(&mut stream_next)
            .await
            .context("writing stage1 message")?;
        remote
            .recv(&mut stream_prev)
            .await
            .context("reading stage1 message")?;
        let up = upgrader.clone();
        let (local, mut remote) =
            tokio::task::spawn_blocking(move || up.stage2(local, remote)).await??;
        remote
            .send(&mut stream_next)
            .await
            .context("writing stage2 message")?;
        remote
            .recv(&mut stream_prev)
            .await
            .context("reading stage2 message")?;
        let up = upgrader.clone();
        let [m0, m1, m2] = tokio::task::spawn_blocking(move || up.stage3(local, remote)).await??;

        let [m0, mut m1, mut m2] = match party_id {
            PartyID::ID0 => [m0, m1, m2],
            PartyID::ID1 => [m1, m2, m0],
            PartyID::ID2 => [m2, m0, m1],
        };

        m1.send(&mut stream_next)
            .await
            .context("writing stage3 message")?;
        m2.send(&mut stream_prev)
            .await
            .context("writing stage3 message")?;
        m1.recv(&mut stream_next)
            .await
            .context("reading stage3 message")?;
        m2.recv(&mut stream_prev)
            .await
            .context("reading stage3 message")?;
        let sink = sink.clone();
        tokio::task::spawn_blocking(move || IrisCodeUpgrader::finalize([m0, m1, m2], masks, &sink))
            .await??;
        finished_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    Ok(())
}
