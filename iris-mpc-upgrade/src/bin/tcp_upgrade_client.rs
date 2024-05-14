use std::net::SocketAddr;

use clap::Parser;
use futures_concurrency::future::Join;
use iris_mpc_upgrade::{
    packets::{ShamirSharesMessage, TwoToThreeIrisCodeMessage},
    shamir::share_bool_vec,
    share::share_to_rep_shares,
    OldIrisShareSource,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use tokio::{
    io::{AsyncWriteExt, BufWriter},
    net::TcpStream,
};

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(long)]
    pub server1: SocketAddr,

    #[clap(long)]
    pub server2: SocketAddr,

    #[clap(long)]
    pub server3: SocketAddr,

    #[clap(long)]
    pub db_size: u64,

    #[clap(long)]
    pub party_id: u8,
}

struct MockOldDbParty1 {
    rng: ChaCha20Rng,
}

struct MockOldDbParty2 {
    rng: ChaCha20Rng,
}

// Generate some random iris code shares by using the same rng seed and using X as one share and -X as the other share

fn old_dbs() -> (MockOldDbParty1, MockOldDbParty2) {
    let seed = [0u8; 32];
    let rng = ChaCha20Rng::from_seed(seed);
    (
        MockOldDbParty1 { rng: rng.clone() },
        MockOldDbParty2 { rng },
    )
}

impl OldIrisShareSource for MockOldDbParty1 {
    fn load_code_share(&self, share_id: u64) -> std::io::Result<Vec<u16>> {
        let mut rng = self.rng.clone();
        rng.set_word_pos(share_id as u128 * 12800);
        Ok((0..12800)
            .map(|i| rng.gen::<u16>().wrapping_add((1 - (i % 3)) as u16))
            .collect())
    }

    fn load_mask(&self, _share_id: u64) -> std::io::Result<Vec<bool>> {
        Ok((0..12800).map(|i| (1 - (i % 3)) != 0).collect())
    }
}

impl OldIrisShareSource for MockOldDbParty2 {
    fn load_code_share(&self, share_id: u64) -> std::io::Result<Vec<u16>> {
        let mut rng = self.rng.clone();
        rng.set_word_pos(share_id as u128 * 12800);
        Ok((0..12800)
            .map(|_| 0u16.wrapping_sub(rng.gen::<u16>()))
            .collect())
    }

    fn load_mask(&self, _share_id: u64) -> std::io::Result<Vec<bool>> {
        Ok((0..12800).map(|i| (1 - (i % 3)) != 0).collect())
    }
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

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    install_tracing();
    let args = Args::parse();

    if args.party_id > 1 {
        panic!("Party id must be 0, 1");
    }

    let (db1, db2) = old_dbs();

    let mut server1 = BufWriter::new(TcpStream::connect(args.server1).await?);
    let mut server2 = BufWriter::new(TcpStream::connect(args.server2).await?);
    let mut server3 = BufWriter::new(TcpStream::connect(args.server3).await?);
    server1.write_u8(args.party_id).await?;
    server2.write_u8(args.party_id).await?;
    server3.write_u8(args.party_id).await?;
    let start = 0u64;
    let end = args.db_size;
    server1.write_u64(start).await?;
    server2.write_u64(start).await?;
    server3.write_u64(start).await?;
    server1.write_u64(end).await?;
    server2.write_u64(end).await?;
    server3.write_u64(end).await?;

    server1.flush().await?;
    server2.flush().await?;
    server3.flush().await?;

    // wrap into framed
    tracing::info!("Connected to all servers");
    tracing::info!("Starting processing...");

    for id in 0..args.db_size {
        tracing::trace!("Processing id: {}", id);
        let [a, b, c] = if args.party_id == 0 {
            let shares = db1.load_code_share(id)?;
            share_to_rep_shares(&shares, &mut rand::thread_rng())
        } else {
            let shares = db2.load_code_share(id)?;
            share_to_rep_shares(&shares, &mut rand::thread_rng())
        };

        let message_a = TwoToThreeIrisCodeMessage {
            id,
            party_id: 0,
            from: args.party_id,
            data: a,
        };
        let message_b = TwoToThreeIrisCodeMessage {
            id,
            party_id: 1,
            from: args.party_id,
            data: b,
        };
        let message_c = TwoToThreeIrisCodeMessage {
            id,
            party_id: 2,
            from: args.party_id,
            data: c,
        };
        let (a, b, c) = (
            message_a.send(&mut server1),
            message_b.send(&mut server2),
            message_c.send(&mut server3),
        )
            .join()
            .await;
        a?;
        b?;
        c?;

        if args.party_id == 0 {
            let masks = db1.load_mask(id)?;
            let [ma, mb, mc] = share_bool_vec(&masks, &mut rand::thread_rng());

            let masks_a = ShamirSharesMessage {
                id,
                party_id: 0,
                from: args.party_id,
                data: ma,
            };
            let masks_b = ShamirSharesMessage {
                id,
                party_id: 1,
                from: args.party_id,
                data: mb,
            };
            let masks_c = ShamirSharesMessage {
                id,
                party_id: 2,
                from: args.party_id,
                data: mc,
            };

            let (a, b, c) = (
                masks_a.send(&mut server1),
                masks_b.send(&mut server2),
                masks_c.send(&mut server3),
            )
                .join()
                .await;
            a?;
            b?;
            c?;
        }
        tracing::trace!("Finished id: {}", id);
    }
    tracing::info!("Processing done!");

    Ok(())
}
