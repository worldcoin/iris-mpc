use clap::Parser;
use eyre::{Context, ContextCompat};
use futures::{Stream, StreamExt};
use futures_concurrency::future::Join;
use indicatif::{ProgressBar, ProgressStyle};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    IRIS_CODE_LENGTH,
};
use iris_mpc_upgrade::{
    config::UpgradeClientConfig,
    db::V1Db,
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    OldIrisShareSource,
};
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rustls::{pki_types::ServerName, ClientConfig};
use std::{array, convert::TryFrom, pin::Pin, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};
use tokio_rustls::{client::TlsStream, TlsConnector};

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

async fn prepare_tls_stream_for_writing(
    address: &str,
    client_config: Arc<ClientConfig>,
) -> eyre::Result<TlsStream<TcpStream>> {
    // Create a TCP connection
    let stream = TcpStream::connect(address).await?;

    let tls_connector = TlsConnector::from(client_config);

    // Hostname for SNI (Server Name Indication)
    // throw away the port number if there is one (e.g. "localhost:8080" ->
    // "localhost")
    let address = address.split(":").next().context("splitting address")?;
    let dns_name =
        ServerName::try_from(address.to_owned()).context("trying to convert address to SNI")?;

    // Perform the TLS handshake to establish a secure connection
    let tls_stream: TlsStream<TcpStream> = tls_connector.connect(dns_name, stream).await?;

    Ok(tls_stream)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    install_tracing();
    let args = UpgradeClientConfig::parse();

    if args.party_id > 1 {
        panic!("Party id must be 0, 1");
    }

    // read the trusted cert
    let mut root_cert_store = rustls::RootCertStore::empty();
    root_cert_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let client_config = Arc::new(
        ClientConfig::builder()
            .with_root_certificates(root_cert_store)
            .with_no_client_auth(),
    );

    let mut server1 = prepare_tls_stream_for_writing(&args.server1, client_config.clone()).await?;
    let mut server2 = prepare_tls_stream_for_writing(&args.server2, client_config.clone()).await?;
    let mut server3 = prepare_tls_stream_for_writing(&args.server3, client_config).await?;

    tracing::info!("Connecting to servers and syncing migration task parameters...");
    server1.write_u8(args.party_id).await?;
    server2.write_u8(args.party_id).await?;
    server3.write_u8(args.party_id).await?;
    server1.write_u8(args.eye as u8).await?;
    server2.write_u8(args.eye as u8).await?;
    server3.write_u8(args.eye as u8).await?;
    let start = args.db_start;
    let end = args.db_end;
    let db_range = start..end;
    server1.write_u64(start).await?;
    server2.write_u64(start).await?;
    server3.write_u64(start).await?;
    server1.write_u64(end).await?;
    server2.write_u64(end).await?;
    server3.write_u64(end).await?;

    server1.flush().await?;
    server2.flush().await?;
    server3.flush().await?;

    tracing::info!("Connected to all servers");
    tracing::info!("Starting processing...");

    let mut rng = ChaCha20Rng::from_entropy();

    let (db0, db1) = old_dbs();

    // need this to maybe store the PG db or otherwise the borrow checker complains
    #[allow(unused_assignments)]
    let mut maybe_shares_db = None;

    #[allow(unused_assignments)]
    let mut maybe_masks_db = None;

    #[allow(clippy::type_complexity)]
    let (mut shares_stream, mut mask_stream): (
        Pin<Box<dyn Stream<Item = eyre::Result<(u64, EncodedBits)>>>>,
        Pin<Box<dyn Stream<Item = eyre::Result<(u64, Bits)>>>>,
    ) = if args.mock {
        match args.party_id {
            0 => (
                Box::pin(db0.stream_shares(db_range.clone())?),
                Box::pin(db0.stream_masks(db_range)?),
            ),
            1 => (
                Box::pin(db1.stream_shares(db_range.clone())?),
                Box::pin(db1.stream_masks(db_range)?),
            ),
            _ => unreachable!(),
        }
    } else {
        maybe_shares_db = Some(V1Database {
            db: V1Db::new(&args.shares_db_url).await?,
        });

        maybe_masks_db = Some(V1Database {
            db: V1Db::new(&args.masks_db_url).await?,
        });

        (
            Box::pin(
                maybe_shares_db
                    .as_ref()
                    .unwrap()
                    .stream_shares(db_range.clone())?,
            ),
            Box::pin(maybe_masks_db.as_ref().unwrap().stream_masks(db_range)?),
        )
    };

    let num_iris_codes = end - start;

    let pb = ProgressBar::new(num_iris_codes).with_message("Migrating iris codes and masks");
    let pb_style = ProgressStyle::default_bar()
        .template(
            "{spinner:.green} {msg} [{elapsed_precise}] [{wide_bar:.green}] {pos:>7}/{len:7} \
             ({eta})",
        )
        .expect("Could not create progress bar");
    pb.set_style(pb_style);

    let mut cur = start;

    while let Some(share_res) = shares_stream.next().await {
        let (share_id, share) = share_res?;
        let (mask_id, mask) = mask_stream
            .next()
            .await
            .context("mask stream ended before share stream did")??;
        eyre::ensure!(
            share_id == mask_id,
            "Share and mask streams out of sync: {} != {}",
            share_id,
            mask_id
        );
        let id = share_id;
        tracing::trace!("Processing id: {}", id);
        let [a, b, c] = {
            let galois_shared_iris_code =
                GaloisRingIrisCodeShare::reencode_extended_iris_code(&share.0, &mut rng);
            [
                galois_shared_iris_code[0].coefs,
                galois_shared_iris_code[1].coefs,
                galois_shared_iris_code[2].coefs,
            ]
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
            let extended_masks = array::from_fn(|i| mask[i] as u16);
            let [ma, mb, mc] = {
                let [a, b, c] =
                    GaloisRingIrisCodeShare::reencode_extended_iris_code(&extended_masks, &mut rng);
                [
                    GaloisRingTrimmedMaskCodeShare::from(a).coefs,
                    GaloisRingTrimmedMaskCodeShare::from(b).coefs,
                    GaloisRingTrimmedMaskCodeShare::from(c).coefs,
                ]
            };

            let masks_a = MaskShareMessage {
                id,
                party_id: 0,
                from: args.party_id,
                data: ma,
            };
            let masks_b = MaskShareMessage {
                id,
                party_id: 1,
                from: args.party_id,
                data: mb,
            };
            let masks_c = MaskShareMessage {
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
        let diff = share_id - cur;
        cur = share_id;
        pb.inc(diff);
    }
    tracing::info!("Processing done!");
    let mut buf = [0u8; 1];
    server1.read_exact(&mut buf[..]).await?;
    server2.read_exact(&mut buf[..]).await?;
    server3.read_exact(&mut buf[..]).await?;
    server1.shutdown().await?;
    server2.shutdown().await?;
    server3.shutdown().await?;
    pb.finish();

    Ok(())
}

// Real v1 databases

struct V1Database {
    db: V1Db,
}

impl OldIrisShareSource for V1Database {
    async fn load_code_share(&self, share_id: u64) -> eyre::Result<EncodedBits> {
        self.db.fetch_share(share_id).await.map(|(_, x)| x)
    }

    async fn load_mask(&self, share_id: u64) -> eyre::Result<Bits> {
        self.db.fetch_mask(share_id).await.map(|(_, x)| x)
    }

    fn stream_shares(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl futures::Stream<Item = eyre::Result<(u64, EncodedBits)>>> {
        Ok(self.db.stream_shares(share_id_range).map(|x| match x {
            Ok((idx, share)) => Ok((u64::try_from(idx).expect("share_id fits into u64"), share)),
            Err(e) => Err(e.into()),
        }))
    }

    fn stream_masks(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl futures::Stream<Item = eyre::Result<(u64, Bits)>>> {
        Ok(self.db.stream_masks(share_id_range).map(|x| match x {
            Ok((idx, share)) => Ok((u64::try_from(idx).expect("share_id fits into u64"), share)),
            Err(e) => Err(e.into()),
        }))
    }
}

// Mocking old databases

struct MockOldDbParty1 {
    rng: ChaCha20Rng,
}

struct MockOldDbParty2 {
    rng: ChaCha20Rng,
}

// Generate some random iris code shares by using the same rng seed and using X
// as one share and -X as the other share

fn old_dbs() -> (MockOldDbParty1, MockOldDbParty2) {
    let seed = [0u8; 32];
    let rng = ChaCha20Rng::from_seed(seed);
    (MockOldDbParty1 { rng: rng.clone() }, MockOldDbParty2 {
        rng,
    })
}

impl OldIrisShareSource for MockOldDbParty1 {
    async fn load_code_share(&self, share_id: u64) -> eyre::Result<EncodedBits> {
        let mut rng = self.rng.clone();
        rng.set_word_pos(share_id as u128 * 12800);
        let mut res = [0u16; IRIS_CODE_LENGTH];
        res.iter_mut().enumerate().for_each(|(i, x)| {
            *x = rng.gen::<u16>().wrapping_add((1 - (i % 3)) as u16);
        });
        Ok(EncodedBits(res))
    }

    async fn load_mask(&self, _share_id: u64) -> eyre::Result<Bits> {
        let mut res = Bits::default();
        (0..IRIS_CODE_LENGTH).for_each(|i| {
            res.set(i, (1 - (i % 3)) != 0);
        });
        Ok(res)
    }

    fn stream_shares(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl futures::Stream<Item = eyre::Result<(u64, EncodedBits)>>> {
        let mut id = share_id_range.start;
        Ok(futures::stream::poll_fn(move |_| {
            if id < share_id_range.end {
                let mut rng = self.rng.clone();
                rng.set_word_pos(id as u128 * 12800);
                let mut res = [0u16; IRIS_CODE_LENGTH];
                res.iter_mut().enumerate().for_each(|(i, x)| {
                    *x = rng.gen::<u16>().wrapping_add((1 - (i % 3)) as u16);
                });
                id += 1;
                return futures::task::Poll::Ready(Some(Ok((id - 1, EncodedBits(res)))));
            }
            futures::task::Poll::Ready(None)
        }))
    }

    fn stream_masks(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl futures::Stream<Item = eyre::Result<(u64, Bits)>>> {
        let mut id = share_id_range.start;
        Ok(futures::stream::poll_fn(move |_| {
            if id >= share_id_range.end {
                return futures::task::Poll::Ready(None);
            }

            let mut res = Bits::default();
            (0..IRIS_CODE_LENGTH).for_each(|i| {
                res.set(i, (1 - (i % 3)) != 0);
            });
            id += 1;
            futures::task::Poll::Ready(Some(Ok((id - 1, res))))
        }))
    }
}

impl OldIrisShareSource for MockOldDbParty2 {
    async fn load_code_share(&self, share_id: u64) -> eyre::Result<EncodedBits> {
        let mut rng = self.rng.clone();
        rng.set_word_pos(share_id as u128 * 12800);
        let mut res = [0u16; IRIS_CODE_LENGTH];
        res.iter_mut().for_each(|x| {
            *x = 0u16.wrapping_sub(rng.gen::<u16>());
        });
        Ok(EncodedBits(res))
    }

    async fn load_mask(&self, _share_id: u64) -> eyre::Result<Bits> {
        let mut res = Bits::default();
        (0..IRIS_CODE_LENGTH).for_each(|i| {
            res.set(i, (1 - (i % 3)) != 0);
        });
        Ok(res)
    }

    fn stream_shares(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl futures::Stream<Item = eyre::Result<(u64, EncodedBits)>>> {
        let mut id = share_id_range.start;
        Ok(futures::stream::poll_fn(move |_| {
            if id < share_id_range.end {
                let mut rng = self.rng.clone();
                rng.set_word_pos(id as u128 * 12800);
                let mut res = [0u16; IRIS_CODE_LENGTH];
                res.iter_mut().for_each(|x| {
                    *x = 0u16.wrapping_sub(rng.gen::<u16>());
                });
                id += 1;
                return futures::task::Poll::Ready(Some(Ok((id - 1, EncodedBits(res)))));
            }
            futures::task::Poll::Ready(None)
        }))
    }

    fn stream_masks(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl futures::Stream<Item = eyre::Result<(u64, Bits)>>> {
        let mut id = share_id_range.start;
        Ok(futures::stream::poll_fn(move |_| {
            if id >= share_id_range.end {
                return futures::task::Poll::Ready(None);
            }

            let mut res = Bits::default();
            (0..IRIS_CODE_LENGTH).for_each(|i| {
                res.set(i, (1 - (i % 3)) != 0);
            });
            id += 1;
            futures::task::Poll::Ready(Some(Ok((id - 1, res))))
        }))
    }
}
