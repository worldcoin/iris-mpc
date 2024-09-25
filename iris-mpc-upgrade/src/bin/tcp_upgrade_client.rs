use clap::Parser;
use eyre::{Context, ContextCompat};
use futures::{Stream, StreamExt};
use futures_concurrency::future::Join;
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    IRIS_CODE_LENGTH,
};
use iris_mpc_upgrade::{
    config::{UpgradeClientConfig, BATCH_SUCCESSFUL_ACK, FINAL_BATCH_SUCCESSFUL_ACK},
    db::V1Db,
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    OldIrisShareSource,
};
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rustls::{pki_types::ServerName, ClientConfig};
use std::{array, convert::TryFrom, pin::Pin, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::timeout,
};
use tokio_native_tls::{TlsConnector, TlsStream};
use tracing::error;

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

async fn prepare_tls_stream_for_writing(address: &str) -> eyre::Result<TlsStream<TcpStream>> {
    // Create a TCP connection
    let stream = TcpStream::connect(address).await?;

    // Create a TLS connector using tokio_native_tls
    let native_tls_connector = tokio_native_tls::native_tls::TlsConnector::new()?;
    let tls_connector = TlsConnector::from(native_tls_connector);

    // Perform the TLS handshake to establish a secure connection
    let tls_stream: TlsStream<TcpStream> = tls_connector.connect(address, stream).await?;

    println!("TLS connection established to {}", address);

    Ok(tls_stream)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    install_tracing();
    let args = UpgradeClientConfig::parse();

    if args.party_id > 1 {
        panic!("Party id must be 0, 1");
    }

    let mut server1 = prepare_tls_stream_for_writing(&args.server1).await?;
    let mut server2 = prepare_tls_stream_for_writing(&args.server2).await?;
    let mut server3 = prepare_tls_stream_for_writing(&args.server3).await?;

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
        let shares_db_name = format!("participant{}_{}", args.party_id + 1, args.eye);
        maybe_shares_db = Some(V1Database {
            db: V1Db::new(format!("{}/{}", args.shares_db_url, shares_db_name).as_str()).await?,
        });

        let masks_db_name = format!("coordinator_{}", args.eye);
        maybe_masks_db = Some(V1Database {
            db: V1Db::new(format!("{}/{}", args.masks_db_url, masks_db_name).as_str()).await?,
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
    tracing::info!("Processing {} iris codes", num_iris_codes);

    let batch_size = args.batch_size;
    let mut batch = Vec::with_capacity(batch_size as usize);

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

        // Prepare the shares and masks for this item
        let [mask_share_a, mask_share_b, mask_share_c] =
            get_shares_from_masks(args.party_id, share_id, &mask, &mut rng);
        let [iris_share_a, iris_share_b, iris_share_c] =
            get_shares_from_shares(args.party_id, share_id, &share, &mut rng);

        // Add to batch
        batch.push((
            iris_share_a,
            iris_share_b,
            iris_share_c,
            mask_share_a,
            mask_share_b,
            mask_share_c,
        ));

        // If the batch is full, send it and wait for the ACK
        if batch.len() == batch_size as usize {
            tracing::info!("Sending batch of size {}", batch_size);
            send_batch_and_wait_for_ack(
                args.party_id,
                &mut server1,
                &mut server2,
                &mut server3,
                &batch,
            )
            .await?;
            batch.clear(); // Clear the batch once ACK is received
        }
    }
    // Send the remaining elements in the last batch
    println!("Batch size: {}", batch.len());
    if !batch.is_empty() {
        tracing::info!("Sending final batch of size {}", batch.len());
        send_batch_and_wait_for_ack(
            args.party_id,
            &mut server1,
            &mut server2,
            &mut server3,
            &batch,
        )
        .await?;
        batch.clear();
    }
    tracing::info!("Final batch sent, waiting for acks");
    wait_for_ack(&mut server1).await?;
    tracing::info!("Server 1 ack received");
    wait_for_ack(&mut server2).await?;
    tracing::info!("Server 2 ack received");
    wait_for_ack(&mut server3).await?;
    tracing::info!("Server 3 ack received");
    Ok(())
}

async fn send_batch_and_wait_for_ack(
    party_id: u8,
    server1: &mut TlsStream<TcpStream>,
    server2: &mut TlsStream<TcpStream>,
    server3: &mut TlsStream<TcpStream>,
    batch: &Vec<(
        TwoToThreeIrisCodeMessage,
        TwoToThreeIrisCodeMessage,
        TwoToThreeIrisCodeMessage,
        MaskShareMessage,
        MaskShareMessage,
        MaskShareMessage,
    )>,
) -> eyre::Result<()> {
    let mut errors = Vec::new();
    let batch_size = batch.len();
    // Send the batch size to all servers
    let (batch_size_result_a, batch_size_result_b, batch_size_result_c) = (
        server1.write_u8(batch_size as u8),
        server2.write_u8(batch_size as u8),
        server3.write_u8(batch_size as u8),
    )
        .join()
        .await;

    if let Err(e) = batch_size_result_a {
        error!("Failed to send batch size to server1: {:?}", e);
        errors.push(e.to_string());
    }
    if let Err(e) = batch_size_result_b {
        error!("Failed to send batch size to server2: {:?}", e);
        errors.push(e.to_string());
    }
    if let Err(e) = batch_size_result_c {
        error!("Failed to send batch size to server3: {:?}", e);
        errors.push(e.to_string());
    }

    // Send the batch to all servers
    for (iris1, iris2, iris3, mask1, mask2, mask3) in batch {
        let (result_iris_a, result_iris_b, result_iris_c) = (
            iris1.send(server1),
            iris2.send(server2),
            iris3.send(server3),
        )
            .join()
            .await;

        // Handle sending errors
        if let Err(e) = result_iris_a {
            error!("Failed to send message to server1: {:?}", e);
            errors.push(e.to_string());
        }
        if let Err(e) = result_iris_b {
            error!("Failed to send message to server2: {:?}", e);
            errors.push(e.to_string());
        }
        if let Err(e) = result_iris_c {
            error!("Failed to send message to server3: {:?}", e);
            errors.push(e.to_string());
        }

        // Send mask shares (only by party_id 0)
        if party_id == 0 {
            let (result_mask_a, result_mask_b, result_mask_c) = (
                mask1.send(server1),
                mask2.send(server2),
                mask3.send(server3),
            )
                .join()
                .await;
            if let Err(e) = result_mask_a {
                error!("Failed to send mask to server1: {:?}", e);
                errors.push(e.to_string());
            }
            if let Err(e) = result_mask_b {
                error!("Failed to send mask to server2: {:?}", e);
                errors.push(e.to_string());
            }
            if let Err(e) = result_mask_c {
                error!("Failed to send mask to server3: {:?}", e);
                errors.push(e.to_string());
            }
        }
    }

    if !errors.is_empty() {
        let combined_error = errors.join(" || ");
        return Err(eyre::eyre!(combined_error));
    }

    // Handle acknowledgment from all servers
    wait_for_ack(server1).await?;
    wait_for_ack(server2).await?;
    wait_for_ack(server3).await?;
    Ok(())
}

async fn wait_for_ack(server: &mut TlsStream<TcpStream>) -> eyre::Result<()> {
    match timeout(Duration::from_secs(10), server.read_u8()).await {
        Ok(Ok(BATCH_SUCCESSFUL_ACK)) => {
            // Ack received successfully
            tracing::info!("ACK received for batch");
            Ok(())
        }
        Ok(Ok(FINAL_BATCH_SUCCESSFUL_ACK)) => {
            tracing::info!("ACK received for final batch");
            Ok(())
        }
        Ok(Ok(_)) => {
            error!("Received invalid ACK");
            Err(eyre::eyre!("Invalid ACK received"))
        }
        Ok(Err(e)) => {
            error!("Error reading ACK: {:?}", e);
            Err(e.into())
        }
        Err(_) => {
            error!("ACK timeout");
            Err(eyre::eyre!("ACK timeout"))
        }
    }
}

struct V1Database {
    db: V1Db,
}

fn get_shares_from_shares(
    party_id: u8,
    serial_id: u64,
    share: &EncodedBits,
    rng: &mut ChaCha20Rng,
) -> [TwoToThreeIrisCodeMessage; 3] {
    let [a, b, c] = {
        let galois_shared_iris_code =
            GaloisRingIrisCodeShare::reencode_extended_iris_code(&share.0, rng);
        [
            galois_shared_iris_code[0].coefs,
            galois_shared_iris_code[1].coefs,
            galois_shared_iris_code[2].coefs,
        ]
    };

    let message_a = TwoToThreeIrisCodeMessage {
        id:       serial_id,
        party_id: 0,
        from:     party_id,
        data:     a,
    };
    let message_b = TwoToThreeIrisCodeMessage {
        id:       serial_id,
        party_id: 1,
        from:     party_id,
        data:     b,
    };
    let message_c = TwoToThreeIrisCodeMessage {
        id:       serial_id,
        party_id: 2,
        from:     party_id,
        data:     c,
    };
    [message_a, message_b, message_c]
}

fn get_shares_from_masks(
    party_id: u8,
    serial_id: u64,
    mask: &Bits,
    rng: &mut ChaCha20Rng,
) -> [MaskShareMessage; 3] {
    let extended_masks = array::from_fn(|i| mask[i] as u16);
    let [ma, mb, mc] = {
        let [a, b, c] = GaloisRingIrisCodeShare::reencode_extended_iris_code(&extended_masks, rng);
        [
            GaloisRingTrimmedMaskCodeShare::from(a).coefs,
            GaloisRingTrimmedMaskCodeShare::from(b).coefs,
            GaloisRingTrimmedMaskCodeShare::from(c).coefs,
        ]
    };

    let masks_a = MaskShareMessage {
        id:       serial_id,
        party_id: 0,
        from:     party_id,
        data:     ma,
    };
    let masks_b = MaskShareMessage {
        id:       serial_id,
        party_id: 1,
        from:     party_id,
        data:     mb,
    };
    let masks_c = MaskShareMessage {
        id:       serial_id,
        party_id: 2,
        from:     party_id,
        data:     mc,
    };
    [masks_a, masks_b, masks_c]
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
    ) -> eyre::Result<impl Stream<Item = eyre::Result<(u64, Bits)>>> {
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
