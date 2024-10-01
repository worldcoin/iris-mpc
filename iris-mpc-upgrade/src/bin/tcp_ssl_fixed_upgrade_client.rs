use clap::Parser;
use eyre::ContextCompat;
use futures::{Stream, StreamExt};
use futures_concurrency::future::Join;
use iris_mpc_upgrade::{
    config::{
        UpgradeClientConfig, BATCH_SUCCESSFUL_ACK, BATCH_TIMEOUT_SECONDS,
        FINAL_BATCH_SUCCESSFUL_ACK,
    },
    db::V1Db,
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    utils::{get_shares_from_masks, get_shares_from_shares, install_tracing, V1Database},
    OldIrisShareSource,
};
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rustls::{pki_types::ServerName, ClientConfig, RootCertStore};
use rustls_pemfile::certs;
use std::{
    fs,
    io::{self, Error as IoError, ErrorKind},
    pin::Pin,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    time::timeout,
};
use tokio_rustls::{client::TlsStream, rustls, TlsConnector};
use tracing::error;

fn extract_domain(address: &str) -> Result<String, IoError> {
    // Try to split the address into domain and port parts.
    if let Some((domain, _port)) = address.rsplit_once(':') {
        Ok(domain.to_string())
    } else {
        Err(IoError::new(
            ErrorKind::InvalidInput,
            "Invalid address format",
        ))
    }
}

async fn prepare_tls_stream_for_writing(address: &str) -> eyre::Result<TlsStream<TcpStream>> {
    // The path to your custom CA certificate file (in PEM format)
    let ca_cert_path = "/usr/local/share/ca-certificates/aws_orb_prod_private_ca.crt";

    // Load the custom CA certificate
    let ca_cert = fs::read(ca_cert_path)?;
    let mut ca_reader = &ca_cert[..];

    let mut root_cert_store = RootCertStore::empty();
    let certs = certs(&mut ca_reader)
        .map(|res| res.map_err(|_| IoError::new(ErrorKind::InvalidData, "Invalid certificate")))
        .collect::<Result<Vec<_>, _>>()?;

    root_cert_store.add_parsable_certificates(certs);

    // Create a rustls ClientConfig with the custom root certificates
    let config = ClientConfig::builder()
        .with_root_certificates(root_cert_store)
        .with_no_client_auth();

    let connector = TlsConnector::from(Arc::new(config));

    let domain = extract_domain(address)?;
    println!("Resolving domain {},", domain);
    // Resolve the server name
    let server_name = ServerName::try_from(domain)
        .map_err(|_| IoError::new(io::ErrorKind::InvalidInput, "Invalid domain name"))?;

    println!("TCP connecting to {}", address);
    // Connect to the server over TCP
    let stream = TcpStream::connect(address).await?;

    // Perform the TLS handshake
    println!("TLS connecting to {}", server_name.to_str());
    let tls_stream = connector
        .connect(server_name, stream)
        .await
        .map_err(|e| IoError::new(io::ErrorKind::Other, format!("TLS error: {}", e)))?;
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

    // need this to maybe store the PG db or otherwise the borrow checker complains
    #[allow(unused_assignments)]
    let mut maybe_shares_db = None;

    #[allow(unused_assignments)]
    let mut maybe_masks_db = None;

    #[allow(clippy::type_complexity)]
    let (mut shares_stream, mut mask_stream): (
        Pin<Box<dyn Stream<Item = eyre::Result<(u64, EncodedBits)>>>>,
        Pin<Box<dyn Stream<Item = eyre::Result<(u64, Bits)>>>>,
    ) = {
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
    match timeout(Duration::from_secs(BATCH_TIMEOUT_SECONDS), server.read_u8()).await {
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
