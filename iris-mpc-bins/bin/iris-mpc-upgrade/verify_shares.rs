// WARNING: This tool reconstructs plaintext iris codes from secret shares.
// It is intended strictly for local development and staging environments with synthetic test data.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use eyre::{ensure, Result};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::Store;
use iris_mpc_upgrade::rerandomization::reconstruct_shares;

#[derive(Parser)]
#[command(
    name = "verify-shares",
    about = "Connect to all 3 party databases, reconstruct every iris entry from \
             all party-pair combinations, and produce per-row + overall hashes.\n\n\
             WARNING: This tool reconstructs plaintext iris codes from secret shares. \
             It is intended strictly for local development and staging environments \
             with synthetic test data."
)]
struct Args {
    #[arg(long, env = "PARTY0_DB_URL")]
    party0_db_url: String,

    #[arg(long, env = "PARTY1_DB_URL")]
    party1_db_url: String,

    #[arg(long, env = "PARTY2_DB_URL")]
    party2_db_url: String,

    /// Schema name shared by all parties. Overridden per-party by
    /// --party{0,1,2}-schema if provided.
    #[arg(long, env = "SCHEMA")]
    schema: String,

    #[arg(long, env = "PARTY0_SCHEMA")]
    party0_schema: Option<String>,

    #[arg(long, env = "PARTY1_SCHEMA")]
    party1_schema: Option<String>,

    #[arg(long, env = "PARTY2_SCHEMA")]
    party2_schema: Option<String>,

    /// Output file for the per-row hash list (one hex hash per line).
    #[arg(long, default_value = "iris_hashes.txt")]
    output: PathBuf,
}

async fn connect(url: &str, schema: &str) -> Result<Store> {
    let client = PostgresClient::new(url, schema, AccessMode::ReadOnly).await?;
    Store::new(&client).await
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    tracing::warn!("*** This tool reconstructs plaintext iris codes from secret shares.   ***");
    tracing::warn!("*** Only use with local/staging environments and synthetic test data. ***");

    tracing::info!("Connecting to party databases…");
    let s0 = args.party0_schema.as_deref().unwrap_or(&args.schema);
    let s1 = args.party1_schema.as_deref().unwrap_or(&args.schema);
    let s2 = args.party2_schema.as_deref().unwrap_or(&args.schema);

    let stores = tokio::try_join!(
        connect(&args.party0_db_url, s0),
        connect(&args.party1_db_url, s1),
        connect(&args.party2_db_url, s2),
    )?;
    let stores = [stores.0, stores.1, stores.2];

    let counts: [usize; 3] = [
        stores[0].count_irises().await?,
        stores[1].count_irises().await?,
        stores[2].count_irises().await?,
    ];
    tracing::info!(
        "Row counts: party0={}, party1={}, party2={}",
        counts[0],
        counts[1],
        counts[2]
    );
    ensure!(
        counts[0] == counts[1] && counts[1] == counts[2],
        "Row counts differ across parties: {:?}",
        counts
    );
    let total = counts[0];
    if total == 0 {
        tracing::warn!("Databases are empty, nothing to verify");
        return Ok(());
    }

    let mut overall_hasher = blake3::Hasher::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.output)?);

    let mut verified = 0u64;
    let log_interval = (total / 100).max(1);

    for id in 1..=(total as i64) {
        let rows = tokio::try_join!(
            stores[0].get_iris_data_by_id(id),
            stores[1].get_iris_data_by_id(id),
            stores[2].get_iris_data_by_id(id),
        )?;
        let (r0, r1, r2) = rows;

        let left_code = reconstruct_shares(r0.left_code(), r1.left_code(), r2.left_code());
        let left_mask = reconstruct_shares(r0.left_mask(), r1.left_mask(), r2.left_mask());
        let right_code = reconstruct_shares(r0.right_code(), r1.right_code(), r2.right_code());
        let right_mask = reconstruct_shares(r0.right_mask(), r1.right_mask(), r2.right_mask());

        let mut row_hasher = blake3::Hasher::new();
        row_hasher.update(bytemuck::cast_slice::<u16, u8>(&left_code));
        row_hasher.update(bytemuck::cast_slice::<u16, u8>(&left_mask));
        row_hasher.update(bytemuck::cast_slice::<u16, u8>(&right_code));
        row_hasher.update(bytemuck::cast_slice::<u16, u8>(&right_mask));
        let row_hash = row_hasher.finalize();

        writeln!(out, "{}:{}", id, row_hash.to_hex())?;
        overall_hasher.update(row_hash.as_bytes());

        verified += 1;
        if verified as usize % log_interval == 0 {
            tracing::info!("Verified {}/{} entries", verified, total);
        }
    }

    out.flush()?;
    let overall_hash = overall_hasher.finalize();

    tracing::info!("Verified all {} entries", total);
    tracing::info!("Overall hash: {}", overall_hash.to_hex());
    tracing::info!("Per-row hashes written to {}", args.output.display());

    println!("{}", overall_hash.to_hex());
    Ok(())
}
