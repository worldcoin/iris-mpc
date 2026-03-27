// WARNING: This tool reconstructs plaintext iris codes from secret shares.
// It is intended strictly for local development and staging environments with synthetic test data.

use std::io::Write;
use std::path::PathBuf;

use clap::Parser;
use eyre::{ensure, Result};
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_store::Store;
use iris_mpc_upgrade::rerandomization::{try_reconstruct_shares, ReconstructionMismatch};

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

    /// Output file for detailed verification failures.
    #[arg(long, default_value = "verification-output.txt")]
    failures_output: PathBuf,
}

async fn connect(url: &str, schema: &str) -> Result<Store> {
    let client = PostgresClient::new(url, schema, AccessMode::ReadOnly).await?;
    Store::new(&client).await
}

fn log_mismatch(
    out: &mut impl Write,
    id: i64,
    component: &str,
    mismatch: &ReconstructionMismatch,
    v0: i16,
    v1: i16,
    v2: i16,
) -> std::io::Result<()> {
    // recon(0,1) vs recon(1,2) vs recon(0,2).
    // If two pair-reconstructions agree, the party NOT in both agreeing pairs
    // is the one with the bad share.
    let divergent_party = match (mismatch.pairs_01_vs_12, mismatch.pairs_01_vs_02) {
        // recon(0,1) != recon(1,2), but recon(0,1) == recon(0,2)
        // agreeing pairs share parties 0; party 2 is the outlier
        (true, false) => "party2 (recon(0,1)==recon(0,2), recon(1,2) differs)",
        // recon(0,1) == recon(1,2), but recon(0,1) != recon(0,2)
        // agreeing pairs share party 1; party 0 is the outlier
        (false, true) => "party0 (recon(0,1)==recon(1,2), recon(0,2) differs)",
        // all three disagree — cannot isolate a single bad party
        (true, true) => "unknown (all three pair reconstructions differ)",
        (false, false) => unreachable!(),
    };

    let msg = format!(
        "id={id} component={component} version_ids=[{v0},{v1},{v2}] suspect={divergent_party}"
    );
    tracing::error!("{}", msg);
    writeln!(out, "{}", msg)
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
    let mut failures_out =
        std::io::BufWriter::new(std::fs::File::create(&args.failures_output)?);

    let mut verified = 0u64;
    let mut failed = 0u64;
    let log_interval = (total / 100).max(1);

    for id in 1..=(total as i64) {
        let rows = tokio::try_join!(
            stores[0].get_iris_data_by_id(id),
            stores[1].get_iris_data_by_id(id),
            stores[2].get_iris_data_by_id(id),
        )?;
        let (r0, r1, r2) = rows;

        let components: [(&str, &[u16], &[u16], &[u16]); 4] = [
            ("left_code", r0.left_code(), r1.left_code(), r2.left_code()),
            ("left_mask", r0.left_mask(), r1.left_mask(), r2.left_mask()),
            (
                "right_code",
                r0.right_code(),
                r1.right_code(),
                r2.right_code(),
            ),
            (
                "right_mask",
                r0.right_mask(),
                r1.right_mask(),
                r2.right_mask(),
            ),
        ];

        let mut row_ok = true;
        let mut reconstructed: Vec<Vec<u16>> = Vec::with_capacity(4);

        for (name, s0, s1, s2) in &components {
            match try_reconstruct_shares(s0, s1, s2) {
                Ok(plain) => reconstructed.push(plain),
                Err(mismatch) => {
                    row_ok = false;
                    log_mismatch(
                        &mut failures_out,
                        id,
                        name,
                        &mismatch,
                        r0.version_id(),
                        r1.version_id(),
                        r2.version_id(),
                    )?;
                }
            }
        }

        if row_ok {
            let mut row_hasher = blake3::Hasher::new();
            for plain in &reconstructed {
                row_hasher.update(bytemuck::cast_slice::<u16, u8>(plain));
            }
            let row_hash = row_hasher.finalize();
            writeln!(out, "{}:{}", id, row_hash.to_hex())?;
            overall_hasher.update(row_hash.as_bytes());
        } else {
            failed += 1;
        }

        verified += 1;
        if verified as usize % log_interval == 0 {
            tracing::info!(
                "Progress {}/{} ({} failures so far)",
                verified,
                total,
                failed
            );
        }
    }

    out.flush()?;
    failures_out.flush()?;
    let overall_hash = overall_hasher.finalize();

    if failed > 0 {
        tracing::error!(
            "Verification completed with {} inconsistent rows out of {} (details in {})",
            failed,
            total,
            args.failures_output.display()
        );
        eyre::bail!(
            "{} rows have inconsistent shares across parties. See {}",
            failed,
            args.failures_output.display()
        );
    }

    tracing::info!("Verified all {} entries", total);
    tracing::info!("Overall hash: {}", overall_hash.to_hex());
    tracing::info!("Per-row hashes written to {}", args.output.display());

    println!("{}", overall_hash.to_hex());
    Ok(())
}
