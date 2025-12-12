use clap::Parser;
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::{
    galois_engine::degree4::FullGaloisRingIrisCodeShare, iris_db::iris::IrisCode,
};
use iris_mpc_store::{Store, StoredIrisRef};
use iris_mpc_upgrade::utils::install_tracing;
use itertools::Itertools;
use rand::thread_rng;
use std::cmp::min;
use tracing::{info, warn};

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long, env = "DB_URL_PARTY_0")]
    db_url_party_0: String,

    #[clap(long, env = "DB_URL_PARTY_1")]
    db_url_party_1: String,

    #[clap(long, env = "DB_URL_PARTY_2")]
    db_url_party_2: String,

    #[clap(long, env = "FILL_TO")]
    fill_to: u64,

    #[clap(long, env = "BATCH_SIZE")]
    batch_size: usize,

    #[clap(long, env = "SCHEMA_NAME_PARTY_0")]
    schema_name_party_0: String,

    #[clap(long, env = "SCHEMA_NAME_PARTY_1")]
    schema_name_party_1: String,

    #[clap(long, env = "SCHEMA_NAME_PARTY_2")]
    schema_name_party_2: String,

    // Accept an empty string and parse later into an empty list
    #[clap(long, env = "DELETED_IDENTITIES")]
    deleted_identities: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    install_tracing();
    let args = Args::parse();

    info!(
        fill_to = args.fill_to,
        batch_size = args.batch_size,
        schema0 = %args.schema_name_party_0,
        schema1 = %args.schema_name_party_1,
        schema2 = %args.schema_name_party_2,
        "Starting seed_v2_dbs"
    );

    let party_1_pg_client = PostgresClient::new(
        &args.db_url_party_0,
        &args.schema_name_party_0,
        AccessMode::ReadWrite,
    )
    .await?;
    let party_2_pg_client = PostgresClient::new(
        &args.db_url_party_1,
        &args.schema_name_party_1,
        AccessMode::ReadWrite,
    )
    .await?;
    let party_3_pg_client = PostgresClient::new(
        &args.db_url_party_2,
        &args.schema_name_party_2,
        AccessMode::ReadWrite,
    )
    .await?;

    let store1 = Store::new(&party_1_pg_client).await?;
    let store2 = Store::new(&party_2_pg_client).await?;
    let store3 = Store::new(&party_3_pg_client).await?;
    info!("Initialized stores for all three parties");

    let mut rng = rand::thread_rng();

    let latest_serial_id1 = store1.count_irises().await?;
    let latest_serial_id2 = store2.count_irises().await?;
    let latest_serial_id3 = store3.count_irises().await?;
    let mut latest_serial_id =
        min(min(latest_serial_id1, latest_serial_id2), latest_serial_id3) as u64;

    info!(
        party1 = latest_serial_id1,
        party2 = latest_serial_id2,
        party3 = latest_serial_id3,
        start_from = latest_serial_id,
        "Computed starting serial id"
    );

    if latest_serial_id == args.fill_to {
        warn!("Nothing to do: already at target fill_to");
        return Ok(());
    }
    // TODO: Does this make sense?
    if latest_serial_id == 0 {
        latest_serial_id += 1
    }

    let deleted_serial_ids: Vec<i32> = args
        .deleted_identities
        .as_deref()
        .map(parse_deleted_identities)
        .unwrap_or_default();
    info!(
        deleted_count = deleted_serial_ids.len(),
        "Parsed deleted identities"
    );

    for range_chunk in &(latest_serial_id..args.fill_to).chunks(args.batch_size) {
        let range_chunk = range_chunk.collect_vec();
        if let (Some(begin), Some(end)) = (range_chunk.first(), range_chunk.last()) {
            info!(
                start = *begin,
                end = *end + 1,
                size = range_chunk.len(),
                "Processing batch range"
            );
        }
        let (party1, party2, party3): (Vec<_>, Vec<_>, Vec<_>) = range_chunk
            .iter()
            .map(|serial_id| {
                let (iris_code_left, iris_code_right) =
                    if deleted_serial_ids.contains(&(*serial_id as i32)) {
                        (
                            // TODO: set them to the deleted values
                            IrisCode::random_rng(&mut thread_rng()),
                            IrisCode::random_rng(&mut thread_rng()),
                        )
                    } else {
                        (
                            IrisCode::random_rng(&mut rng),
                            IrisCode::random_rng(&mut rng),
                        )
                    };
                let [left1, left2, left3] =
                    FullGaloisRingIrisCodeShare::encode_iris_code(&iris_code_left, &mut rng);
                let [right1, right2, right3] =
                    FullGaloisRingIrisCodeShare::encode_iris_code(&iris_code_right, &mut rng);
                ((left1, right1), (left2, right2), (left3, right3))
            })
            .multiunzip();
        let party1_insert = party1
            .iter()
            .zip(range_chunk.iter())
            .map(|((left, right), id)| StoredIrisRef {
                id: *id as i64,
                left_code: &left.code.coefs,
                left_mask: &left.mask.coefs,
                right_code: &right.code.coefs,
                right_mask: &right.mask.coefs,
            })
            .collect_vec();

        let mut tx = store1.tx().await?;
        store1
            .insert_irises_overriding(&mut tx, &party1_insert)
            .await?;
        tx.commit().await?;
        info!(count = party1_insert.len(), "Committed batch to party 1");

        let party2_insert = party2
            .iter()
            .zip(range_chunk.iter())
            .map(|((left, right), id)| StoredIrisRef {
                id: *id as i64,
                left_code: &left.code.coefs,
                left_mask: &left.mask.coefs,
                right_code: &right.code.coefs,
                right_mask: &right.mask.coefs,
            })
            .collect_vec();
        let mut tx = store2.tx().await?;
        store2
            .insert_irises_overriding(&mut tx, &party2_insert)
            .await?;
        tx.commit().await?;
        info!(count = party2_insert.len(), "Committed batch to party 2");

        let party3_insert = party3
            .iter()
            .zip(range_chunk.iter())
            .map(|((left, right), id)| StoredIrisRef {
                id: *id as i64,
                left_code: &left.code.coefs,
                left_mask: &left.mask.coefs,
                right_code: &right.code.coefs,
                right_mask: &right.mask.coefs,
            })
            .collect_vec();
        let mut tx = store3.tx().await?;
        store3
            .insert_irises_overriding(&mut tx, &party3_insert)
            .await?;
        tx.commit().await?;
        info!(count = party3_insert.len(), "Committed batch to party 3");
    }

    info!(target = args.fill_to, "Seeding complete");
    Ok(())
}

fn parse_deleted_identities(input: &str) -> Vec<i32> {
    if input.trim().is_empty() {
        return Vec::new();
    }
    input
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                t.parse::<i32>().ok()
            }
        })
        .collect()
}
