use clap::Parser;
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_common::{
    galois_engine::degree4::FullGaloisRingIrisCodeShare, iris_db::iris::IrisCode,
};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::Itertools;
use rand::thread_rng;
use std::cmp::min;

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long)]
    db_url_party1: String,

    #[clap(long)]
    db_url_party2: String,

    #[clap(long)]
    db_url_party3: String,

    #[clap(long)]
    fill_to: u64,

    #[clap(long)]
    batch_size: usize,

    #[clap(long)]
    schema_name_party1: String,

    #[clap(long)]
    schema_name_party2: String,

    #[clap(long)]
    schema_name_party3: String,

    #[clap(long, value_delimiter = ',', num_args = 1..)]
    deleted_identities: Option<Vec<i32>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let party_1_pg_client = PostgresClient::new(
        &args.db_url_party1,
        &args.schema_name_party1,
        AccessMode::ReadWrite,
    )
    .await?;
    let party_2_pg_client = PostgresClient::new(
        &args.db_url_party2,
        &args.schema_name_party1,
        AccessMode::ReadWrite,
    )
    .await?;
    let party_3_pg_client = PostgresClient::new(
        &args.db_url_party3,
        &args.schema_name_party1,
        AccessMode::ReadWrite,
    )
    .await?;

    let store1 = Store::new(&party_1_pg_client).await?;
    let store2 = Store::new(&party_2_pg_client).await?;
    let store3 = Store::new(&party_3_pg_client).await?;

    let mut rng = rand::thread_rng();

    let latest_serial_id1 = store1.count_irises().await?;
    let latest_serial_id2 = store2.count_irises().await?;
    let latest_serial_id3 = store3.count_irises().await?;
    let mut latest_serial_id =
        min(min(latest_serial_id1, latest_serial_id2), latest_serial_id3) as u64;

    if latest_serial_id == args.fill_to {
        return Ok(());
    }
    // TODO: Does this make sense?
    if latest_serial_id == 0 {
        latest_serial_id += 1
    }

    let deleted_serial_ids = args.deleted_identities.unwrap_or_default();

    for range_chunk in &(latest_serial_id..args.fill_to).chunks(args.batch_size) {
        let range_chunk = range_chunk.collect_vec();
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
    }

    Ok(())
}
