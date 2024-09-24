use clap::Parser;
use eyre::Result;
use mpc_uniqueness_check::{
    bits::{Bits, BITS},
    config::DbConfig,
    db::Db,
    distance::EncodedBits,
    template::Template,
};
use rand::Rng;
use sqlx::{postgres::PgPoolOptions, Executor, Postgres, QueryBuilder};
use std::cmp::min;

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long)]
    shares_db_urls: Vec<String>,

    #[clap(long)]
    masks_db_url: String,

    #[clap(long)]
    fill_to: u64,

    #[clap(long)]
    create: bool,

    #[clap(long)]
    migrate: bool,

    #[clap(long)]
    side: String,
}

const SEED_BATCH_SIZE: usize = 200;

async fn insert_masks(pool: sqlx::Pool<Postgres>, masks: &[(u64, Bits)]) -> Result<()> {
    let mut builder = QueryBuilder::new("INSERT INTO masks (id, mask) VALUES ");

    for (idx, (id, mask)) in masks.iter().enumerate() {
        if idx > 0 {
            builder.push(", ");
        }
        builder.push("(");
        builder.push_bind(*id as i64);
        builder.push(", ");
        builder.push_bind(mask);
        builder.push(")");
    }

    builder.push(" ON CONFLICT (id) DO UPDATE SET mask = EXCLUDED.mask");

    let query = builder.build();

    query.execute(&pool).await?;

    Ok(())
}

pub async fn insert_shares(
    pool: sqlx::Pool<Postgres>,
    shares: &[(u64, EncodedBits)],
) -> Result<()> {
    let mut builder = QueryBuilder::new("INSERT INTO shares (id, share) VALUES ");

    for (idx, (id, share)) in shares.iter().enumerate() {
        if idx > 0 {
            builder.push(", ");
        }
        builder.push("(");
        builder.push_bind(*id as i64);
        builder.push(", ");
        builder.push_bind(share);
        builder.push(")");
    }

    builder.push(" ON CONFLICT (id) DO UPDATE SET share = EXCLUDED.share");

    let query = builder.build();

    query.execute(&pool).await?;
    Ok(())
}

pub async fn create_pool(url: &str) -> Result<sqlx::Pool<Postgres>> {
    let pool = PgPoolOptions::new()
        .max_connections(30) // Increase the number of connection
        .after_connect(|conn, _meta| {
            Box::pin(async move {
                conn.execute("SET idle_in_transaction_session_timeout = '60s'")
                    .await?;
                conn.execute("SET statement_timeout = '60s'").await?;
                Ok(())
            })
        }) // Increase the number of connections
        .connect(url)
        .await?;

    Ok(pool)
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = Args::parse();

    if args.shares_db_urls.len() != 2 {
        return Err(eyre::eyre!("Expect 2 shares db urls to be provided"));
    }

    if args.masks_db_url.is_empty() {
        return Err(eyre::eyre!("Expect 1 masks db urls to be provided"));
    }

    if args.side != "left" && args.side != "right" {
        return Err(eyre::eyre!("Expect side to be either 'left' or 'right'"));
    }

    let participant_one_shares_db_name = format!("participant1_{}", args.side);
    let participant_two_shares_db_name = format!("participant2_{}", args.side);
    let participant_one_masks_db_name = format!("coordinator_{}", args.side);

    let participant_one_shares_db_url = format!(
        "{}/{}",
        args.shares_db_urls[0], participant_one_shares_db_name
    );
    let participant_two_shares_db_url = format!(
        "{}/{}",
        args.shares_db_urls[1], participant_two_shares_db_name
    );
    let participant_one_masks_db_url = format!(
        "{}/{}",
        args.masks_db_url.clone(),
        participant_one_masks_db_name
    );

    let shares_db_config0 = DbConfig {
        url:     participant_one_shares_db_url.clone(),
        migrate: args.migrate,
        create:  args.create,
    };
    let shares_db_config1 = DbConfig {
        url:     participant_two_shares_db_url.clone(),
        migrate: args.migrate,
        create:  args.create,
    };
    let masks_db_config = DbConfig {
        url:     participant_one_masks_db_url.clone(),
        migrate: args.migrate,
        create:  args.create,
    };

    let shares_db0 = Db::new(&shares_db_config0).await?;
    let mut latest_shares_id_0 = shares_db0.fetch_latest_share_id().await?;
    let shares_db1 = Db::new(&shares_db_config1).await?;
    let mut latest_shares_id_1 = shares_db1.fetch_latest_share_id().await?;
    let masks_db = Db::new(&masks_db_config).await?;
    let latest_masks_id = masks_db.fetch_latest_mask_id().await?;

    let mut rng = rand::thread_rng();

    let mut masks = Vec::with_capacity(args.fill_to as usize);
    let mut shares0 = Vec::with_capacity(args.fill_to as usize);
    let mut shares1 = Vec::with_capacity(args.fill_to as usize);

    if latest_shares_id_0 == 0 {
        latest_shares_id_0 += 1;
        latest_shares_id_1 += 1;
    }
    let latest_serial_id = min(latest_masks_id, min(latest_shares_id_0, latest_shares_id_1));

    let shares_1_pool = create_pool(&participant_one_shares_db_url.clone()).await?;
    let shares_2_pool = create_pool(&participant_two_shares_db_url.clone()).await?;
    let masks_pool = create_pool(&participant_one_masks_db_url.clone()).await?;

    for i in latest_serial_id..args.fill_to {
        let mut iris_code = rng.gen::<Template>();
        // fix the iris code mask to be valid: all chunks of 2 bits are equal, since
        // they mask the real/imaginary party of the same bit
        for j in (0..BITS).step_by(2) {
            iris_code.mask.set(j + 1, iris_code.mask.get(j))
        }
        let encoded = mpc_uniqueness_check::distance::encode(&iris_code).share(2, &mut rng);
        shares0.push((i, encoded[0]));
        shares1.push((i, encoded[1]));
        masks.push((i, iris_code.mask));

        if shares0.len() == SEED_BATCH_SIZE {
            println!("Inserting {} shares", shares0.len());
            insert_shares(shares_1_pool.clone(), &shares0).await?;
            shares0.clear();
        }
        if shares1.len() == SEED_BATCH_SIZE {
            println!("Inserting {} shares", shares1.len());
            insert_shares(shares_2_pool.clone(), &shares1).await?;
            shares1.clear();
        }
        if masks.len() == SEED_BATCH_SIZE {
            println!("Inserting {} masks", masks.len());
            insert_masks(masks_pool.clone(), &masks).await?;
            masks.clear();
        }
    }

    if !shares0.is_empty() {
        shares_db0.insert_shares(&shares0).await?;
    }
    if !shares1.is_empty() {
        shares_db1.insert_shares(&shares1).await?;
    }
    if !masks.is_empty() {
        masks_db.insert_masks(&masks).await?;
    }

    Ok(())
}
