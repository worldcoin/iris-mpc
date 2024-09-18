use clap::Parser;
use mpc_uniqueness_check::{bits::BITS, config::DbConfig, db::Db, template::Template};
use rand::Rng;

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long)]
    shares_db_urls: Vec<String>,

    #[clap(long)]
    masks_db_url: String,

    #[clap(long)]
    num_elements: u64,

    #[clap(long)]
    side: String,
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

    let shares_db_config0 = DbConfig {
        url:     format!(
            "{}/{}",
            args.shares_db_urls[0], participant_one_shares_db_name
        ),
        migrate: true,
        create:  true,
    };

    let shares_db_config1 = DbConfig {
        url:     format!(
            "{}/{}",
            args.shares_db_urls[1], participant_two_shares_db_name
        ),
        migrate: true,
        create:  true,
    };

    let masks_db_config = DbConfig {
        url:     format!(
            "{}/{}",
            args.masks_db_url.clone(),
            participant_one_masks_db_name
        ),
        migrate: true,
        create:  true,
    };

    let shares_db0 = Db::new(&shares_db_config0).await?;
    let shares_db1 = Db::new(&shares_db_config1).await?;
    let masks_db = Db::new(&masks_db_config).await?;

    let mut rng = rand::thread_rng();

    let mut masks = Vec::with_capacity(args.num_elements as usize);
    let mut shares0 = Vec::with_capacity(args.num_elements as usize);
    let mut shares1 = Vec::with_capacity(args.num_elements as usize);

    for i in 1..args.num_elements + 1 {
        let mut iris_code = rng.gen::<Template>();
        // fix the iris code mask to be valid: all chunks of 2 bits are equal, since
        // they mask the real/imaginary party of the same bit
        for j in (0..BITS).step_by(2) {
            iris_code.mask.set(j + 1, iris_code.mask.get(j))
        }
        let encoded = mpc_uniqueness_check::distance::encode(&iris_code).share(2, &mut rng);
        masks.push((i, iris_code.mask));
        shares0.push((i, encoded[0]));
        shares1.push((i, encoded[1]));
    }
    masks_db.insert_masks(&masks).await?;

    shares_db0.insert_shares(&shares0).await?;
    shares_db1.insert_shares(&shares1).await?;

    Ok(())
}
