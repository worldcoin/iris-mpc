use clap::Parser;
use mpc_uniqueness_check::{bits::BITS, config::DbConfig, db::Db, template::Template};
use rand::Rng;

#[derive(Debug, Clone, Parser)]
struct Args {
    #[clap(long)]
    db_urls:      Vec<String>,
    #[clap(long)]
    num_elements: u64,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let args = Args::parse();

    if args.db_urls.len() != 2 {
        return Err(eyre::eyre!("Expect 2 db urls to be provided"));
    }

    let db_config0 = DbConfig {
        url:     args.db_urls[0].clone(),
        migrate: true,
        create:  true,
    };
    let db_config1 = DbConfig {
        url:     args.db_urls[1].clone(),
        migrate: true,
        create:  true,
    };
    let db0 = Db::new(&db_config0).await?;
    let db1 = Db::new(&db_config1).await?;

    let mut rng = rand::thread_rng();

    let mut masks = Vec::with_capacity(args.num_elements as usize);
    let mut shares0 = Vec::with_capacity(args.num_elements as usize);
    let mut shares1 = Vec::with_capacity(args.num_elements as usize);

    for i in 0..args.num_elements {
        let mut iris_code = rng.gen::<Template>();
        // fix the iris code mask to be valid: all chunks of 2 bits are equal, since
        // they mask the real/imaginary party of the same bit
        for i in (0..BITS).step_by(2) {
            iris_code.mask.set(i + 1, iris_code.mask.get(i))
        }
        let encoded = mpc_uniqueness_check::distance::encode(&iris_code).share(2, &mut rng);
        masks.push((i, iris_code.mask));
        shares0.push((i, encoded[0]));
        shares1.push((i, encoded[1]));
    }
    db0.insert_masks(&masks).await?;
    db1.insert_masks(&masks).await?;
    db0.insert_shares(&shares0).await?;
    db1.insert_shares(&shares1).await?;

    Ok(())
}
