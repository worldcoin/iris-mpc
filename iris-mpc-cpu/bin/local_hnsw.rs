use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_cpu::hawkers::galois_store::LocalNetAby3NgStoreProtocol;
use rand::SeedableRng;
use std::error::Error;

#[derive(Parser)]
struct Args {
    #[clap(short = 'n', default_value = "1000")]
    database_size: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let database_size = args.database_size;

    println!("Starting Local HNSW with {} vectors", database_size);
    let mut rng = AesRng::seed_from_u64(0_u64);

    LocalNetAby3NgStoreProtocol::shared_random_setup_with_grpc(&mut rng, database_size).await?;

    Ok(())
}
