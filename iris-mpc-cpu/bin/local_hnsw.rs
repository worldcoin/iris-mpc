use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_cpu::hawkers::aby3::test_utils::shared_random_setup_with_grpc;
use rand::SeedableRng;
use std::{error::Error, fs::File};
use tracing_forest::{tag::NoTag, ForestLayer, PrettyPrinter};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

#[derive(Parser)]
struct Args {
    #[clap(short = 'n', default_value = "1000")]
    database_size: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let database_size = args.database_size;

    let file = File::create("searcher_network_tree.txt")?;
    let file_processor = PrettyPrinter::new().writer(std::sync::Mutex::new(file));

    tracing_subscriber::registry()
        .with(
            ForestLayer::new(file_processor, NoTag {})
                .with_filter(EnvFilter::new("searcher::network")),
        )
        .init();

    println!("Starting Local HNSW with {} vectors", database_size);
    let mut rng = AesRng::seed_from_u64(0_u64);

    shared_random_setup_with_grpc(&mut rng, database_size).await?;

    Ok(())
}
