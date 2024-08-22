use clap::Parser;
use iris_mpc_common::galois_engine::degree4::GaloisRingIrisCodeShare;
use iris_mpc_common::iris_db::iris::IrisCodeArray;
use rand::prelude::StdRng;
use rand::SeedableRng;

const RNG_SEED: u64 = 42; // Replace with your seed value

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    iris_base_64: Option<String>,

    #[arg(short, long)]
    mask_base_64: Option<String>,

    #[arg(short, long, env)]
    rng_seed: Option<u64>,
}


fn main() {
    let args = Args::parse();
    let mut rng = if let Some(seed_rng) = args.rng_seed {
        StdRng::seed_from_u64(seed_rng)
    } else {
        StdRng::seed_from_u64(RNG_SEED)
    };

    let iris_code = if let Some(iris_base_64) = args.iris_base_64 {
        IrisCodeArray::from_base64(&iris_base_64).unwrap()
    } else {
        IrisCodeArray::random_rng(&mut rng)
    };

    let mask_code = if let Some(mask_base_64) = args.mask_base_64 {
        IrisCodeArray::from_base64(&mask_base_64).unwrap()
    } else {
        IrisCodeArray::default()
    };

    let shares = GaloisRingIrisCodeShare::encode_iris_code(&iris_code, &mask_code, &mut rng);

    println!("Generated Shares: {:?}", shares);
}