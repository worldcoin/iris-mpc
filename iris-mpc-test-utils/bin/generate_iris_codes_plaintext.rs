use iris_mpc_test_utils::resources::write_plaintext_iris_codes;
use std::error::Error;

/// Default RNG seed to use when generating plaintext iris codes.
const RNG_SEED: u64 = 93;

/// Default number of Iris codes to generate.
const N_TO_GENERATE: usize = 1000;

/// Default graph sizes.
const GRAPH_SIZE_RANGE: [usize; 4] = [1, 10, 100, 1000];

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    write_plaintext_iris_codes(RNG_SEED, N_TO_GENERATE, GRAPH_SIZE_RANGE.to_vec()).await;

    Ok(())
}
