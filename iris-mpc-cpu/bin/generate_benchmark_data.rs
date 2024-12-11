use aes_prng::AesRng;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    py_bindings::{io::write_bin, plaintext_store::to_ndjson_file},
};
use rand::SeedableRng;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // create a folder ./iris-mpc-cpu/data if it is non-existent
    let crate_root = env!("CARGO_MANIFEST_DIR");
    std::fs::create_dir_all(format!("{crate_root}/data"))?;
    let mut rng = AesRng::seed_from_u64(0_u64);
    println!("Generating plaintext store with 100_000 irises");
    let mut store = PlaintextStore::create_random_store(&mut rng, 100_000).await?;
    println!("Writing store to file");
    to_ndjson_file(&store, &format!("{crate_root}/data/store.ndjson"))?;

    for graph_size in [1, 10, 100, 1000, 10_000, 100_000] {
        println!("Generating graph with {} vertices", graph_size);
        let graph = store.create_graph(&mut rng, graph_size).await?;
        write_bin(&graph, &format!("{crate_root}/data/graph_{graph_size}.dat"))?;
    }
    Ok(())
}
