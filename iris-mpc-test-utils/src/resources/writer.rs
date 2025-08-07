use crate::utils::fsys;
use aes_prng::AesRng;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{HnswParams, HnswSearcher},
    py_bindings::{io::write_bin, plaintext_store::to_ndjson_file},
};
use rand::SeedableRng;

/// Writes to data directory an ndjson file plus associated data files.
///
/// # Arguments
///
/// * `rng_seed` - RNG seed used when generating Iris codes.
/// * `n_to_generate` - Number of Iris codes to generate.
/// * `graph_size_range` - Range of graph sizes to generate.
/// * `outdir` - Optional output directory.
///
/// # Returns
///
/// An iterator over Iris code pairs.
///
pub async fn write_plaintext_iris_codes(
    rng_seed: u64,
    n_to_generate: usize,
    graph_size_range: Vec<usize>,
    outdir: Option<&str>,
) {
    // Set output directory.
    let outdir = format!(
        "{}/iris-shares-plaintext",
        outdir.unwrap_or(fsys::get_data_root().as_str())
    );
    std::fs::create_dir_all(&outdir).unwrap();

    // Set RNG from seed.
    let mut rng = AesRng::seed_from_u64(rng_seed);

    // Write plaintext store.
    let resource_path = format!("{}/store.ndjson", outdir);
    let mut store = PlaintextStore::new_random(&mut rng, n_to_generate);
    println!("HNSW :: Writing plaintext store: {}", resource_path);
    to_ndjson_file(&store, resource_path.as_str()).unwrap();

    // Write graphs.
    let searcher = HnswSearcher {
        params: HnswParams::new(320, 256, 256),
    };
    for graph_size in graph_size_range {
        let resource_path = format!("{}/graph_{graph_size}.dat", outdir);
        println!(
            "HNSW :: Generating graph: vertices={} :: output={}",
            graph_size, resource_path
        );
        write_bin(
            &store
                .generate_graph(&mut rng, graph_size, &searcher)
                .await
                .unwrap(),
            resource_path.as_str(),
        )
        .unwrap();
    }
}
