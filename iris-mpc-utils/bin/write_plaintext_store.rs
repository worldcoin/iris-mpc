use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{HnswParams, HnswSearcher},
};
use iris_mpc_utils as utils;
use rand::SeedableRng;
use std::{error::Error, path::PathBuf};

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    /// HNSW param: `M` parameter for HNSW insertion.
    ///
    /// Specifies the base size of graph neighborhoods for newly inserted nodes in the graph.
    #[clap(long("hnsw-m"), default_value = "256")]
    hnsw_M: usize,

    /// HNSW param: `ef_construction` parameter for HNSW insertion.
    ///
    /// Specifies size of active search neighborhood for insertion layers during the HNSW insertion process.
    #[clap(long("hnsw-ef-construction"), default_value = "320")]
    hnsw_ef_construction: usize,

    /// HNSW param: `ef_search` parameter for HNSW insertion.
    ///
    /// Specifies exploration factor `ef` during search.
    #[clap(long("hnsw-ef-search"), default_value = "256")]
    hnsw_ef_search: usize,

    /// Path to output directory.  Default=crate data sub-directory.
    #[clap(short, long("output-dir"))]
    output_dir: Option<String>,

    /// Random number generator seed.
    #[clap(long("seed"), default_value = "0")]
    rng_seed: u64,

    /// Size of plaintext Iris code store to generate.
    #[clap(long("size"), default_value = "1000")]
    store_size: usize,
}

// Convertor: Args -> HnswSearcher.
impl From<&Args> for HnswSearcher {
    fn from(args: &Args) -> Self {
        HnswSearcher {
            params: HnswParams::new(args.hnsw_ef_construction, args.hnsw_ef_search, args.hnsw_M),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();
    tracing::info!("Initialized tracing subscriber");

    // Parse args.
    let args = Args::parse();
    assert!(utils::constants::GRAPH_SIZE_RANGE.contains(&args.store_size));
    let output_dir = match args.output_dir {
        Some(ref path) => {
            let path = PathBuf::from(path);
            assert!(path.exists());
            path
        }
        None => {
            // Default to local data directory.
            let path = utils::fsys::get_path_to_subdir("data/iris-codes-plaintext");
            std::fs::create_dir_all(&path).unwrap();
            path
        }
    };

    // Write plaintext store.
    tracing::info!(
        "Writing plaintext store of  {} irises -> {:?}/store.ndjson",
        args.store_size,
        output_dir
    );
    let mut rng = AesRng::seed_from_u64(args.rng_seed);
    let mut store = PlaintextStore::new_random(&mut rng, args.store_size);
    let out_file = output_dir.join("store.ndjson");
    utils::resources::write_iris_codes(&store, &out_file)?;

    // Write graphs.
    let searcher = HnswSearcher::from(&args);
    for graph_size in utils::constants::GRAPH_SIZE_RANGE {
        if graph_size > args.store_size {
            break;
        }
        tracing::info!(
            "Writing graph with {graph_size} vertices -> {:?}/graph_{graph_size}.dat",
            output_dir
        );
        let graph = store
            .generate_graph(&mut rng, graph_size, &searcher)
            .await?;
        let out_file = output_dir.join(format!("graph_{graph_size}.dat"));
        utils::fsys::write_bin(&graph, &out_file)?;
    }

    Ok(())
}
