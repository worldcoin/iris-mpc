use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::iris_db::iris::IrisCodeBase64;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{HnswParams, HnswSearcher},
    utils::serialization::write_bin,
};
use rand::SeedableRng;
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

/// Test graph sizes.
const GRAPH_SIZE_RANGE: [usize; 8] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000];

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
    assert!(GRAPH_SIZE_RANGE.contains(&args.store_size));
    let output_dir = match args.output_dir {
        Some(ref path) => {
            let path = PathBuf::from(path);
            assert!(path.exists());
            path
        }
        None => {
            // Default to local data directory.
            let path = get_path_to_default_outdir();
            std::fs::create_dir_all(&path).unwrap();
            path
        }
    };

    // Write plaintext store.
    let out_file = output_dir.join("store.ndjson");
    tracing::info!(
        "Writing plaintext store of {} irises -> {:?}",
        args.store_size,
        out_file
    );
    let mut rng = AesRng::seed_from_u64(args.rng_seed);
    let mut store = PlaintextStore::new_random(&mut rng, args.store_size);
    write_plaintext_store(&store, &out_file)?;

    // Write graphs.
    let searcher = HnswSearcher::from(&args);
    for graph_size in GRAPH_SIZE_RANGE {
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
        write_bin(&graph, out_file.to_str().unwrap())?;
    }

    Ok(())
}

/// Returns path to default output directory.
fn get_path_to_default_outdir() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR").to_string()).join("data")
}

/// Writes an ndjson file of plaintext Iris codes.
fn write_plaintext_store(vector: &PlaintextStore, path_to_ndjson: &Path) -> std::io::Result<()> {
    // Set serial identifiers.
    let serial_ids: Vec<_> = vector.storage.get_sorted_serial_ids();

    // Write Iris codes only - ensures files are backwards compatible.
    let handle = File::create(path_to_ndjson)?;
    let mut writer = BufWriter::new(handle);
    for serial_id in serial_ids {
        let pt = vector
            .storage
            .get_vector_by_serial_id(serial_id)
            .expect("key not found in store");
        let json_pt: IrisCodeBase64 = (&**pt).into();
        serde_json::to_writer(&mut writer, &json_pt)?;
        writer.write_all(b"\n")?; // Write a newline after each JSON object
    }
    writer.flush()?;

    Ok(())
}
