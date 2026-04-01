use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::{iris_db::iris::IrisCodeBase64, vector_id::VectorId};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{GraphMem, HnswSearcher, SortedNeighborhood},
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
        HnswSearcher::new_linear_scan(
            args.hnsw_ef_construction,
            args.hnsw_ef_search,
            args.hnsw_M,
            1,
        )
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

    // Build one graph iteratively, snapshotting at each checkpoint in GRAPH_SIZE_RANGE.
    let searcher = HnswSearcher::from(&args);
    let mut graph = GraphMem::new();
    let mut insert_rng = AesRng::from_rng(rng.clone())?;

    let serial_ids: Vec<_> = store.storage.get_sorted_serial_ids();
    let mut snapshot_iter = GRAPH_SIZE_RANGE.iter().copied().peekable();

    for (i, serial_id) in serial_ids.iter().copied().enumerate() {
        // Skip snapshots beyond current store size.
        while snapshot_iter.peek().is_some_and(|&s| s > args.store_size) {
            snapshot_iter.next();
        }
        if snapshot_iter.peek().is_none() {
            break;
        }

        let query = store
            .storage
            .get_vector_by_serial_id(serial_id)
            .unwrap()
            .clone();
        let query_id = VectorId::from_serial_id(serial_id);
        let insertion_layer = searcher.gen_layer_rng(&mut insert_rng)?;
        let (neighbors, update_ep) = searcher
            .search_to_insert::<_, SortedNeighborhood<_>>(
                &mut store,
                &graph,
                &query,
                insertion_layer,
            )
            .await?;
        searcher
            .insert_from_search_results(&mut store, &mut graph, query_id, neighbors, update_ep)
            .await?;

        let inserted_count = i + 1;
        if snapshot_iter.peek() == Some(&inserted_count) {
            let out_file = output_dir.join(format!("graph_{inserted_count}.dat"));
            tracing::info!(
                "Snapshotting graph with {inserted_count} vertices -> {:?}",
                out_file
            );
            write_bin(&graph, out_file.to_str().unwrap())?;
            snapshot_iter.next();
        }
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
