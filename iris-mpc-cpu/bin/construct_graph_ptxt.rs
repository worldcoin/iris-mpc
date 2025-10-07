use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
    sync::Arc,
};

use clap::Parser;
use eyre::Result;
use tracing::info;

use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{vector_store::VectorStoreMut, GraphMem, HnswParams, HnswSearcher},
    py_bindings::{limited_iterator, plaintext_store::Base64IrisCode},
};

#[allow(non_snake_case)]
#[derive(Parser)]
struct Args {
    /// The source file for plaintext iris codes, in NDJSON file format.
    #[clap(long("source"))]
    iris_codes_path: PathBuf,

    /// The target file for the constructed HNSW graph.
    #[clap(long("target"))]
    graph_path: PathBuf,

    /// Target maximum number of iris codes from the source file to use for
    /// constructing the graph.  If this parameter is omitted, then all entries
    /// in the source iris code file will be used.
    #[clap(short('s'), long)]
    graph_size: Option<usize>,

    /// `M` parameter for HNSW insertion.
    ///
    /// Specifies the base size of graph neighborhoods for newly inserted
    /// nodes in the graph.
    #[clap(short, long("hnsw-m"), default_value = "256")]
    M: usize,

    /// `ef` parameter for HNSW insertion.
    ///
    /// Specifies the size of active search neighborhood for insertion layers
    /// during the HNSW insertion process.
    #[clap(long("hnsw-ef"), short, default_value = "320")]
    ef: usize,

    /// The probability that an inserted element is promoted to a higher layer
    /// of the HNSW graph hierarchy.
    #[clap(long("hnsw-p"), short('p'))]
    layer_probability: Option<f64>,

    /// PRF key for HNSW insertion, used to select the layer at which new
    /// elements are inserted into the hierarchical graph structure.
    #[clap(long, default_value = "0")]
    hnsw_prf_key: u64,
}

#[allow(non_snake_case)]
#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();

    // parse args
    info!("Parsing CLI arguments");
    let args = Args::parse();

    let mut params = HnswParams::new(args.ef, args.ef, args.M);
    if let Some(q) = args.layer_probability {
        params.layer_probability = q;
    }

    info!("Opening iris codes input stream");
    let file = File::open(args.iris_codes_path.as_path()).unwrap();
    let reader = BufReader::new(file);

    let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Base64IrisCode>();
    let stream = limited_iterator(stream, args.graph_size);

    info!("Building HNSW graph over iris codes...");
    let searcher = HnswSearcher { params };
    let mut counter = 0usize;

    let mut vector_store = PlaintextStore::new();
    let mut graph = GraphMem::new();
    let prf_seed = (args.hnsw_prf_key as u128).to_le_bytes();

    for json_ptxt in stream {
        let query = Arc::new((&json_ptxt.unwrap()).into());

        let inserted_id = vector_store.insert(&query).await;

        let insertion_layer = searcher.select_layer_prf(&prf_seed, &inserted_id)?;
        let (neighbors, set_ep) = searcher
            .search_to_insert(&mut vector_store, &graph, &query, insertion_layer)
            .await?;
        searcher
            .insert_from_search_results(
                &mut vector_store,
                &mut graph,
                inserted_id,
                neighbors,
                set_ep,
            )
            .await?;

        counter += 1;
        if counter % 1000 == 0 {
            info!("Inserted {} plaintext entries", counter);
        }
    }

    info!("Persisting HNSW graph to file");
    let file = File::create(args.graph_path.as_path()).unwrap();
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &graph)?;

    Ok(())
}
