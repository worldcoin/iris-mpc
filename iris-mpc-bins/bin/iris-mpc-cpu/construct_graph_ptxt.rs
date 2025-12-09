use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use iris_mpc_common::IrisVectorId;
use tracing::info;

use iris_mpc_cpu::{
    hawkers::build_plaintext::plaintext_parallel_batch_insert,
    hnsw::{searcher::LayerDistribution, HnswSearcher},
    utils::serialization::{
        iris_ndjson::{irises_from_ndjson, IrisSelection},
        write_bin,
    },
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

    let mut searcher = HnswSearcher::new_standard(args.ef, args.ef, args.M);
    if let Some(q) = args.layer_probability {
        match &mut searcher.layer_distribution {
            LayerDistribution::Geometric { layer_probability } => *layer_probability = q,
        }
    }

    let prf_seed = (args.hnsw_prf_key as u128).to_le_bytes();

    info!("Reading iris codes from file");
    let irises = irises_from_ndjson(
        args.iris_codes_path.as_path(),
        args.graph_size,
        IrisSelection::All,
    )?;
    let irises = irises
        .into_iter()
        .enumerate()
        .map(|(idx, code)| (IrisVectorId::from_0_index(idx as u32), code))
        .collect();

    info!("Building HNSW graph over iris codes...");
    let (graph, _) =
        plaintext_parallel_batch_insert(None, None, irises, &searcher, 1, &prf_seed).await?;

    info!("Persisting HNSW graph to file");
    write_bin(&graph, args.graph_path.as_os_str().to_str().unwrap())?;

    Ok(())
}
