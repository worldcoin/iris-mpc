use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{
        metrics::ops_counter::{
            OpCountersLayer, Operation, ParamVertexOpeningsCounter, StaticCounter,
        },
        searcher::LayerDistribution,
        GraphMem, HnswSearcher,
    },
};
use rand::SeedableRng;
use std::{error::Error, fs::File, sync::Arc};
use tracing::Level;
use tracing_forest::{tag::NoTag, ForestLayer, PrettyPrinter};
use tracing_subscriber::{filter::Targets, prelude::*, EnvFilter};

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    #[clap(short, default_value = "384")]
    M: usize,
    #[clap(long("efc"), default_value = "512")]
    ef_constr: usize,
    #[clap(long("efs"), default_value = "512")]
    ef_search: usize,
    #[clap(short, default_value = "10000")]
    database_size: usize,
    #[clap(short('p'))]
    layer_probability: Option<f64>,
}

#[tokio::main]
#[allow(non_snake_case)]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let M = args.M;
    let ef_constr = args.ef_constr;
    let ef_search = args.ef_search;
    let database_size = args.database_size;
    let layer_probability = args.layer_probability;

    // Configure tracing Subscriber for counters

    let dist_evaluations = StaticCounter::new();
    let dist_evaluations_counter = dist_evaluations.get_counter();

    let dist_comparisons = StaticCounter::new();
    let dist_comparisons_counter = dist_comparisons.get_counter();

    let layer_searches = StaticCounter::new();
    let layer_searches_counter = layer_searches.get_counter();

    let node_openings = StaticCounter::new();
    let node_openings_counter = node_openings.get_counter();

    // TODO support several counters and output formats using CLI options

    let param_openings = ParamVertexOpeningsCounter::new();
    let (param_openings_map, _) = param_openings.get_counters();

    let counting_layer = OpCountersLayer::builder()
        .register_static(dist_evaluations, Operation::EvaluateDistance)
        .register_static(dist_comparisons, Operation::CompareDistance)
        .register_static(layer_searches, Operation::LayerSearch)
        .register_static(node_openings, Operation::OpenNode)
        .register_dynamic(param_openings, Operation::OpenNode)
        .init();

    let filter = Targets::new()
        .with_target("iris_mpc_cpu::hnsw", Level::DEBUG)
        .with_target("iris_mpc_cpu::hawkers", Level::DEBUG);
    let file = File::create("searcher_time_tree.txt")?;
    let file_processor = PrettyPrinter::new().writer(std::sync::Mutex::new(file));

    tracing_subscriber::registry()
        .with(counting_layer)
        .with(filter)
        .with(
            ForestLayer::new(file_processor, NoTag {})
                .with_filter(EnvFilter::new("searcher::cpu_time")),
        )
        .init();

    // Run HNSW construction

    let mut rng = AesRng::seed_from_u64(42_u64);
    let mut vector = PlaintextStore::new();
    let mut graph = GraphMem::new();
    let mut searcher = HnswSearcher::new_standard(ef_constr, ef_search, M);
    if let Some(q) = layer_probability {
        match &mut searcher.layer_distribution {
            LayerDistribution::Geometric { layer_probability } => *layer_probability = q,
        }
    }

    for idx in 0..database_size {
        let raw_query = IrisCode::random_rng(&mut rng);
        let query = Arc::new(raw_query.clone());
        let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
        searcher
            .insert(&mut vector, &mut graph, &query, insertion_layer)
            .await?;

        if idx % 1000 == 999 {
            println!(
                "insertions: {:?}, evaluations: {:?}, comparisons: {:?}, openings: {:?}, \
                 searches: {:?}",
                idx + 1,
                dist_evaluations_counter,
                dist_comparisons_counter,
                node_openings_counter,
                layer_searches_counter,
            );
        }
    }

    println!("Node opening counts, by associated layer search params:");
    let counter_map = param_openings_map.read().unwrap();
    for ((lc, ef), value) in counter_map.iter() {
        println!("  lc={lc},ef={ef}: {:?}", value);
    }

    Ok(())
}
