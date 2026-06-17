use aes_prng::AesRng;
use clap::{Parser, ValueEnum};
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::aby3::aby3_store::FhdOps,
    hawkers::plaintext_deep_id_store::{Int4Vector, PlaintextDeepIDStore},
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{
        metrics::ops_counter::{
            OpCountersLayer, Operation, ParamVertexOpeningsCounter, StaticCounter,
        },
        searcher::LayerDistribution,
        GraphMem, HnswSearcher, LINEAR_SCAN_MAX_GRAPH_LAYER,
    },
};
use rand::SeedableRng;
use std::{error::Error, fs::File, sync::Arc};
use tracing::Level;
use tracing_forest::{tag::NoTag, ForestLayer, PrettyPrinter};
use tracing_subscriber::{filter::Targets, prelude::*, EnvFilter};

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum StoreKindArg {
    Iris,
    DeepId,
}

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
    #[clap(long, value_enum, default_value = "iris")]
    store_kind: StoreKindArg,
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
    let mut graph = GraphMem::new();
    let mut searcher =
        HnswSearcher::new_linear_scan(ef_constr, ef_search, M, LINEAR_SCAN_MAX_GRAPH_LAYER);
    if let Some(q) = layer_probability {
        match &mut searcher.layer_distribution {
            LayerDistribution::Geometric { layer_probability } => *layer_probability = q,
        }
    }

    match args.store_kind {
        StoreKindArg::Iris => {
            let mut vector = PlaintextStore::<FhdOps>::new();
            for idx in 0..database_size {
                let raw_query = IrisCode::random_rng(&mut rng);
                let query = Arc::new(raw_query.clone());
                let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
                searcher
                    .insert(&mut vector, &mut graph, &query, insertion_layer)
                    .await?;

                if idx % 1000 == 999 {
                    println!(
                        "[{:?}] insertions: {:?}, evaluations: {:?}, comparisons: {:?}, \
                         openings: {:?}, searches: {:?}",
                        args.store_kind,
                        idx + 1,
                        dist_evaluations_counter,
                        dist_comparisons_counter,
                        node_openings_counter,
                        layer_searches_counter,
                    );
                }
            }
        }
        StoreKindArg::DeepId => {
            // Threshold is irrelevant during insertion (is_match is not invoked
            // by searcher.insert), so any value works. Pick 0 for clarity.
            let mut vector = PlaintextDeepIDStore::new(0);
            for idx in 0..database_size {
                let raw_query = Int4Vector::random(&mut rng);
                let query = Arc::new(raw_query.clone());
                let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
                searcher
                    .insert(&mut vector, &mut graph, &query, insertion_layer)
                    .await?;

                if idx % 1000 == 999 {
                    println!(
                        "[{:?}] insertions: {:?}, evaluations: {:?}, comparisons: {:?}, \
                         openings: {:?}, searches: {:?}",
                        args.store_kind,
                        idx + 1,
                        dist_evaluations_counter,
                        dist_comparisons_counter,
                        node_openings_counter,
                        layer_searches_counter,
                    );
                }
            }
        }
    }

    println!("Node opening counts, by associated layer search params:");
    let counter_map = param_openings_map.read().unwrap();
    for ((lc, ef), value) in counter_map.iter() {
        println!("  lc={lc},ef={ef}: {:?}", value);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn default_store_kind_is_iris() {
        let args = Args::parse_from(["bin"]);
        assert_eq!(args.store_kind, StoreKindArg::Iris);
    }

    #[test]
    fn store_kind_can_be_deep_id() {
        let args = Args::parse_from(["bin", "--store-kind", "deep-id"]);
        assert_eq!(args.store_kind, StoreKindArg::DeepId);
    }
}
