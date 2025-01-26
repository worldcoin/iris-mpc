use aes_prng::AesRng;
use clap::Parser;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{
        graph::layered_graph::GraphMem,
        metrics::ops_counter::{
            OpCountersLayer, Operation, ParamVertexOpeningsCounter, StaticCounter,
        },
        searcher::{HnswParams, HnswSearcher},
    },
};
use rand::SeedableRng;
use std::error::Error;
use tracing_subscriber::prelude::*;

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    #[clap(default_value = "384")]
    M:                 usize,
    #[clap(default_value = "512")]
    ef_constr:         usize,
    #[clap(default_value = "512")]
    ef_search:         usize,
    #[clap(default_value = "10000")]
    database_size:     usize,
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

    let node_openings = StaticCounter::new();
    let node_openings_counter = node_openings.get_counter();

    // TODO support several counters and output formats using CLI options

    // let param_openings = ParamVertexOpeningsCounter::new();
    // let (param_openings_map, _) = param_openings.get_counters();

    let counting_layer = OpCountersLayer::builder()
        .register_static(dist_evaluations, Operation::EvaluateDistance)
        .register_static(dist_comparisons, Operation::CompareDistance)
        .register_static(node_openings, Operation::OpenNode)
        // .register_dynamic(param_openings, Operation::OpenNode)
        .init();

    tracing_subscriber::registry().with(counting_layer).init();

    // Run HNSW construction

    let mut rng = AesRng::seed_from_u64(42_u64);
    // let mut rng = rand::thread_rng();
    let mut vector = PlaintextStore::default();
    let mut graph = GraphMem::new();
    let params = if let Some(p) = layer_probability {
        HnswParams::new_with_layer_probability(ef_constr, ef_search, M, p)
    } else {
        HnswParams::new(ef_constr, ef_search, M)
    };
    let searcher = HnswSearcher { params };

    for idx in 0..database_size {
        let raw_query = IrisCode::random_rng(&mut rng);
        let query = vector.prepare_query(raw_query.clone());
        searcher
            .insert(&mut vector, &mut graph, &query, &mut rng)
            .await;

        if idx % 1000 == 999 {
            println!(
                "insertions: {:?}, evaluations: {:?}, comparisons: {:?}, openings: {:?}",
                idx + 1,
                dist_evaluations_counter,
                dist_comparisons_counter,
                node_openings_counter,
            );
        }
    }

    // println!("Layer search counts:");
    // let counter_map = param_openings_map.read().unwrap();
    // for ((lc, ef), value) in counter_map.iter() {
    //     println!("  lc={lc},ef={ef}: {:?}", value);
    // }

    Ok(())
}
