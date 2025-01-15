use aes_prng::AesRng;
use clap::Parser;
use hawk_pack::graph_store::GraphMem;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore,
    hnsw::{
        metrics::{
            CounterLayer, OpCounters,
            VertexOpeningsLayer, Operation
        },
        searcher::{HnswParams, HnswSearcher},
    },
};
use rand::SeedableRng;
use std::{
    collections::HashMap,
    error::Error,
    sync::{Arc, Mutex},
};
use tracing_subscriber::prelude::*;

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    #[clap(default_value = "64")]
    M:                 usize,
    #[clap(default_value = "128")]
    ef_constr:         usize,
    #[clap(default_value = "64")]
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

    let (op_counters, counter_map) = configure_tracing();

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
            print!("{}, ", idx + 1);
            print_stats(&op_counters, false);
        }
    }

    println!("Final counts:");
    print_stats(&op_counters, true);

    println!("Layer search counts:");
    for ((lc, ef), value) in counter_map.lock().unwrap().iter() {
        println!("  lc={lc},ef={ef}: {value}");
    }

    Ok(())
}

fn print_stats(counters: &Arc<OpCounters>, verbose: bool) {
    let layer_searches = counters.get(Operation::LayerSearch.id() as usize).unwrap();
    let opened_nodes = counters.get(Operation::OpenNode.id() as usize).unwrap();
    let distance_evals = counters.get(Operation::EvaluateDistance.id() as usize).unwrap();
    let distance_comps = counters.get(Operation::CompareDistance.id() as usize).unwrap();

    if verbose {
        println!("  Layer search events: {:?}", layer_searches);
        println!("  Open node events: {:?}", opened_nodes);
        println!("  Evaluate distance events: {:?}", distance_evals);
        println!("  Compare distance events: {:?}", distance_comps);
    } else {
        println!(
            "{:?}, {:?}, {:?}",
            opened_nodes, distance_evals, distance_comps
        );
    }
}

fn configure_tracing() -> (
    Arc<OpCounters>,
    Arc<Mutex<HashMap<(usize, usize), usize>>>,
) {
    let count_ops_layer = CounterLayer::new();
    let ops_counters = count_ops_layer.get_counters();

    // let compare_dist_layer = SingletonCounterLayer::new("compare_distance");
    // let compare_dist_counter = compare_dist_layer.get_counter();


    // let counters = Arc::new(EventCounter::default());

    // let counting_layer = SingletonCounterLayer {
    //     counter: counters.clone(),
    // };

    let counter_map: Arc<Mutex<HashMap<(usize, usize), usize>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let vertex_openings_layer = VertexOpeningsLayer {
        counter_map: counter_map.clone(),
    };

    tracing_subscriber::registry()
        .with(vertex_openings_layer)
        .with(count_ops_layer)
        .init();

    // tracing_subscriber::fmt()
    //    .init();

    (ops_counters, counter_map)
}
