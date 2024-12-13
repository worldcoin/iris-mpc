use aes_prng::AesRng;
use clap::Parser;
use hawk_pack::graph_store::GraphMem;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::hawkers::{
    iris_searcher::{tracing::{EventCounter, HnswEventCounterLayer, COMPARE_DIST_EVENT, EVAL_DIST_EVENT, LAYER_SEARCH_EVENT, OPEN_NODE_EVENT}, HnswParams, HnswSearcher},
    plaintext_store::PlaintextStore};
use rand::SeedableRng;
use std::{error::Error, sync::Arc};

use tracing_subscriber::prelude::*;

#[derive(Parser)]
#[allow(non_snake_case)]
struct Args {
    #[clap(default_value = "64")]
    M: usize,
    #[clap(default_value = "128")]
    ef_constr: usize,
    #[clap(default_value = "64")]
    ef_search: usize,
    #[clap(default_value = "10000")]
    database_size: usize,
}

#[tokio::main]
#[allow(non_snake_case)]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let M = args.M;
    let ef_constr = args.ef_constr;
    let ef_search = args.ef_search;
    let database_size = args.database_size;

    let counters = configure_tracing();

    let mut rng = AesRng::seed_from_u64(0_u64);
    let mut vector = PlaintextStore::default();
    let mut graph = GraphMem::new();
    let searcher = HnswSearcher {
        params: HnswParams::new(M, ef_constr, ef_search)
    };

    for idx in 0..database_size {
        let raw_query = IrisCode::random_rng(&mut rng);
        let query = vector.prepare_query(raw_query.clone());
        searcher
            .insert(&mut vector, &mut graph, &query, &mut rng)
            .await;

        if idx % 1000 == 999 {
            print!("{}, ", idx + 1);
            print_stats(&counters, false);
        }
    }

    println!("Final counts:");
    print_stats(&counters, true);

    Ok(())
}

fn print_stats(counters: &Arc<EventCounter>, verbose: bool) {
    let layer_searches = counters.counters.get(LAYER_SEARCH_EVENT as usize).unwrap();
    let opened_nodes = counters.counters.get(OPEN_NODE_EVENT as usize).unwrap();
    let distance_evals = counters.counters.get(EVAL_DIST_EVENT as usize).unwrap();
    let distance_comps = counters.counters.get(COMPARE_DIST_EVENT as usize).unwrap();

    if verbose {
        println!("  Layer search events: {:?}", layer_searches);
        println!("  Open node events: {:?}", opened_nodes);
        println!("  Evaluate distance events: {:?}", distance_evals);
        println!("  Compare distance events: {:?}", distance_comps);
    } else {
        println!("{:?}, {:?}, {:?}", opened_nodes, distance_evals, distance_comps);
    }
}

fn configure_tracing() -> Arc<EventCounter> {
    let counters = Arc::new(EventCounter::default());

    let layer = HnswEventCounterLayer {
        counters: counters.clone(),
    };
    // Set up how `tracing-subscriber` will deal with tracing data.
    tracing_subscriber::registry().with(layer).init();

    counters
}

// mod custom_layer {
//     use tracing_subscriber::Layer;

//     pub struct CustomLayer;

//     impl<S> Layer<S> for CustomLayer where S: tracing::Subscriber {
//         fn on_event(
//             &self,
//             event: &tracing::Event<'_>,
//             _ctx: tracing_subscriber::layer::Context<'_, S>,
//         ) {
//             println!("Got event!");
//             println!("  level={:?}", event.metadata().level());
//             println!("  target={:?}", event.metadata().target());
//             println!("  name={:?}", event.metadata().name());
//             let mut visitor = PrintlnVisitor;
//             event.record(&mut visitor);
//         }
//     }

//     struct PrintlnVisitor;

//     impl tracing::field::Visit for PrintlnVisitor {
//         fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
//             println!("  field={} value={}", field.name(), value)
//         }

//         fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
//             println!("  field={} value={}", field.name(), value)
//         }

//         fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
//             println!("  field={} value={}", field.name(), value)
//         }

//         fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
//             println!("  field={} value={}", field.name(), value)
//         }

//         fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
//             println!("  field={} value={}", field.name(), value)
//         }

//         fn record_error(
//             &mut self,
//             field: &tracing::field::Field,
//             value: &(dyn std::error::Error + 'static),
//         ) {
//             println!("  field={} value={}", field.name(), value)
//         }

//         fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
//             println!("  field={} value={:?}", field.name(), value)
//         }
//     }

// }
// use custom_layer::CustomLayer;