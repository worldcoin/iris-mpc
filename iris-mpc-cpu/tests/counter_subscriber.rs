//! The `OpCountersLayer` subscriber for the tracing library is tested using an
//! integration test because of limitations in the way that tracing subscribers
//! are installed in a process.
//!
//! Specifically, functionality is available in tracing to install a subscriber
//! either globally, or in a per-thread fashion, meaning that a) there is no
//! support for testing a subscriber for a particular tokio Task, and b) other
//! tests running concurrently result in additional tracing events and spans
//! that interfere with clean monitoring of the subscriber functionality.
//!
//! Integration tests are run in their own executable, so are isolated from
//! parallel operation with other tests, and allows a clean execution
//! environment for testing tracing subscriber functionality.

use aes_prng::AesRng;
use eyre::Result;
use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::{
    hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef},
    hnsw::{
        graph::neighborhood::SortedNeighborhood,
        metrics::ops_counter::{
            OpCountersLayer, Operation, ParamCounterRef, ParamVertexOpeningsCounter, StaticCounter,
        },
        GraphMem, HnswSearcher,
    },
};
use rand::SeedableRng;
use std::{
    collections::HashMap,
    sync::{atomic::Ordering, Arc},
};
use tokio::{self, task::JoinSet};
use tracing_subscriber::prelude::*;

#[tokio::test]
async fn test_counter_subscriber() -> Result<()> {
    let rng = &mut AesRng::seed_from_u64(0_u64);
    let (searcher, vector_store, graph_store, query1, query2) = init_hnsw(200, rng).await?;

    // Set up tracing Subscriber for counting operations

    // counter from events
    let dist_evaluations = StaticCounter::new();
    let dist_evaluations_counter = dist_evaluations.get_counter();

    // counter from new spans
    let layer_searches = StaticCounter::new();
    let layer_searches_counter = layer_searches.get_counter();

    let param_openings = ParamVertexOpeningsCounter::new();
    let (param_openings_map, _) = param_openings.get_counters();

    let counting_layer = OpCountersLayer::builder()
        .register_static(dist_evaluations, Operation::EvaluateDistance)
        .register_static(layer_searches, Operation::LayerSearch)
        .register_dynamic(param_openings, Operation::OpenNode)
        .init();

    tracing_subscriber::registry().with(counting_layer).init();

    // Test consistency of static counters between serial and parallel operation

    let seq_static_counters = {
        let start = (
            dist_evaluations_counter.load(Ordering::Relaxed),
            layer_searches_counter.load(Ordering::Relaxed),
        );

        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();
        hnsw_search_queries_seq(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1.clone(),
            query2.clone(),
        )
        .await?;

        let end = (
            dist_evaluations_counter.load(Ordering::Relaxed),
            layer_searches_counter.load(Ordering::Relaxed),
        );

        (end.0 - start.0, end.1 - start.1)
    };

    let par_static_counters = {
        let start = (
            dist_evaluations_counter.load(Ordering::Relaxed),
            layer_searches_counter.load(Ordering::Relaxed),
        );

        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();
        hnsw_search_queries_par(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1.clone(),
            query2.clone(),
        )
        .await;

        let end = (
            dist_evaluations_counter.load(Ordering::Relaxed),
            layer_searches_counter.load(Ordering::Relaxed),
        );

        (end.0 - start.0, end.1 - start.1)
    };

    assert!(seq_static_counters.0 > 0);
    assert!(seq_static_counters.1 > 0);
    assert_eq!(seq_static_counters, par_static_counters);

    // Test consistency of dynamic counters between serial and parallel operation

    let seq_openings_map = {
        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();

        let start = clone_counter_map(&param_openings_map).await;
        hnsw_search_queries_seq(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1.clone(),
            query2.clone(),
        )
        .await?;
        let end = clone_counter_map(&param_openings_map).await;

        map_diffs(&end, &start)
    };

    let par_openings_map = {
        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();

        let start = clone_counter_map(&param_openings_map).await;
        hnsw_search_queries_par(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1.clone(),
            query2.clone(),
        )
        .await;
        let end = clone_counter_map(&param_openings_map).await;

        map_diffs(&end, &start)
    };

    for (key, par_val) in par_openings_map.iter() {
        let seq_val = seq_openings_map.get(key).unwrap_or(&0);
        assert!(*seq_val > 0);
        assert_eq!(seq_val, par_val);
    }

    Ok(())
}

async fn init_hnsw(
    db_size: usize,
    rng: &mut AesRng,
) -> Result<(
    HnswSearcher,
    PlaintextStore,
    GraphMem<PlaintextVectorRef>,
    Arc<IrisCode>,
    Arc<IrisCode>,
)> {
    let searcher = HnswSearcher::new_with_test_parameters();
    let mut vector_store = PlaintextStore::new_random(rng, db_size);
    let graph_store = vector_store.generate_graph(rng, db_size, &searcher).await?;
    let query1 = Arc::new(IrisCode::random_rng(rng));
    let query2 = Arc::new(IrisCode::random_rng(rng));
    Ok((searcher, vector_store, graph_store, query1, query2))
}

async fn hnsw_search_queries_seq(
    searcher: &HnswSearcher,
    vector_store: &mut PlaintextStore,
    graph_store: &mut GraphMem<PlaintextVectorRef>,
    query1: Arc<IrisCode>,
    query2: Arc<IrisCode>,
) -> Result<()> {
    for q in [query1, query2].into_iter() {
        searcher
            .search::<_, SortedNeighborhood<PlaintextStore>>(vector_store, graph_store, &q, 1)
            .await?;
    }

    Ok(())
}

async fn hnsw_search_queries_par(
    searcher: &HnswSearcher,
    vector_store: &mut PlaintextStore,
    graph_store: &mut GraphMem<PlaintextVectorRef>,
    query1: Arc<IrisCode>,
    query2: Arc<IrisCode>,
) {
    let mut jobs: JoinSet<Result<()>> = JoinSet::new();
    for q in [query1, query2].into_iter() {
        let searcher = searcher.clone();
        let mut vector_store = vector_store.clone();
        let graph_store = graph_store.clone();
        jobs.spawn(async move {
            searcher
                .search::<_, SortedNeighborhood<PlaintextStore>>(
                    &mut vector_store,
                    &graph_store,
                    &q,
                    1,
                )
                .await?;

            Ok(())
        });
    }
    jobs.join_all().await;
}

async fn clone_counter_map<K: Eq + std::hash::Hash + Copy>(
    async_map: &ParamCounterRef<K>,
) -> HashMap<K, usize> {
    let async_map_read = async_map.read().unwrap();
    let mut copied_map: HashMap<K, usize> = HashMap::new();
    for (key, value) in async_map_read.iter() {
        copied_map.insert(*key, value.load(Ordering::Relaxed));
    }

    copied_map
}

fn map_diffs<K: Eq + std::hash::Hash + Copy>(
    end: &HashMap<K, usize>,
    start: &HashMap<K, usize>,
) -> HashMap<K, usize> {
    let mut diffs: HashMap<K, usize> = HashMap::new();
    for (key, end_val) in end.iter() {
        let start_val = start.get(key).unwrap_or(&0);
        diffs.insert(*key, end_val - start_val);
    }
    diffs
}
