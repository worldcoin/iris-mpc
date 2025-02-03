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
    hawkers::plaintext_store::{PlaintextStore, PointId},
    hnsw::{
        metrics::ops_counter::{
            OpCountersLayer, Operation, ParamCounterRef, ParamVertexOpeningsCounter, StaticCounter,
        },
        GraphMem, HnswParams, HnswSearcher,
    },
};
use rand::SeedableRng;
use std::{collections::HashMap, sync::atomic::Ordering};
use tokio::{self, task::JoinSet};
use tracing_subscriber::prelude::*;

#[tokio::test]
async fn test_counter_subscriber() -> Result<()> {
    let rng = &mut AesRng::seed_from_u64(0_u64);
    let (searcher, vector_store, graph_store, query1, query2) = init_hnsw(200, rng).await;

    // Set up tracing Subscriber for counting operations

    let dist_evaluations = StaticCounter::new();
    let dist_evaluations_counter = dist_evaluations.get_counter();

    let param_openings = ParamVertexOpeningsCounter::new();
    let (param_openings_map, _) = param_openings.get_counters();

    let counting_layer = OpCountersLayer::builder()
        .register_static(dist_evaluations, Operation::EvaluateDistance)
        .register_dynamic(param_openings, Operation::OpenNode)
        .init();

    tracing_subscriber::registry().with(counting_layer).init();

    // Test consistency of static counters between serial and parallel operation

    let seq_dist_evaluations = {
        let start = dist_evaluations_counter.load(Ordering::Relaxed);

        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();
        hnsw_search_queries_seq(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1,
            query2,
        )
        .await;

        let end = dist_evaluations_counter.load(Ordering::Relaxed);
        end - start
    };

    let par_dist_evaluations = {
        let start = dist_evaluations_counter.load(Ordering::Relaxed);

        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();
        hnsw_search_queries_par(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1,
            query2,
        )
        .await;

        let end = dist_evaluations_counter.load(Ordering::Relaxed);
        end - start
    };

    assert!(seq_dist_evaluations > 0);
    assert_eq!(seq_dist_evaluations, par_dist_evaluations);

    // Test consistency of dynamic counters between serial and parallel operation

    let seq_openings_map = {
        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();

        let start = clone_counter_map(&param_openings_map).await;
        hnsw_search_queries_seq(
            &searcher,
            &mut vector_store,
            &mut graph_store,
            query1,
            query2,
        )
        .await;
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
            query1,
            query2,
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
) -> (
    HnswSearcher,
    PlaintextStore,
    GraphMem<PlaintextStore>,
    PointId,
    PointId,
) {
    let searcher = HnswSearcher {
        params: HnswParams::new(64, 64, 32),
    };
    let (mut vector_store, graph_store) = PlaintextStore::create_random(rng, db_size, &searcher)
        .await
        .unwrap();
    let queries: Vec<_> = (0..=1)
        .map(|_| vector_store.prepare_query(IrisCode::random_rng(rng)))
        .collect();
    (searcher, vector_store, graph_store, queries[0], queries[1])
}

async fn hnsw_search_queries_seq(
    searcher: &HnswSearcher,
    vector_store: &mut PlaintextStore,
    graph_store: &mut GraphMem<PlaintextStore>,
    query1: PointId,
    query2: PointId,
) {
    for q in [query1, query2].into_iter() {
        searcher.search(vector_store, graph_store, &q, 1).await;
    }
}

async fn hnsw_search_queries_par(
    searcher: &HnswSearcher,
    vector_store: &mut PlaintextStore,
    graph_store: &mut GraphMem<PlaintextStore>,
    query1: PointId,
    query2: PointId,
) {
    let mut jobs = JoinSet::new();
    for q in [query1, query2].into_iter() {
        let searcher = searcher.clone();
        let mut vector_store = vector_store.clone();
        let mut graph_store = graph_store.clone();
        jobs.spawn(async move {
            searcher
                .search(&mut vector_store, &mut graph_store, &q, 1)
                .await;
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
