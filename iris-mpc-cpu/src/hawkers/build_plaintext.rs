use std::sync::Arc;

use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
use itertools::Itertools;
use tokio::task::JoinSet;
use tracing::info;

use crate::hawkers::plaintext_deep_id_store::{Int4Vector, SharedPlaintextDeepIDStore};
use crate::{
    execution::hawk_main::insert::{self, InsertPlanV},
    hawkers::{
        aby3::aby3_store::DistanceOps,
        plaintext_store::{PlaintextVectorRef, SharedPlaintextStore},
    },
    hnsw::{graph::neighborhood::Neighborhood, GraphMem, HnswSearcher, SortedNeighborhood},
};

/// Number of entries to insert before reporting a new info log entry
const REPORTING_INTERVAL: usize = 1000;

pub async fn plaintext_parallel_batch_insert<D: DistanceOps>(
    graph: Option<GraphMem<PlaintextVectorRef>>,
    store: Option<SharedPlaintextStore<D>>,
    irises: Vec<(IrisVectorId, IrisCode)>,
    searcher: &HnswSearcher,
    batch_size: usize,
    prf_seed: &[u8; 16],
) -> Result<(GraphMem<PlaintextVectorRef>, SharedPlaintextStore<D>)> {
    assert!(graph.is_none() == store.is_none());
    let mut graph = Arc::new(graph.unwrap_or_default());
    let mut store = store.unwrap_or_default();

    let mut inserted_count: usize = 0;
    let mut reported_count: usize = 0;

    for batch in &irises.into_iter().enumerate().chunks(batch_size) {
        let mut jobs: JoinSet<Result<_>> = JoinSet::new();

        for (_, iris) in batch {
            let query = Arc::new(iris.1);
            let vector_id = iris.0;
            let prf_seed = *prf_seed;
            let mut store = store.clone();
            let graph = graph.clone();
            let searcher = searcher.clone();

            jobs.spawn(async move {
                let insertion_layer = searcher.gen_layer_prf(&prf_seed, &(vector_id))?;

                let (links, update_ep) = searcher
                    .search_to_insert::<_, SortedNeighborhood<_>>(
                        &mut store,
                        &graph,
                        &query,
                        insertion_layer,
                    )
                    .await?;

                // Trim and extract unstructured vector lists
                let mut links_unstructured = Vec::new();
                for (lc, mut l) in links.into_iter().enumerate() {
                    let m = searcher.params.get_M(lc);
                    l.trim(&mut store, m).await?;
                    links_unstructured.push(l.edge_ids())
                }

                let insert_plan: InsertPlanV<SharedPlaintextStore<D>> = InsertPlanV {
                    query,
                    links: links_unstructured,
                    update_ep,
                };
                Ok((vector_id, insert_plan))
            });

            inserted_count += 1;
        }

        let mut results: Vec<_> = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<_, _>>()?;

        results.sort_by_key(|(vector_id, _)| vector_id.serial_id());

        let (ids, plans): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let ids = ids.into_iter().map(Some).collect_vec();
        let plans = plans.into_iter().map(Some).collect_vec();

        // Unwrap Arc while inserting, then wrap again for the next batch
        let mut graph_temp = Arc::try_unwrap(graph).unwrap();
        insert::insert(&mut store, &mut graph_temp, searcher, plans, &ids).await?;
        graph = Arc::new(graph_temp);

        if inserted_count.saturating_sub(reported_count) >= REPORTING_INTERVAL {
            info!("Inserted {inserted_count} iris codes...");
            reported_count = inserted_count;
        }
    }

    Ok((Arc::try_unwrap(graph).unwrap(), store))
}

pub async fn deep_id_parallel_batch_insert(
    graph: GraphMem<IrisVectorId>,
    mut store: SharedPlaintextDeepIDStore,
    vectors: Vec<(IrisVectorId, Int4Vector)>,
    searcher: &HnswSearcher,
    batch_size: usize,
    prf_seed: &[u8; 16],
) -> Result<(GraphMem<IrisVectorId>, SharedPlaintextDeepIDStore)> {
    let mut graph = Arc::new(graph);

    let mut inserted_count: usize = 0;
    let mut reported_count: usize = 0;

    for batch in &vectors.into_iter().enumerate().chunks(batch_size) {
        let mut jobs: JoinSet<Result<_>> = JoinSet::new();

        for (_, entry) in batch {
            let query = Arc::new(entry.1);
            let vector_id = entry.0;
            let prf_seed = *prf_seed;
            let mut store = store.clone();
            let graph = graph.clone();
            let searcher = searcher.clone();

            jobs.spawn(async move {
                let insertion_layer = searcher.gen_layer_prf(&prf_seed, &(vector_id))?;

                let (links, update_ep) = searcher
                    .search_to_insert::<_, SortedNeighborhood<_>>(
                        &mut store,
                        &graph,
                        &query,
                        insertion_layer,
                    )
                    .await?;

                let mut links_unstructured = Vec::new();
                for (lc, mut l) in links.into_iter().enumerate() {
                    let m = searcher.params.get_M(lc);
                    l.trim(&mut store, m).await?;
                    links_unstructured.push(l.edge_ids())
                }

                let insert_plan: InsertPlanV<SharedPlaintextDeepIDStore> = InsertPlanV {
                    query,
                    links: links_unstructured,
                    update_ep,
                };
                Ok((vector_id, insert_plan))
            });

            inserted_count += 1;
        }

        let mut results: Vec<_> = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<_, _>>()?;
        results.sort_by_key(|(vector_id, _)| vector_id.serial_id());

        let (ids, plans): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let ids = ids.into_iter().map(Some).collect_vec();
        let plans = plans.into_iter().map(Some).collect_vec();

        let mut graph_temp = Arc::try_unwrap(graph).unwrap();
        insert::insert(&mut store, &mut graph_temp, searcher, plans, &ids).await?;
        graph = Arc::new(graph_temp);

        if inserted_count.saturating_sub(reported_count) >= REPORTING_INTERVAL {
            info!("Inserted {inserted_count} deep-ID vectors...");
            reported_count = inserted_count;
        }
    }

    Ok((Arc::try_unwrap(graph).unwrap(), store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{hawkers::plaintext_store::PlaintextStore, hnsw::HnswSearcher};
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;

    async fn setup_test_data(
        database_size: usize,
        to_insert: usize,
    ) -> Result<(
        HnswSearcher,
        GraphMem<PlaintextVectorRef>,
        SharedPlaintextStore,
        Vec<(IrisVectorId, IrisCode)>,
        [u8; 16],
    )> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let searcher = HnswSearcher::new_with_test_parameters();
        let prf_seed = [0u8; 16];

        let mut ptxt_vector = PlaintextStore::new_random(&mut rng, database_size);
        let ptxt_graph = ptxt_vector
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;

        let shared_store = SharedPlaintextStore::from(ptxt_vector);
        let graph = ptxt_graph;

        let irises_to_insert = IrisDB::new_random_rng(to_insert, &mut rng).db;

        let irises: Vec<(IrisVectorId, IrisCode)> = irises_to_insert
            .into_iter()
            .enumerate()
            .map(|(id, iris_code)| {
                (
                    IrisVectorId::from_serial_id((id + database_size + 1).try_into().unwrap()),
                    iris_code,
                )
            })
            .collect();

        Ok((searcher, graph, shared_store, irises, prf_seed))
    }

    async fn check_results(
        mut store: SharedPlaintextStore,
        graph: GraphMem<PlaintextVectorRef>,
        irises: Vec<(IrisVectorId, IrisCode)>,
        searcher: &HnswSearcher,
        expected_total_size: usize,
    ) -> Result<()> {
        assert_eq!(
            store.len().await,
            expected_total_size,
            "Final store size is incorrect"
        );

        // Check if each inserted iris can be found and matched correctly
        for (_vector_id, iris_code) in irises {
            let query = Arc::new(iris_code);
            let result = searcher
                .search::<_, SortedNeighborhood<_>>(&mut store, &graph, &query, 1)
                .await?;
            assert!(
                searcher.is_match(&mut store, &[result]).await?,
                "Match verification failed for an inserted iris"
            );
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_insert_with_batches() -> Result<()> {
        let database_size = 64;
        let to_insert = 64;
        let batch_size = 8;
        let (searcher, graph, store, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        let (final_graph, final_store) = plaintext_parallel_batch_insert(
            Some(graph),
            Some(store),
            irises.clone(),
            &searcher,
            batch_size,
            &prf_seed,
        )
        .await?;

        check_results(
            final_store,
            final_graph,
            irises,
            &searcher,
            database_size + to_insert,
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_deep_id_insert_with_batches() -> Result<()> {
        use crate::hawkers::plaintext_deep_id_store::{Int4Vector, PlaintextDeepIDStore};
        use crate::hnsw::vector_store::VectorStoreMut;
        use aes_prng::AesRng;
        use rand::SeedableRng;

        let database_size = 32_usize;
        let to_insert = 32_usize;
        let batch_size = 8;
        let prf_seed = [0u8; 16];
        let mut rng = AesRng::seed_from_u64(0);

        let searcher = HnswSearcher::new_with_test_parameters();

        // Seed store with `database_size` vectors and build a small graph.
        let mut store = PlaintextDeepIDStore::new(/* threshold */ 0);
        for _ in 0..database_size {
            let v = Arc::new(Int4Vector::random(&mut rng));
            VectorStoreMut::insert(&mut store, &v).await;
        }
        let graph = store
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;

        let shared_store: SharedPlaintextDeepIDStore = store.into();

        // Build the to-insert batch with explicit ids past the seeded range.
        let to_insert_vectors: Vec<(IrisVectorId, Int4Vector)> = (0..to_insert)
            .map(|i| {
                (
                    IrisVectorId::from_serial_id((i + database_size + 1).try_into().unwrap()),
                    Int4Vector::random(&mut rng),
                )
            })
            .collect();

        let (final_graph, mut final_store) = deep_id_parallel_batch_insert(
            graph,
            shared_store,
            to_insert_vectors.clone(),
            &searcher,
            batch_size,
            &prf_seed,
        )
        .await?;

        assert_eq!(final_store.len().await, database_size + to_insert);

        // Each inserted vector should be its own top-1 neighbor.
        for (id, v) in to_insert_vectors {
            let query = Arc::new(v);
            let results: SortedNeighborhood<_> = searcher
                .search(&mut final_store, &final_graph, &query, /* k */ 1)
                .await?;
            let pairs = results.as_vec_ref();
            assert!(!pairs.is_empty());
            assert_eq!(pairs[0].0, id, "expected top-1 to be self for id {id}");
            assert!(
                pairs[0].1 >= 0,
                "self-dot should be non-negative, got {}",
                pairs[0].1
            );
        }

        Ok(())
    }
}
