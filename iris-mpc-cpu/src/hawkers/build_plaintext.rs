use std::sync::Arc;

use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
use itertools::Itertools;
use tokio::task::JoinSet;

use crate::{
    execution::hawk_main::{
        insert::{self, InsertPlanV},
        BothEyes,
    },
    hawkers::plaintext_store::{PlaintextVector, SharedPlaintextStore},
    hnsw::{GraphMem, HnswSearcher},
};

pub type SharedPlaintextGraphs = BothEyes<GraphMem<PlaintextVector>>;
pub type SharedPlaintextStores = BothEyes<SharedPlaintextStore>;

pub async fn plaintext_parallel_batch_insert(
    graph: Option<GraphMem<PlaintextVector>>,
    store: Option<SharedPlaintextStore>,
    irises: Vec<(IrisVectorId, IrisCode)>,
    params: crate::hnsw::HnswParams,
    batch_size: usize,
    prf_seed: &[u8; 16],
) -> Result<(GraphMem<PlaintextVector>, SharedPlaintextStore)> {
    assert!(graph.is_none() == store.is_none());
    let mut graph = Arc::new(graph.unwrap_or_default());
    let mut store = store.unwrap_or_default();

    let searcher = HnswSearcher { params };

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
                let insertion_layer = searcher.select_layer_prf(&prf_seed, &(vector_id))?;

                let (links, set_ep) = searcher
                    .search_to_insert(&mut store, &graph, &query, insertion_layer)
                    .await?;

                let insert_plan: InsertPlanV<SharedPlaintextStore> = InsertPlanV {
                    query,
                    links,
                    set_ep,
                };
                Ok((vector_id, insert_plan))
            });
        }

        let mut results: Vec<_> = jobs
            .join_all()
            .await
            .into_iter()
            .collect::<Result<_, _>>()?;

        results.sort_by_key(|(vector_id, _)| (vector_id.serial_id()));

        let (ids, plans): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let ids = ids.into_iter().map(Some).collect_vec();
        let plans = plans.into_iter().map(Some).collect_vec();

        // Unwrap Arc while inserting, then wrap again for the next batch
        let mut graph_temp = Arc::try_unwrap(graph).unwrap();
        insert::insert(&mut store, &mut graph_temp, &searcher, plans, &ids).await?;
        graph = Arc::new(graph_temp);
    }

    Ok((Arc::try_unwrap(graph).unwrap(), store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hawkers::plaintext_store::PlaintextStore,
        hnsw::{HnswParams, HnswSearcher},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;

    async fn setup_test_data(
        database_size: usize,
        to_insert: usize,
    ) -> Result<(
        HnswSearcher,
        GraphMem<PlaintextVector>,
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
        graph: GraphMem<PlaintextVector>,
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
            let result = searcher.search(&mut store, &graph, &query, 1).await?;
            assert!(
                searcher.is_match(&mut store, &[result]).await?,
                "Match verification failed for an inserted iris"
            );
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_insert_with_small_batches() -> Result<()> {
        let database_size = 256;
        let to_insert = 512;
        let batch_size = 32;
        let (searcher, graph, store, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        let (final_graph, final_store) = plaintext_parallel_batch_insert(
            Some(graph),
            Some(store),
            irises.clone(),
            HnswParams::new(64, 32, 32),
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
    async fn test_insert_with_large_batches() -> Result<()> {
        let database_size = 256;
        let to_insert = 512;
        let batch_size = 100;
        let (searcher, graph, store, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        let (final_graph, final_store) = plaintext_parallel_batch_insert(
            Some(graph),
            Some(store),
            irises.clone(),
            HnswParams::new(64, 32, 32),
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
}
