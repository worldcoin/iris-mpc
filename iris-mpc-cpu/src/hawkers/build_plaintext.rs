use std::sync::Arc;

use eyre::Result;
use iris_mpc_common::{iris_db::iris::IrisCode, IrisVectorId};
use itertools::{izip, Itertools};
use tokio::task::JoinSet;

use crate::{
    execution::hawk_main::{
        insert::{self, InsertPlanV},
        BothEyes, STORE_IDS,
    },
    genesis::BatchSize,
    hawkers::plaintext_store::SharedPlaintextStore,
    hnsw::{GraphMem, HnswSearcher},
};

pub type SharedPlaintextGraphs = BothEyes<GraphMem<SharedPlaintextStore>>;
pub type SharedPlaintextStores = BothEyes<SharedPlaintextStore>;

pub async fn plaintext_parallel_batch_insert(
    graph: Option<GraphMem<SharedPlaintextStore>>,
    store: Option<SharedPlaintextStore>,
    irises: Vec<(IrisVectorId, IrisCode)>,
    params: crate::hnsw::HnswParams,
    batch_size: usize,
    prf_seed: &[u8; 16],
) -> Result<(GraphMem<SharedPlaintextStore>, SharedPlaintextStore)> {
    // Checks for same option case, but otherwise assumes graphs and stores are in sync.
    assert!(graph.is_none() == store.is_none());
    let graph = Arc::new(graph.unwrap_or_else(|| GraphMem::new()));
    let mut store = store.unwrap_or_else(|| SharedPlaintextStore::new());

    let mut jobs: JoinSet<Result<_>> = JoinSet::new();
    let searcher = HnswSearcher { params };

    for batch in &irises.into_iter().enumerate().chunks(batch_size) {
        let prf_seed = *prf_seed;
        let mut store = store.clone();
        let graph = graph.clone();
        let searcher = searcher.clone();
        let batch = batch.collect_vec();

        jobs.spawn({
            async move {
                let mut results = Vec::new();
                for (_, iris) in batch {
                    let query = Arc::new(iris.1);
                    let vector_id = iris.0;
                    let insertion_layer = searcher.select_layer_prf(&prf_seed, &(vector_id))?;

                    let (links, set_ep) = searcher
                        .search_to_insert(&mut store, &graph, &query, insertion_layer)
                        .await?;

                    let insert_plan: InsertPlanV<SharedPlaintextStore> = InsertPlanV {
                        query,
                        links,
                        set_ep,
                    };

                    results.push((vector_id, insert_plan));
                }
                Ok(results)
            }
        });
    }

    // Flatten all results, sort by side and index to recover order
    let results: Vec<_> = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<_, _>>()?;

    let mut results: Vec<(IrisVectorId, InsertPlanV<SharedPlaintextStore>)> =
        results.into_iter().flatten().collect();
    results.sort_by_key(|(vector_id, _)| (vector_id.serial_id()));

    let (ids, plans): (Vec<_>, Vec<_>) = results.into_iter().unzip();
    let ids = ids.into_iter().map(Some).collect_vec();
    let plans = plans.into_iter().map(Some).collect_vec();

    // Should be able to take ownership from Arc, as all threads have finished before
    let mut graph = Arc::try_unwrap(graph).unwrap();
    insert::insert(&mut store, &mut graph, &searcher, plans, &ids).await?;

    Ok((graph, store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hawkers::plaintext_store::PlaintextStore,
        hnsw::{graph::layered_graph::migrate, HnswParams, HnswSearcher},
    };
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;

    /// Prepares the arguments for the `plaintext_parallel_batch_insert` function.
    /// This now sets up a single graph and store.
    async fn setup_test_data(
        database_size: usize,
        to_insert: usize,
    ) -> Result<(
        HnswSearcher,
        GraphMem<SharedPlaintextStore>,
        SharedPlaintextStore,
        Vec<(IrisVectorId, IrisCode)>,
        [u8; 16],
    )> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let searcher = HnswSearcher::new_with_test_parameters();
        let prf_seed = [0u8; 16];

        // Generate the initial database (graph and store)
        let mut ptxt_vector = PlaintextStore::new_random(&mut rng, database_size);
        let ptxt_graph = ptxt_vector
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;

        // Convert to the shared, thread-safe versions
        let shared_store = SharedPlaintextStore::from(ptxt_vector);
        let graph = migrate(ptxt_graph, |id| id);

        // Generate the new iris codes to be inserted
        let irises_to_insert = IrisDB::new_random_rng(to_insert, &mut rng).db;

        // Create unique IrisVectorIds for the new irises, ensuring they don't overlap
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

    /// Verifies the state of the store after insertion and checks that searches work correctly.
    async fn check_results(
        mut store: SharedPlaintextStore,
        graph: GraphMem<SharedPlaintextStore>,
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
    async fn test_insert_with_existing_data() -> Result<()> {
        let database_size = 256;
        let to_insert = 256;
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
    async fn test_insert_into_empty_db() -> Result<()> {
        let database_size = 0; // Start with an empty database
        let to_insert = 256;
        let batch_size = 64;
        let (searcher, _graph, _store, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        // Test the code path where graph and store are created from scratch
        let (final_graph, final_store) = plaintext_parallel_batch_insert(
            None,
            None,
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
    async fn test_insert_with_fixed_batch_size() -> Result<()> {
        let database_size = 512;
        let to_insert = 256;
        let batch_size = 64; // A fixed batch size
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
