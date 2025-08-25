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
    graphs: Option<SharedPlaintextGraphs>,
    stores: Option<SharedPlaintextStores>,
    irises: Vec<(IrisVectorId, IrisCode, IrisCode)>,
    params: crate::hnsw::HnswParams,
    batch_size: Option<usize>,
    batch_size_error_rate: usize,
    prf_seed: &[u8; 16],
) -> Result<(SharedPlaintextGraphs, SharedPlaintextStores)> {
    // Checks for same option case, but otherwise assumes graphs and stores are in sync.
    assert!(graphs.is_none() == stores.is_none());
    let graphs = graphs
        .unwrap_or_else(|| [GraphMem::new(), GraphMem::new()])
        .map(Arc::new);
    let stores =
        stores.unwrap_or_else(|| [SharedPlaintextStore::new(), SharedPlaintextStore::new()]);

    let batch_size = match batch_size {
        None => BatchSize::get_dynamic_size(
            stores[0].len().await.try_into().unwrap(),
            batch_size_error_rate,
            params.M[1],
        ),
        Some(batch_size) => batch_size,
    };

    let irises_by_side: [Vec<(IrisVectorId, IrisCode)>; 2] = [
        irises
            .iter()
            .map(|(id, left, _)| (*id, left.clone()))
            .collect(),
        irises
            .iter()
            .map(|(id, _, right)| (*id, right.clone()))
            .collect(),
    ];

    let mut jobs: JoinSet<Result<_>> = JoinSet::new();
    let searcher = HnswSearcher { params };

    for (side, graph, store, irises) in izip!(
        STORE_IDS,
        graphs.clone().into_iter(),
        stores.clone().into_iter(),
        irises_by_side.into_iter()
    ) {
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
                        let insertion_layer =
                            searcher.select_layer_prf(&prf_seed, &(vector_id, side))?;

                        let (links, set_ep) = searcher
                            .search_to_insert(&mut store, &graph, &query, insertion_layer)
                            .await?;

                        let insert_plan: InsertPlanV<SharedPlaintextStore> = InsertPlanV {
                            query,
                            links,
                            set_ep,
                        };

                        let iside = STORE_IDS.iter().position(|&s| s == side).unwrap();
                        results.push((iside, vector_id, insert_plan));
                    }
                    Ok(results)
                }
            });
        }
    }

    // Flatten all results, sort by side and index to recover order
    let results: Vec<_> = jobs
        .join_all()
        .await
        .into_iter()
        .collect::<Result<_, _>>()?;

    let mut results: Vec<(usize, IrisVectorId, InsertPlanV<SharedPlaintextStore>)> =
        results.into_iter().flatten().collect();
    results.sort_by_key(|(iside, vector_id, _)| (*iside, vector_id.serial_id()));

    let mut results_by_side: [Vec<_>; 2] = [Vec::new(), Vec::new()];
    for (iside, index, insert_plan) in results {
        results_by_side[iside].push((index, insert_plan));
    }

    let mut ret_graphs = Vec::new();
    let mut ret_stores = Vec::new();

    for (_side, graph, mut store, insert_plans) in izip!(
        STORE_IDS,
        graphs.into_iter(),
        stores.into_iter(),
        results_by_side.into_iter()
    ) {
        // Mocking these because I'm not sure what they're supposed to be

        let (ids, plans): (Vec<_>, Vec<_>) = insert_plans.into_iter().unzip();
        let ids = ids.into_iter().map(Some).collect_vec();
        let plans = plans.into_iter().map(Some).collect_vec();

        // Should be able to take ownership from Arc, as all threads have finished before
        let mut graph = Arc::try_unwrap(graph).unwrap();
        insert::insert(&mut store, &mut graph, &searcher, plans, &ids).await?;

        ret_graphs.push(graph);
        ret_stores.push(store);
    }

    Ok((
        ret_graphs.try_into().unwrap(),
        ret_stores.try_into().unwrap(),
    ))
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
    async fn setup_test_data(
        database_size: usize,
        to_insert: usize,
    ) -> Result<(
        HnswSearcher,
        [GraphMem<SharedPlaintextStore>; 2],
        [SharedPlaintextStore; 2],
        Vec<(IrisVectorId, IrisCode, IrisCode)>,
        [u8; 16],
    )> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let searcher = HnswSearcher::new_with_test_parameters();
        let prf_seed = [0u8; 16];

        let mut graphs = Vec::new();
        let mut stores = Vec::new();
        let mut irises_by_side = Vec::new();

        for _side in STORE_IDS {
            let mut ptxt_vector = PlaintextStore::new_random(&mut rng, database_size);
            let ptxt_graph = ptxt_vector
                .generate_graph(&mut rng, database_size, &searcher)
                .await?;

            let shared_vector = SharedPlaintextStore::from(ptxt_vector);
            let graph = migrate(ptxt_graph, |id| id);
            let irises_ = IrisDB::new_random_rng(to_insert, &mut rng).db;

            graphs.push(graph);
            stores.push(shared_vector);
            irises_by_side.push(irises_);
        }

        let irises: Vec<(IrisVectorId, IrisCode, IrisCode)> =
            izip!(irises_by_side[0].clone(), irises_by_side[1].clone())
                .enumerate()
                .map(|(id, (left, right))| {
                    (
                        IrisVectorId::from_serial_id(
                            // Ensure new IDs are unique and don't overlap with existing ones
                            (id + database_size + 1).try_into().unwrap(),
                        ),
                        left,
                        right,
                    )
                })
                .collect();

        Ok((
            searcher,
            graphs.try_into().unwrap(),
            stores.try_into().unwrap(),
            irises,
            prf_seed,
        ))
    }

    /// Verifies the state of the stores after insertion, and checks that searches and matches work correctly.
    async fn check_results(
        mut stores: [SharedPlaintextStore; 2],
        graphs: [GraphMem<SharedPlaintextStore>; 2],
        irises: Vec<(IrisVectorId, IrisCode, IrisCode)>,
        searcher: &HnswSearcher,
        expected_total_size: usize,
    ) -> Result<()> {
        for side in STORE_IDS {
            assert_eq!(
                stores[side as usize].len().await,
                expected_total_size,
                "Store size mismatch on side {}",
                side
            );
        }

        for eye in irises {
            for (side, iris_code) in izip!(STORE_IDS, [eye.1, eye.2]) {
                let query = Arc::new(iris_code);
                let result = searcher
                    .search(
                        &mut stores[side as usize],
                        &graphs[side as usize],
                        &query,
                        1,
                    )
                    .await?;
                assert!(
                    searcher
                        .is_match(&mut stores[side as usize], &[result])
                        .await?,
                    "Match verification failed for side {}",
                    side
                );
            }
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_insert_with_existing_data_and_dynamic_batch() -> Result<()> {
        let database_size = 256;
        let to_insert = 256;
        let (searcher, graphs, stores, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        let (final_graphs, final_stores) = plaintext_parallel_batch_insert(
            Some(graphs),
            Some(stores),
            irises.clone(),
            HnswParams::new(64, 32, 32),
            None, // Dynamic batch size
            1,
            &prf_seed,
        )
        .await?;

        check_results(
            final_stores,
            final_graphs,
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
        let (searcher, graphs, stores, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        let (final_graphs, final_stores) = plaintext_parallel_batch_insert(
            Some(graphs),
            Some(stores),
            irises.clone(),
            HnswParams::new(64, 32, 32),
            None,
            1,
            &prf_seed,
        )
        .await?;

        check_results(
            final_stores,
            final_graphs,
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
        let (searcher, graphs, stores, irises, prf_seed) =
            setup_test_data(database_size, to_insert).await?;

        let (final_graphs, final_stores) = plaintext_parallel_batch_insert(
            Some(graphs),
            Some(stores),
            irises.clone(),
            HnswParams::new(64, 32, 32),
            Some(64), // Set a fixed batch size
            1,
            &prf_seed,
        )
        .await?;

        check_results(
            final_stores,
            final_graphs,
            irises,
            &searcher,
            database_size + to_insert,
        )
        .await
    }
}
