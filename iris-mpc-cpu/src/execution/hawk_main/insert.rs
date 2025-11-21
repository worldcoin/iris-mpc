
use crate::hnsw::{
        graph::neighborhood::SortedNeighborhoodV,
        searcher::{ConnectPlanV, SetEntryPoint},
        sorting::quickselect::run_quickselect_with_store,
        vector_store::VectorStoreMut,
        GraphMem, HnswSearcher, VectorStore,
    };

use super::VecRequests;

use eyre::Result;
use itertools::{izip, Itertools};

/// InsertPlan specifies where a query may be inserted into the HNSW graph.
/// That is lists of neighbors for each layer.
#[derive(Debug)]
pub struct InsertPlanV<V: VectorStore> {
    pub query: V::QueryRef,
    pub links: Vec<SortedNeighborhoodV<V>>,
    pub set_ep: SetEntryPoint,
}

// Manual implementation of Clone for InsertPlanV, since derive(Clone) does not
// propagate the nested Clone bounds on V::QueryRef via TransientRef.
impl<V: VectorStore> Clone for InsertPlanV<V> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            links: self.links.clone(),
            set_ep: self.set_ep.clone(),
        }
    }
}

/// Insert a collection `plans` of `InsertPlanV` structs into the graph and vector store,
/// adjusting the insertion plans as needed to repair any conflict from parallel searches.
///
/// The `ids` argument consists of `Option<VectorId>`s which are `Some(id)` if the associated
/// plan is to be inserted with a specific identifier (e.g. for updates or for insertions
/// which need to parallel an existing iris code database), and `None` if the associated plan
/// is to be inserted at the next available serial ID, with version 0.
pub async fn insert<V: VectorStoreMut>(
    store: &mut V,
    graph: &mut GraphMem<<V as VectorStore>::VectorRef>,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<V>>>,
    ids: &VecRequests<Option<V::VectorRef>>,
) -> Result<VecRequests<Option<ConnectPlanV<V>>>> {
    let insert_plans = join_plans(plans);
    let mut connect_plans = vec![None; insert_plans.len()];
    let mut inserted_ids = vec![];
    let m = searcher.params.get_M(0);

    for (plan, update_id, cp) in izip!(insert_plans, ids, &mut connect_plans) {
        if let Some(InsertPlanV {
            query,
            links,
            set_ep,
        }) = plan
        {
            let extended_links =
                add_batch_neighbors(&mut *store, &query, links, &inserted_ids, m).await?;

            let inserted = {
                match update_id {
                    None => store.insert(&query).await,
                    Some(id) => store.insert_at(id, &query).await?,
                }
            };

            let connect_plan = searcher
                .insert_prepare(store, graph, inserted, extended_links, set_ep)
                .await?;
            graph.insert_apply(connect_plan.clone()).await;
            inserted_ids.push(connect_plan.inserted_vector.clone());
            cp.replace(connect_plan);
        }
    }

    Ok(connect_plans)
}

// todo: switch with oblivious min-k and random shuffle
/// enforce size constraints on the neighborhood in an oblivious manner
async fn neighborhood_compaction<V: VectorStore>(
    store: &mut V,
    query: V::VectorRef,
    neighborhood: &[V::VectorRef],
    max_size: usize,
) -> Result<Vec<V::VectorRef>> {
    let r = store.vectors_as_queries(vec![query]).await;
    let query = &r[0];
    let link_distances = store.eval_distance_batch(query, neighborhood).await?;
    let sorted_idxs = run_quickselect_with_store(&mut (*store), &link_distances, max_size).await?;

    let trimmed_neighborhood = sorted_idxs
        .into_iter()
        .take(max_size)
        .map(|idx| neighborhood[idx].clone())
        .collect();
    Ok(trimmed_neighborhood)
}

/// Extends the bottom layer of links with additional vectors in `extra_ids` if there is room.
async fn add_batch_neighbors<V: VectorStore>(
    store: &mut V,
    query: &V::QueryRef,
    mut links: Vec<SortedNeighborhoodV<V>>,
    extra_ids: &[V::VectorRef],
    target_n_neighbors: usize,
) -> Result<Vec<SortedNeighborhoodV<V>>> {
    if let Some(bottom_layer) = links.first_mut() {
        if bottom_layer.len() < target_n_neighbors {
            let distances = store.eval_distance_batch(query, extra_ids).await?;

            let ids_dists = izip!(extra_ids.iter().cloned(), distances)
                .map(|(id, dist)| (id, dist))
                .collect_vec();

            bottom_layer.insert_batch(&mut *store, &ids_dists).await?;
        }
    }

    Ok(links)
}

/// Combine insert plans from parallel searches, repairing any conflict.
fn join_plans<V: VectorStore>(
    mut plans: Vec<Option<InsertPlanV<V>>>,
) -> Vec<Option<InsertPlanV<V>>> {
    let ep_layers: Vec<_> = plans
        .iter()
        .flatten()
        .filter(|plan| plan.set_ep != SetEntryPoint::False)
        .map(|plan| plan.links.len())
        .collect();

    if !ep_layers.is_empty() {
        let max_insertion_layer = ep_layers.into_iter().max().unwrap_or_default();

        // for TopLevelSearchMode::Default, SetEntryPoint::NewLayer is used.
        // for TopLevelSearchMode::LinearScan, SetEntryPoint::AddToLayer is used.
        // if multiple plans have SetEntryPoint::NewLayer, an arbitrary one is chosen as the entry point.
        let mut set_ep_new_layer = false;
        for plan in plans.iter_mut() {
            let Some(plan) = plan else { continue };

            if plan.set_ep == SetEntryPoint::False {
                continue;
            }

            let current_layer = plan.links.len();
            if current_layer < max_insertion_layer {
                plan.set_ep = SetEntryPoint::False;
                continue;
            }

            if plan.set_ep == SetEntryPoint::NewLayer {
                if !set_ep_new_layer {
                    set_ep_new_layer = true;
                } else {
                    plan.set_ep = SetEntryPoint::False;
                }
            }
        }
    }
    plans
}

#[cfg(test)]
mod tests {
    use crate::hawkers::plaintext_store::{
        PlaintextStore, PlaintextVectorRef, SharedPlaintextStore,
    };
    use crate::hnsw::graph::layered_graph::Layer;
    use crate::hnsw::graph::neighborhood::SortedNeighborhoodV;
    use crate::hnsw::SortedNeighborhood;
    use aes_prng::AesRng;
    use iris_mpc_common::{
        iris_db::{db::IrisDB, iris::IrisCode},
        vector_id::VectorId,
    };
    use rand::{thread_rng, SeedableRng};
    use std::sync::Arc;
    use std::sync::LazyLock;

    use super::*;

    /// Helper function to test join_plans with multiple scenarios
    fn test_join_plans_helper(
        test_cases: &[(usize, SetEntryPoint)],
        expected_results: &[SetEntryPoint],
    ) {
        static QUERY: LazyLock<Arc<IrisCode>> = LazyLock::new(|| {
            let mut rng = thread_rng();
            let iris_code = IrisCode::random_rng(&mut rng);
            Arc::new(iris_code)
        });

        let query = QUERY.clone();
        let mut plans = Vec::new();
        for (num_layers, input_set_ep) in test_cases {
            let plan: InsertPlanV<PlaintextStore> = InsertPlanV {
                query: query.clone(),
                links: vec![SortedNeighborhoodV::<PlaintextStore>::new(); *num_layers],
                set_ep: input_set_ep.clone(),
            };
            plans.push(Some(plan));
        }
        plans.push(None); // Add a None plan as in the original tests
        let result = join_plans(plans);
        assert_eq!(result.len(), expected_results.len() + 1);
        for (idx, expected_set_ep) in expected_results.iter().enumerate() {
            assert_eq!(result[idx].as_ref().unwrap().set_ep, *expected_set_ep);
        }
    }

    // test adding entry points at two different layers
    #[test]
    fn test_join_plans1() {
        test_join_plans_helper(
            &[(9, SetEntryPoint::NewLayer), (10, SetEntryPoint::NewLayer)],
            &[SetEntryPoint::False, SetEntryPoint::NewLayer],
        );

        // try reversing the order
        test_join_plans_helper(
            &[(10, SetEntryPoint::NewLayer), (9, SetEntryPoint::NewLayer)],
            &[SetEntryPoint::NewLayer, SetEntryPoint::False],
        );
    }

    // test adding entry points on the same layers
    #[test]
    fn test_join_plans2() {
        test_join_plans_helper(
            &[(10, SetEntryPoint::NewLayer), (10, SetEntryPoint::NewLayer)],
            &[SetEntryPoint::NewLayer, SetEntryPoint::False],
        );
    }

    // test AddToLayer on different layers
    #[test]
    fn test_join_plans3() {
        test_join_plans_helper(
            &[
                (9, SetEntryPoint::AddToLayer),
                (10, SetEntryPoint::AddToLayer),
            ],
            &[SetEntryPoint::False, SetEntryPoint::AddToLayer],
        );

        // change the order
        test_join_plans_helper(
            &[
                (10, SetEntryPoint::AddToLayer),
                (9, SetEntryPoint::AddToLayer),
            ],
            &[SetEntryPoint::AddToLayer, SetEntryPoint::False],
        );
    }

    // test AddToLayer on same layers
    #[test]
    fn test_join_plans4() {
        test_join_plans_helper(
            &[
                (10, SetEntryPoint::AddToLayer),
                (10, SetEntryPoint::AddToLayer),
            ],
            &[SetEntryPoint::AddToLayer, SetEntryPoint::AddToLayer],
        );
    }
}
