use std::{future::Future, sync::Arc};

use crate::{
    execution::hawk_main::{scheduler::parallelize, HawkSession},
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::{
        graph::neighborhood::SortedNeighborhoodV,
        searcher::{ConnectPlanV, SetEntryPoint},
        sorting::quickselect::run_quickselect_with_store,
        vector_store::VectorStoreMut,
        GraphMem, HnswParams, HnswSearcher, VectorStore,
    },
};

use super::VecRequests;

use eyre::Result;
use itertools::{izip, Itertools};
use tokio::sync::{mpsc, RwLock};

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

            let mut connect_plan = searcher
                .insert_prepare(store, graph, inserted, extended_links, set_ep)
                .await?;
            //trim_neighborhoods(&searcher.params, store, &mut connect_plan).await?;
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

    fn make_layer<V: VectorStore>(nodes: &[V::VectorRef]) -> Layer<V::VectorRef> {
        let mut ret = Layer::new();
        for i in 0..nodes.len() {
            let mut neighbors = Vec::new();
            for (j, node) in nodes.iter().enumerate() {
                if i != j {
                    neighbors.push(node.clone());
                }
            }
            ret.set_links(nodes[i].clone(), neighbors);
        }

        ret
    }

    async fn trim_neighborhoods_test(
        params: &HnswParams,
        sessions: Vec<Arc<RwLock<SharedPlaintextStore>>>,
        connect_plan: &mut ConnectPlanV<SharedPlaintextStore>,
    ) -> Result<()> {
        let (session_tasks, mut result_rx) =
            trim_neighborhoods_generic::<SharedPlaintextStore>(params, sessions, connect_plan)
                .await?;

        // Run all session tasks in parallel and collect results
        let _ = parallelize(session_tasks.into_iter()).await?;

        // Apply all the trimmed neighborhoods back to the connect plans
        while let Some(trim_result) = result_rx.recv().await {
            if let Some(layer) = connect_plan.layers.get_mut(trim_result.layer_idx) {
                if let Some(neighborhood) = layer.nb_links.get_mut(trim_result.neighborhood_idx) {
                    *neighborhood = trim_result.neighborhood;
                }
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_serial_compaction() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(42);
        let iris_db = IrisDB::new_random_rng(32, &mut rng);
        let queries: Vec<_> = iris_db.db.into_iter().collect();

        let mut searcher = HnswSearcher::new_with_test_parameters_and_linear_scan();
        searcher.params.M_max[0] = 16;
        searcher.params.M_max[1] = 8;
        searcher.params.M_max[2] = 4;
        searcher.params.M_max_extra[0] = 16;
        searcher.params.M_max_extra[1] = 8;
        searcher.params.M_max_extra[2] = 4;
        let mut vector_store = PlaintextStore::new();
        let mut graph_store = GraphMem::<PlaintextVectorRef>::new();

        let vector_ids: Vec<_> = queries
            .iter()
            .enumerate()
            .map(|(id, query)| {
                vector_store.insert_with_id(VectorId::new(id as _, 3), Arc::new(query.clone()))
            })
            .collect();

        let l0 = make_layer::<SharedPlaintextStore>(&vector_ids[..searcher.params.get_M_max(0)]);
        let l1 = make_layer::<SharedPlaintextStore>(&vector_ids[..searcher.params.get_M_max(1)]);
        let l2 = make_layer::<SharedPlaintextStore>(&vector_ids[..searcher.params.get_M_max(2)]);

        let max_size = l0.links.len();
        let query_vector_id = vector_ids[max_size];

        // save these for later
        let distances = vector_store
            .eval_distance_batch(
                &Arc::new(queries[max_size].clone()),
                &vector_ids[..max_size],
            )
            .await?;

        let new_l0: Vec<_> = vector_ids[..max_size + 1].to_vec();
        let compacted1 = neighborhood_compaction(
            &mut vector_store,
            query_vector_id,
            &new_l0[..max_size],
            max_size,
        )
        .await?;
        assert_eq!(compacted1.len(), max_size);

        let compacted2 =
            neighborhood_compaction(&mut vector_store, query_vector_id, &new_l0, max_size).await?;

        assert_eq!(compacted2.len(), max_size);

        let mut vector_store: SharedPlaintextStore = vector_store.into();
        graph_store.layers = vec![l0, l1, l2];

        // pick neighbors for the vector at each layer
        let nbs: Vec<SortedNeighborhood<_, _>> = (0..3)
            .map(|idx| {
                SortedNeighborhood::from_ascending_vec(vec![(vector_ids[idx], distances[idx])])
            })
            .collect();

        let connect_plans = searcher
            .insert_prepare(
                &mut vector_store,
                &graph_store,
                query_vector_id,
                nbs,
                SetEntryPoint::False,
            )
            .await?;

        // assert that the neighbors were built correctly
        assert_eq!(connect_plans.layers[0].neighbors[0], vector_ids[0].clone());
        assert_eq!(connect_plans.layers[1].neighbors[0], vector_ids[1].clone());
        assert_eq!(connect_plans.layers[2].neighbors[0], vector_ids[2].clone());

        assert_eq!(connect_plans.layers.len(), 3);
        assert_eq!(connect_plans.layers[0].neighbors.len(), 1);
        assert_eq!(connect_plans.layers[1].neighbors.len(), 1);
        assert_eq!(connect_plans.layers[2].neighbors.len(), 1);

        let mut nb_l0: Vec<_> = vector_ids[1..searcher.params.get_M_max(0)].to_vec(); // removes idx 0
        nb_l0.push(query_vector_id);
        assert_eq!(nb_l0.len(), 16);
        assert_eq!(connect_plans.layers[0].nb_links[0], nb_l0);

        let mut nb_l1: Vec<_> = vector_ids[0..searcher.params.get_M_max(1)].to_vec();
        nb_l1.remove(1); // removes idx 1
        nb_l1.push(query_vector_id);
        assert_eq!(connect_plans.layers[1].nb_links[0], nb_l1);

        let mut nb_l2: Vec<_> = vector_ids[0..searcher.params.get_M_max(2)].to_vec();
        nb_l2.remove(2); // removes idx 2
        nb_l2.push(query_vector_id);
        assert_eq!(connect_plans.layers[2].nb_links[0], nb_l2);

        // call trim when neighbors = M_max and M_max_extra
        let mut trim_nb_plans = connect_plans.clone();
        trim_neighborhoods(&searcher.params, &mut vector_store, &mut trim_nb_plans).await?;

        // nothing should have been trimmed.
        assert_eq!(trim_nb_plans, connect_plans);

        searcher.params.M_max[0] = 15;
        searcher.params.M_max[2] = 3;
        searcher.params.M_max_extra[0] = 15;
        searcher.params.M_max_extra[2] = 3;

        // compare serial trim to parallel trim
        let mut trim_nb_plans = connect_plans.clone();
        trim_neighborhoods(&searcher.params, &mut vector_store, &mut trim_nb_plans).await?;

        let vector_store: Arc<RwLock<SharedPlaintextStore>> = Arc::new(RwLock::new(vector_store));
        let fake_sessions: Vec<_> = (0..16).map(|_| vector_store.clone()).collect();

        let mut trim_nb_plans_par = connect_plans.clone();
        trim_neighborhoods_test(&searcher.params, fake_sessions, &mut trim_nb_plans_par).await?;

        assert_ne!(trim_nb_plans, connect_plans);
        // layer 1 was unchanged
        assert_eq!(trim_nb_plans.layers[1], connect_plans.layers[1]);
        assert_eq!(trim_nb_plans, trim_nb_plans_par);

        Ok(())
    }

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
