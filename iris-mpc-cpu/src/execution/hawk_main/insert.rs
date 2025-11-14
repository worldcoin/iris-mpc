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

pub async fn insert<V: VectorStoreMut>(
    store: &mut V,
    graph: &mut GraphMem<<V as VectorStore>::VectorRef>,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<V>>>,
    ids: &VecRequests<Option<V::VectorRef>>,
) -> Result<VecRequests<Option<ConnectPlanV<V>>>> {
    let mut connect_plans = insert_prepare(store, graph, searcher, plans, ids).await?;
    trim_neighborhoods(&searcher.params, store, &mut connect_plans).await?;
    insert_finalize::<V>(graph, &connect_plans).await?;
    Ok(connect_plans)
}

pub async fn insert_hawk(
    sessions: &[HawkSession],
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<Aby3Store>>>,
    ids: &VecRequests<Option<<Aby3Store as VectorStore>::VectorRef>>,
) -> Result<VecRequests<Option<ConnectPlanV<Aby3Store>>>> {
    let session = &sessions[0];

    let mut connect_plans = {
        let mut store = session.aby3_store.write().await;
        let mut graph = session.graph_store.write().await;
        insert_prepare(&mut (*store), &mut (*graph), searcher, plans, ids).await
    }?;

    trim_neighborhoods_par(&searcher.params, sessions, &mut connect_plans).await?;

    let mut graph = session.graph_store.write().await;
    insert_finalize::<Aby3Store>(&mut (*graph), &connect_plans).await?;

    Ok(connect_plans)
}

/// for each insertion, determines the node's neighbors at each layer, and the neighbors' neighbors,
/// which will now include the new node.
///
/// The `ids` argument consists of `Option<VectorId>`s which are `Some(id)` if the associated
/// plan is to be inserted with a specific identifier (e.g. for updates or for insertions
/// which need to parallel an existing iris code database), and `None` if the associated plan
/// is to be inserted at the next available serial ID, with version 0.
///
/// This is the first step in the three-phase insertion process:
/// 1. `insert_prepare`
/// 2. `trim_neighborhoods`
/// 3. `insert_finalize`
async fn insert_prepare<V: VectorStoreMut>(
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

            *cp = {
                let connect_plan = searcher
                    .insert_prepare(store, graph, inserted, extended_links, set_ep)
                    .await?;
                inserted_ids.push(connect_plan.inserted_vector.clone());
                Some(connect_plan)
            };
        }
    }

    Ok(connect_plans)
}

/// enforce neighborhood size constraints on the connect plans
///
/// Examines each layer of each connection plan and trims neighborhoods that have grown
/// beyond the optimal size (110% of max_links) down to the maximum allowed connections.
///
/// This is the second step in the three-phase insertion process:
/// 1. `insert_prepare`
/// 2. `trim_neighborhoods`
/// 3. `insert_finalize`
async fn trim_neighborhoods<V: VectorStoreMut>(
    params: &HnswParams,
    store: &mut V,
    connect_plans: &mut VecRequests<Option<ConnectPlanV<V>>>,
) -> Result<()> {
    for cp in connect_plans.iter_mut() {
        let Some(connect_plan) = cp.as_mut() else {
            continue;
        };

        for (lc, layer) in connect_plan.layers.iter_mut().enumerate() {
            let max_links = params.get_M_max(lc);
            let max_extra = params.get_M_extra(lc);
            for (idx, neighborhood) in layer.nb_links.iter_mut().enumerate() {
                if neighborhood.len() > max_extra {
                    let r = neighborhood_compaction(
                        store,
                        layer.neighbors[idx].clone(),
                        neighborhood,
                        max_links,
                    )
                    .await?;
                    *neighborhood = r;
                }
            }
        }
    }

    Ok(())
}

/// Finalize the insertion process by applying connection plans to the HNSW graph.
///
/// This is the final step in the three-phase insertion process:
/// 1. `insert_prepare`
/// 2. `trim_neighborhoods`
/// 3. `insert_finalize`
async fn insert_finalize<V: VectorStoreMut>(
    graph: &mut GraphMem<<V as VectorStore>::VectorRef>,
    connect_plans: &VecRequests<Option<ConnectPlanV<V>>>,
) -> Result<()> {
    for cp in connect_plans.iter() {
        let Some(connect_plan) = cp.as_ref() else {
            continue;
        };

        graph.insert_apply(connect_plan.clone()).await;
    }

    Ok(())
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

/// parallelizes trim_neighborhoods using the hawk sessions.
///
/// when trim_neighborhoods_generic makes the async closures to pass to tokio tasks (via parallelize), it is required that
/// the futures returned by the functions on VectorStore implement Send. For concrete types, this works out. For generic
/// types, this is not guaranteed and would require changes to the VectorStore trait. To avoid this, parallelize is called
/// with a concrete type.
async fn trim_neighborhoods_par(
    params: &HnswParams,
    sessions: &[HawkSession],
    connect_plans: &mut VecRequests<Option<ConnectPlanV<Aby3Store>>>,
) -> Result<()> {
    let sessions: Vec<Arc<RwLock<Aby3Store>>> =
        sessions.iter().map(|s| s.aby3_store.clone()).collect();
    let (session_tasks, mut result_rx) =
        trim_neighborhoods_generic::<Aby3Store>(params, sessions, connect_plans).await?;

    // Run all session tasks in parallel and collect results
    let _ = parallelize(session_tasks.into_iter()).await?;

    // Apply all the trimmed neighborhoods back to the connect plans
    while let Some(trim_result) = result_rx.recv().await {
        if let Some(connect_plan) = connect_plans[trim_result.plan_idx].as_mut() {
            if let Some(layer) = connect_plan.layers.get_mut(trim_result.layer_idx) {
                if let Some(neighborhood) = layer.nb_links.get_mut(trim_result.neighborhood_idx) {
                    *neighborhood = trim_result.neighborhood;
                }
            }
        }
    }

    Ok(())
}

/// Result of a trimming job
#[derive(Debug)]
struct TrimResult<V: VectorStore> {
    /// Index of the connect plan in the original vector
    plan_idx: usize,
    /// Layer index within the connect plan
    layer_idx: usize,
    /// Neighborhood index within the layer
    neighborhood_idx: usize,
    /// The trimmed neighborhood
    neighborhood: Vec<V::VectorRef>,
}

/// allows for testing without having to create HawkSessions.
async fn trim_neighborhoods_generic<V: VectorStore>(
    params: &HnswParams,
    sessions: Vec<Arc<RwLock<V>>>,
    connect_plans: &mut VecRequests<Option<ConnectPlanV<V>>>,
) -> Result<(
    Vec<impl Future<Output = Result<()>>>,
    mpsc::UnboundedReceiver<TrimResult<V>>,
)> {
    /// Job representing a neighborhood trimming task
    #[derive(Debug)]
    struct Job<V: VectorStore> {
        /// Index of the connect plan in the original vector
        plan_idx: usize,
        /// Layer index within the connect plan
        layer_idx: usize,
        /// Neighborhood index within the layer
        neighborhood_idx: usize,
        /// the neighbor being updated
        neighbor: V::VectorRef,
        /// The neighborhood data to trim
        neighborhood: Vec<V::VectorRef>,
        /// Maximum number of links allowed
        max_links: usize,
    }

    // Collect all neighborhoods that need trimming
    let mut jobs = Vec::new();

    for (plan_idx, cp) in connect_plans.iter_mut().enumerate() {
        let Some(connect_plan) = cp.as_mut() else {
            continue;
        };

        for (layer_idx, layer) in connect_plan.layers.iter_mut().enumerate() {
            let max_links = params.get_M_max(layer_idx);
            let max_extra = params.get_M_extra(layer_idx);

            for (neighborhood_idx, neighborhood) in layer.nb_links.iter_mut().enumerate() {
                if neighborhood.len() > max_extra {
                    let nb = std::mem::take(neighborhood);
                    jobs.push(Job {
                        plan_idx,
                        layer_idx,
                        neighborhood_idx,
                        neighbor: layer.neighbors[neighborhood_idx].clone(),
                        neighborhood: nb,
                        max_links,
                    });
                }
            }
        }
    }

    // Create single result channel for collecting all results
    let (result_tx, result_rx) = mpsc::unbounded_channel::<TrimResult<V>>();

    // If no trimming is needed, return early
    if jobs.is_empty() {
        return Ok((vec![], result_rx));
    }

    // Create separate channels for each session
    let n_sessions = sessions.len();
    let mut job_senders = Vec::with_capacity(n_sessions);
    let mut job_receivers = Vec::with_capacity(n_sessions);

    for _ in 0..n_sessions {
        let (tx, rx) = mpsc::unbounded_channel::<Job<V>>();
        job_senders.push(tx);
        job_receivers.push(rx);
    }

    // Distribute jobs round-robin to each session's channel
    for (job_idx, job) in jobs.into_iter().enumerate() {
        let session_idx = job_idx % n_sessions;
        job_senders[session_idx]
            .send(job)
            .map_err(|_| eyre::eyre!("Failed to send trim job"))?;
    }

    // Close all job senders to signal no more jobs
    drop(job_senders);

    // Create parallel tasks for each session with its own job receiver
    let session_tasks = sessions
        .into_iter()
        .zip(job_receivers.into_iter())
        .map(|(store, mut job_rx)| {
            let result_tx = result_tx.clone();

            async move {
                // Get the ABY3 store for this session
                let mut store = store.write().await;

                while let Some(job) = job_rx.recv().await {
                    let trimmed_neighborhood = neighborhood_compaction(
                        &mut (*store),
                        job.neighbor,
                        &job.neighborhood,
                        job.max_links,
                    )
                    .await?;

                    // Send the result back
                    let result = TrimResult {
                        plan_idx: job.plan_idx,
                        layer_idx: job.layer_idx,
                        neighborhood_idx: job.neighborhood_idx,
                        neighborhood: trimmed_neighborhood,
                    };

                    result_tx
                        .send(result)
                        .map_err(|_| eyre::eyre!("Failed to send trim result"))?;
                }

                Ok::<(), eyre::Error>(())
            }
        })
        .collect();

    Ok((session_tasks, result_rx))
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
        connect_plans: &mut VecRequests<Option<ConnectPlanV<SharedPlaintextStore>>>,
    ) -> Result<()> {
        let (session_tasks, mut result_rx) =
            trim_neighborhoods_generic::<SharedPlaintextStore>(params, sessions, connect_plans)
                .await?;

        // Run all session tasks in parallel and collect results
        let _ = parallelize(session_tasks.into_iter()).await?;

        // Apply all the trimmed neighborhoods back to the connect plans
        while let Some(trim_result) = result_rx.recv().await {
            if let Some(connect_plan) = connect_plans[trim_result.plan_idx].as_mut() {
                if let Some(layer) = connect_plan.layers.get_mut(trim_result.layer_idx) {
                    if let Some(neighborhood) = layer.nb_links.get_mut(trim_result.neighborhood_idx)
                    {
                        *neighborhood = trim_result.neighborhood;
                    }
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

        let connect_plans = vec![Some(connect_plans)];

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
        assert_eq!(
            trim_nb_plans[0].as_ref().unwrap().layers[1],
            connect_plans[0].as_ref().unwrap().layers[1]
        );
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
