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
use iris_mpc_common::IrisVectorId;
use itertools::{izip, Itertools};
use tokio::sync::mpsc;

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
    connect_plans = trim_neighborhoods(&searcher.params, store, connect_plans).await?;
    connect_plans = insert_finalize::<V>(graph, connect_plans).await?;
    Ok(connect_plans)
}

pub async fn insert_hawk(
    sessions: &[HawkSession],
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<Aby3Store>>>,
    ids: &VecRequests<Option<<Aby3Store as VectorStore>::VectorRef>>,
) -> Result<VecRequests<Option<ConnectPlanV<Aby3Store>>>> {
    let session = &sessions[0];
    let mut store = session.aby3_store.write().await;
    let mut graph = session.graph_store.write().await;

    let mut connect_plans =
        insert_prepare(&mut (*store), &mut (*graph), searcher, plans, ids).await?;
    connect_plans = trim_neighborhoods(&searcher.params, &mut (*store), connect_plans).await?;
    connect_plans = insert_finalize::<Aby3Store>(&mut (*graph), connect_plans).await?;
    Ok(connect_plans)
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

/// enforce neighborhood size constraints on the connect plans
///
/// Examines each layer of each connection plan and trims neighborhoods that have grown
/// beyond the optimal size (110% of max_links) down to the maximum allowed connections.
///
/// This is the second step in the three-phase insertion process:
/// 1. `insert_prepare`
/// 2. `trim_neighborhoods`
/// 3. `insert_finalize`
pub async fn trim_neighborhoods<V: VectorStoreMut>(
    params: &HnswParams,
    store: &mut V,
    mut connect_plans: VecRequests<Option<ConnectPlanV<V>>>,
) -> Result<VecRequests<Option<ConnectPlanV<V>>>> {
    for cp in connect_plans.iter_mut() {
        let Some(connect_plan) = cp.as_mut() else {
            continue;
        };

        for (lc, layer) in connect_plan.layers.iter_mut().enumerate() {
            let max_links = params.get_M_max(lc);
            let max_extra = params.get_M_extra(lc);
            for (idx, neighborhood) in layer.nb_links.iter_mut().enumerate() {
                if neighborhood.len() >= max_extra {
                    let r = neighborhood_compaction(
                        store,
                        layer.neighbors[idx].clone(),
                        &neighborhood,
                        max_links,
                    )
                    .await?;
                    *neighborhood = r;
                }
            }
        }
    }

    Ok(connect_plans)
}

/// parallelizes trim_neighborhoods using the hawk sessions.
pub async fn trim_neighborhoods_par(
    params: &HnswParams,
    sessions: &[HawkSession],
    mut connect_plans: VecRequests<Option<ConnectPlanV<Aby3Store>>>,
) -> Result<VecRequests<Option<ConnectPlanV<Aby3Store>>>> {
    /// Job representing a neighborhood trimming task
    #[derive(Debug)]
    struct Job {
        /// Index of the connect plan in the original vector
        plan_idx: usize,
        /// Layer index within the connect plan
        layer_idx: usize,
        /// Neighborhood index within the layer
        neighborhood_idx: usize,
        /// the neighbor being updated
        neighbor: IrisVectorId,
        /// The neighborhood data to trim
        neighborhood: Vec<IrisVectorId>,
        /// Maximum number of links allowed
        max_links: usize,
    }

    /// Result of a trimming job
    #[derive(Debug)]
    struct TrimResult {
        /// Index of the connect plan in the original vector
        plan_idx: usize,
        /// Layer index within the connect plan
        layer_idx: usize,
        /// Neighborhood index within the layer
        neighborhood_idx: usize,
        /// The trimmed neighborhood
        neighborhood: Vec<IrisVectorId>,
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
                let nb = std::mem::take(neighborhood);
                if neighborhood.len() >= max_extra {
                    jobs.push(Job {
                        plan_idx,
                        layer_idx,
                        neighborhood_idx,
                        neighbor: layer.neighbors[neighborhood_idx],
                        neighborhood: nb,
                        max_links,
                    });
                }
            }
        }
    }

    // If no trimming is needed, return early
    if jobs.is_empty() {
        return Ok(connect_plans);
    }

    // Create separate channels for each session
    let n_sessions = sessions.len();
    let mut job_senders = Vec::with_capacity(n_sessions);
    let mut job_receivers = Vec::with_capacity(n_sessions);

    for _ in 0..n_sessions {
        let (tx, rx) = mpsc::unbounded_channel::<Job>();
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

    // Create single result channel for collecting all results
    let (result_tx, mut result_rx) = mpsc::unbounded_channel::<TrimResult>();

    // Create parallel tasks for each session with its own job receiver
    let session_tasks =
        sessions
            .iter()
            .cloned()
            .zip(job_receivers.drain(..))
            .map(|(session, mut job_rx)| {
                let result_tx = result_tx.clone();
                let store = session.aby3_store.clone();

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
            });

    // Run all session tasks in parallel and collect results
    let _ = parallelize(session_tasks).await?;

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

    Ok(connect_plans)
}

/// Finalize the insertion process by applying connection plans to the HNSW graph.
///
/// This is the final step in the three-phase insertion process:
/// 1. `insert_prepare`
/// 2. `trim_neighborhoods`
/// 3. `insert_finalize`
pub async fn insert_finalize<V: VectorStoreMut>(
    graph: &mut GraphMem<<V as VectorStore>::VectorRef>,
    connect_plans: VecRequests<Option<ConnectPlanV<V>>>,
) -> Result<VecRequests<Option<ConnectPlanV<V>>>> {
    for cp in connect_plans.iter() {
        let Some(connect_plan) = cp.as_ref() else {
            continue;
        };

        graph.insert_apply(connect_plan.clone()).await;
    }

    Ok(connect_plans)
}

#[cfg(test)]
mod tests {
    use crate::hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef};
    use crate::hnsw::graph::layered_graph::Layer;
    use crate::hnsw::vector_store::Ref;
    use crate::hnsw::{graph::neighborhood::SortedNeighborhoodV, vector_store::VectorStoreMut};
    use aes_prng::AesRng;
    use iris_mpc_common::{
        iris_db::{db::IrisDB, iris::IrisCode},
        vector_id::VectorId,
    };
    use rand::{thread_rng, SeedableRng};
    use std::cmp;
    use std::fmt::Display;
    use std::str::FromStr;
    use std::sync::Arc;
    use std::sync::LazyLock;

    use super::*;

    fn make_layer<V: VectorStore>(nodes: &[V::VectorRef]) -> Layer<V::VectorRef> {
        let mut ret = Layer::new();
        let mut neighborhoods: Vec<V::VectorRef> = Vec::new();
        for i in 0..nodes.len() {
            let mut neighbors = Vec::new();
            for j in 0..nodes.len() {
                if i != j {
                    neighbors.push(nodes[j].clone());
                }
            }
            ret.set_links(nodes[i].clone(), neighbors);
        }

        ret
    }

    #[tokio::test]
    async fn test_serial_compaction() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(42);
        let iris_db = IrisDB::new_random_rng(32, &mut rng);
        let queries: Vec<_> = iris_db.db.into_iter().collect();

        let searcher = HnswSearcher::new_with_test_parameters_and_linear_scan_m4();
        let vector_store = &mut PlaintextStore::new();
        let graph_store = &mut GraphMem::<PlaintextVectorRef>::new();

        let vector_ids: Vec<_> = queries
            .iter()
            .enumerate()
            .map(|(id, query)| {
                vector_store.insert_with_id(VectorId::new(id as _, 3), Arc::new(query.clone()))
            })
            .collect();

        let l0 = make_layer::<PlaintextStore>(&vector_ids[..16]);
        let l1 = make_layer::<PlaintextStore>(&vector_ids[..l0.links.len() / 2]);
        let l2 = make_layer::<PlaintextStore>(&vector_ids[..l0.links.len() / 4]);

        let max_size = l0.links.len();
        let query_vector_id = vector_ids[max_size].clone();
        let new_l0: Vec<_> = (&vector_ids[..max_size + 1]).iter().cloned().collect();
        let compacted1 =
            neighborhood_compaction(vector_store, query_vector_id, &new_l0[..max_size], max_size)
                .await?;
        assert_eq!(compacted1.len(), max_size);

        let compacted2 =
            neighborhood_compaction(vector_store, query_vector_id, &new_l0, max_size).await?;

        assert_eq!(compacted2.len(), max_size);

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
