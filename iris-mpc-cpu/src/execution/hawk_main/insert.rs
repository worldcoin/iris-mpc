use std::ops::Deref;

use crate::hnsw::{
    graph::neighborhood::SortedNeighborhoodV, searcher::ConnectPlanV, GraphMem, HnswSearcher,
    VectorStore,
};

use super::{ConnectPlan, HawkSession, InsertPlan, VecRequests};

use eyre::Result;
use iris_mpc_common::vector_id::VectorId;
use itertools::{izip, Itertools};

/// Insert a collection `plans` of `InsertPlan` structs into the graph and vector store
/// represented by `session`, adjusting the insertion plans as needed to repair any conflict
/// from parallel searches.
///
/// The `ids` argument consists of `Option<VectorId>`s which are `Some(id)` if the associated
/// plan is to be inserted with a specific identifier (e.g. for updates or for insertions
/// which need to parallel an existing iris code database), and `None` if the associated plan
/// is to be inserted at the next available serial ID, with version 0.
///
/// Parallel insertions are not supported, so only one session is needed.
pub async fn insert(
    session: &HawkSession,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlan>>,
    ids: &VecRequests<Option<VectorId>>,
) -> Result<VecRequests<Option<ConnectPlan>>> {
    let insert_plans = join_plans(plans);
    let mut connect_plans = vec![None; insert_plans.len()];
    let mut inserted_ids = vec![];
    let m = searcher.params.get_M(0);

    for (plan, update_id, cp) in izip!(insert_plans, ids, &mut connect_plans) {
        if let Some(InsertPlan {
            query,
            links,
            set_ep,
            ..
        }) = plan
        {
            let mut store = session.aby3_store.write().await;

            let extended_links =
                add_batch_neighbors(&mut *store, &query, links, &inserted_ids, m).await?;

            let inserted = {
                let storage = &mut store.storage;

                match update_id {
                    None => storage.append(&query.iris).await,
                    Some(id) => storage.insert(*id, &query.iris).await,
                }
            };

            let graph = &mut session.graph_store.write().await;
            *cp = Some(
                insert_one(
                    &mut *store,
                    graph,
                    searcher,
                    inserted,
                    extended_links,
                    set_ep,
                )
                .await?,
            );

            inserted_ids.push(cp.as_ref().unwrap().inserted_vector);
        }
    }

    Ok(connect_plans)
}

/// Extends the bottom layer of links with additional vectors in `extra_ids` if there
/// is room.
async fn add_batch_neighbors<V: VectorStore>(
    store: &mut V,
    query: &V::QueryRef,
    mut links: Vec<SortedNeighborhoodV<V>>,
    extra_ids: &[V::VectorRef],
    target_n_neighbors: usize,
) -> Result<Vec<SortedNeighborhoodV<V>>> {
    if let Some(bottom_layer) = links.first_mut() {
        if bottom_layer.len() < target_n_neighbors {
            let distances = store
                .eval_distance_batch(&[query.clone()], extra_ids)
                .await?;

            let ids_dists = izip!(extra_ids.iter().cloned(), distances)
                .map(|(id, dist)| (id, dist))
                .collect_vec();

            bottom_layer.insert_batch(&mut *store, &ids_dists).await?;
        }
    }

    Ok(links)
}

/// Insert a single `InsertPlan` into the vector store and graph of `session`.  If
/// `insert_id` is `Some(id)`, then `id` is used as the identifier for the inserted
/// query.  Otherwise, the query is inserted at the next unused serial id, as version 0.
async fn insert_one<V: VectorStore>(
    // session: &mut HawkSession,
    store: &mut V,
    graph: &mut GraphMem<V>,
    searcher: &HnswSearcher,
    inserted: V::VectorRef,
    links: Vec<SortedNeighborhoodV<V>>,
    set_ep: bool,
) -> Result<ConnectPlanV<V>> {
    let connect_plan = searcher
        .insert_prepare(store, graph.deref(), inserted, links, set_ep)
        .await?;

    graph.insert_apply(connect_plan.clone()).await;

    Ok(connect_plan)
}

/// Combine insert plans from parallel searches, repairing any conflict.
pub fn join_plans(mut plans: Vec<Option<InsertPlan>>) -> Vec<Option<InsertPlan>> {
    let set_ep = plans.iter().flatten().any(|plan| plan.set_ep);
    if set_ep {
        // Find a unique instance of the max insertion layer.
        let set_ep_idx = plans
            .iter()
            .map(|plan| match plan {
                Some(plan) => plan.links.len(),
                None => 0,
            })
            .position_max()
            .unwrap();

        // Set the entry point to false for all but the highest.
        for (i, plan) in plans.iter_mut().enumerate() {
            if let Some(plan) = plan {
                plan.set_ep = i == set_ep_idx;
            }
        }
    }
    plans
}
