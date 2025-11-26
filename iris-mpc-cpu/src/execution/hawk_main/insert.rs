use crate::hnsw::{
    graph::neighborhood::Neighborhood,
    searcher::{ConnectPlanV, SetEntryPoint},
    vector_store::VectorStoreMut,
    GraphMem, HnswSearcher, SortedNeighborhood, VectorStore,
};

use super::VecRequests;

use eyre::Result;
use itertools::{izip, Itertools};

/// InsertPlan specifies where a query may be inserted into the HNSW graph.
/// That is lists of neighbors for each layer.
#[derive(Debug)]
pub struct InsertPlanV<V: VectorStore> {
    pub query: V::QueryRef,
    pub links: Vec<SortedNeighborhood<V>>,
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

            *cp = {
                let connect_plan = searcher
                    .insert_prepare(store, graph, inserted, extended_links, set_ep)
                    .await?;

                graph.insert_apply(connect_plan.clone()).await;

                Some(connect_plan)
            };

            inserted_ids.push(cp.as_ref().unwrap().inserted_vector.clone());
        }
    }

    Ok(connect_plans)
}

/// Extends the bottom layer of links with additional vectors in `extra_ids` if there is room.
async fn add_batch_neighbors<V: VectorStore>(
    store: &mut V,
    query: &V::QueryRef,
    mut links: Vec<SortedNeighborhood<V>>,
    extra_ids: &[V::VectorRef],
    target_n_neighbors: usize,
) -> Result<Vec<SortedNeighborhood<V>>> {
    if let Some(bottom_layer) = links.first_mut() {
        if bottom_layer.as_ref().len() < target_n_neighbors {
            let distances = store.eval_distance_batch(query, extra_ids).await?;

            let ids_dists = izip!(extra_ids.iter().cloned(), distances)
                .map(|(id, dist)| (id, dist))
                .collect_vec();

            bottom_layer
                .insert_batch_and_trim(&mut *store, &ids_dists, None)
                .await?;
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
