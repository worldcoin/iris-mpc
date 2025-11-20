use crate::hnsw::{
    graph::neighborhood::{Neighborhood, SortedNeighborhoodV},
    searcher::ConnectPlanV,
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
    pub set_ep: bool,
}

// Manual implementation of Clone for InsertPlanV, since derive(Clone) does not
// propagate the nested Clone bounds on V::QueryRef via TransientRef.
impl<V: VectorStore> Clone for InsertPlanV<V> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            links: self.links.clone(),
            set_ep: self.set_ep,
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
    let set_ep = plans.iter().flatten().any(|plan| plan.set_ep);
    if set_ep {
        let insertion_layers = plans
            .iter()
            .map(|plan| match plan {
                Some(plan) => plan.links.len(),
                None => 0,
            })
            .collect_vec();

        let max_insertion_layer = *insertion_layers.iter().max().unwrap();

        let set_ep_idx = insertion_layers
            .into_iter()
            .position(|layer| layer == max_insertion_layer)
            .unwrap();

        // Set the entry point to false for all but the first instance of the
        // highest insertion layer.
        for (i, plan) in plans.iter_mut().enumerate() {
            if let Some(plan) = plan {
                plan.set_ep = i == set_ep_idx;
            }
        }
    }
    plans
}
