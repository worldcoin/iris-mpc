use std::ops::Deref;

use crate::hnsw::HnswSearcher;

use super::{ConnectPlan, HawkSession, HawkSessionRef, InsertPlan, VecRequests};

use eyre::Result;
use iris_mpc_common::vector_id::VectorId;
use itertools::{izip, Itertools};

/// Insert a collection of `InsertPlan` structs into the graph and vector store represented
/// by `session`, adjusting the insertion plans as needed to repair any conflict from
/// parallel searches.
pub async fn insert(
    plans: VecRequests<Option<InsertPlan>>,
    searcher: &HnswSearcher,
    session: &HawkSessionRef,
) -> Result<VecRequests<Option<ConnectPlan>>> {
    insert_several(plans, None, searcher, session).await
}

/// Insert a collection of `InsertPlan` structs into the graph and vector store represented
/// by `session`, adjusting the insertion plans as needed to repair any conflict from
/// parallel searches.
/// 
/// The `identifiers` argument gives explicit `VectorId`s which will be used for the newly
/// inserted entries, for use when the insertions need to match with a pre-existing database
/// of iris codes with associated identifiers.
pub async fn insert_with_ids(
    plans: VecRequests<Option<InsertPlan>>,
    identifiers: VecRequests<VectorId>,
    searcher: &HnswSearcher,
    session: &HawkSessionRef,
) -> Result<VecRequests<Option<ConnectPlan>>> {
    insert_several(plans, Some(identifiers), searcher, session).await
}

async fn insert_several(
    plans: VecRequests<Option<InsertPlan>>,
    identifiers: Option<VecRequests<VectorId>>,
    searcher: &HnswSearcher,
    session: &HawkSessionRef,
) -> Result<VecRequests<Option<ConnectPlan>>> {
    let insert_plans = join_plans(plans);
    let mut connect_plans = vec![None; insert_plans.len()];

    let mut session = session.write().await;

    // Iterator returns `Some(id)` if identifiers specified, otherwise repeats `None`
    let id_iter: Box<dyn Iterator<Item = Option<_>> + Send> = match identifiers {
        Some(v) => Box::new(v.into_iter().map(Some)),
        None => Box::new(std::iter::repeat(None)),
    };

    for (plan, id, cp) in izip!(insert_plans, id_iter, &mut connect_plans) {
        if let Some(plan) = plan {
            *cp = Some(insert_one(plan, searcher, &mut session, id).await?);
        }
    }
    Ok(connect_plans)
}

/// Insert a query into the `Aby3Store` and `GraphMem` associated with `session`.
/// 
/// If `identifier` is `Some(id)`, then the node is inserted as this `VectorId`.
async fn insert_one(
    insert_plan: InsertPlan,
    searcher: &HnswSearcher,
    session: &mut HawkSession,
    identifier: Option<VectorId>,
) -> Result<ConnectPlan> {
    let inserted = if let Some(id) = identifier {
        session.aby3_store.storage.insert_with_id(id, &insert_plan.query).await;
        id
    } else {
        session.aby3_store.storage.insert(&insert_plan.query).await
    };
    let mut graph_store = session.graph_store.write().await;

    let connect_plan = searcher
        .insert_prepare(
            &mut session.aby3_store,
            graph_store.deref(),
            inserted,
            insert_plan.links,
            insert_plan.set_ep,
        )
        .await?;

    graph_store.insert_apply(connect_plan.clone()).await;

    Ok(connect_plan)
}

/// Combine insert plans from parallel searches, repairing any conflict.
fn join_plans(mut plans: Vec<Option<InsertPlan>>) -> Vec<Option<InsertPlan>> {
    let set_ep = plans.iter().flatten().any(|plan| plan.set_ep);
    if set_ep {
        // There can be at most one new entry point.
        let highest = plans
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
                plan.set_ep = i == highest;
            }
        }
    }
    plans
}
