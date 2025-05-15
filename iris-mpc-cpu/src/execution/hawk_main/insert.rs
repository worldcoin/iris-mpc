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
    sessions: &[HawkSessionRef],
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlan>>,
    update_ids: &VecRequests<Option<VectorId>>,
) -> Result<VecRequests<Option<ConnectPlan>>> {
    let insert_plans = join_plans(plans);
    let mut connect_plans = vec![None; insert_plans.len()];

    // Parallel insertions are not supported, so only one session is needed.
    let mut session = sessions[0].write().await;

    for (plan, update_id, cp) in izip!(insert_plans, update_ids, &mut connect_plans) {
        if let Some(plan) = plan {
            *cp = Some(insert_one(&mut session, searcher, plan, *update_id).await?);
        }
    }
    Ok(connect_plans)
}

async fn insert_one(
    session: &mut HawkSession,
    searcher: &HnswSearcher,
    insert_plan: InsertPlan,
    update_id: Option<VectorId>,
) -> Result<ConnectPlan> {
    let inserted = {
        let storage = &mut session.aby3_store.storage;

        match update_id {
            None => storage.append(&insert_plan.query).await,
            Some(id) => storage.update(id, &insert_plan.query).await,
        }
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
    let insert_plans = join_plans(plans);
    let mut connect_plans = vec![None; insert_plans.len()];

    let mut session = session.write().await;

    for (plan, id, cp) in izip!(insert_plans, identifiers, &mut connect_plans) {
        if let Some(plan) = plan {
            *cp = Some(insert_one_with_id(plan, searcher, &mut session, id).await?);
        }
    }
    Ok(connect_plans)
}

/// Insert a query into the `Aby3Store` and `GraphMem` associated with `session`.
///
/// If `identifier` is `Some(id)`, then the node is inserted as this `VectorId`.
async fn insert_one_with_id(
    insert_plan: InsertPlan,
    searcher: &HnswSearcher,
    session: &mut HawkSession,
    id: VectorId,
) -> Result<ConnectPlan> {
    session
        .aby3_store
        .storage
        .insert(id, &insert_plan.query)
        .await;

    let mut graph_store = session.graph_store.write().await;

    let connect_plan = searcher
        .insert_prepare(
            &mut session.aby3_store,
            graph_store.deref(),
            id,
            insert_plan.links,
            insert_plan.set_ep,
        )
        .await?;

    graph_store.insert_apply(connect_plan.clone()).await;

    Ok(connect_plan)
}

/// Combine insert plans from parallel searches, repairing any conflict.
pub fn join_plans(mut plans: Vec<Option<InsertPlan>>) -> Vec<Option<InsertPlan>> {
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
