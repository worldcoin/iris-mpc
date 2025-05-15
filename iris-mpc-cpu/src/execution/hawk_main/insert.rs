use std::ops::Deref;

use crate::hnsw::HnswSearcher;

use super::{ConnectPlan, HawkSession, HawkSessionRef, InsertPlan, VecRequests};

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
pub async fn insert(
    session: &HawkSessionRef,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlan>>,
    ids: &VecRequests<Option<VectorId>>,
) -> Result<VecRequests<Option<ConnectPlan>>> {
    let insert_plans = join_plans(plans);
    let mut connect_plans = vec![None; insert_plans.len()];

    // Parallel insertions are not supported, so only one session is needed.
    let mut session = session.write().await;

    for (plan, update_id, cp) in izip!(insert_plans, ids, &mut connect_plans) {
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
    insert_id: Option<VectorId>,
) -> Result<ConnectPlan> {
    let inserted = {
        let storage = &mut session.aby3_store.storage;

        match insert_id {
            None => storage.append(&insert_plan.query).await,
            Some(id) => storage.insert(id, &insert_plan.query).await,
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
