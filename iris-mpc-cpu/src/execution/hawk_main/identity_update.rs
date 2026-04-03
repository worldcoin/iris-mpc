use std::time::Instant;

use super::{
    rot::CENTER_ONLY_MASK,
    search::{self, SearchParams, SearchQueries, SearchResults},
    BothEyes, HawkActor, HawkRequest, HawkSession, LEFT, RIGHT,
};
use crate::execution::hawk_main::{
    iris_worker::{IrisWorkerPool, QueryId},
    search::SearchIds,
    NEIGHBORHOOD_MODE,
};
use eyre::Result;
use iris_mpc_common::vector_id::VectorId;

pub struct IdentityUpdateRequests {
    pub vector_ids: Vec<VectorId>,
    pub request_ids: SearchIds,
    pub queries: SearchQueries<{ CENTER_ONLY_MASK }>,
}

pub struct IdentityUpdatePlan {
    pub vector_ids: Vec<VectorId>,
    pub search_results: SearchResults<{ CENTER_ONLY_MASK }>,
    /// QueryIds from identity update irises, for eviction after batch processing.
    pub cached_query_ids: Vec<QueryId>,
}

pub async fn search_to_identity_update(
    hawk_actor: &mut HawkActor,
    sessions: &BothEyes<Vec<HawkSession>>,
    request: &HawkRequest,
) -> Result<IdentityUpdatePlan> {
    let start = Instant::now();

    let (updates, id_update_cache) = {
        let reg = hawk_actor.registry[LEFT].read().await;
        request.identity_updates(&reg)
    };
    // Cache identity update irises in the worker pools.
    futures::try_join!(
        hawk_actor.worker_pools[LEFT].cache_queries(id_update_cache[LEFT].clone()),
        hawk_actor.worker_pools[RIGHT].cache_queries(id_update_cache[RIGHT].clone()),
    )?;

    let search_params = SearchParams::new_no_match(hawk_actor.searcher());

    // Search the central rotation to determine how to insert the update vectors.
    let search_results = search::search::<{ CENTER_ONLY_MASK }>(
        sessions,
        &updates.queries,
        &updates.request_ids,
        search_params,
        NEIGHBORHOOD_MODE,
    )
    .await?;

    // Collect all identity update query IDs for eviction.
    let cached_query_ids: Vec<QueryId> = id_update_cache[LEFT]
        .iter()
        .chain(id_update_cache[RIGHT].iter())
        .map(|(qid, _)| *qid)
        .collect();

    metrics::histogram!("search_to_identity_update_duration").record(start.elapsed().as_secs_f64());
    Ok(IdentityUpdatePlan {
        vector_ids: updates.vector_ids,
        search_results,
        cached_query_ids,
    })
}

pub async fn apply_deletions(hawk_actor: &mut HawkActor, request: &HawkRequest) -> Result<()> {
    if hawk_actor.args.hnsw_disable_memory_persistence {
        tracing::debug!("In-memory persistence disabled, skipping deletions");
        return Ok(());
    }

    let del_ids = {
        let reg = hawk_actor.registry[LEFT].read().await;
        request.deletion_ids(&reg)
    };

    if del_ids.is_empty() {
        return Ok(());
    }

    // Workers write party-specific dummy sentinels at the deleted VectorIds.
    futures::try_join!(
        hawk_actor.worker_pools[LEFT].delete_irises(hawk_actor.party_id, del_ids.clone()),
        hawk_actor.worker_pools[RIGHT].delete_irises(hawk_actor.party_id, del_ids.clone()),
    )?;

    // Update registries (metadata only — version bump + checksum).
    let mut registries = [
        hawk_actor.registry[LEFT].write().await,
        hawk_actor.registry[RIGHT].write().await,
    ];
    for del_id in del_ids {
        for reg in registries.iter_mut() {
            reg.update(del_id, ());
        }
    }
    Ok(())
}
