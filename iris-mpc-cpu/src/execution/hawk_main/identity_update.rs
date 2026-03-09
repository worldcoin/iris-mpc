use std::{sync::Arc, time::Instant};

use super::{
    rot::CENTER_ONLY_MASK,
    search::{self, SearchParams, SearchQueries, SearchResults},
    BothEyes, HawkActor, HawkRequest, HawkSession, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::{search::SearchIds, NEIGHBORHOOD_MODE},
    protocol::shared_iris::GaloisRingSharedIris,
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
}

pub async fn search_to_identity_update(
    hawk_actor: &mut HawkActor,
    sessions: &BothEyes<Vec<HawkSession>>,
    request: &HawkRequest,
) -> Result<IdentityUpdatePlan> {
    let start = Instant::now();

    let updates = {
        let store = hawk_actor.iris_store[LEFT].read().await;
        request.identity_updates(&store)
    };

    let search_params = SearchParams {
        hnsw: hawk_actor.searcher(),
        do_match: false,
    };

    // Search the central rotation to determine how to insert the update vectors.
    let search_results = search::search::<{ CENTER_ONLY_MASK }>(
        sessions,
        &updates.queries,
        &updates.request_ids,
        search_params,
        NEIGHBORHOOD_MODE,
    )
    .await?;

    metrics::histogram!("search_to_identity_update_duration").record(start.elapsed().as_secs_f64());
    Ok(IdentityUpdatePlan {
        vector_ids: updates.vector_ids,
        search_results,
    })
}

pub async fn apply_deletions(hawk_actor: &mut HawkActor, request: &HawkRequest) -> Result<()> {
    if hawk_actor.args.hnsw_disable_memory_persistence {
        tracing::debug!("In-memory persistence disabled, skipping deletions");
        return Ok(());
    }

    let dummy = Arc::new(GaloisRingSharedIris::dummy_for_party(hawk_actor.party_id));

    let mut stores = [
        hawk_actor.iris_store[LEFT].write().await,
        hawk_actor.iris_store[RIGHT].write().await,
    ];

    let del_ids = request.deletion_ids(&stores[LEFT]);

    for del_id in del_ids {
        for store in &mut stores {
            store.update(del_id, dummy.clone());
        }
    }
    Ok(())
}
