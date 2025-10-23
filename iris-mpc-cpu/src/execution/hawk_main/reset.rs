use std::{sync::Arc, time::Instant};

use super::{
    rot::CenterOnly,
    search::{self, SearchParams, SearchQueries, SearchResults},
    BothEyes, HawkActor, HawkRequest, HawkSession, LEFT, RIGHT,
};
use crate::{execution::hawk_main::search::SearchIds, protocol::shared_iris::GaloisRingSharedIris};
use eyre::Result;
use iris_mpc_common::vector_id::VectorId;

pub struct ResetRequests {
    pub vector_ids: Vec<VectorId>,
    pub request_ids: SearchIds,
    pub queries: SearchQueries<CenterOnly>,
}

pub struct ResetPlan {
    pub vector_ids: Vec<VectorId>,
    pub search_results: SearchResults<CenterOnly>,
}

pub async fn search_to_reset(
    hawk_actor: &mut HawkActor,
    sessions: &BothEyes<Vec<HawkSession>>,
    request: &HawkRequest,
) -> Result<ResetPlan> {
    let start = Instant::now();

    // Get the reset updates from the request.
    let updates = {
        // The store to find vector ids (same left or right).
        let store = hawk_actor.iris_store[LEFT].read().await;
        request.reset_updates(&store)
    };

    let search_params = SearchParams {
        hnsw: hawk_actor.searcher(),
        do_match: false,
    };

    let search_results = search::search::<CenterOnly>(
        sessions,
        &updates.queries,
        &updates.request_ids,
        search_params,
    )
    .await?;

    metrics::histogram!("search_to_reset_duration").record(start.elapsed().as_secs_f64());
    Ok(ResetPlan {
        vector_ids: updates.vector_ids,
        search_results,
    })
}

pub async fn apply_deletions(hawk_actor: &mut HawkActor, request: &HawkRequest) -> Result<()> {
    let dummy = Arc::new(GaloisRingSharedIris::dummy_for_party(hawk_actor.party_id));

    let mut stores = [
        hawk_actor.iris_store[LEFT].write().await,
        hawk_actor.iris_store[RIGHT].write().await,
    ];

    // Map the deletion indices to IDs of the iris stores (same left or right).
    let del_ids = request.deletion_ids(&stores[LEFT]);

    for del_id in del_ids {
        for store in &mut stores {
            store.update(del_id, dummy.clone());
        }
    }
    Ok(())
}
