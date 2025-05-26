use std::sync::Arc;

use super::{
    rot::WithoutRot,
    search::{self, SearchParams, SearchQueries, SearchResults},
    BothEyes, HawkActor, HawkRequest, HawkSessionRef, LEFT, RIGHT,
};
pub use crate::hawkers::aby3::aby3_store::VectorId;
use crate::protocol::shared_iris::GaloisRingSharedIris;
use eyre::Result;

pub struct ResetRequests {
    pub vector_ids: Vec<VectorId>,
    pub queries: SearchQueries<WithoutRot>,
}

pub struct ResetPlan {
    pub vector_ids: Vec<VectorId>,
    pub search_results: SearchResults<WithoutRot>,
}

pub async fn search_to_reset(
    hawk_actor: &mut HawkActor,
    sessions: &BothEyes<Vec<HawkSessionRef>>,
    request: &HawkRequest,
) -> Result<ResetPlan> {
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

    Ok(ResetPlan {
        vector_ids: updates.vector_ids,
        search_results: search::search(sessions, &updates.queries, search_params).await?,
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
            store.overwrite(del_id, dummy.clone());
        }
    }
    Ok(())
}
