use super::{
    search::{self, SearchParams, SearchQueries, SearchResults},
    BothEyes, HawkActor, HawkRequest, HawkSessionRef, LEFT,
};
pub use crate::hawkers::aby3::aby3_store::VectorId;
use eyre::Result;

pub struct ResetRequests {
    pub vector_ids: Vec<VectorId>,
    pub queries: SearchQueries,
}

pub struct ResetPlan {
    pub vector_ids: Vec<VectorId>,
    pub search_results: SearchResults,
}

pub async fn search_to_reset(
    hawk_actor: &mut HawkActor,
    sessions: &BothEyes<Vec<HawkSessionRef>>,
    request: &HawkRequest,
) -> Result<ResetPlan> {
    // Get the reset updates from the request.
    let updates = {
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
