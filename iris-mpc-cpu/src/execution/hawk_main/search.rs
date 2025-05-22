use super::{
    rot::{Rotations, VecRots, WithRot},
    scheduler::{Batch, Schedule, TaskId},
    BothEyes, HawkSession, HawkSessionRef, InsertPlan, VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::scheduler::{collect_results, parallelize},
    hawkers::aby3::aby3_store::{Aby3Store, QueryRef},
    hnsw::{GraphMem, HnswSearcher},
};
use eyre::Result;
use std::sync::Arc;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub type SearchQueries<ROT = WithRot> = Arc<BothEyes<VecRequests<VecRots<QueryRef, ROT>>>>;
pub type SearchResults<ROT = WithRot> = BothEyes<VecRequests<VecRots<InsertPlan, ROT>>>;

#[derive(Clone)]
pub struct SearchParams {
    pub hnsw: Arc<HnswSearcher>,
    pub do_match: bool,
}

pub async fn search<ROT>(
    sessions: &BothEyes<Vec<HawkSessionRef>>,
    search_queries: &SearchQueries<ROT>,
    search_params: SearchParams,
) -> Result<SearchResults<ROT>>
where
    ROT: Rotations,
{
    let n_sessions = sessions[LEFT].len();
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_requests = search_queries[LEFT].len();
    assert_eq!(n_requests, search_queries[RIGHT].len());

    let (tx, rx) = unbounded_channel::<(TaskId, InsertPlan)>();

    let per_session = |batch: Batch| {
        let session = sessions[batch.i_eye][batch.i_session].clone();
        let search_queries = search_queries.clone();
        let search_params = search_params.clone();
        let tx = tx.clone();
        async move {
            let mut session = session.write().await;
            per_session(&mut session, &search_queries, &search_params, tx, batch).await
        }
    };

    let schedule = Schedule::new(n_sessions, n_requests, ROT::n_rotations());

    parallelize(schedule.batches().into_iter().map(per_session)).await?;

    let results = collect_results(rx).await?;

    schedule.organize_results(results)
}

async fn per_session<ROT>(
    session: &mut HawkSession,
    search_queries: &BothEyes<VecRequests<VecRots<QueryRef, ROT>>>,
    search_params: &SearchParams,
    tx: UnboundedSender<(TaskId, InsertPlan)>,
    batch: Batch,
) -> Result<()> {
    let graph_store = session.graph_store.clone().read_owned().await;

    for task in batch.tasks {
        let query = search_queries[batch.i_eye][task.i_request][task.i_rotation].clone();
        let result = per_query(session, query, search_params, &graph_store).await?;
        tx.send((task.id(), result))?;
    }

    Ok(())
}

async fn per_query(
    session: &mut HawkSession,
    query: QueryRef,
    search_params: &SearchParams,
    graph_store: &GraphMem<Aby3Store>,
) -> Result<InsertPlan> {
    let insertion_layer = search_params.hnsw.select_layer(&mut session.shared_rng)?;

    let (links, set_ep) = search_params
        .hnsw
        .search_to_insert(
            &mut session.aby3_store,
            graph_store,
            &query,
            insertion_layer,
        )
        .await?;

    let match_count = if search_params.do_match {
        search_params
            .hnsw
            .match_count(&mut session.aby3_store, &links)
            .await?
    } else {
        0
    };

    Ok(InsertPlan {
        query,
        links,
        match_count,
        set_ep,
    })
}

/// Search for a single query with the given session and searcher, without
/// calculating the match count of the results.
///
/// (The `match_count` field returned is always set to 0.)
pub async fn search_single_query_no_match_count(
    session: HawkSessionRef,
    query: QueryRef,
    searcher: &HnswSearcher,
) -> Result<InsertPlan> {
    let mut session = session.write().await;

    let graph = session.graph_store.clone().read_owned().await;
    let insertion_layer = searcher.select_layer(&mut session.shared_rng)?;

    let (links, set_ep) = searcher
        .search_to_insert(&mut session.aby3_store, &graph, &query, insertion_layer)
        .await?;

    Ok(InsertPlan {
        query,
        links,
        // TODO consider refactoring this field out of InsertPlan
        match_count: 0,
        set_ep,
    })
}
