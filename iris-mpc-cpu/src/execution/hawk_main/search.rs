use super::{
    rot::VecRots,
    scheduler::{schedule, Batch, TaskId},
    BothEyes, HawkSession, HawkSessionRef, InsertPlan, VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::scheduler::{collect_results, parallelize},
    hawkers::aby3::aby3_store::{Aby3Store, QueryRef},
    hnsw::{GraphMem, HnswSearcher},
};
use eyre::Result;
use iris_mpc_common::ROTATIONS;
use std::sync::Arc;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub async fn search(
    sessions: &BothEyes<Vec<HawkSessionRef>>,
    search_queries: &Arc<BothEyes<VecRequests<VecRots<QueryRef>>>>,
    search_params: Arc<HnswSearcher>,
) -> Result<BothEyes<VecRequests<VecRots<InsertPlan>>>> {
    let n_sessions = sessions[LEFT].len();
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_eyes = 2;
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

    let sched = schedule(n_sessions, n_eyes, n_requests, ROTATIONS);

    parallelize(sched.batches.iter().cloned().map(per_session)).await?;

    let results = collect_results(rx).await?;

    sched.organize_results(results)
}

async fn per_session(
    session: &mut HawkSession,
    search_queries: &BothEyes<VecRequests<VecRots<QueryRef>>>,
    search_params: &HnswSearcher,
    tx: UnboundedSender<(TaskId, InsertPlan)>,
    batch: Batch,
) {
    let graph_store = session.graph_store.clone().read_owned().await;

    for task in batch.tasks {
        let query = search_queries[batch.i_eye][task.i_request][task.i_rotation].clone();
        let result = per_query(session, query, search_params, &graph_store).await;
        tx.send((task.id(), result)).expect("infallible send");
    }
}

async fn per_query(
    session: &mut HawkSession,
    query: QueryRef,
    search_params: &HnswSearcher,
    graph_store: &GraphMem<Aby3Store>,
) -> InsertPlan {
    let insertion_layer = search_params.select_layer(&mut session.shared_rng);

    let (links, set_ep) = search_params
        .search_to_insert(
            &mut session.aby3_store,
            graph_store,
            &query,
            insertion_layer,
        )
        .await;

    let match_count = search_params
        .match_count(&mut session.aby3_store, &links)
        .await;

    InsertPlan {
        query,
        links,
        match_count,
        set_ep,
    }
}
