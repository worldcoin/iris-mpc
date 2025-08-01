use super::{
    rot::{Rotations, VecRots, WithRot},
    scheduler::{Batch, Schedule, TaskId},
    BothEyes, HawkInsertPlan, HawkSession, VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::{
        scheduler::{collect_results, parallelize},
        InsertPlanV, StoreId,
    },
    hawkers::aby3::aby3_store::{Aby3Query, Aby3Store},
    hnsw::{GraphMem, HnswSearcher},
};
use eyre::{OptionExt, Result};
use std::sync::Arc;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub type SearchQueries<ROT = WithRot> = Arc<BothEyes<VecRequests<VecRots<Aby3Query, ROT>>>>;
pub type SearchResults<ROT = WithRot> = BothEyes<VecRequests<VecRots<HawkInsertPlan, ROT>>>;

/// Identifiers of requests
pub type SearchIds = Arc<VecRequests<String>>;

#[derive(Clone)]
pub struct SearchParams {
    pub hnsw: Arc<HnswSearcher>,
    pub do_match: bool,
}

pub async fn search<ROT>(
    sessions: &BothEyes<Vec<HawkSession>>,
    search_queries: &SearchQueries<ROT>,
    search_ids: &SearchIds,
    search_params: SearchParams,
) -> Result<SearchResults<ROT>>
where
    ROT: Rotations,
{
    let n_sessions = sessions[LEFT].len();
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_requests = search_queries[LEFT].len();
    assert_eq!(n_requests, search_queries[RIGHT].len());

    let (tx, rx) = unbounded_channel::<(TaskId, HawkInsertPlan)>();

    let per_session = |batch: Batch| {
        let session = sessions[batch.i_eye][batch.i_session].clone();
        let search_queries = search_queries.clone();
        let search_ids = search_ids.clone();
        let search_params = search_params.clone();
        let tx = tx.clone();
        async move {
            per_session(
                &session,
                &search_queries,
                &search_ids,
                &search_params,
                tx,
                batch,
            )
            .await
        }
    };

    let schedule = Schedule::new(n_sessions, n_requests, ROT::N_ROTATIONS);

    parallelize(schedule.batches().into_iter().map(per_session)).await?;

    let results = collect_results(rx).await?;

    schedule.organize_results(results)
}

async fn per_session<ROT>(
    session: &HawkSession,
    search_queries: &SearchQueries<ROT>,
    search_ids: &SearchIds,
    search_params: &SearchParams,
    tx: UnboundedSender<(TaskId, HawkInsertPlan)>,
    batch: Batch,
) -> Result<()> {
    let mut vector_store = session.aby3_store.write().await;
    let graph_store = session.graph_store.clone().read_owned().await;

    for task in batch.tasks {
        let query = search_queries[batch.i_eye][task.i_request][task.i_rotation].clone();
        let mut insertion_layer = 0;
        if task.is_central {
            let query_uuid = search_ids
                .get(task.i_request)
                .ok_or_eyre("Invalid request id for uuid lookup")?
                .clone();
            let side: StoreId = batch.i_eye.try_into()?;
            let layer_selection_value = (query_uuid, side);
            insertion_layer = search_params
                .hnsw
                .select_layer_prf(&session.hnsw_prf_key, &layer_selection_value)?
        }
        let result = per_query(
            query,
            search_params,
            &mut vector_store,
            &graph_store,
            insertion_layer,
        )
        .await?;
        tx.send((task.id(), result))?;
    }

    Ok(())
}

async fn per_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store,
    graph_store: &GraphMem<Aby3Store>,
    insertion_layer: usize,
) -> Result<HawkInsertPlan> {
    let (links, set_ep) = search_params
        .hnsw
        .search_to_insert(aby3_store, graph_store, &query, insertion_layer)
        .await?;

    let match_count = if search_params.do_match {
        search_params.hnsw.match_count(aby3_store, &links).await?
    } else {
        0
    };

    Ok(HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links,
            set_ep,
        },
        match_count,
    })
}

/// Search for a single query with the given session and searcher, without
/// calculating the match count of the results.
///
/// (The `match_count` field returned is always set to 0.)
pub async fn search_single_query_no_match_count<H: std::hash::Hash>(
    session: HawkSession,
    query: Aby3Query,
    searcher: &HnswSearcher,
    identifier: &H,
) -> Result<InsertPlanV<Aby3Store>> {
    let mut store = session.aby3_store.write().await;
    let graph = session.graph_store.clone().read_owned().await;

    let insertion_layer = searcher.select_layer_prf(&session.hnsw_prf_key, identifier)?;

    let (links, set_ep) = searcher
        .search_to_insert(&mut *store, &graph, &query, insertion_layer)
        .await?;

    Ok(InsertPlanV {
        query,
        links,
        set_ep,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::setup_hawk_actors;
    use super::super::VectorId;
    use super::*;
    use crate::execution::hawk_main::test_utils::{init_graph, init_iris_db, make_request};
    use crate::execution::hawk_main::{HawkActor, Orientation};

    #[tokio::test]
    async fn test_search() -> Result<()> {
        let actors = setup_hawk_actors().await?;

        parallelize(actors.into_iter().map(go_search)).await?;

        Ok(())
    }

    async fn go_search(mut actor: HawkActor) -> Result<HawkActor> {
        init_iris_db(&mut actor).await?;
        init_graph(&mut actor).await?;

        let [sessions, _mirror] = actor.new_sessions_orient().await?;
        HawkSession::state_check([&sessions[LEFT][0], &sessions[RIGHT][0]]).await?;

        let batch_size = 3;
        let request = make_request(batch_size, actor.party_id);
        let search_queries = &request.queries(Orientation::Normal);
        let search_params = SearchParams {
            hnsw: actor.searcher(),
            do_match: true,
        };

        let result = search(&sessions, search_queries, &request.ids, search_params).await?;

        for side in result {
            assert_eq!(side.len(), batch_size);
            for (i, rotations) in side.iter().enumerate() {
                // Match because i from make_request is the same as i from init_db.
                assert_eq!(rotations.center().match_count, 1);
                assert_eq!(
                    rotations.center().plan.links[0].edges[0].0,
                    VectorId::from_0_index(i as u32)
                );
            }
        }

        Ok(actor)
    }
}
