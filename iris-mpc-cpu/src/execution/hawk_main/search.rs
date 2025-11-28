use super::{
    rot::{Rotations, VecRotationSupport},
    scheduler::{Batch, Schedule, TaskId},
    BothEyes, HawkInsertPlan, HawkSession, VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::{
        scheduler::{collect_results, parallelize},
        InsertPlanV, StoreId,
    },
    hawkers::aby3::aby3_store::{Aby3Query, Aby3Store, Aby3VectorRef},
    hnsw::{
        graph::neighborhood::{Neighborhood, UnsortedNeighborhood},
        searcher::{NeighborhoodMode, UpdateEntryPoint},
        GraphMem, HnswSearcher, SortedNeighborhood,
    },
};
use eyre::{OptionExt, Result};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub type SearchQueries<ROT> = Arc<BothEyes<VecRequests<VecRotationSupport<Aby3Query, ROT>>>>;
pub type SearchResults<ROT> = BothEyes<VecRequests<VecRotationSupport<HawkInsertPlan, ROT>>>;

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
    mode: NeighborhoodMode,
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
        let mode = mode.clone();

        async move {
            match mode {
                NeighborhoodMode::Sorted => {
                    per_session::<_, SortedNeighborhood<_>>(
                        &session,
                        &search_queries,
                        &search_ids,
                        &search_params,
                        tx,
                        batch,
                    )
                    .await
                }
                NeighborhoodMode::Unsorted => {
                    per_session::<_, UnsortedNeighborhood<_>>(
                        &session,
                        &search_queries,
                        &search_ids,
                        &search_params,
                        tx,
                        batch,
                    )
                    .await
                }
            }
        }
    };

    let schedule = Schedule::new(n_sessions, n_requests, ROT::N_ROTATIONS);

    parallelize(schedule.search_batches().into_iter().map(per_session)).await?;

    let results = schedule.organize_results(collect_results(rx).await?)?;

    Ok(results)
}

async fn per_session<ROT, N: Neighborhood<Aby3Store>>(
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
        let result = if task.is_central {
            // search_to_insert for centers
            let query_uuid = search_ids
                .get(task.i_request)
                .ok_or_eyre("Invalid request id for uuid lookup")?
                .clone();
            let side: StoreId = batch.i_eye.try_into()?;
            let layer_selection_value = (query_uuid, side);
            let insertion_layer = search_params
                .hnsw
                .gen_layer_prf(&session.hnsw_prf_key, &layer_selection_value)?;
            per_insert_query::<N>(
                query,
                search_params,
                &mut vector_store,
                &graph_store,
                insertion_layer,
            )
            .await?
        } else {
            // plain search for non-centers
            per_search_query(query, search_params, &mut vector_store, &graph_store).await?
        };

        tx.send((task.id(), result))?;
    }

    Ok(())
}

async fn per_insert_query<N: Neighborhood<Aby3Store>>(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store,
    graph_store: &GraphMem<Aby3VectorRef>,
    insertion_layer: usize,
) -> Result<HawkInsertPlan> {
    let start = Instant::now();

    let (links, update_ep) = search_params
        .hnsw
        .search_to_insert::<_, N>(aby3_store, graph_store, &query, insertion_layer)
        .await?;

    let matches = if search_params.do_match {
        search_params.hnsw.matches(aby3_store, &links).await?
    } else {
        vec![]
    };

    // Trim and extract unstructured vector lists
    let mut links_unstructured = Vec::new();
    for (lc, mut l) in links.iter().cloned().enumerate() {
        let m = search_params.hnsw.params.get_M(lc);
        l.trim(aby3_store, Some(m)).await?;
        links_unstructured.push(l.edge_ids())
    }

    metrics::histogram!("search_query_duration").record(start.elapsed().as_secs_f64());
    Ok(HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links: links_unstructured,
            update_ep,
        },
        matches,
    })
}

async fn per_search_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store,
    graph_store: &GraphMem<Aby3VectorRef>,
) -> Result<HawkInsertPlan> {
    let start = Instant::now();

    let layer_0_neighbors = search_params
        .hnsw
        .search::<_, SortedNeighborhood<_>>(
            aby3_store,
            graph_store,
            &query,
            search_params.hnsw.params.get_ef_search(0),
        )
        .await?;

    let links_unstructured = vec![layer_0_neighbors.edge_ids()];
    let links = vec![layer_0_neighbors];

    let matches = if search_params.do_match {
        search_params.hnsw.matches(aby3_store, &links).await?
    } else {
        vec![]
    };

    metrics::histogram!("search_query_duration").record(start.elapsed().as_secs_f64());
    Ok(HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links: links_unstructured,
            update_ep: UpdateEntryPoint::False,
        },
        matches,
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

    let insertion_layer = searcher.gen_layer_prf(&session.hnsw_prf_key, identifier)?;

    let (links, update_ep) = searcher
        .search_to_insert::<_, SortedNeighborhood<_>>(&mut *store, &graph, &query, insertion_layer)
        .await?;

    // Trim and extract unstructured vector lists
    let mut links_unstructured = Vec::new();
    for (lc, mut l) in links.iter().cloned().enumerate() {
        let m = searcher.params.get_M(lc);
        l.trim(&mut store, Some(m)).await?;
        links_unstructured.push(l.edge_ids());
    }

    Ok(InsertPlanV {
        query,
        links: links_unstructured,
        update_ep,
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

        let sessions = actor.new_sessions().await?;
        HawkSession::state_check([&sessions[LEFT][0], &sessions[RIGHT][0]]).await?;

        let batch_size = 3;
        let request = make_request(batch_size, actor.party_id);
        let search_queries = &request.queries(Orientation::Normal);
        let search_params = SearchParams {
            hnsw: actor.searcher(),
            do_match: true,
        };

        let result = search(
            &sessions,
            search_queries,
            &request.ids,
            search_params,
            NeighborhoodMode::Sorted,
        )
        .await?;

        for side in result {
            assert_eq!(side.len(), batch_size);
            for (i, rotations) in side.iter().enumerate() {
                // Match because i from make_request is the same as i from init_db.
                assert_eq!(rotations.center().matches.len(), 1);
                assert!(rotations
                    .center()
                    .matches
                    .iter()
                    .position(|(v, _)| *v == VectorId::from_0_index(i as u32))
                    .is_some());
            }
        }
        actor.sync_peers().await?;
        Ok(actor)
    }
}
