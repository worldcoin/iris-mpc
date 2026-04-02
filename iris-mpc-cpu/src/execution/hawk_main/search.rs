use super::{
    rot::VecRotationSupport,
    scheduler::{Batch, Schedule, TaskId},
    BothEyes, ClassifiedMatches, HawkInsertPlan, HawkOps, HawkSession, SaturableMatches,
    VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::{
        scheduler::{collect_results, parallelize},
        InsertPlanV, StoreId,
    },
    hawkers::aby3::aby3_store::{Aby3DistanceRef, Aby3Query, Aby3Store, Aby3VectorRef},
    hnsw::{
        graph::neighborhood::{Neighborhood, UnsortedNeighborhood},
        searcher::{NeighborhoodMode, UpdateEntryPoint},
        GraphMem, HnswSearcher, SortedNeighborhood,
    },
};
use eyre::{OptionExt, Result};
use iris_mpc_common::iris_db::iris::Threshold;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tracing::instrument;

pub type SearchQueries<const ROTMASK: u32> =
    Arc<BothEyes<VecRequests<VecRotationSupport<Aby3Query, ROTMASK>>>>;
pub type SearchResults<const ROTMASK: u32> =
    BothEyes<VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>>;

/// Identifiers of requests
pub type SearchIds = Arc<VecRequests<String>>;

#[derive(Clone)]
pub struct SearchParams {
    pub hnsw: Arc<HnswSearcher>,
    /// Searcher with layer-0 ef params overridden to `ef_supermatch`, for supermatcher re-search.
    hnsw_supermatch: Option<Arc<HnswSearcher>>,
    pub do_match: bool,
    /// How many non-matches to tolerate before considering results "not saturated".
    /// With margin=0 (default), all `ef` results must match to trigger extended search or to detect a supermatcher.
    /// A small margin (e.g. 1-30) accounts for imprecision in the HNSW neighbor tail.
    pub saturation_margin: usize,
    /// Orientation label for phase tracing (e.g. 'N' or 'M').
    #[cfg(feature = "phase_trace")]
    pub orient: char,
}

impl SearchParams {
    pub fn new(
        hnsw: Arc<HnswSearcher>,
        do_match: bool,
        ef_supermatch: Option<usize>,
        ef_saturation_margin: usize,
        #[cfg(feature = "phase_trace")] orient: char,
    ) -> Self {
        let ef = hnsw.params.get_ef_search(0);
        let hnsw_supermatch = ef_supermatch.map(|ef_sm| {
            if ef_sm <= ef {
                tracing::warn!(
                    "ef_supermatch ({ef_sm}) <= ef_search ({ef}): \
                     saturated results will not be extended"
                );
            }
            let mut searcher = (*hnsw).clone();
            let p = &mut searcher.params;
            p.ef_search[0] = p.ef_search[0].max(ef_sm);
            p.ef_constr_search[0] = p.ef_constr_search[0].max(ef_sm);
            p.ef_constr_insert[0] = p.ef_constr_insert[0].max(ef_sm);
            Arc::new(searcher)
        });
        Self {
            hnsw,
            hnsw_supermatch,
            do_match,
            saturation_margin: ef_saturation_margin,
            #[cfg(feature = "phase_trace")]
            orient,
        }
    }

    pub fn new_no_match(hnsw: Arc<HnswSearcher>) -> Self {
        Self::new(
            hnsw,
            false,
            None,
            0,
            #[cfg(feature = "phase_trace")]
            'U',
        )
    }
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn search<const ROTMASK: u32>(
    sessions: &BothEyes<Vec<HawkSession>>,
    search_queries: &SearchQueries<ROTMASK>,
    search_ids: &SearchIds,
    search_params: SearchParams,
    mode: NeighborhoodMode,
) -> Result<SearchResults<ROTMASK>> {
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

    let schedule = Schedule::new(n_sessions, n_requests, ROTMASK.count_ones() as usize);

    parallelize(schedule.search_batches().into_iter().map(per_session)).await?;

    let results = schedule.organize_results(collect_results(rx).await?)?;

    Ok(results)
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn per_session<const ROTMASK: u32, N: Neighborhood<Aby3Store<HawkOps>>>(
    session: &HawkSession,
    search_queries: &SearchQueries<ROTMASK>,
    search_ids: &SearchIds,
    search_params: &SearchParams,
    tx: UnboundedSender<(TaskId, HawkInsertPlan)>,
    batch: Batch,
) -> Result<()> {
    let inner = async {
        let mut vector_store = session.aby3_store.write().await;
        let graph_store = session.graph_store.clone().read_owned().await;

        for task in batch.tasks {
            let query = search_queries[batch.i_eye][task.i_request][task.i_rotation];
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
    };

    #[cfg(feature = "phase_trace")]
    {
        use super::phase_tracer::{SessionContext, SESSION_CTX};
        let ctx = SessionContext {
            i_eye: batch.i_eye,
            i_session: batch.i_session,
            orient: search_params.orient,
        };
        SESSION_CTX.scope(ctx, inner).await
    }
    #[cfg(not(feature = "phase_trace"))]
    {
        inner.await
    }
}

/// Classify search results at two thresholds and optionally re-search with extended ef.
///
/// Two thresholds (GPU parity):
/// - **Match threshold** (0.345): determines uniqueness decisions (match/no-match).
/// - **Anon stats threshold** (0.375): higher threshold whose matches feed anonymous
///   statistics. Also used as the saturation trigger: if all `ef` results are below
///   this threshold, the query is a potential supermatcher and we re-search with a
///   larger `ef` to get a more complete picture.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn classify_and_extend(
    edges: &[(Aby3VectorRef, Aby3DistanceRef)],
    query: &Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem<Aby3VectorRef>,
    ef: usize,
) -> Result<ClassifiedMatches> {
    let margin = search_params.saturation_margin;
    let classified = classify_edges(edges, aby3_store, ef, margin).await?;

    // Extended search if anon stats threshold is saturated (supermatcher)
    if let Some((ef_supermatch, hnsw_supermatch)) = classified
        .anon_stats_matches
        .saturated
        .then(|| {
            search_params
                .hnsw_supermatch
                .as_ref()
                .map(|s| (s.params.get_ef_search(0), s))
        })
        .flatten()
        .filter(|(ef_sm, _)| *ef_sm > ef)
    {
        tracing::info!(
            "Potential supermatcher: all {ef} results below anon stats threshold, \
             re-searching with ef={ef_supermatch} to confirm",
        );
        metrics::counter!("supermatcher_extended_searches").increment(1);

        let supermatch_neighbors = hnsw_supermatch
            .search::<_, SortedNeighborhood<_>>(aby3_store, graph_store, query, ef_supermatch)
            .await?;

        let supermatch_classified = classify_edges(
            &supermatch_neighbors.edges,
            aby3_store,
            ef_supermatch,
            margin,
        )
        .await?;

        if supermatch_classified.anon_stats_matches.saturated {
            tracing::warn!(
                "Supermatcher still saturated after extended search (ef={ef_supermatch})",
            );
            metrics::counter!("supermatcher_still_saturated_after_extended").increment(1);
        }

        return Ok(supermatch_classified);
    }

    Ok(classified)
}

/// Batch-classify edges at both the match threshold and the anon stats threshold.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn classify_edges(
    edges: &[(Aby3VectorRef, Aby3DistanceRef)],
    aby3_store: &mut Aby3Store<HawkOps>,
    ef: usize,
    saturation_margin: usize,
) -> Result<ClassifiedMatches> {
    let all_distances: Vec<_> = edges.iter().map(|(_, d)| *d).collect();

    // Step 1: Batch-check all edges at anon stats threshold (weaker, fewer passes)
    let anon_bits = aby3_store
        .is_match_at(&all_distances, Threshold::AnonStats)
        .await?;
    let anon_stats_matches: Vec<_> = edges
        .iter()
        .zip(&anon_bits)
        .filter(|(_, &b)| b)
        .map(|(edge, _)| *edge)
        .collect();
    let anon_stats_saturated = anon_stats_matches.len() + saturation_margin >= ef;

    // Step 2: Batch-check anon stats matches at match threshold (stricter, smaller set)
    let anon_distances: Vec<_> = all_distances
        .iter()
        .zip(&anon_bits)
        .filter(|(_, &b)| b)
        .map(|(d, _)| *d)
        .collect();
    let matches = if anon_distances.is_empty() {
        vec![]
    } else {
        let match_bits = aby3_store
            .is_match_at(&anon_distances, Threshold::Match)
            .await?;
        anon_stats_matches
            .iter()
            .zip(match_bits)
            .filter(|(_, b)| *b)
            .map(|(edge, _)| *edge)
            .collect()
    };
    let matches_saturated = matches.len() + saturation_margin >= ef;

    Ok(ClassifiedMatches {
        matches: SaturableMatches {
            results: matches,
            saturated: matches_saturated,
        },
        anon_stats_matches: SaturableMatches {
            results: anon_stats_matches,
            saturated: anon_stats_saturated,
        },
    })
}

async fn per_insert_query<N: Neighborhood<Aby3Store<HawkOps>>>(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem<Aby3VectorRef>,
    insertion_layer: usize,
) -> Result<HawkInsertPlan> {
    let start = Instant::now();

    let (links, update_ep) = search_params
        .hnsw
        .search_to_insert::<_, N>(aby3_store, graph_store, &query, insertion_layer)
        .await?;

    let classified = if search_params.do_match {
        match links.first() {
            Some(bottom_layer) => {
                let ef = search_params.hnsw.params.get_ef_constr_insert(0);
                classify_and_extend(
                    bottom_layer.as_ref(),
                    &query,
                    search_params,
                    aby3_store,
                    graph_store,
                    ef,
                )
                .await?
            }
            None => ClassifiedMatches::default(),
        }
    } else {
        ClassifiedMatches::default()
    };

    // Trim and extract unstructured vector lists
    let mut links_unstructured = Vec::new();
    for (lc, mut l) in links.iter().cloned().enumerate() {
        let m = search_params.hnsw.params.get_M(lc);
        l.trim(aby3_store, m).await?;
        links_unstructured.push(l.edge_ids())
    }

    metrics::histogram!("search_query_duration").record(start.elapsed().as_secs_f64());
    Ok(HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links: links_unstructured,
            update_ep,
        },
        classified,
    })
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn per_search_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem<Aby3VectorRef>,
) -> Result<HawkInsertPlan> {
    let start = Instant::now();

    let ef_search = search_params.hnsw.params.get_ef_search(0);
    let layer_0_neighbors = search_params
        .hnsw
        .search::<_, SortedNeighborhood<_>>(aby3_store, graph_store, &query, ef_search)
        .await?;

    let links_unstructured = vec![layer_0_neighbors.edge_ids()];

    let classified = if search_params.do_match {
        classify_and_extend(
            &layer_0_neighbors.edges,
            &query,
            search_params,
            aby3_store,
            graph_store,
            ef_search,
        )
        .await?
    } else {
        ClassifiedMatches::default()
    };

    metrics::histogram!("search_query_duration").record(start.elapsed().as_secs_f64());
    Ok(HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links: links_unstructured,
            update_ep: UpdateEntryPoint::False,
        },
        classified,
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
) -> Result<InsertPlanV<Aby3Store<HawkOps>>> {
    let start = Instant::now();

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
        l.trim(&mut store, m).await?;
        links_unstructured.push(l.edge_ids());
    }

    metrics::histogram!("search_query_duration").record(start.elapsed().as_secs_f64());

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
    use iris_mpc_common::iris_db::iris::Threshold;

    #[test]
    fn match_threshold_is_stricter_than_anon_stats() {
        assert!(
            Threshold::Match.ratio() <= Threshold::AnonStats.ratio(),
            "Match threshold must be stricter (lower) than anon stats threshold"
        );
    }

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
        request.cache_into(&actor.worker_pools).await?;
        let search_queries = &request.queries(Orientation::Normal);
        let search_params = SearchParams::new(
            actor.searcher(),
            true,
            Some(4000),
            0,
            #[cfg(feature = "phase_trace")]
            'T',
        );

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
                assert_eq!(rotations.center().classified.matches.results.len(), 1);
                assert!(rotations
                    .center()
                    .classified
                    .matches
                    .results
                    .iter()
                    .any(|(v, _)| *v == VectorId::from_0_index(i as u32)));
            }
        }
        actor.sync_peers().await?;
        Ok(actor)
    }
}
