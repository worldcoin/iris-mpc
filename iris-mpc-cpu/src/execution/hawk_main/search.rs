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
        vector_store::VectorStore,
        GraphMem, HnswSearcher, SortedNeighborhood,
    },
};
use eyre::{OptionExt, Result};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

pub type SearchQueries<const ROTMASK: u32> =
    Arc<BothEyes<VecRequests<VecRotationSupport<Aby3Query, ROTMASK>>>>;
pub type SearchResults<const ROTMASK: u32> =
    BothEyes<VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>>;

/// Identifiers of requests
pub type SearchIds = Arc<VecRequests<String>>;

/// Set of `(i_request, i_eye, i_rotation)` tuples that should force-trigger
/// extended-ef search regardless of saturation.
pub type ForceExtendSet = Arc<HashSet<(usize, usize, usize)>>;

/// Build force-extend sets for both orientations from the two benchmark knobs.
///
/// Each request has `2 × 2 × n_rotations` independent searches (2 orientations
/// × 2 eyes × n_rotations). The deterministic order used for selecting which
/// searches to force is: orientation(normal first) → eye(left first) →
/// rotation(center first, then outward alternating).
///
/// Returns `[normal_set, mirror_set]`.
pub fn build_force_extend_sets(
    n_requests_to_force: usize,
    n_searches_per_request: usize,
    n_rotations: usize,
) -> [ForceExtendSet; 2] {
    // Deterministic ordering of (eye, rotation) within one orientation.
    let center = n_rotations / 2;
    let rotation_order: Vec<usize> = std::iter::once(center)
        .chain((1..n_rotations).map(|delta| {
            if delta % 2 == 1 {
                center.wrapping_sub((delta + 1) / 2)
            } else {
                center + delta / 2
            }
        }))
        .filter(|&r| r < n_rotations)
        .collect();

    let per_orient: Vec<(usize, usize)> = (0..2_usize)
        .flat_map(|eye| rotation_order.iter().map(move |&rot| (eye, rot)))
        .collect();

    // Full ordering across orientations: normal's (eye,rot) pairs first, then mirror's.
    // orient_index 0 = normal, 1 = mirror
    let all_coords: Vec<(usize, usize, usize)> = (0..2_usize)
        .flat_map(|orient| per_orient.iter().map(move |&(eye, rot)| (orient, eye, rot)))
        .collect();

    let mut sets = [HashSet::new(), HashSet::new()];
    for i_request in 0..n_requests_to_force {
        for &(orient, i_eye, i_rotation) in all_coords.iter().take(n_searches_per_request) {
            sets[orient].insert((i_request, i_eye, i_rotation));
        }
    }

    sets.map(|s| Arc::new(s))
}

#[derive(Clone)]
pub struct SearchParams {
    pub hnsw: Arc<HnswSearcher>,
    /// Searcher with layer-0 ef params overridden to `ef_supermatch`, for supermatcher re-search.
    hnsw_supermatch: Option<Arc<HnswSearcher>>,
    pub do_match: bool,
    /// Searches in this set bypass the saturation check and always trigger
    /// extended-ef re-search. Used for benchmarking.
    force_extend: ForceExtendSet,
}

impl SearchParams {
    pub fn new(
        hnsw: Arc<HnswSearcher>,
        do_match: bool,
        ef_supermatch: Option<usize>,
        force_extend: ForceExtendSet,
    ) -> Self {
        let hnsw_supermatch = ef_supermatch.map(|ef_sm| {
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
            force_extend,
        }
    }
}

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

async fn per_session<const ROTMASK: u32, N: Neighborhood<Aby3Store<HawkOps>>>(
    session: &HawkSession,
    search_queries: &SearchQueries<ROTMASK>,
    search_ids: &SearchIds,
    search_params: &SearchParams,
    tx: UnboundedSender<(TaskId, HawkInsertPlan)>,
    batch: Batch,
) -> Result<()> {
    let mut vector_store = session.aby3_store.write().await;
    let graph_store = session.graph_store.clone().read_owned().await;

    for task in batch.tasks {
        let query = search_queries[batch.i_eye][task.i_request][task.i_rotation].clone();
        let force_extend = search_params
            .force_extend
            .contains(&(task.i_request, batch.i_eye, task.i_rotation));
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
                force_extend,
            )
            .await?
        } else {
            // plain search for non-centers
            per_search_query(query, search_params, &mut vector_store, &graph_store, force_extend)
                .await?
        };

        tx.send((task.id(), result))?;
    }

    Ok(())
}

/// Classify search results at both thresholds and optionally re-search with extended ef.
///
/// When `force_extend` is true, the saturation check is bypassed and the
/// extended search is always triggered (for benchmarking).
async fn classify_and_extend(
    edges: &[(Aby3VectorRef, Aby3DistanceRef)],
    query: &Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem<Aby3VectorRef>,
    ef: usize,
    force_extend: bool,
) -> Result<ClassifiedMatches> {
    let classified = classify_edges(edges, aby3_store, ef).await?;

    let should_extend = classified.anon_stats_matches.saturated || force_extend;

    // Extended search if anon stats threshold is saturated (supermatcher) or forced
    if let Some((ef_supermatch, hnsw_supermatch)) = should_extend
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
            "Supermatcher detected: all {ef} results below anon stats threshold, \
             re-searching with ef={ef_supermatch}",
        );
        metrics::counter!("supermatcher_extended_searches").increment(1);

        let supermatch_neighbors = hnsw_supermatch
            .search::<_, SortedNeighborhood<_>>(aby3_store, graph_store, query, ef_supermatch)
            .await?;

        let supermatch_classified =
            classify_edges(&supermatch_neighbors.edges, aby3_store, ef_supermatch).await?;

        if supermatch_classified.anon_stats_matches.saturated {
            tracing::warn!(
                "Supermatcher still saturated after extended search (ef={ef_supermatch})",
            );
        }

        return Ok(supermatch_classified);
    }

    Ok(classified)
}

/// Batch-classify edges at both the match threshold and the anon stats threshold.
async fn classify_edges(
    edges: &[(Aby3VectorRef, Aby3DistanceRef)],
    aby3_store: &mut Aby3Store<HawkOps>,
    ef: usize,
) -> Result<ClassifiedMatches> {
    let all_distances: Vec<_> = edges.iter().map(|(_, d)| *d).collect();

    // Step 1: Batch-check all edges at match threshold
    let match_bits = aby3_store.is_match_batch(&all_distances).await?;
    let matches: Vec<_> = edges
        .iter()
        .zip(&match_bits)
        .filter(|(_, &b)| b)
        .map(|(edge, _)| *edge)
        .collect();
    let matches_saturated = matches.len() == ef;

    // Step 2: Batch-check non-matched edges at anon stats threshold
    let remaining_distances: Vec<_> = all_distances
        .iter()
        .zip(&match_bits)
        .filter(|(_, &b)| !b)
        .map(|(d, _)| *d)
        .collect();
    let anon_stats_matches = if remaining_distances.is_empty() {
        matches.clone()
    } else {
        let anon_bits = aby3_store
            .is_match_anon_stats_batch(&remaining_distances)
            .await?;
        let anon_extra: Vec<_> = edges
            .iter()
            .zip(&match_bits)
            .filter(|(_, &b)| !b)
            .zip(anon_bits)
            .filter(|(_, is_match)| *is_match)
            .map(|((edge, _), _)| *edge)
            .collect();
        let mut all = matches.clone();
        all.extend(anon_extra);
        all
    };
    let anon_stats_saturated = anon_stats_matches.len() == ef;

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
    force_extend: bool,
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
                    force_extend,
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

async fn per_search_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem<Aby3VectorRef>,
    force_extend: bool,
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
            force_extend,
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
        let search_params =
            SearchParams::new(actor.searcher(), true, Some(4000), Default::default());

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

    #[test]
    fn test_build_force_extend_sets() {
        let n_rotations = 3; // rotations 0, 1(center), 2

        // No forcing → both sets empty
        let [normal, mirror] = build_force_extend_sets(0, 12, n_rotations);
        assert!(normal.is_empty());
        assert!(mirror.is_empty());

        // 1 request, 1 search → only normal, left eye, center rotation
        let [normal, mirror] = build_force_extend_sets(1, 1, n_rotations);
        assert_eq!(normal.len(), 1);
        assert!(normal.contains(&(0, 0, 1))); // request 0, eye 0 (left), rotation 1 (center)
        assert!(mirror.is_empty());

        // 1 request, 6 searches → all 6 normal searches, no mirror
        let [normal, mirror] = build_force_extend_sets(1, 6, n_rotations);
        assert_eq!(normal.len(), 6);
        for eye in 0..2 {
            for rot in 0..3 {
                assert!(normal.contains(&(0, eye, rot)));
            }
        }
        assert!(mirror.is_empty());

        // 1 request, 7 searches → all 6 normal + 1 mirror
        let [normal, mirror] = build_force_extend_sets(1, 7, n_rotations);
        assert_eq!(normal.len(), 6);
        assert_eq!(mirror.len(), 1);
        assert!(mirror.contains(&(0, 0, 1))); // mirror, left eye, center

        // 1 request, 12 searches → all 12
        let [normal, mirror] = build_force_extend_sets(1, 12, n_rotations);
        assert_eq!(normal.len(), 6);
        assert_eq!(mirror.len(), 6);

        // 2 requests, 1 search each
        let [normal, mirror] = build_force_extend_sets(2, 1, n_rotations);
        assert_eq!(normal.len(), 2);
        assert!(normal.contains(&(0, 0, 1)));
        assert!(normal.contains(&(1, 0, 1)));
        assert!(mirror.is_empty());

        // Rotation order: center(1) first, then 0, then 2
        let [normal, _] = build_force_extend_sets(1, 3, n_rotations);
        assert!(normal.contains(&(0, 0, 1))); // left, center
        assert!(normal.contains(&(0, 0, 0))); // left, rotation 0
        assert!(normal.contains(&(0, 0, 2))); // left, rotation 2
    }
}
