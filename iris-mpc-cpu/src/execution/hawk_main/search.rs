use super::{
    rot::VecRotationSupport,
    scheduler::{Batch, Schedule, TaskId},
    BothEyes, ClassifiedMatches, HawkInsertPlan, HawkOps, HawkSearchMode, HawkSession, MapEdges,
    Orientation, SaturableMatches, VecEdges, VecRequests, LEFT, RIGHT,
};
use crate::{
    execution::hawk_main::{
        iris_worker::IrisWorkerPool,
        scheduler::{collect_results, parallelize},
        InsertPlanV, StoreId,
    },
    hawkers::aby3::aby3_store::{Aby3DistanceRef, Aby3Query, Aby3Store, DistanceOps},
    hnsw::{graph::UpdateEntryPoint, GraphMem, HnswSearcher},
};
use ampc_anon_stats::types::Eye;
use eyre::{OptionExt, Result};
use iris_mpc_common::iris_db::iris::Threshold;
use iris_mpc_common::{VectorId, ROTATIONS};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tracing::instrument;

/// Keep enough records in each MPC call to amortize its fixed costs without
/// making one session monopolize the scan. On the production r8g.24xlarge,
/// 4K records per session keeps the dot-product workers, MPC circuit, and TCP
/// streams overlapped throughout a large scan.
#[cfg(not(test))]
const LINEAR_SCAN_CHUNK_SIZE: usize = 1 << 12;
// Exercise chunk sharding and result merging in the small in-process tests.
#[cfg(test)]
const LINEAR_SCAN_CHUNK_SIZE: usize = 2;

/// Global safety ceiling for independent full-scan chunks in flight.
///
/// Operational concurrency is normally lower and is configured through the
/// number of sessions in a search group. With the current three base rotations,
/// that is `3 * hawk_request_parallelism`. Keep this ceiling hardware-neutral
/// so deployments can tune through existing config.
pub(super) const LINEAR_SCAN_MAX_IN_FLIGHT_CHUNKS: usize = 256;

#[derive(Clone, Debug)]
struct LinearScanChunk {
    i_request: usize,
    i_chunk: usize,
    range: std::ops::Range<usize>,
}

#[derive(Clone, Copy, Debug)]
enum LinearScanStage {
    Full,
    Candidate,
}

#[derive(Clone, Copy, Debug)]
struct LinearScanEyeContext {
    eye: Eye,
    stage: LinearScanStage,
    orientation: Orientation,
}

impl LinearScanStage {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Candidate => "candidate",
        }
    }
}

const fn eye_index(eye: Eye) -> usize {
    match eye {
        Eye::Left => LEFT,
        Eye::Right => RIGHT,
    }
}

const fn eye_label(eye: Eye) -> &'static str {
    match eye {
        Eye::Left => "left",
        Eye::Right => "right",
    }
}

const fn orientation_label(orientation: Orientation) -> &'static str {
    match orientation {
        Orientation::Normal => "normal",
        Orientation::Mirror => "mirror",
    }
}

#[derive(Clone)]
struct LinearScanPrefetch {
    worker: Arc<dyn IrisWorkerPool>,
}

pub type SearchQueries<const ROTMASK: u32> =
    Arc<BothEyes<VecRequests<VecRotationSupport<Aby3Query, ROTMASK>>>>;
pub type SearchResults<const ROTMASK: u32> =
    BothEyes<VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>>;

/// Identifiers of requests
pub type SearchIds = Arc<VecRequests<String>>;

#[derive(Clone)]
pub struct SearchParams {
    pub hnsw: Arc<HnswSearcher>,
    pub mode: HawkSearchMode,
    /// Searcher with layer-0 ef params overridden to `ef_supermatch`, for supermatcher re-search.
    hnsw_supermatch: Option<Arc<HnswSearcher>>,
    pub do_match: bool,
    pub return_partial_results: bool,
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
        mode: HawkSearchMode,
        do_match: bool,
        ef_supermatch: Option<usize>,
        ef_saturation_margin: usize,
        return_partial_results: bool,
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
            mode,
            hnsw_supermatch,
            do_match,
            return_partial_results,
            saturation_margin: ef_saturation_margin,
            #[cfg(feature = "phase_trace")]
            orient,
        }
    }

    pub fn new_no_match(hnsw: Arc<HnswSearcher>, mode: HawkSearchMode) -> Self {
        Self::new(
            hnsw,
            mode,
            false,
            None,
            0,
            false,
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
) -> Result<SearchResults<ROTMASK>> {
    let n_sessions = sessions[LEFT].len();
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_requests = search_queries[LEFT].len();
    assert_eq!(n_requests, search_queries[RIGHT].len());

    // `search_to_identity_update` invokes this function for every batch, even
    // when the batch has no reset/recovery updates. Avoid spawning one empty
    // session per eye in that common case.
    if n_requests == 0 {
        return Ok([Vec::new(), Vec::new()]);
    }

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

    let schedule = Schedule::new(n_sessions, n_requests, ROTMASK.count_ones() as usize);

    parallelize(schedule.search_batches().into_iter().map(per_session)).await?;

    let results = schedule.organize_results(collect_results(rx).await?)?;

    Ok(results)
}

/// Run the GPU-compatible two-eye linear-scan cascade.
///
/// The configured first eye is evaluated against every live vector. Only IDs
/// passing the wider anonymous-statistics threshold (plus request-specific OR
/// and reauthentication targets) are evaluated on the other eye. Both stages
/// still use the same fused 31-rotation AMPC primitive. As each first-eye chunk
/// completes, its public candidates are queued for bounded cold-eye database
/// prefetch so stage-two I/O overlaps the remainder of stage one.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn linear_scan_cascade<const ROTMASK: u32>(
    sessions: &BothEyes<Vec<HawkSession>>,
    search_queries: &SearchQueries<ROTMASK>,
    search_params: SearchParams,
    orientation: Orientation,
    full_scan_side: Eye,
    extra_candidate_ids: &VecRequests<Vec<VectorId>>,
    forced_anon_stats_ids: &VecRequests<Vec<VectorId>>,
) -> Result<SearchResults<ROTMASK>> {
    let cascade_start = Instant::now();
    debug_assert_eq!(search_params.mode, HawkSearchMode::LinearScan);

    let n_sessions = sessions[LEFT].len();
    assert!(n_sessions > 0, "linear scan requires at least one session");
    assert_eq!(n_sessions, sessions[RIGHT].len());
    let n_requests = search_queries[LEFT].len();
    assert_eq!(n_requests, search_queries[RIGHT].len());
    assert_eq!(n_requests, extra_candidate_ids.len());
    assert_eq!(n_requests, forced_anon_stats_ids.len());

    let first_eye = eye_index(full_scan_side);
    let second_eye_side = full_scan_side.other();
    let second_eye = eye_index(second_eye_side);

    // Both eye registries contain the same live VectorIds. Build this list
    // once, then share it across all requests and sessions in the full stage.
    let live_ids = {
        let vector_store = sessions[first_eye][0].aby3_store.read().await;
        let registry = vector_store.registry.read().await;
        Arc::<[VectorId]>::from(
            registry
                .get_points()
                .iter()
                .enumerate()
                .filter_map(|(serial_id, entry)| {
                    entry
                        .as_ref()
                        .map(|(version, ())| VectorId::new(serial_id as u32, *version))
                })
                .collect::<Vec<_>>(),
        )
    };
    let full_scan_ids = Arc::new(vec![live_ids.clone(); n_requests]);
    let first_eye_comparisons = live_ids.len() * n_requests;

    tracing::info!(
        eye = %full_scan_side,
        orientation = orientation_label(orientation),
        requests = n_requests,
        vectors = live_ids.len(),
        "Running full linear-scan stage"
    );
    let prefetch_worker = {
        let store = sessions[second_eye][0].aby3_store.read().await;
        store.workers.clone()
    };
    // Known OR/reauth candidates are second-stage candidates regardless of
    // the first-eye result, so their database reads can start before the
    // resident-eye scan. Reservations are single-use and idempotent: a record
    // hinted again by a first-eye chunk or by the other orientation is not
    // reserved twice, and leftovers are released when the batch completes.
    for extras in extra_candidate_ids {
        let prefetch_ids = collect_live_second_stage_ids(&live_ids, [], extras);
        prefetch_worker.prefetch_irises(prefetch_ids).await?;
    }
    let first_results = linear_scan_eye(
        sessions,
        search_queries,
        &search_params,
        LinearScanEyeContext {
            eye: full_scan_side,
            stage: LinearScanStage::Full,
            orientation,
        },
        full_scan_ids,
        Arc::new(forced_anon_stats_ids.clone()),
        Some(LinearScanPrefetch {
            worker: prefetch_worker.clone(),
        }),
    )
    .await?;
    let prefetch_wait_start = Instant::now();
    prefetch_worker.wait_for_prefetch().await?;
    let prefetch_wait_seconds = prefetch_wait_start.elapsed().as_secs_f64();
    metrics::histogram!("linear_scan_cold_prefetch_wait_duration").record(prefetch_wait_seconds);
    metrics::histogram!(
        "linear_scan_cascade_prefetch_wait_duration",
        "eye" => eye_label(second_eye_side),
        "orientation" => orientation_label(orientation),
    )
    .record(prefetch_wait_seconds);

    // The CUDA path uses the first eye's wider prefilter bitmap and explicitly
    // unions OR-rule and reauth IDs. Retain only live IDs, matching its bounds
    // check, and sort for deterministic MPC/database access order. `live_ids`
    // is strictly ordered: it was collected above by enumerating the registry's
    // serial-ID-indexed vector, which holds at most one current version per
    // serial ID. Avoid materializing a full-database HashSet for this sparse
    // candidate membership check.
    debug_assert!(live_ids.windows(2).all(|pair| pair[0] < pair[1]));
    let mut second_stage_ids = Vec::with_capacity(n_requests);
    let mut candidate_count = 0usize;
    for (plans, extras) in first_results.iter().zip(extra_candidate_ids) {
        let ids = collect_live_second_stage_ids(
            &live_ids,
            plans
                .iter()
                .flat_map(|plan| &plan.classified.anon_stats_matches.results)
                .map(|(id, _)| *id),
            extras,
        );
        candidate_count += ids.len();
        second_stage_ids.push(Arc::<[VectorId]>::from(ids));
    }

    tracing::info!(
        eye = %second_eye_side,
        orientation = orientation_label(orientation),
        requests = n_requests,
        candidates = candidate_count,
        "Running candidate-only linear-scan stage"
    );
    metrics::counter!("linear_scan_second_eye_candidates_total").increment(candidate_count as u64);
    let second_results = linear_scan_eye(
        sessions,
        search_queries,
        &search_params,
        LinearScanEyeContext {
            eye: second_eye_side,
            stage: LinearScanStage::Candidate,
            orientation,
        },
        Arc::new(second_stage_ids),
        Arc::new(forced_anon_stats_ids.clone()),
        None,
    )
    .await?;

    let total_comparisons = first_eye_comparisons + candidate_count;
    let elapsed_seconds = cascade_start.elapsed().as_secs_f64();
    let comparisons_per_second = total_comparisons as f64 / elapsed_seconds.max(f64::EPSILON);
    let second_eye_candidate_fraction = if first_eye_comparisons == 0 {
        0.0
    } else {
        candidate_count as f64 / first_eye_comparisons as f64
    };
    metrics::counter!(
        "linear_scan_cascade_comparisons_total",
        "orientation" => orientation_label(orientation),
    )
    .increment(total_comparisons as u64);
    metrics::histogram!(
        "linear_scan_cascade_duration",
        "orientation" => orientation_label(orientation),
    )
    .record(elapsed_seconds);
    metrics::histogram!(
        "linear_scan_cascade_comparisons_per_second",
        "orientation" => orientation_label(orientation),
    )
    .record(comparisons_per_second);
    metrics::histogram!(
        "linear_scan_second_eye_candidate_fraction",
        "eye" => eye_label(second_eye_side),
        "orientation" => orientation_label(orientation),
    )
    .record(second_eye_candidate_fraction);
    tracing::info!(
        orientation = orientation_label(orientation),
        full_scan_eye = eye_label(full_scan_side),
        candidate_eye = eye_label(second_eye_side),
        requests = n_requests,
        database_records = live_ids.len(),
        rotations_per_comparison = ROTATIONS,
        first_eye_comparisons,
        second_eye_comparisons = candidate_count,
        total_comparisons,
        second_eye_candidate_fraction,
        prefetch_wait_seconds,
        elapsed_seconds,
        comparisons_per_second,
        "LINEAR_SCAN_CASCADE_SUMMARY"
    );

    Ok(match full_scan_side {
        Eye::Left => [first_results, second_results],
        Eye::Right => [second_results, first_results],
    })
}

/// Resolve the matching module's "compare the other eye" requests from the
/// cascade results instead of a second MPC pass.
///
/// Every ID the matching module asks about was already evaluated on both eyes
/// by [`linear_scan_cascade`]: one-eyed strict matches are a subset of the
/// first eye's anonymous-statistics prefilter, and LUC / OR-rule / reauth IDs
/// are unioned into the candidate stage exactly like the CUDA actor. An ID
/// absent from an eye's strict matches therefore did not match on that eye.
/// Re-running `is_match_batch` would consult a different circuit (the 11
/// rotation min-rotation path) and, for the cold eye, fetch every ID from the
/// database again per query rotation inside the MPC critical path.
pub fn linear_scan_comparison_results<const ROTMASK: u32>(
    search_results: &SearchResults<ROTMASK>,
    ids_to_compare: BothEyes<VecRequests<VecEdges<VectorId>>>,
) -> BothEyes<VecRequests<MapEdges<bool>>> {
    let [ids_left, ids_right] = ids_to_compare;
    [(LEFT, ids_left), (RIGHT, ids_right)].map(|(eye, ids_per_request)| {
        assert_eq!(
            ids_per_request.len(),
            search_results[eye].len(),
            "comparison requests must align with search results"
        );
        search_results[eye]
            .iter()
            .zip(ids_per_request)
            .map(|(plans, ids)| {
                let matched = plans
                    .iter()
                    .flat_map(|plan| plan.classified.matches.results.iter())
                    .map(|(id, _)| *id)
                    .collect::<HashSet<_>>();
                ids.into_iter()
                    .map(|id| (id, matched.contains(&id)))
                    .collect::<MapEdges<bool>>()
            })
            .collect()
    })
}

fn collect_live_second_stage_ids(
    live_ids: &[VectorId],
    threshold_candidates: impl IntoIterator<Item = VectorId>,
    extra_candidate_ids: &[VectorId],
) -> Vec<VectorId> {
    // `live_ids` is the serial-ID-ordered registry snapshot asserted by the
    // caller; binary search also checks the exact current version.
    let mut ids = threshold_candidates
        .into_iter()
        .chain(extra_candidate_ids.iter().copied())
        .filter(|id| live_ids.binary_search(id).is_ok())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids
}

async fn linear_scan_eye<const ROTMASK: u32>(
    sessions: &BothEyes<Vec<HawkSession>>,
    search_queries: &SearchQueries<ROTMASK>,
    search_params: &SearchParams,
    context: LinearScanEyeContext,
    candidate_ids: Arc<VecRequests<Arc<[VectorId]>>>,
    forced_anon_stats_ids: Arc<VecRequests<Vec<VectorId>>>,
    prefetch: Option<LinearScanPrefetch>,
) -> Result<VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>> {
    let stage_start = Instant::now();
    let eye_index = eye_index(context.eye);
    let n_requests = search_queries[eye_index].len();
    assert_eq!(n_requests, candidate_ids.len());
    assert_eq!(n_requests, forced_anon_stats_ids.len());
    // One logical comparison means one query eye against one database iris,
    // including all 31 rotations and both threshold checks. This is the same
    // unit reported by the standalone full-protocol benchmark.
    let comparisons = candidate_ids.iter().map(|ids| ids.len()).sum::<usize>();
    let n_rotations = ROTMASK.count_ones() as usize;
    let central_rotation = n_rotations / 2;

    // The HNSW scheduler distributes query rotations, but the exact scan has
    // only one useful query rotation: its center query already evaluates all
    // 31 database rotations. Instead, shard every request's candidate list so
    // batch-size=1 can use the whole machine and pipeline dot/MPC/network work.
    let mut chunks_per_request = Vec::with_capacity(n_requests);
    let mut chunks = Vec::new();
    for (i_request, ids) in candidate_ids.iter().enumerate() {
        let n_chunks = ids.len().div_ceil(LINEAR_SCAN_CHUNK_SIZE).max(1);
        chunks_per_request.push(n_chunks);
        for i_chunk in 0..n_chunks {
            let start = (i_chunk * LINEAR_SCAN_CHUNK_SIZE).min(ids.len());
            let end = (start + LINEAR_SCAN_CHUNK_SIZE).min(ids.len());
            chunks.push(LinearScanChunk {
                i_request,
                i_chunk,
                range: start..end,
            });
        }
    }

    let chunk_count = chunks.len();
    let configured_sessions = sessions[eye_index].len();
    let n_workers = configured_sessions
        .min(LINEAR_SCAN_MAX_IN_FLIGHT_CHUNKS)
        .min(chunks.len())
        .max(1);
    let mut batches = vec![Vec::new(); n_workers];
    for (index, chunk) in chunks.into_iter().enumerate() {
        batches[index % n_workers].push(chunk);
    }
    let min_chunks_per_session = batches.iter().map(Vec::len).min().unwrap_or(0);
    let max_chunks_per_session = batches.iter().map(Vec::len).max().unwrap_or(0);

    let jobs = batches
        .into_iter()
        .enumerate()
        .filter(|(_, batch)| !batch.is_empty())
        .map(|(i_session, batch)| {
            let session = sessions[eye_index][i_session].clone();
            let search_queries = search_queries.clone();
            let search_params = search_params.clone();
            let candidate_ids = candidate_ids.clone();
            let forced_anon_stats_ids = forced_anon_stats_ids.clone();
            let prefetch = prefetch.clone();
            async move {
                let mut vector_store = session.aby3_store.write().await;
                let graph_store = session.graph_store.clone().read_owned().await;
                let mut results = Vec::with_capacity(batch.len());
                for chunk in batch {
                    let query = search_queries[eye_index][chunk.i_request][central_rotation];
                    let result = per_linear_scan_query(
                        query,
                        &search_params,
                        &mut vector_store,
                        &graph_store,
                        &candidate_ids[chunk.i_request][chunk.range.clone()],
                        &forced_anon_stats_ids[chunk.i_request],
                    )
                    .await?;
                    if let Some(prefetch) = &prefetch {
                        let chunk_ids = &candidate_ids[chunk.i_request][chunk.range.clone()];
                        let prefetch_ids = collect_live_second_stage_ids(
                            chunk_ids,
                            result
                                .classified
                                .anon_stats_matches
                                .results
                                .iter()
                                .map(|(id, _)| *id),
                            &[],
                        );
                        prefetch.worker.prefetch_irises(prefetch_ids).await?;
                    }
                    results.push((chunk.i_request, chunk.i_chunk, result));
                }
                Ok(results)
            }
        });

    let mut chunk_results = chunks_per_request
        .iter()
        .map(|&len| vec![None; len])
        .collect::<Vec<Vec<Option<HawkInsertPlan>>>>();
    for (i_request, i_chunk, result) in parallelize(jobs).await?.into_iter().flatten() {
        chunk_results[i_request][i_chunk] = Some(result);
    }

    let graph_store = sessions[eye_index][0].graph_store.read().await;
    let results = chunk_results
        .into_iter()
        .enumerate()
        .map(|(i_request, results)| {
            let mut results = results.into_iter();
            let mut merged = results
                .next()
                .flatten()
                .ok_or_eyre("missing first linear-scan chunk result")?;
            for result in results {
                merge_linear_scan_plan(
                    &mut merged,
                    result.ok_or_eyre("missing linear-scan chunk result")?,
                );
            }

            let mut merged = Some(merged);
            (0..n_rotations)
                .map(|i_rotation| {
                    if i_rotation == central_rotation {
                        Ok(merged
                            .take()
                            .expect("central linear-scan result must be consumed once"))
                    } else {
                        Ok(empty_linear_scan_plan(
                            search_queries[eye_index][i_request][i_rotation],
                            &graph_store,
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(VecRotationSupport::from)
        })
        .collect::<Result<Vec<_>>>()?;

    let strict_match_records = results
        .iter()
        .map(|rotations| rotations[central_rotation].classified.matches.results.len())
        .sum::<usize>();
    let anon_stats_rotation_matches = results
        .iter()
        .map(|rotations| {
            rotations[central_rotation]
                .classified
                .anon_stats_matches
                .results
                .len()
        })
        .sum::<usize>();

    let elapsed_seconds = stage_start.elapsed().as_secs_f64();
    let comparisons_per_second = comparisons as f64 / elapsed_seconds.max(f64::EPSILON);
    let session_utilization = n_workers as f64 / configured_sessions as f64;
    let stage_label = context.stage.as_str();
    let eye_label = eye_label(context.eye);
    let orientation_label = orientation_label(context.orientation);
    metrics::counter!(
        "linear_scan_eye_comparisons_total",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .increment(comparisons as u64);
    metrics::histogram!(
        "linear_scan_eye_duration",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .record(elapsed_seconds);
    metrics::histogram!(
        "linear_scan_eye_comparisons",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .record(comparisons as f64);
    metrics::histogram!(
        "linear_scan_eye_comparisons_per_second",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .record(comparisons_per_second);
    metrics::histogram!(
        "linear_scan_eye_chunks",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .record(chunk_count as f64);
    metrics::histogram!(
        "linear_scan_eye_session_utilization",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .record(session_utilization);
    metrics::histogram!(
        "linear_scan_eye_active_sessions",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .record(n_workers as f64);
    metrics::counter!(
        "linear_scan_eye_strict_match_records_total",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .increment(strict_match_records as u64);
    metrics::counter!(
        "linear_scan_eye_anon_stats_rotation_matches_total",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .increment(anon_stats_rotation_matches as u64);
    tracing::info!(
        eye = eye_label,
        stage = stage_label,
        orientation = orientation_label,
        requests = n_requests,
        comparisons,
        rotations_per_comparison = ROTATIONS,
        chunks = chunk_count,
        chunk_size = LINEAR_SCAN_CHUNK_SIZE,
        configured_sessions,
        active_sessions = n_workers,
        session_utilization,
        min_chunks_per_session,
        max_chunks_per_session,
        strict_match_records,
        anon_stats_rotation_matches,
        elapsed_seconds,
        comparisons_per_second,
        "LINEAR_SCAN_EYE_SUMMARY"
    );

    Ok(results)
}

fn merge_linear_scan_plan(target: &mut HawkInsertPlan, mut source: HawkInsertPlan) {
    fn merge_matches(target: &mut SaturableMatches, mut source: SaturableMatches) {
        target.results.append(&mut source.results);
        target.saturated |= source.saturated;
    }

    merge_matches(&mut target.classified.matches, source.classified.matches);
    merge_matches(
        &mut target.classified.anon_stats_matches,
        source.classified.anon_stats_matches,
    );
    match (
        &mut target.classified.pre_extension,
        source.classified.pre_extension,
    ) {
        (Some(target), Some(source)) => merge_matches(target, source),
        (target @ None, source @ Some(_)) => *target = source,
        _ => {}
    }
    target.classified.linear_scan_supermatch_threshold = target
        .classified
        .linear_scan_supermatch_threshold
        .or(source.classified.linear_scan_supermatch_threshold);
    target
        .classified
        .partial_match_rotations
        .append(&mut source.classified.partial_match_rotations);
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn per_session<const ROTMASK: u32>(
    session: &HawkSession,
    search_queries: &SearchQueries<ROTMASK>,
    search_ids: &SearchIds,
    search_params: &SearchParams,
    tx: UnboundedSender<(TaskId, HawkInsertPlan)>,
    batch: Batch,
) -> Result<()> {
    let inner = async {
        // Linear scan does not build graph links for identity updates. The
        // shared HNSW path expresses that operation as a no-match search, but
        // materializing every live VectorId would be pure overhead here.
        if search_params.mode == HawkSearchMode::LinearScan && !search_params.do_match {
            let graph_store = session.graph_store.clone().read_owned().await;
            for task in batch.tasks {
                let query = search_queries[batch.i_eye][task.i_request][task.i_rotation];
                tx.send((task.id(), empty_linear_scan_plan(query, &graph_store)))?;
            }
            return Ok(());
        }

        // Matching linear scans go through `linear_scan_cascade`, which owns
        // the two-eye candidate logic; the HNSW scheduler below would run the
        // wrong algorithm for them.
        eyre::ensure!(
            search_params.mode == HawkSearchMode::Hnsw,
            "linear-scan matching must use linear_scan_cascade, not search()"
        );

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
                per_insert_query(
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

/// Evaluate the center query against every live vector using one fused
/// 31-rotation pass and the same AMPC distance/threshold primitives as HNSW.
/// Results are accumulated in deterministic serial-ID order so output ordering
/// matches the CUDA actor.
async fn per_linear_scan_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem,
    vector_ids: &[VectorId],
    forced_anon_stats_ids: &[VectorId],
) -> Result<HawkInsertPlan> {
    let start = Instant::now();
    let mut classified = ClassifiedMatches::default();

    if search_params.do_match {
        for ids in vector_ids.chunks(LINEAR_SCAN_CHUNK_SIZE) {
            let forced_anon_stats_vectors = forced_anon_stats_ids
                .iter()
                .filter_map(|id| ids.binary_search(id).ok())
                .collect::<Vec<_>>();
            let thresholds = aby3_store
                .eval_distance_batch_full_rotation_thresholds_with_forced_anon_stats(
                    &query,
                    ids,
                    &forced_anon_stats_vectors,
                )
                .await?;
            classified.matches.results.extend(
                ids.iter()
                    .copied()
                    .zip(&thresholds.matches)
                    .filter_map(|(id, &distance)| distance.map(|distance| (id, distance))),
            );
            classified.anon_stats_matches.results.extend(
                thresholds
                    .anon_stats_matches
                    .into_iter()
                    .map(|(vector, _rotation, distance)| (ids[vector], distance)),
            );

            if search_params.return_partial_results {
                classified.partial_match_rotations.extend(
                    ids.iter()
                        .copied()
                        .zip(thresholds.match_rotations)
                        .filter_map(|(id, rotations)| {
                            (!rotations.is_empty()).then(|| {
                                let rotations = rotations
                                    .into_iter()
                                    .map(|rotation| rotation as i8 - 15)
                                    .collect();
                                (id, rotations)
                            })
                        }),
                );
            }
        }

        // The CUDA actor compares its per-eye match counters against
        // SUPERMATCH_THRESHOLD, but those counters are only fetched when
        // return_partial_results is set; otherwise they stay at zero and no
        // query is ever treated as a supermatcher. Apply the same gate.
        classified.linear_scan_supermatch_threshold = search_params
            .hnsw_supermatch
            .as_ref()
            .filter(|_| search_params.return_partial_results)
            .map(|searcher| searcher.params.get_ef_search(0));
    }

    metrics::histogram!("linear_scan_query_duration").record(start.elapsed().as_secs_f64());
    metrics::counter!("linear_scan_vectors_total").increment(vector_ids.len() as u64);

    Ok(HawkInsertPlan {
        // Linear scan does not consume graph edges. Keep a minimal plan so the
        // common mutation pipeline can continue assigning stable VectorIds and
        // persisting the same request-level mutations as the GPU actor.
        plan: InsertPlanV {
            query,
            links: Vec::new(),
            update_ep: UpdateEntryPoint::False,
            as_of: graph_store.last_update_seq_no,
        },
        classified,
    })
}

/// Preserve the three-slot HNSW-shaped result container without rescanning the
/// database for the two non-central base rotations. Matching merges all slots,
/// so placing the full 31-rotation result in the center is behaviorally
/// equivalent to the old overlapping 3 x 11 representation.
fn empty_linear_scan_plan(query: Aby3Query, graph_store: &GraphMem) -> HawkInsertPlan {
    HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links: Vec::new(),
            update_ep: UpdateEntryPoint::False,
            as_of: graph_store.last_update_seq_no,
        },
        classified: ClassifiedMatches::default(),
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
    edges: &[(VectorId, Aby3DistanceRef<<HawkOps as DistanceOps>::Ring>)],
    query: &Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem,
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
            .search(aby3_store, graph_store, query, ef_supermatch)
            .await?;

        let mut supermatch_classified = classify_edges(
            &supermatch_neighbors.edges,
            aby3_store,
            ef_supermatch,
            margin,
        )
        .await?;
        supermatch_classified.pre_extension = Some(classified.matches);

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
    edges: &[(VectorId, Aby3DistanceRef<<HawkOps as DistanceOps>::Ring>)],
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
        pre_extension: None,
        linear_scan_supermatch_threshold: None,
        partial_match_rotations: Vec::new(),
    })
}

async fn per_insert_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem,
    insertion_layer: usize,
) -> Result<HawkInsertPlan> {
    let start = Instant::now();

    let (links, update_ep, as_of) = search_params
        .hnsw
        .search_to_insert(aby3_store, graph_store, &query, insertion_layer)
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
            as_of,
        },
        classified,
    })
}

#[instrument(level = "trace", target = "searcher::network", skip_all)]
async fn per_search_query(
    query: Aby3Query,
    search_params: &SearchParams,
    aby3_store: &mut Aby3Store<HawkOps>,
    graph_store: &GraphMem,
) -> Result<HawkInsertPlan> {
    let start = Instant::now();

    let ef_search = search_params.hnsw.params.get_ef_search(0);
    let as_of = graph_store.last_update_seq_no;
    let layer_0_neighbors = search_params
        .hnsw
        .search(aby3_store, graph_store, &query, ef_search)
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
            as_of,
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

    let (links, update_ep, as_of) = searcher
        .search_to_insert(&mut *store, &graph, &query, insertion_layer)
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
        as_of,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{setup_hawk_actors, setup_linear_scan_actors};
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
    async fn empty_search_does_not_require_sessions() -> Result<()> {
        let sessions: BothEyes<Vec<HawkSession>> = [Vec::new(), Vec::new()];
        let queries: SearchQueries<0> = Arc::new([Vec::new(), Vec::new()]);
        let request_ids: SearchIds = Arc::new(Vec::new());
        let params = SearchParams::new_no_match(
            Arc::new(HnswSearcher::new_with_test_parameters()),
            HawkSearchMode::LinearScan,
        );

        let results = search(&sessions, &queries, &request_ids, params).await?;

        assert!(results.iter().all(Vec::is_empty));
        Ok(())
    }

    #[test]
    fn second_stage_ids_retain_only_current_live_versions() {
        let live_v1 = VectorId::new(2, 1);
        let live_ids = vec![
            VectorId::new(1, 0),
            live_v1,
            // Serial ID 3 is tombstoned and therefore absent.
            VectorId::new(4, 0),
        ];
        let threshold_candidates = [VectorId::new(4, 0), VectorId::new(3, 0), live_v1];
        let extras = [
            VectorId::new(1, 0),
            live_v1,
            VectorId::new(2, 0), // stale version of a live serial ID
            VectorId::new(3, 0), // tombstoned
            VectorId::new(5, 0), // outside the registry
        ];

        assert_eq!(
            collect_live_second_stage_ids(&live_ids, threshold_candidates, &extras),
            vec![VectorId::new(1, 0), live_v1, VectorId::new(4, 0)]
        );
    }

    #[test]
    fn second_stage_ids_merge_deduplicate_and_sort_thresholds_and_extras() {
        let live_ids = (1..=5).map(VectorId::from_serial_id).collect::<Vec<_>>();
        let threshold_candidates = [
            VectorId::from_serial_id(5),
            VectorId::from_serial_id(2),
            VectorId::from_serial_id(2),
        ];
        let extras = [
            VectorId::from_serial_id(4),
            VectorId::from_serial_id(2),
            VectorId::from_serial_id(1),
        ];

        assert_eq!(
            collect_live_second_stage_ids(&live_ids, threshold_candidates, &extras),
            vec![
                VectorId::from_serial_id(1),
                VectorId::from_serial_id(2),
                VectorId::from_serial_id(4),
                VectorId::from_serial_id(5),
            ]
        );
    }

    #[tokio::test]
    async fn test_search() -> Result<()> {
        let actors = setup_hawk_actors().await?;

        parallelize(actors.into_iter().map(go_search)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn linear_scan_finds_matches_without_a_graph() -> Result<()> {
        let actors = setup_linear_scan_actors().await?;

        parallelize(actors.into_iter().map(go_linear_scan)).await?;

        Ok(())
    }

    async fn go_linear_scan(mut actor: HawkActor) -> Result<HawkActor> {
        init_iris_db(&mut actor).await?;

        // Deliberately leave GraphMem empty. Exact scan candidates come from
        // the versioned registry, proving the result is independent of HNSW.
        let sessions = actor.new_sessions().await?;
        let batch_size = 3;
        let request = make_request(batch_size, actor.party_id);
        request.cache_into(&actor.worker_pools).await?;
        let search_params = SearchParams::new(
            actor.searcher(),
            HawkSearchMode::LinearScan,
            true,
            None,
            0,
            false,
            #[cfg(feature = "phase_trace")]
            'L',
        );

        let extra_candidate_ids = vec![Vec::new(); batch_size];
        let forced_anon_stats_ids = vec![Vec::new(); batch_size];
        let result = linear_scan_cascade(
            &sessions,
            &request.queries(Orientation::Normal),
            search_params,
            Orientation::Normal,
            Eye::Left,
            &extra_candidate_ids,
            &forced_anon_stats_ids,
        )
        .await?;

        for side in &result {
            for (query_index, rotations) in side.iter().enumerate() {
                assert!(rotations.iter().any(|rotation| {
                    rotation
                        .classified
                        .matches
                        .results
                        .iter()
                        .any(|(id, _)| *id == VectorId::from_0_index(query_index as u32))
                }));
                assert!(rotations
                    .iter()
                    .all(|rotation| !rotation.classified.matches.saturated));
            }
        }

        // The matching module's other-eye comparisons are answered from the
        // cascade: each query matches its own record on both eyes, while a
        // different record and a stale version are reported as non-matches
        // without another MPC round.
        let ids_to_compare = [LEFT, RIGHT].map(|_| {
            (0..batch_size)
                .map(|query_index| {
                    vec![
                        VectorId::from_0_index(query_index as u32),
                        VectorId::from_0_index(((query_index + 1) % batch_size) as u32),
                        VectorId::from_0_index(query_index as u32).next_version(),
                    ]
                })
                .collect::<Vec<_>>()
        });
        let comparisons = linear_scan_comparison_results(&result, ids_to_compare);
        for side in &comparisons {
            assert_eq!(side.len(), batch_size);
            for (query_index, is_match) in side.iter().enumerate() {
                let own_id = VectorId::from_0_index(query_index as u32);
                assert_eq!(is_match.len(), 3);
                assert!(is_match[&own_id]);
                assert!(
                    !is_match[&VectorId::from_0_index(((query_index + 1) % batch_size) as u32)]
                );
                assert!(!is_match[&own_id.next_version()]);
            }
        }

        actor.sync_peers().await?;
        Ok(actor)
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
            HawkSearchMode::Hnsw,
            true,
            Some(4000),
            0,
            false,
            #[cfg(feature = "phase_trace")]
            'T',
        );

        let result = search(&sessions, search_queries, &request.ids, search_params).await?;

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
