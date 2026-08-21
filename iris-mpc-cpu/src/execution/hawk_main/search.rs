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
    hawkers::aby3::aby3_store::{
        Aby3DistanceRef, Aby3Query, Aby3Store, DistanceOps, FullRotationThresholdResult,
    },
    hnsw::{graph::UpdateEntryPoint, GraphMem, HnswSearcher},
    shares::RingElement,
};
use ampc_anon_stats::types::Eye;
use eyre::{OptionExt, Result};
use iris_mpc_common::iris_db::iris::Threshold;
use iris_mpc_common::{VectorId, ROTATIONS};
use std::collections::HashSet;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;
use tokio::sync::{
    mpsc::{unbounded_channel, UnboundedSender},
    Notify,
};
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

#[derive(Clone)]
struct LinearScanPrefetch {
    worker: Arc<dyn IrisWorkerPool>,
    /// Candidate IDs already consumed by the concurrent known-candidate
    /// stage. They must not reserve prefetch slots that no later stage reads.
    excluded_ids: Arc<VecRequests<Vec<VectorId>>>,
}

#[derive(Clone, Default)]
struct LinearScanHooks {
    prefetch: Option<LinearScanPrefetch>,
    progress: Option<Arc<LinearScanProgress>>,
}

struct LinearScanProgress {
    completed_comparisons: AtomicUsize,
    start_candidate_after: usize,
    notify: Notify,
}

impl LinearScanProgress {
    fn new(comparisons: usize) -> Self {
        Self {
            completed_comparisons: AtomicUsize::new(0),
            // Start the sparse second-eye work while the final 10% of full-eye
            // chunks drain. This leaves enough overlap to hide its latency but
            // avoids competing with the peak dot/network fan-out.
            start_candidate_after: comparisons.saturating_mul(9).div_ceil(10),
            notify: Notify::new(),
        }
    }

    fn record(&self, comparisons: usize) {
        let previous = self
            .completed_comparisons
            .fetch_add(comparisons, Ordering::AcqRel);
        if previous < self.start_candidate_after
            && previous.saturating_add(comparisons) >= self.start_candidate_after
        {
            self.notify.notify_waiters();
        }
    }

    async fn wait_for_candidate_start(&self) {
        loop {
            let notified = self.notify.notified();
            if self.completed_comparisons.load(Ordering::Acquire) >= self.start_candidate_after {
                return;
            }
            notified.await;
        }
    }
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
    let known_second_stage_ids = Arc::new(
        extra_candidate_ids
            .iter()
            .map(|extras| {
                Arc::<[VectorId]>::from(collect_live_second_stage_ids(
                    &live_ids,
                    std::iter::empty(),
                    extras,
                ))
            })
            .collect::<Vec<_>>(),
    );

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
    let first_eye_progress = Arc::new(LinearScanProgress::new(first_eye_comparisons));
    // LUC, OR-rule, and reauthentication candidates are public before the
    // scan starts. Check them on the cold eye while the full resident-eye scan
    // is running; any database I/O is thereby hidden behind the long stage.
    // Candidates discovered by the anonymous-statistics threshold are still
    // prefetched chunk by chunk and checked below.
    let first_eye_scan = linear_scan_eye(
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
        LinearScanHooks {
            prefetch: Some(LinearScanPrefetch {
                worker: prefetch_worker.clone(),
                excluded_ids: Arc::new(
                    known_second_stage_ids
                        .iter()
                        .map(|ids| ids.to_vec())
                        .collect(),
                ),
            }),
            progress: Some(first_eye_progress.clone()),
        },
    );
    let known_second_eye_scan = async {
        first_eye_progress.wait_for_candidate_start().await;
        linear_scan_eye(
            sessions,
            search_queries,
            &search_params,
            LinearScanEyeContext {
                eye: second_eye_side,
                stage: LinearScanStage::Candidate,
                orientation,
            },
            known_second_stage_ids.clone(),
            Arc::new(forced_anon_stats_ids.clone()),
            LinearScanHooks::default(),
        )
        .await
    };
    let (first_results, mut second_results) =
        tokio::try_join!(first_eye_scan, known_second_eye_scan)?;
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
    let mut discovered_second_stage_ids = Vec::with_capacity(n_requests);
    let mut discovered_candidate_count = 0usize;
    for (plans, known_ids) in first_results.iter().zip(known_second_stage_ids.iter()) {
        let mut ids = collect_live_second_stage_ids(
            &live_ids,
            plans
                .iter()
                .flat_map(|plan| &plan.classified.anon_stats_matches.results)
                .map(|(id, _)| *id),
            &[],
        );
        ids.retain(|id| known_ids.binary_search(id).is_err());
        discovered_candidate_count += ids.len();
        discovered_second_stage_ids.push(Arc::<[VectorId]>::from(ids));
    }
    let known_candidate_count = known_second_stage_ids
        .iter()
        .map(|ids| ids.len())
        .sum::<usize>();
    let candidate_count = known_candidate_count + discovered_candidate_count;

    tracing::info!(
        eye = %second_eye_side,
        orientation = orientation_label(orientation),
        requests = n_requests,
        candidates = candidate_count,
        known_candidates = known_candidate_count,
        discovered_candidates = discovered_candidate_count,
        "Running candidate-only linear-scan stage"
    );
    metrics::counter!("linear_scan_second_eye_candidates_total").increment(candidate_count as u64);
    if discovered_candidate_count > 0 {
        let discovered_results = linear_scan_eye(
            sessions,
            search_queries,
            &search_params,
            LinearScanEyeContext {
                eye: second_eye_side,
                stage: LinearScanStage::Candidate,
                orientation,
            },
            Arc::new(discovered_second_stage_ids),
            Arc::new(forced_anon_stats_ids.clone()),
            LinearScanHooks::default(),
        )
        .await?;
        merge_linear_scan_results(&mut second_results, discovered_results);
    }

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

fn exclude_known_second_stage_ids(ids: &mut Vec<VectorId>, known_ids: &[VectorId]) {
    debug_assert!(known_ids.windows(2).all(|pair| pair[0] < pair[1]));
    ids.retain(|id| known_ids.binary_search(id).is_err());
}

async fn linear_scan_eye<const ROTMASK: u32>(
    sessions: &BothEyes<Vec<HawkSession>>,
    search_queries: &SearchQueries<ROTMASK>,
    search_params: &SearchParams,
    context: LinearScanEyeContext,
    candidate_ids: Arc<VecRequests<Arc<[VectorId]>>>,
    forced_anon_stats_ids: Arc<VecRequests<Vec<VectorId>>>,
    hooks: LinearScanHooks,
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
            let hooks = hooks.clone();
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
                    if let Some(prefetch) = &hooks.prefetch {
                        let chunk_ids = &candidate_ids[chunk.i_request][chunk.range.clone()];
                        let mut prefetch_ids = collect_live_second_stage_ids(
                            chunk_ids,
                            result
                                .classified
                                .anon_stats_matches
                                .results
                                .iter()
                                .map(|(id, _)| *id),
                            &[],
                        );
                        exclude_known_second_stage_ids(
                            &mut prefetch_ids,
                            &prefetch.excluded_ids[chunk.i_request],
                        );
                        prefetch.worker.prefetch_irises(prefetch_ids).await?;
                    }
                    if let Some(progress) = &hooks.progress {
                        progress.record(chunk.range.len());
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
    let results =
        assemble_linear_scan_results(chunk_results, &search_queries[eye_index], &graph_store)?;

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

/// Merge per-chunk plans into per-request results and package them in the
/// three-slot rotation container (full result in the center slot).
fn assemble_linear_scan_results<const ROTMASK: u32>(
    chunk_results: Vec<Vec<Option<HawkInsertPlan>>>,
    queries: &VecRequests<VecRotationSupport<Aby3Query, ROTMASK>>,
    graph_store: &GraphMem,
) -> Result<VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>> {
    let n_rotations = ROTMASK.count_ones() as usize;
    let central_rotation = n_rotations / 2;
    chunk_results
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
                            queries[i_request][i_rotation],
                            graph_store,
                        ))
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(VecRotationSupport::from)
        })
        .collect()
}

/// Scheduling shape of one eye-stage scan, shared by the metric emitters.
#[derive(Clone, Copy)]
struct LinearScanStageShape {
    n_requests: usize,
    comparisons: usize,
    chunk_count: usize,
    configured_sessions: usize,
    n_workers: usize,
    min_chunks_per_session: usize,
    max_chunks_per_session: usize,
}

/// Emit the per-stage metrics and `LINEAR_SCAN_EYE_SUMMARY` line for one
/// orientation. Shared by the single and paired stage implementations so both
/// produce identical observability output.
fn emit_linear_scan_eye_summary<const ROTMASK: u32>(
    context: LinearScanEyeContext,
    shape: LinearScanStageShape,
    results: &VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>,
    elapsed_seconds: f64,
) {
    let n_rotations = ROTMASK.count_ones() as usize;
    let central_rotation = n_rotations / 2;
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

    let comparisons_per_second = shape.comparisons as f64 / elapsed_seconds.max(f64::EPSILON);
    let session_utilization = shape.n_workers as f64 / shape.configured_sessions as f64;
    let stage_label = context.stage.as_str();
    let eye_label = eye_label(context.eye);
    let orientation_label = orientation_label(context.orientation);
    metrics::counter!(
        "linear_scan_eye_comparisons_total",
        "eye" => eye_label,
        "stage" => stage_label,
        "orientation" => orientation_label,
    )
    .increment(shape.comparisons as u64);
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
    .record(shape.comparisons as f64);
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
    .record(shape.chunk_count as f64);
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
    .record(shape.n_workers as f64);
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
        requests = shape.n_requests,
        comparisons = shape.comparisons,
        rotations_per_comparison = ROTATIONS,
        chunks = shape.chunk_count,
        chunk_size = LINEAR_SCAN_CHUNK_SIZE,
        configured_sessions = shape.configured_sessions,
        active_sessions = shape.n_workers,
        session_utilization,
        min_chunks_per_session = shape.min_chunks_per_session,
        max_chunks_per_session = shape.max_chunks_per_session,
        strict_match_records,
        anon_stats_rotation_matches,
        elapsed_seconds,
        comparisons_per_second,
        "LINEAR_SCAN_EYE_SUMMARY"
    );
}

/// Fused full-eye stage for both orientations: one chunk grid over the shared
/// live-ID list, with paired sessions. Each chunk streams its targets once for
/// both orientations' dot products; each orientation's threshold rounds then
/// run on that orientation's own session. Chunk-to-session assignment matches
/// [`linear_scan_eye`], so every per-orientation session sees the same chunk
/// sequence (and therefore the same network transcript) as two independent
/// stages.
#[allow(clippy::too_many_arguments)]
async fn linear_scan_full_stage_paired<const ROTMASK: u32>(
    sessions_both: [&BothEyes<Vec<HawkSession>>; 2],
    search_queries_both: [&SearchQueries<ROTMASK>; 2],
    search_params_both: [&SearchParams; 2],
    contexts: [LinearScanEyeContext; 2],
    full_scan_ids: Arc<VecRequests<Arc<[VectorId]>>>,
    forced_anon_stats_ids: Arc<VecRequests<Vec<VectorId>>>,
    hooks: LinearScanHooks,
) -> Result<[VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>; 2]> {
    let stage_start = Instant::now();
    let eye_index = eye_index(contexts[0].eye);
    debug_assert_eq!(eye_index, self::eye_index(contexts[1].eye));
    let n_requests = search_queries_both[0][eye_index].len();
    assert_eq!(n_requests, search_queries_both[1][eye_index].len());
    assert_eq!(n_requests, full_scan_ids.len());
    assert_eq!(n_requests, forced_anon_stats_ids.len());
    let comparisons = full_scan_ids.iter().map(|ids| ids.len()).sum::<usize>();
    let central_rotation = ROTMASK.count_ones() as usize / 2;

    let mut chunks_per_request = Vec::with_capacity(n_requests);
    let mut chunks = Vec::new();
    for (i_request, ids) in full_scan_ids.iter().enumerate() {
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
    let configured_sessions = sessions_both[0][eye_index]
        .len()
        .min(sessions_both[1][eye_index].len());
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
            let session_a = sessions_both[0][eye_index][i_session].clone();
            let session_b = sessions_both[1][eye_index][i_session].clone();
            let search_queries_a = search_queries_both[0].clone();
            let search_queries_b = search_queries_both[1].clone();
            let search_params_a = search_params_both[0].clone();
            let search_params_b = search_params_both[1].clone();
            let full_scan_ids = full_scan_ids.clone();
            let forced_anon_stats_ids = forced_anon_stats_ids.clone();
            let hooks = hooks.clone();
            async move {
                let mut store_a = session_a.aby3_store.write().await;
                let mut store_b = session_b.aby3_store.write().await;
                let graph_a = session_a.graph_store.clone().read_owned().await;
                let graph_b = session_b.graph_store.clone().read_owned().await;
                let mut results = Vec::with_capacity(batch.len());
                // Software-pipeline this lane: the next chunk's fused dot
                // products run on the worker pool while the current chunk's
                // threshold rounds are in flight, so the dot workers stay fed
                // instead of idling for a round trip per chunk.
                let queries_for = |chunk: &LinearScanChunk| {
                    (
                        search_queries_a[eye_index][chunk.i_request][central_rotation],
                        search_queries_b[eye_index][chunk.i_request][central_rotation],
                    )
                };
                let do_match = search_params_a.do_match;
                let dispatch = |store: &Aby3Store<HawkOps>, chunk: &LinearScanChunk| {
                    let (query_a, query_b) = queries_for(chunk);
                    store.spawn_full_rotation_dot_contributions_pair(
                        [&query_a, &query_b],
                        &full_scan_ids[chunk.i_request][chunk.range.clone()],
                    )
                };
                // Keep one chunk of dot work buffered per lane: the next
                // chunk's dot products run while this chunk's threshold
                // rounds are in flight. Deeper buffering measures worse — all
                // dot work then completes early and the stage drains on
                // thresholds alone with idle dot workers.
                const DOT_PIPELINE_DEPTH: usize = 1;
                // Handles abort their task when dropped, so an error anywhere
                // in this lane (or a sibling lane failing `try_join!`) also
                // cancels the lookahead chunk instead of leaving it running.
                let mut pending_dots = std::collections::VecDeque::new();
                if do_match {
                    for chunk in batch.iter().take(DOT_PIPELINE_DEPTH) {
                        pending_dots.push_back(dispatch(&store_a, chunk)?);
                    }
                }
                for (index, chunk) in batch.iter().enumerate() {
                    let contributions = match pending_dots.pop_front() {
                        Some(handle) => handle
                            .await
                            .map_err(|error| eyre::eyre!("fused dot task failed: {error}"))??,
                        None => [Vec::new(), Vec::new()],
                    };
                    if do_match {
                        if let Some(next) = batch.get(index + DOT_PIPELINE_DEPTH) {
                            pending_dots.push_back(dispatch(&store_a, next)?);
                        }
                    }
                    let (query_a, query_b) = queries_for(chunk);
                    let chunk_ids = &full_scan_ids[chunk.i_request][chunk.range.clone()];
                    let plans = per_linear_scan_chunk_pair(
                        [query_a, query_b],
                        [&search_params_a, &search_params_b],
                        (&mut store_a, &mut store_b),
                        (&graph_a, &graph_b),
                        contributions,
                        chunk_ids,
                        &forced_anon_stats_ids[chunk.i_request],
                    )
                    .await?;
                    if let Some(prefetch) = &hooks.prefetch {
                        // One union prefetch warms the cold eye for both
                        // orientations' second-stage candidates.
                        let mut prefetch_ids = collect_live_second_stage_ids(
                            chunk_ids,
                            plans.iter().flat_map(|plan| {
                                plan.classified
                                    .anon_stats_matches
                                    .results
                                    .iter()
                                    .map(|(id, _)| *id)
                            }),
                            &[],
                        );
                        exclude_known_second_stage_ids(
                            &mut prefetch_ids,
                            &prefetch.excluded_ids[chunk.i_request],
                        );
                        prefetch.worker.prefetch_irises(prefetch_ids).await?;
                    }
                    if let Some(progress) = &hooks.progress {
                        progress.record(chunk.range.len());
                    }
                    results.push((chunk.i_request, chunk.i_chunk, plans));
                }
                Ok(results)
            }
        });

    let mut chunk_results: [Vec<Vec<Option<HawkInsertPlan>>>; 2] = [
        chunks_per_request
            .iter()
            .map(|&len| vec![None; len])
            .collect(),
        chunks_per_request
            .iter()
            .map(|&len| vec![None; len])
            .collect(),
    ];
    for (i_request, i_chunk, plans) in parallelize(jobs).await?.into_iter().flatten() {
        let [plan_a, plan_b] = plans;
        chunk_results[0][i_request][i_chunk] = Some(plan_a);
        chunk_results[1][i_request][i_chunk] = Some(plan_b);
    }
    let [chunk_results_a, chunk_results_b] = chunk_results;

    let graph_a = sessions_both[0][eye_index][0].graph_store.read().await;
    let graph_b = sessions_both[1][eye_index][0].graph_store.read().await;
    let results = [
        assemble_linear_scan_results(
            chunk_results_a,
            &search_queries_both[0][eye_index],
            &graph_a,
        )?,
        assemble_linear_scan_results(
            chunk_results_b,
            &search_queries_both[1][eye_index],
            &graph_b,
        )?,
    ];

    let elapsed_seconds = stage_start.elapsed().as_secs_f64();
    let shape = LinearScanStageShape {
        n_requests,
        comparisons,
        chunk_count,
        configured_sessions,
        n_workers,
        min_chunks_per_session,
        max_chunks_per_session,
    };
    for (context, results) in contexts.iter().zip(&results) {
        emit_linear_scan_eye_summary(*context, shape, results, elapsed_seconds);
    }

    Ok(results)
}

/// Run both orientations' two-eye linear-scan cascades with a fused first-eye
/// stage: the resident full-scan eye is streamed once and every loaded target
/// feeds both orientations' 31-rotation dot products. All MPC threshold work
/// stays on each orientation's own sessions, so results and per-orientation
/// transcripts are identical to two concurrent [`linear_scan_cascade`] calls.
/// The sparse second-eye candidate stages remain per-orientation.
#[instrument(level = "trace", target = "searcher::network", skip_all)]
pub async fn linear_scan_cascade_paired<const ROTMASK: u32>(
    sessions_both: [&BothEyes<Vec<HawkSession>>; 2],
    search_queries_both: [&SearchQueries<ROTMASK>; 2],
    search_params_both: [SearchParams; 2],
    orientations: [Orientation; 2],
    full_scan_side: Eye,
    extra_candidate_ids_both: [&VecRequests<Vec<VectorId>>; 2],
    forced_anon_stats_ids: &VecRequests<Vec<VectorId>>,
) -> Result<[SearchResults<ROTMASK>; 2]> {
    let cascade_start = Instant::now();
    for params in &search_params_both {
        debug_assert_eq!(params.mode, HawkSearchMode::LinearScan);
    }

    let first_eye = eye_index(full_scan_side);
    let second_eye_side = full_scan_side.other();
    let second_eye = eye_index(second_eye_side);

    let n_requests = search_queries_both[0][first_eye].len();
    for sessions in sessions_both {
        let n_sessions = sessions[LEFT].len();
        assert!(n_sessions > 0, "linear scan requires at least one session");
        assert_eq!(n_sessions, sessions[RIGHT].len());
    }
    for search_queries in search_queries_both {
        assert_eq!(n_requests, search_queries[LEFT].len());
        assert_eq!(n_requests, search_queries[RIGHT].len());
    }
    for extra_candidate_ids in extra_candidate_ids_both {
        assert_eq!(n_requests, extra_candidate_ids.len());
    }
    assert_eq!(n_requests, forced_anon_stats_ids.len());

    // Both eye registries and both orientation groups observe the same live
    // VectorIds. Build the list once and share it across the fused stage.
    let live_ids = {
        let vector_store = sessions_both[0][first_eye][0].aby3_store.read().await;
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
    debug_assert!(live_ids.windows(2).all(|pair| pair[0] < pair[1]));

    let known_second_stage_ids_both = extra_candidate_ids_both.map(|extra_candidate_ids| {
        Arc::new(
            extra_candidate_ids
                .iter()
                .map(|extras| {
                    Arc::<[VectorId]>::from(collect_live_second_stage_ids(
                        &live_ids,
                        std::iter::empty(),
                        extras,
                    ))
                })
                .collect::<Vec<_>>(),
        )
    });
    let prefetch_excluded_ids = Arc::new(
        (0..n_requests)
            .map(|i_request| {
                let mut ids = known_second_stage_ids_both[0][i_request].to_vec();
                ids.extend_from_slice(&known_second_stage_ids_both[1][i_request]);
                ids.sort_unstable();
                ids.dedup();
                ids
            })
            .collect(),
    );

    for orientation in orientations {
        tracing::info!(
            eye = %full_scan_side,
            orientation = orientation_label(orientation),
            requests = n_requests,
            vectors = live_ids.len(),
            "Running full linear-scan stage"
        );
    }
    let prefetch_worker = {
        let store = sessions_both[0][second_eye][0].aby3_store.read().await;
        store.workers.clone()
    };
    let first_eye_progress = Arc::new(LinearScanProgress::new(first_eye_comparisons));
    let contexts = orientations.map(|orientation| LinearScanEyeContext {
        eye: full_scan_side,
        stage: LinearScanStage::Full,
        orientation,
    });
    let forced_anon_stats_ids_shared = Arc::new(forced_anon_stats_ids.clone());
    let first_eye_scan = linear_scan_full_stage_paired(
        sessions_both,
        search_queries_both,
        [&search_params_both[0], &search_params_both[1]],
        contexts,
        full_scan_ids,
        forced_anon_stats_ids_shared.clone(),
        LinearScanHooks {
            prefetch: Some(LinearScanPrefetch {
                worker: prefetch_worker.clone(),
                excluded_ids: prefetch_excluded_ids,
            }),
            progress: Some(first_eye_progress.clone()),
        },
    );
    // LUC and reauthentication candidates are public before the scan starts.
    // Check them on the cold eye per orientation while the fused resident-eye
    // scan is running.
    let known_second_eye_scan = |index: usize| {
        let progress = first_eye_progress.clone();
        let known_ids = known_second_stage_ids_both[index].clone();
        let forced = forced_anon_stats_ids_shared.clone();
        let search_params = search_params_both[index].clone();
        async move {
            progress.wait_for_candidate_start().await;
            linear_scan_eye(
                sessions_both[index],
                search_queries_both[index],
                &search_params,
                LinearScanEyeContext {
                    eye: second_eye_side,
                    stage: LinearScanStage::Candidate,
                    orientation: orientations[index],
                },
                known_ids,
                forced,
                LinearScanHooks::default(),
            )
            .await
        }
    };
    let (first_results_both, known_results_a, known_results_b) = tokio::try_join!(
        first_eye_scan,
        known_second_eye_scan(0),
        known_second_eye_scan(1),
    )?;
    let mut second_results_both = [known_results_a, known_results_b];
    let prefetch_wait_start = Instant::now();
    prefetch_worker.wait_for_prefetch().await?;
    let prefetch_wait_seconds = prefetch_wait_start.elapsed().as_secs_f64();
    metrics::histogram!("linear_scan_cold_prefetch_wait_duration").record(prefetch_wait_seconds);
    for orientation in orientations {
        metrics::histogram!(
            "linear_scan_cascade_prefetch_wait_duration",
            "eye" => eye_label(second_eye_side),
            "orientation" => orientation_label(orientation),
        )
        .record(prefetch_wait_seconds);
    }

    // Discovered candidates per orientation, exactly as in the single-cascade
    // path: retain live IDs, drop already-checked known candidates.
    let mut discovered_ids_both = Vec::with_capacity(2);
    let mut candidate_counts = [0usize; 2];
    for index in 0..2 {
        let mut discovered_second_stage_ids = Vec::with_capacity(n_requests);
        let mut discovered_candidate_count = 0usize;
        for (plans, known_ids) in first_results_both[index]
            .iter()
            .zip(known_second_stage_ids_both[index].iter())
        {
            let mut ids = collect_live_second_stage_ids(
                &live_ids,
                plans
                    .iter()
                    .flat_map(|plan| &plan.classified.anon_stats_matches.results)
                    .map(|(id, _)| *id),
                &[],
            );
            ids.retain(|id| known_ids.binary_search(id).is_err());
            discovered_candidate_count += ids.len();
            discovered_second_stage_ids.push(Arc::<[VectorId]>::from(ids));
        }
        let known_candidate_count = known_second_stage_ids_both[index]
            .iter()
            .map(|ids| ids.len())
            .sum::<usize>();
        let candidate_count = known_candidate_count + discovered_candidate_count;
        candidate_counts[index] = candidate_count;

        tracing::info!(
            eye = %second_eye_side,
            orientation = orientation_label(orientations[index]),
            requests = n_requests,
            candidates = candidate_count,
            known_candidates = known_candidate_count,
            discovered_candidates = discovered_candidate_count,
            "Running candidate-only linear-scan stage"
        );
        metrics::counter!("linear_scan_second_eye_candidates_total")
            .increment(candidate_count as u64);
        discovered_ids_both.push((discovered_candidate_count, discovered_second_stage_ids));
    }

    let discovered_scan = |index: usize, discovered: Vec<Arc<[VectorId]>>| {
        let search_params = search_params_both[index].clone();
        let forced = forced_anon_stats_ids_shared.clone();
        async move {
            linear_scan_eye(
                sessions_both[index],
                search_queries_both[index],
                &search_params,
                LinearScanEyeContext {
                    eye: second_eye_side,
                    stage: LinearScanStage::Candidate,
                    orientation: orientations[index],
                },
                Arc::new(discovered),
                forced,
                LinearScanHooks::default(),
            )
            .await
        }
    };
    let mut discovered_iter = discovered_ids_both.into_iter();
    let (discovered_count_a, discovered_ids_a) = discovered_iter.next().expect("two orientations");
    let (discovered_count_b, discovered_ids_b) = discovered_iter.next().expect("two orientations");
    let (discovered_results_a, discovered_results_b) = tokio::try_join!(
        async {
            if discovered_count_a > 0 {
                discovered_scan(0, discovered_ids_a).await.map(Some)
            } else {
                Ok(None)
            }
        },
        async {
            if discovered_count_b > 0 {
                discovered_scan(1, discovered_ids_b).await.map(Some)
            } else {
                Ok(None)
            }
        },
    )?;
    if let Some(results) = discovered_results_a {
        merge_linear_scan_results(&mut second_results_both[0], results);
    }
    if let Some(results) = discovered_results_b {
        merge_linear_scan_results(&mut second_results_both[1], results);
    }

    let elapsed_seconds = cascade_start.elapsed().as_secs_f64();
    for index in 0..2 {
        let orientation = orientation_label(orientations[index]);
        let candidate_count = candidate_counts[index];
        let total_comparisons = first_eye_comparisons + candidate_count;
        let comparisons_per_second = total_comparisons as f64 / elapsed_seconds.max(f64::EPSILON);
        let second_eye_candidate_fraction = if first_eye_comparisons == 0 {
            0.0
        } else {
            candidate_count as f64 / first_eye_comparisons as f64
        };
        metrics::counter!(
            "linear_scan_cascade_comparisons_total",
            "orientation" => orientation,
        )
        .increment(total_comparisons as u64);
        metrics::histogram!(
            "linear_scan_cascade_duration",
            "orientation" => orientation,
        )
        .record(elapsed_seconds);
        metrics::histogram!(
            "linear_scan_cascade_comparisons_per_second",
            "orientation" => orientation,
        )
        .record(comparisons_per_second);
        metrics::histogram!(
            "linear_scan_second_eye_candidate_fraction",
            "eye" => eye_label(second_eye_side),
            "orientation" => orientation,
        )
        .record(second_eye_candidate_fraction);
        tracing::info!(
            orientation = orientation,
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
    }

    let [first_a, first_b] = first_results_both;
    let [second_a, second_b] = second_results_both;
    let pack = |first_results, second_results| match full_scan_side {
        Eye::Left => [first_results, second_results],
        Eye::Right => [second_results, first_results],
    };
    Ok([pack(first_a, second_a), pack(first_b, second_b)])
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

fn merge_linear_scan_results<const ROTMASK: u32>(
    target: &mut VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>,
    source: VecRequests<VecRotationSupport<HawkInsertPlan, ROTMASK>>,
) {
    for (target, source) in target.iter_mut().zip(source) {
        let target = target.center_mut();
        merge_linear_scan_plan(target, source.into_center());

        // The early known set and the later threshold-discovered set are each
        // serial-ID ordered but may interleave. Restore the single sorted order
        // produced by the non-overlapped scan; stable sorting also preserves
        // rotation order for repeated anonymous-statistics IDs.
        target.classified.matches.results.sort_by_key(|(id, _)| *id);
        target
            .classified
            .anon_stats_matches
            .results
            .sort_by_key(|(id, _)| *id);
        if let Some(pre_extension) = &mut target.classified.pre_extension {
            pre_extension.results.sort_by_key(|(id, _)| *id);
        }
        target
            .classified
            .partial_match_rotations
            .sort_by_key(|(id, _)| *id);
    }
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
        classified
            .matches
            .results
            .reserve(vector_ids.len().min(4096));
        classified
            .anon_stats_matches
            .results
            .reserve(vector_ids.len().min(4096));

        for ids in vector_ids.chunks(LINEAR_SCAN_CHUNK_SIZE) {
            let forced_anon_stats_vectors = forced_anon_stats_ids
                .iter()
                .filter_map(|id| ids.binary_search(id).ok())
                .collect::<Vec<_>>();
            let thresholds = aby3_store
                .eval_distance_batch_full_rotation_thresholds_fused_with_forced_anon_stats(
                    &query,
                    ids,
                    &forced_anon_stats_vectors,
                )
                .await?;
            extend_classified_from_thresholds(
                &mut classified,
                ids,
                thresholds,
                search_params.return_partial_results,
            );
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

/// Classify one chunk's opened threshold results into the accumulating
/// [`ClassifiedMatches`]. Shared by the single-orientation and paired scans.
fn extend_classified_from_thresholds(
    classified: &mut ClassifiedMatches,
    ids: &[VectorId],
    thresholds: FullRotationThresholdResult,
    return_partial_results: bool,
) {
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

    if return_partial_results {
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

/// Threshold rounds and classification for one fused chunk whose local dot
/// contributions were already computed (typically pipelined on the worker
/// pool while the previous chunk's thresholds ran). Each orientation's
/// threshold protocol runs on its own session, so the per-orientation network
/// transcript is identical to the unfused path.
#[allow(clippy::too_many_arguments)]
async fn per_linear_scan_chunk_pair(
    queries: [Aby3Query; 2],
    search_params: [&SearchParams; 2],
    stores: (&mut Aby3Store<HawkOps>, &mut Aby3Store<HawkOps>),
    graph_stores: (&GraphMem, &GraphMem),
    contributions: [Vec<RingElement<u16>>; 2],
    vector_ids: &[VectorId],
    forced_anon_stats_ids: &[VectorId],
) -> Result<[HawkInsertPlan; 2]> {
    let start = Instant::now();
    let (store_a, store_b) = stores;
    let mut classified = [ClassifiedMatches::default(), ClassifiedMatches::default()];
    debug_assert_eq!(search_params[0].do_match, search_params[1].do_match);
    debug_assert!(vector_ids.len() <= LINEAR_SCAN_CHUNK_SIZE);

    if search_params[0].do_match {
        let ids = vector_ids;
        let forced_anon_stats_vectors = forced_anon_stats_ids
            .iter()
            .filter_map(|id| ids.binary_search(id).ok())
            .collect::<Vec<_>>();
        let [contributions_a, contributions_b] = contributions;
        let (thresholds_a, thresholds_b) = tokio::try_join!(
            store_a.eval_full_rotation_thresholds_fused_from_contributions_with_forced_anon_stats(
                contributions_a,
                ids.len(),
                &forced_anon_stats_vectors,
            ),
            store_b.eval_full_rotation_thresholds_fused_from_contributions_with_forced_anon_stats(
                contributions_b,
                ids.len(),
                &forced_anon_stats_vectors,
            ),
        )?;
        extend_classified_from_thresholds(
            &mut classified[0],
            ids,
            thresholds_a,
            search_params[0].return_partial_results,
        );
        extend_classified_from_thresholds(
            &mut classified[1],
            ids,
            thresholds_b,
            search_params[1].return_partial_results,
        );

        for (side, params) in classified.iter_mut().zip(&search_params) {
            side.linear_scan_supermatch_threshold = params
                .hnsw_supermatch
                .as_ref()
                .map(|searcher| searcher.params.get_ef_search(0));
        }
    }

    metrics::histogram!("linear_scan_query_duration").record(start.elapsed().as_secs_f64());
    metrics::counter!("linear_scan_vectors_total").increment(2 * vector_ids.len() as u64);

    let [classified_a, classified_b] = classified;
    let as_plan = |query: Aby3Query, as_of, classified| HawkInsertPlan {
        plan: InsertPlanV {
            query,
            links: Vec::new(),
            update_ep: UpdateEntryPoint::False,
            as_of,
        },
        classified,
    };
    Ok([
        as_plan(queries[0], graph_stores.0.last_update_seq_no, classified_a),
        as_plan(queries[1], graph_stores.1.last_update_seq_no, classified_b),
    ])
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

    #[test]
    fn prefetch_excludes_candidates_already_consumed_by_known_stage() {
        let mut discovered = vec![
            VectorId::from_serial_id(1),
            VectorId::from_serial_id(3),
            VectorId::from_serial_id(5),
        ];
        let known = [
            VectorId::from_serial_id(2),
            VectorId::from_serial_id(3),
            VectorId::from_serial_id(4),
        ];

        exclude_known_second_stage_ids(&mut discovered, &known);

        assert_eq!(
            discovered,
            vec![VectorId::from_serial_id(1), VectorId::from_serial_id(5)]
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

        // Exercise both overlapped known candidates and candidates discovered
        // only by the full-eye threshold. Even-numbered requests include their
        // exact match up front; odd-numbered requests include a different
        // live record so their exact match must be merged in afterward.
        let extra_candidate_ids = (0..batch_size)
            .map(|index| {
                let extra = if index.is_multiple_of(2) {
                    index
                } else {
                    (index + 1) % batch_size
                };
                vec![VectorId::from_0_index(extra as u32)]
            })
            .collect::<Vec<_>>();
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
                for rotation in rotations.iter() {
                    assert!(rotation
                        .classified
                        .matches
                        .results
                        .windows(2)
                        .all(|pair| pair[0].0 <= pair[1].0));
                    assert!(rotation
                        .classified
                        .anon_stats_matches
                        .results
                        .windows(2)
                        .all(|pair| pair[0].0 <= pair[1].0));
                }
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

    #[tokio::test]
    async fn paired_linear_scan_matches_single_cascades() -> Result<()> {
        let actors = setup_linear_scan_actors().await?;

        parallelize(actors.into_iter().map(go_paired_linear_scan)).await?;

        Ok(())
    }

    /// Opened-result projection of one plan: everything that is deterministic
    /// across protocol runs. Distance *shares* differ between session sets by
    /// construction, so they are excluded.
    #[allow(clippy::type_complexity)]
    fn opened_projection<const ROTMASK: u32>(
        results: &SearchResults<ROTMASK>,
    ) -> Vec<
        Vec<
            Vec<(
                Aby3Query,
                Vec<VectorId>,
                bool,
                Vec<VectorId>,
                bool,
                Vec<(VectorId, Vec<i8>)>,
                Option<usize>,
            )>,
        >,
    > {
        results
            .iter()
            .map(|side| {
                side.iter()
                    .map(|rotations| {
                        rotations
                            .iter()
                            .map(|plan| {
                                (
                                    plan.plan.query,
                                    plan.classified
                                        .matches
                                        .results
                                        .iter()
                                        .map(|(id, _)| *id)
                                        .collect(),
                                    plan.classified.matches.saturated,
                                    plan.classified
                                        .anon_stats_matches
                                        .results
                                        .iter()
                                        .map(|(id, _)| *id)
                                        .collect(),
                                    plan.classified.anon_stats_matches.saturated,
                                    plan.classified.partial_match_rotations.clone(),
                                    plan.classified.linear_scan_supermatch_threshold,
                                )
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    async fn go_paired_linear_scan(mut actor: HawkActor) -> Result<HawkActor> {
        init_iris_db(&mut actor).await?;

        let sessions_normal = actor.new_sessions().await?;
        let sessions_mirror = actor.new_sessions().await?;
        let sessions_reference = actor.new_sessions().await?;
        let batch_size = 3;
        let request = make_request(batch_size, actor.party_id);
        request.cache_into(&actor.worker_pools).await?;
        let search_params = SearchParams::new(
            actor.searcher(),
            HawkSearchMode::LinearScan,
            true,
            None,
            0,
            true,
            #[cfg(feature = "phase_trace")]
            'L',
        );

        // Same candidate shapes as `go_linear_scan`: overlapped known
        // candidates plus threshold-discovered ones.
        let extra_candidate_ids = (0..batch_size)
            .map(|index| {
                let extra = if index.is_multiple_of(2) {
                    index
                } else {
                    (index + 1) % batch_size
                };
                vec![VectorId::from_0_index(extra as u32)]
            })
            .collect::<Vec<_>>();
        let forced_anon_stats_ids = vec![Vec::new(); batch_size];

        let queries_normal = request.queries(Orientation::Normal);
        let queries_mirror = request.queries(Orientation::Mirror);
        let [paired_normal, paired_mirror] = linear_scan_cascade_paired(
            [&sessions_normal, &sessions_mirror],
            [&queries_normal, &queries_mirror],
            [search_params.clone(), search_params.clone()],
            [Orientation::Normal, Orientation::Mirror],
            Eye::Left,
            [&extra_candidate_ids, &extra_candidate_ids],
            &forced_anon_stats_ids,
        )
        .await?;

        let reference_normal = linear_scan_cascade(
            &sessions_reference,
            &queries_normal,
            search_params.clone(),
            Orientation::Normal,
            Eye::Left,
            &extra_candidate_ids,
            &forced_anon_stats_ids,
        )
        .await?;
        let reference_mirror = linear_scan_cascade(
            &sessions_reference,
            &queries_mirror,
            search_params,
            Orientation::Mirror,
            Eye::Left,
            &extra_candidate_ids,
            &forced_anon_stats_ids,
        )
        .await?;

        assert_eq!(
            opened_projection(&paired_normal),
            opened_projection(&reference_normal),
            "normal orientation"
        );
        assert_eq!(
            opened_projection(&paired_mirror),
            opened_projection(&reference_mirror),
            "mirror orientation"
        );
        // The fused pass must find the planted matches, not merely agree with
        // an equally-empty reference.
        for side in &paired_normal {
            for (query_index, rotations) in side.iter().enumerate() {
                assert!(rotations.iter().any(|rotation| {
                    rotation
                        .classified
                        .matches
                        .results
                        .iter()
                        .any(|(id, _)| *id == VectorId::from_0_index(query_index as u32))
                }));
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
