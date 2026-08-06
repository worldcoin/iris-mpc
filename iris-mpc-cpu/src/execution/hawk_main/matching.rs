//! Turning search results into match decisions.
//!
//! The two eyes are searched by separate HNSW indices, so each returns its own set of
//! matching ids. We cannot AND/OR them directly: an id returned for one eye may not have
//! been considered at all by the other, which does not mean it would not match. So for
//! every id seen on one eye only, we compare the other eye explicitly via MPC, and only
//! then apply the final matching logic.
//!
//! The pipeline runs in two stages:
//!
//! 1. [`PendingBatch::new`] organizes the nearest-neighbour results for one orientation.
//!    Then:
//!    1. [`PendingBatch::ids_to_compare`] gives the vectors found on one eye only.
//!    2. The caller computes their `is_match` on the other eye with MPC.
//!    3. [`PendingBatch::resolve`] takes those results back, producing a
//!       [`ResolvedBatch`].
//! 2. [`ResolvedBatch::decide`] combines the normal and mirror orientations and makes the
//!    final decision for every request. It is also the only place per-batch supermatcher
//!    metrics are emitted. The resulting [`MatchResults`] exposes those decisions and the
//!    matched ids for reporting.
//!
//! Terms used throughout:
//!
//! - **join**: the per-request merge of both eyes' hits, keyed by vector id and recording
//!   which eye(s) matched. "Unresolved" before the MPC comparisons, "resolved" after.
//! - **saturation / supermatch**: a search whose results were (nearly) all matches, so
//!   more matches likely exist beyond `ef`. Treated as a match for uniqueness purposes.
//! - **effective vs. baseline**: when saturation triggers the supermatcher, the search is
//!   re-run with a larger `ef` and those extended results become the *effective* ones.
//!   The original results are kept as the *baseline*, used only for A/B metrics.

use super::{
    intra_batch::IntraMatch, BothEyes, HawkInsertPlan, MapEdges, Orientation, StoreId, UseOrRule,
    VecEdges, VecRequests, VecRotations, VectorId, LEFT, RIGHT,
};
use itertools::{chain, izip, Itertools};
use std::collections::HashMap;

use Decision::*;
use MatchId::*;
use OnlyOrBoth::{Both, Only};
use Orientation::{Mirror, Normal};
use StoreId::{Left, Right};

// ===========================================================================
// Vocabulary: filters, match ids, decisions, request types, search variants
// ===========================================================================

/// Search *AND* policy: only match if both eyes match (like `mergeDbResults`).
///
/// LUC *OR* policy: "Local" irises match if either side matches.
///
/// Intra-batch *AND* policy: match against requests before this request in the same batch.
///
/// Partial matches: set `eyes: Only(Left)` or `eyes: Only(Right)`.
///
/// Mirror matches: set `orient: Only(Mirror)`.
#[derive(Copy, Clone)]
pub struct Filter {
    pub eyes: OnlyOrBoth<StoreId>,
    pub orient: OnlyOrBoth<Orientation>,
    pub intra_batch: bool,
}

#[derive(Copy, Clone)]
pub enum OnlyOrBoth<T> {
    Only(T),
    Both,
}

impl Filter {
    fn search_rule(&self, left: bool, right: bool) -> bool {
        match self.eyes {
            Only(Left) => left,
            Only(Right) => right,
            Both => left && right,
        }
    }

    fn luc_rule(&self, left: bool, right: bool) -> bool {
        match self.eyes {
            Only(Left) => left,
            Only(Right) => right,
            Both => left || right,
        }
    }

    /// Decide if this is a successful reauth based on left and right matches.
    /// Use the OR or AND rule as specified in the reauth request.
    fn reauth_rule(&self, or_rule: UseOrRule, [left, right]: BothEyes<bool>) -> bool {
        match self.eyes {
            Only(Left) => left,
            Only(Right) => right,
            Both if or_rule => left || right,
            Both => left && right,
        }
    }

    fn intra_rule(&self, left: bool, right: bool) -> bool {
        self.intra_batch && self.search_rule(left, right)
    }

    /// Supermatch uses OR policy: saturated on either eye is a supermatch.
    fn supermatch_rule(&self, [left, right]: BothEyes<bool>) -> bool {
        match self.eyes {
            Only(Left) => left,
            Only(Right) => right,
            Both => left || right,
        }
    }
}

/// Wide filter: any match in any orientation, including intra-batch peers.
pub const DECISION_FILTER: Filter = Filter {
    eyes: OnlyOrBoth::Both,
    orient: OnlyOrBoth::Both,
    intra_batch: true,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MatchId {
    Search(VectorId),
    Luc(VectorId),
    Reauth(VectorId),
    IntraBatch(usize),
    /// Search results were saturated (supermatcher).
    Supermatch,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    UniqueInsert,
    UniqueInsertSkipped,
    ReauthUpdate(VectorId),
    NoMutation,
}

impl Decision {
    pub fn is_mutation(&self) -> bool {
        match self {
            UniqueInsert | ReauthUpdate(_) => true,
            UniqueInsertSkipped | NoMutation => false,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RequestType {
    /// A request to check if a vector is unique.
    Uniqueness { skip_persistence: bool },
    /// A request to check if a vector is unique without inserting it.
    IdentityMatchCheck,
    /// A request to check if a vector matches a target and replace it.
    Reauth {
        /// Target vector id and whether to use an OR-rule for comparison
        target: Option<(VectorId, UseOrRule)>,
    },
    /// Other features.
    Unsupported,
}

/// A value in the form decisions are actually made on, plus the pre-extension
/// counterfactual retained for supermatcher A/B metrics.
///
/// `baseline` is `Some` only when the supermatcher re-searched with a larger `ef` for at
/// least one rotation of this request. When it is `None`, `effective` *is* the baseline:
/// no extension happened, so the two would be identical.
#[derive(Clone, Debug, PartialEq, Eq)]
struct WithBaseline<T> {
    effective: T,
    baseline: Option<T>,
}

impl<T> WithBaseline<T> {
    /// The requested form, falling back to `effective` when no extension happened.
    fn get(&self, variant: SearchVariant) -> &T {
        match variant {
            SearchVariant::Effective => &self.effective,
            SearchVariant::Baseline => self.baseline.as_ref().unwrap_or(&self.effective),
        }
    }

    /// Every stored form: one item when no extension happened, two when it did.
    fn variants(&self) -> impl Iterator<Item = &T> + '_ {
        chain!(std::iter::once(&self.effective), &self.baseline)
    }

    /// True if the supermatcher extended this request's search.
    fn was_extended(&self) -> bool {
        self.baseline.is_some()
    }

    fn map<U>(&self, f: impl Fn(&T) -> U) -> WithBaseline<U> {
        WithBaseline {
            effective: f(&self.effective),
            baseline: self.baseline.as_ref().map(f),
        }
    }
}

/// Which form of the search results to evaluate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SearchVariant {
    /// The results decisions are made on: extended, if the supermatcher ran.
    Effective,
    /// What the original search alone would have produced.
    Baseline,
}

// ===========================================================================
// Stage 1: join the two eyes' search results, then resolve one-eyed matches via MPC
// ===========================================================================

/// One orientation's per-request search results, not yet resolved against the other
/// eye. Consumed by [`PendingBatch::resolve`].
pub struct PendingBatch(VecRequests<PendingRequest>);

impl PendingBatch {
    pub fn new(
        plans: &BothEyes<VecRequests<VecRotations<HawkInsertPlan>>>,
        luc_ids: &VecRequests<Vec<VectorId>>,
        request_types: VecRequests<RequestType>,
    ) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT], luc_ids, request_types)
                .map(|(left, right, luc, rt)| PendingRequest::new([left, right], luc.clone(), rt))
                .collect_vec(),
        )
    }

    /// The vectors that need an MPC comparison on each eye before matches can be resolved.
    /// Indexed `[eye_to_compare][request]`.
    pub fn ids_to_compare(&self) -> BothEyes<VecRequests<VecEdges<VectorId>>> {
        [LEFT, RIGHT].map(|eye| {
            self.0
                .iter()
                .map(|request| request.ids_to_compare(eye))
                .collect_vec()
        })
    }

    pub fn resolve(
        self,
        comparison_results: &BothEyes<VecRequests<MapEdges<bool>>>,
        intra_matches: VecRequests<Vec<IntraMatch>>,
    ) -> ResolvedBatch {
        assert_eq!(self.0.len(), comparison_results[LEFT].len());
        assert_eq!(self.0.len(), comparison_results[RIGHT].len());
        assert_eq!(self.0.len(), intra_matches.len());
        ResolvedBatch(
            izip!(
                self.0,
                &comparison_results[LEFT],
                &comparison_results[RIGHT],
                intra_matches,
            )
            .map(|(request, left, right, intra_matches)| {
                request.resolve([left, right], intra_matches)
            })
            .collect_vec(),
        )
    }
}

/// One request's search results for one orientation, not yet resolved against the
/// other eye.
struct PendingRequest {
    /// Search matches, in the form decisions are made on plus the pre-extension baseline.
    search: WithBaseline<UnresolvedJoin>,
    luc_ids: Vec<VectorId>,
    request_type: RequestType,
}

impl PendingRequest {
    fn new(
        search_results: BothEyes<&VecRotations<HawkInsertPlan>>,
        luc_ids: Vec<VectorId>,
        request_type: RequestType,
    ) -> PendingRequest {
        let effective = UnresolvedJoin::from_rotations(search_results, SearchVariant::Effective);

        // Only build the baseline join when at least one rotation was actually extended
        // by the supermatcher; otherwise it would equal `effective`.
        let was_extended = search_results.iter().any(|rotations| {
            rotations
                .iter()
                .any(|r| r.classified.pre_extension.is_some())
        });
        let baseline = was_extended
            .then(|| UnresolvedJoin::from_rotations(search_results, SearchVariant::Baseline));

        PendingRequest {
            search: WithBaseline {
                effective,
                baseline,
            },
            luc_ids,
            request_type,
        }
    }

    fn reauth_id(&self) -> Option<(VectorId, UseOrRule)> {
        match self.request_type {
            RequestType::Reauth { target } => target,
            _ => None,
        }
    }

    /// The vectors whose `eye_to_compare` side must be compared via MPC before this
    /// request's matches can be resolved.
    ///
    /// Baseline one-sided matches are included alongside the effective ones, so the
    /// pre-extension outcome can be resolved from the same MPC results. This is needed
    /// because a) a one-sided match in the baseline may become a two-sided match after
    /// extension, and so would not appear in the effective `matched_one_eye`, and
    /// b) baseline and extended results come from independent searches, so they can
    /// diverge slightly given the approximate nature of HNSW search.
    fn ids_to_compare(&self, eye_to_compare: usize) -> VecEdges<VectorId> {
        let matched_eye = 1 - eye_to_compare;
        let one_eyed = self
            .search
            .variants()
            .flat_map(|join| join.matched_one_eye[matched_eye].iter());

        // Always add the reauth target so is_match is computed even if the search missed it.
        let reauth_id = self.reauth_id().map(|(id, _)| id);

        chain!(one_eyed, &self.luc_ids, &reauth_id)
            .cloned()
            .unique()
            .collect_vec()
    }

    fn resolve(
        self,
        comparison_results: BothEyes<&MapEdges<bool>>,
        intra_matches: Vec<IntraMatch>,
    ) -> ResolvedRequest {
        let luc_results = self
            .luc_ids
            .iter()
            .map(|id| {
                let is_match =
                    [LEFT, RIGHT].map(|side| *comparison_results[side].get(id).unwrap_or(&false));
                (*id, is_match)
            })
            .collect_vec();

        let reauth_result = self.reauth_id().map(|(id, or_rule)| {
            let is_match =
                [LEFT, RIGHT].map(|side| *comparison_results[side].get(&id).unwrap_or(&false));
            tracing::info!("Reauth ID: {id}, or_rule: {or_rule}, is_match: {is_match:?}");
            (id, or_rule, is_match)
        });

        ResolvedRequest {
            search: self.search.map(|join| join.resolve(comparison_results)),
            luc_results,
            reauth_result,
            intra_matches,
            request_type: self.request_type,
        }
    }
}

/// One request's search matches, split by how many eyes matched, plus per-eye
/// saturation.
///
/// `matched_both_eyes` holds vectors that matched on both eyes directly in the search
/// results. `matched_one_eye[side]` holds vectors that matched only on `side`;
/// the other eye is resolved later via `resolve` using the MPC `comparison_results`.
#[derive(Clone, Debug)]
struct UnresolvedJoin {
    matched_both_eyes: VecEdges<VectorId>,
    matched_one_eye: BothEyes<VecEdges<VectorId>>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
}

impl UnresolvedJoin {
    /// Build by merging match results across all rotations of both eyes, reading the
    /// requested form of each rotation's results.
    fn from_rotations(
        search_results: BothEyes<&VecRotations<HawkInsertPlan>>,
        variant: SearchVariant,
    ) -> UnresolvedJoin {
        let mut hits_by_id: MapEdges<BothEyes<bool>> = HashMap::new();

        let mut saturated = [false, false];
        for (side, rotations) in izip!([LEFT, RIGHT], search_results) {
            // Merge matches from all rotations.
            for rotation in rotations.iter() {
                let matches = match variant {
                    SearchVariant::Effective => &rotation.classified.matches,
                    SearchVariant::Baseline => rotation
                        .classified
                        .pre_extension
                        .as_ref()
                        .unwrap_or(&rotation.classified.matches),
                };
                if matches.saturated {
                    saturated[side] = true;
                }
                for (vector_id, _) in matches.results.iter() {
                    hits_by_id.entry(*vector_id).or_default()[side] = true;
                }
            }
        }

        let partial_hits_sorted: Vec<_> = hits_by_id
            .into_iter()
            .filter(|(_, [is_match_l, is_match_r])| *is_match_l || *is_match_r)
            .sorted()
            .collect();

        let mut matched_both_eyes = Vec::new();
        let mut matched_one_eye: BothEyes<VecEdges<VectorId>> = [Vec::new(), Vec::new()];
        for (vector_id, is_match_lr) in partial_hits_sorted {
            match is_match_lr {
                [true, true] => matched_both_eyes.push(vector_id),
                [true, false] => matched_one_eye[LEFT].push(vector_id),
                [false, true] => matched_one_eye[RIGHT].push(vector_id),
                [false, false] => {}
            }
        }

        UnresolvedJoin {
            matched_both_eyes,
            matched_one_eye,
            saturated,
        }
    }

    /// Resolve the one-eyed matches into a full join, using the MPC comparison results
    /// for the opposite eye.
    fn resolve(&self, comparison_results: BothEyes<&MapEdges<bool>>) -> ResolvedJoin {
        let mut matches: Vec<_> = self
            .matched_both_eyes
            .iter()
            .map(|id| (*id, [true, true]))
            .collect();
        for id in &self.matched_one_eye[LEFT] {
            if let Some(right) = comparison_results[RIGHT].get(id) {
                matches.push((*id, [true, *right]));
            }
        }
        for id in &self.matched_one_eye[RIGHT] {
            if let Some(left) = comparison_results[LEFT].get(id) {
                matches.push((*id, [*left, true]));
            }
        }
        ResolvedJoin {
            matches,
            saturated: self.saturated,
        }
    }
}

// ===========================================================================
// Resolved types: request and batch state produced by Stage 1
// ===========================================================================

/// All requests for one orientation, with the other-eye comparisons applied. Consumed
/// by [`Self::decide`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedBatch(VecRequests<ResolvedRequest>);

/// One resolved request: search, LUC, reauth, and intra-batch matches for one
/// orientation, after MPC resolution.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedRequest {
    /// Resolved search matches, in the form decisions are made on plus the baseline.
    search: WithBaseline<ResolvedJoin>,
    luc_results: VecEdges<(VectorId, BothEyes<bool>)>,
    reauth_result: Option<(VectorId, UseOrRule, BothEyes<bool>)>,
    intra_matches: Vec<IntraMatch>,
    request_type: RequestType,
}

impl ResolvedRequest {
    /// The IDs of the vectors that matched this request.
    ///
    /// The luc, reauth and intra-batch contributions are unaffected by supermatcher
    /// extension, so only the search join varies with `variant`.
    fn select(&self, filter: Filter, variant: SearchVariant) -> impl Iterator<Item = MatchId> + '_ {
        let join = self.search.get(variant);

        let search = join
            .matches
            .iter()
            .filter(move |(_, [l, r])| filter.search_rule(*l, *r))
            .map(|(id, _)| Search(*id));

        let luc = self
            .luc_results
            .iter()
            .filter(move |(_, [l, r])| filter.luc_rule(*l, *r))
            .map(|(id, _)| Luc(*id));

        let reauth = self
            .reauth_result
            .filter(move |(_, or_rule, matches)| filter.reauth_rule(*or_rule, *matches))
            .map(|(id, _, _)| Reauth(id));

        let intra = self
            .intra_matches
            .iter()
            .filter(move |m| filter.intra_rule(m.is_match[LEFT], m.is_match[RIGHT]))
            .map(|m| IntraBatch(m.other_request_i));

        let supermatch = filter.supermatch_rule(join.saturated).then_some(Supermatch);

        chain!(search, luc, reauth, intra, supermatch)
    }
}

/// A search join after the missing-eye MPC comparisons have been resolved, bundled with
/// its per-eye saturation (supermatcher) flags.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolvedJoin {
    matches: VecEdges<(VectorId, BothEyes<bool>)>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
}

// ===========================================================================
// Stage 2: combine orientations, decide, and expose results
// ===========================================================================

/// Combines the results from mirrored checks.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestMatches {
    normal: ResolvedRequest,
    mirror: ResolvedRequest,
}

impl RequestMatches {
    /// The IDs of the vectors that matched at least partially, in both orientations.
    fn select(&self, filter: Filter, variant: SearchVariant) -> impl Iterator<Item = MatchId> + '_ {
        chain!(
            matches!(filter.orient, Only(Normal) | Both)
                .then_some(self.normal.select(filter, variant)),
            matches!(filter.orient, Only(Mirror) | Both)
                .then_some(self.mirror.select(filter, variant)),
        )
        .flatten()
    }

    /// True if the supermatcher extended this request's search in either orientation,
    /// so a baseline comparison is meaningful.
    fn was_extended(&self) -> bool {
        self.normal.search.was_extended() || self.mirror.search.was_extended()
    }
}

impl ResolvedBatch {
    /// Combine both orientations and make the final decision for every request.
    ///
    /// Emulates inserting entries one by one: intra-batch matches only count if the
    /// request they matched is itself being inserted or updated. Applies supermatcher
    /// rejection — if any rotation's match results were saturated on either eye, and
    /// nothing else matched, the decision is `NoMutation`.
    ///
    /// This is the one place decisions are computed and the one place the per-request
    /// supermatcher metrics are emitted; callers read the stored vector afterwards via
    /// `MatchResults::decisions()`.
    ///
    /// Note that `normal` and `mirror` inputs play non-equivalent roles in the decision
    /// procedure, so it is important to provide the inputs in the correct order.
    pub fn decide(normal: Self, mirror: Self) -> MatchResults {
        assert_eq!(normal.0.len(), mirror.0.len());
        let requests = izip!(normal.0, mirror.0)
            .map(|(normal, mirror)| RequestMatches { normal, mirror })
            .collect_vec();

        tracing::info!(
            "Calculating decisions for batch of {} requests",
            requests.len()
        );
        let filter = DECISION_FILTER;
        let mut decisions = Vec::<Decision>::with_capacity(requests.len());

        for request in &requests {
            tracing::info!(
                "Processing request type normal: {:?} mirror {:?}",
                request.normal.request_type,
                request.mirror.request_type,
            );
            let mut only_supermatch = false;

            let decision = match request.normal.request_type {
                RequestType::Uniqueness { skip_persistence } => {
                    let outcome = evaluate_uniqueness(
                        request.select(filter, SearchVariant::Effective),
                        &decisions,
                    );
                    only_supermatch = outcome.only_supermatch;

                    if request.was_extended() {
                        record_extension_metrics(&extension_outcome(request, filter, &decisions));
                    }

                    if outcome.is_match {
                        NoMutation
                    } else if skip_persistence {
                        UniqueInsertSkipped
                    } else {
                        UniqueInsert
                    }
                }
                // Identity Match Check request. Nothing to do.
                RequestType::IdentityMatchCheck => NoMutation,
                // Reauth request.
                RequestType::Reauth { .. } => match request.normal.reauth_result {
                    Some((id, or_rule, matches)) if filter.reauth_rule(or_rule, matches) => {
                        ReauthUpdate(id)
                    }
                    _ => NoMutation,
                },
                // Unsupported request. Nothing to do.
                RequestType::Unsupported => NoMutation,
            };

            if only_supermatch {
                tracing::info!("Supermatcher rejection");
                metrics::counter!("supermatcher_rejections").increment(1);
            }
            tracing::info!("Pushing decision: {decision:?}");
            decisions.push(decision);
        }

        MatchResults {
            requests,
            decisions,
        }
    }
}

/// The final match results for a batch: what each request matched, and what to do
/// about it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchResults {
    requests: VecRequests<RequestMatches>,
    decisions: VecRequests<Decision>,
}

impl MatchResults {
    /// The final decision of what to do with each request, decided by `decide`.
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// The IDs of the vectors that matched at least partially.
    pub fn select(&self, filter: Filter) -> VecRequests<Vec<MatchId>> {
        self.requests
            .iter()
            .map(|request| {
                request
                    .select(filter, SearchVariant::Effective)
                    .collect_vec()
            })
            .collect_vec()
    }
}

/// The outcome of evaluating a uniqueness request against its selected match ids.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct UniquenessOutcome {
    /// True if anything at all blocks insertion.
    is_match: bool,
    /// True if search saturation was the *only* thing blocking insertion.
    only_supermatch: bool,
}

/// Evaluate whether a uniqueness request matched, given its selected match ids and the
/// decisions already made for earlier requests in the batch.
///
/// Attribution to saturation is order-independent: a supermatch is the *only* reason for
/// a match only when no ordinary id matched, in either orientation.
///
/// Iterating the full set of ids is deliberate: a short-circuiting `any()` would
/// mis-attribute `only_supermatch`, because `RequestMatches::select` yields all of the
/// normal orientation's ids — including its `Supermatch` — before any of the mirror's.
fn evaluate_uniqueness(
    ids: impl IntoIterator<Item = MatchId>,
    prior_decisions: &[Decision],
) -> UniquenessOutcome {
    let mut ordinary_match = false;
    let mut supermatch = false;

    for id in ids {
        match id {
            Search(_) | Luc(_) | Reauth(_) => ordinary_match = true,
            Supermatch => supermatch = true,
            IntraBatch(request_i) => {
                // We are blocked by an intra-batch match only if the request we matched
                // with will itself be inserted or updated. A request after us in the
                // batch has no decision yet, so it does not block us.
                if prior_decisions
                    .get(request_i)
                    .is_some_and(Decision::is_mutation)
                {
                    ordinary_match = true;
                }
            }
        }
    }

    UniquenessOutcome {
        is_match: ordinary_match || supermatch,
        only_supermatch: supermatch && !ordinary_match,
    }
}

// ===========================================================================
// Supermatcher A/B metrics
// ===========================================================================

/// Supermatcher A/B comparison: the three match determinations an extended search is
/// judged by. Only meaningful for requests where `RequestMatches::was_extended()`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ExtensionOutcome {
    /// Final call on the extended results, including rejection due to saturation.
    extended_decision: bool,
    /// The effective search results, ignoring saturation: whether they contain a real
    /// (non-supermatch) neighbour match.
    extended_match: bool,
    /// What the original search alone would have concluded. Saturation is excluded here
    /// too, since those results were never expanded.
    baseline_match: bool,
}

fn extension_outcome(
    request: &RequestMatches,
    filter: Filter,
    prior_decisions: &[Decision],
) -> ExtensionOutcome {
    let not_supermatch = |id: &MatchId| !matches!(id, Supermatch);
    let is_match =
        |variant| evaluate_uniqueness(request.select(filter, variant), prior_decisions).is_match;
    let is_real_match = |variant| {
        evaluate_uniqueness(
            request.select(filter, variant).filter(not_supermatch),
            prior_decisions,
        )
        .is_match
    };

    ExtensionOutcome {
        extended_decision: is_match(SearchVariant::Effective),
        extended_match: is_real_match(SearchVariant::Effective),
        baseline_match: is_real_match(SearchVariant::Baseline),
    }
}

/// Emits the A/B counters for one extended request.
fn record_extension_metrics(outcome: &ExtensionOutcome) {
    match (outcome.baseline_match, outcome.extended_decision) {
        (false, true) => {
            metrics::counter!("supermatcher_extended_search_changed_decision_to_reject")
                .increment(1);
        }
        (true, false) => {
            metrics::counter!("supermatcher_extended_search_changed_decision_to_accept")
                .increment(1);
        }
        _ => {}
    }

    match (outcome.baseline_match, outcome.extended_match) {
        (false, true) => {
            metrics::counter!("supermatcher_extended_search_found_new_match").increment(1);
        }
        (true, false) => {
            metrics::counter!("supermatcher_extended_search_lost_match").increment(1);
        }
        _ => {}
    }

    // Unconditionally count when a request had an extended search.
    metrics::counter!("supermatcher_extended_search_requests").increment(1);
}

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests;
