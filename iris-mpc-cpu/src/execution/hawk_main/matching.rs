use super::{
    intra_batch::IntraMatch, BothEyes, HawkInsertPlan, MapEdges, Orientation, StoreId, UseOrRule,
    VecEdges, VecRequests, VectorId, LEFT, RIGHT,
};
use crate::execution::hawk_main::VecRotations;
use itertools::{chain, izip, Itertools};
use std::collections::HashMap;

/// Since the two separate HSNW for left and right return separate vectors of matching ids, we
/// cannot do the trivial AND/OR matching procedure from v2, since the other side might not have
/// considered that id at all. This however does not mean it would not match, so for all ids that
/// are given back for one side we do a manual comparison in the other side to get a full
/// left-right match pair. Only then do we continue to the final matching logic.
///
/// The matching algorithm follows these steps:
///
/// 1. Organize the results of the nearest neighbor search with
///    `PendingBatch::new`. Then:
///
///    1.a. Get the vectors found on only one side with `ids_to_compare()`.
///    1.b. Fetch the other side and calculate their `is_match` with MPC.
///    1.c. Give this back to `resolve(comparison_results)`.
///
/// 2. `ResolvedBatch::is_matches`: Combine it all into the final match decisions.
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

/// Which form of the search results to evaluate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SearchVariant {
    /// The results decisions are made on: extended, if the supermatcher ran.
    Effective,
    /// What the original search alone would have produced.
    Baseline,
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

struct PendingRequest {
    /// Search matches, in the form decisions are made on plus the pre-extension baseline.
    search: WithBaseline<UnresolvedJoin>,
    luc_ids: Vec<VectorId>,
    request_type: RequestType,
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
            RequestType::Reauth(r) => r,
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

/// Results for a batch of requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedBatch(VecRequests<ResolvedRequest>);

impl ResolvedBatch {
    pub fn decide(self, mirror: Self) -> MatchResults {
        assert_eq!(self.0.len(), mirror.0.len());
        MatchResults(
            izip!(self.0, mirror.0)
                .map(|(normal, mirror)| RequestMatches { normal, mirror })
                .collect_vec(),
        )
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    UniqueInsert,
    UniqueInsertSkipped,
    ReauthUpdate(VectorId),
    NoMutation,
}
use Decision::*;

impl Decision {
    pub fn is_mutation(&self) -> bool {
        match self {
            UniqueInsert | ReauthUpdate(_) => true,
            UniqueInsertSkipped | NoMutation => false,
        }
    }
}

/// Wide filter: any match in any orientation, including intra-batch peers.
pub const DECISION_FILTER: Filter = Filter {
    eyes: OnlyOrBoth::Both,
    orient: OnlyOrBoth::Both,
    intra_batch: true,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchResults(VecRequests<RequestMatches>);

/// Evaluate whether a uniqueness request matched, given its selected match ids
/// and the decisions already made for earlier requests in the batch.
///
/// Returns `(is_match, because_supermatch)` where `because_supermatch` is true
/// if a `Supermatch` (saturation) id was the reason a match was found. Note this
/// depends on `Supermatch` being yielded last by `select`, so it is only set
/// when no ordinary match short-circuited the search first.
fn uniqueness_is_match(
    ids: impl IntoIterator<Item = MatchId>,
    prior_decisions: &[Decision],
) -> (bool, bool) {
    let mut because_supermatch = false;
    let is_match = ids.into_iter().any(|id| match id {
        Search(_) | Luc(_) | Reauth(_) => true,
        Supermatch => {
            because_supermatch = true;
            true
        }
        IntraBatch(request_i) => {
            match prior_decisions.get(request_i) {
                // If the request we matched with will be inserted or updated,
                // then we are blocked by this intra-batch match.
                Some(decision) => decision.is_mutation(),
                // The request we matched with is after us in the batch, so we are not blocked by it.
                None => false,
            }
        }
    });
    (is_match, because_supermatch)
}

/// Supermatcher A/B comparison: for requests whose search was extended by the
/// supermatcher, compare the extended search-match outcome against what the
/// pre-extension search alone would have produced. The `Supermatch` (saturation)
/// signal is excluded so we isolate whether the extended search surfaced a
/// *real* neighbor match that the original search missed.
fn record_extension_metrics(
    request: &RequestMatches,
    filter: Filter,
    prior_decisions: &[Decision],
) {
    if !request.was_extended() {
        return;
    }

    let (extended_decision, _) = uniqueness_is_match(
        request.select(filter, SearchVariant::Effective),
        prior_decisions,
    );

    let not_supermatch = |id: &MatchId| !matches!(id, Supermatch);

    let (extended_match, _) = uniqueness_is_match(
        request
            .select(filter, SearchVariant::Effective)
            .filter(not_supermatch),
        prior_decisions,
    );

    let (pre_match, _) = uniqueness_is_match(
        request
            .select(filter, SearchVariant::Baseline)
            .filter(not_supermatch),
        prior_decisions,
    );

    match (pre_match, extended_decision) {
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

    match (pre_match, extended_match) {
        (false, true) => {
            metrics::counter!("supermatcher_extended_search_found_new_match").increment(1);
        }
        (true, false) => {
            metrics::counter!("supermatcher_extended_search_lost_match").increment(1);
        }
        _ => {}
    }

    // Unconditionally count when a request had some pre-extension in one of its searches.
    metrics::counter!("supermatcher_extended_search_requests").increment(1);
}

impl MatchResults {
    /// The final decision of what to do with a request.
    ///
    /// Emulate the behavior of inserting entries one by one. Intra-batch matches
    /// only count if they are being inserted themselves.
    ///
    /// Applies supermatcher rejection: if any rotation's match results were
    /// saturated on either eye, the decision is forced to `NoMutation`.
    pub fn decisions(&self) -> VecRequests<Decision> {
        tracing::info!(
            "Calculating decisions for batch of {} requests",
            self.0.len()
        );
        use Decision::*;

        let filter = DECISION_FILTER;

        let mut decisions = Vec::<Decision>::with_capacity(self.0.len());

        for request in &self.0 {
            tracing::info!(
                "Processing request type normal: {:?} mirror {:?}",
                request.normal.request_type,
                request.mirror.request_type,
            );
            let mut because_supermatch = false;

            let decision = match request.normal.request_type {
                RequestType::Uniqueness(UniquenessRequest { skip_persistence }) => {
                    let (is_match, bsm) = uniqueness_is_match(
                        request.select(filter, SearchVariant::Effective),
                        &decisions,
                    );
                    because_supermatch = bsm;

                    record_extension_metrics(request, filter, &decisions);

                    if is_match {
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
                RequestType::Reauth(_) => match request.normal.reauth_result {
                    Some((id, or_rule, matches)) if filter.reauth_rule(or_rule, matches) => {
                        ReauthUpdate(id)
                    }
                    _ => NoMutation,
                },
                // Unsupported request. Nothing to do.
                RequestType::Unsupported => NoMutation,
            };

            if because_supermatch {
                tracing::info!("Supermatcher rejection");
                metrics::counter!("supermatcher_rejections").increment(1);
            }
            tracing::info!("Pushing decision: {decision:?}");
            decisions.push(decision);
        }

        decisions
    }

    /// The IDs of the vectors that matched at least partially.
    pub fn select(&self, filter: Filter) -> VecRequests<Vec<MatchId>> {
        self.0
            .iter()
            .map(|request| {
                request
                    .select(filter, SearchVariant::Effective)
                    .collect_vec()
            })
            .collect_vec()
    }
}

/// Results for one request.
#[derive(Clone, Debug, PartialEq, Eq)]
/// A search join after the missing-side MPC comparisons have been resolved,
/// bundled with its per-eye saturation (supermatcher) flags.
struct ResolvedJoin {
    matches: VecEdges<(VectorId, BothEyes<bool>)>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
}

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
            .map(|(id, _)| MatchId::Search(*id));

        let luc = self
            .luc_results
            .iter()
            .filter(move |(_, [l, r])| filter.luc_rule(*l, *r))
            .map(|(id, _)| MatchId::Luc(*id));

        let reauth = self
            .reauth_result
            .filter(move |(_, or_rule, matches)| filter.reauth_rule(*or_rule, *matches))
            .map(|(id, _, _)| MatchId::Reauth(id));

        let intra = self
            .intra_matches
            .iter()
            .filter(move |m| filter.intra_rule(m.is_match[LEFT], m.is_match[RIGHT]))
            .map(|m| MatchId::IntraBatch(m.other_request_i));

        let supermatch = filter
            .supermatch_rule(join.saturated)
            .then_some(MatchId::Supermatch);

        chain!(search, luc, reauth, intra, supermatch)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MatchId {
    Search(VectorId),
    Luc(VectorId),
    Reauth(VectorId),
    IntraBatch(usize),
    /// Search results were saturated (supermatcher).
    Supermatch,
}
use MatchId::*;

// TODO: This could move to `BatchQuery` and maybe use the original types in `smpc_request.rs`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RequestType {
    /// A request to check if a vector is unique.
    Uniqueness(UniquenessRequest),
    /// A request to check if a vector is unique without inserting it.
    IdentityMatchCheck,
    /// A request to check if a vector matches a target and replace it.
    Reauth(Option<(VectorId, UseOrRule)>),
    /// Other features.
    Unsupported,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UniquenessRequest {
    pub skip_persistence: bool,
}

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

use OnlyOrBoth::{Both, Only};
use Orientation::{Mirror, Normal};
use StoreId::{Left, Right};

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
        tracing::info!("left: {left}, right: {right}, or_rule: {or_rule}");
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

#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests;
