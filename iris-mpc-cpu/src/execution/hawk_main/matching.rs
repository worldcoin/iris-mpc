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
///    `BatchStep1::new`. Then:
///
///    1.a. Get the vectors found on only one side with `missing_vector_ids()`.
///    1.b. Fetch the other side and calculate their `is_match` with MPC.
///    1.c. Give this back to `step2(missing_is_match)`.
///
/// 2. `BatchStep2::is_matches`: Combine it all into the final match decisions.
pub struct BatchStep1(VecRequests<Step1>);

impl BatchStep1 {
    pub fn new(
        plans: &BothEyes<VecRequests<VecRotations<HawkInsertPlan>>>,
        luc_ids: &VecRequests<Vec<VectorId>>,
        request_types: VecRequests<RequestType>,
    ) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT], luc_ids, request_types)
                .map(|(left, right, luc, rt)| Step1::new([left, right], luc.clone(), rt))
                .collect_vec(),
        )
    }

    pub fn missing_vector_ids(&self) -> BothEyes<VecRequests<VecEdges<VectorId>>> {
        [LEFT, RIGHT].map(|side| {
            self.0
                .iter()
                .map(|step| step.missing_vector_ids(side))
                .collect_vec()
        })
    }

    pub fn step2(
        self,
        missing_is_match: &BothEyes<VecRequests<MapEdges<bool>>>,
        intra_matches: VecRequests<Vec<IntraMatch>>,
    ) -> BatchStep2 {
        assert_eq!(self.0.len(), missing_is_match[LEFT].len());
        assert_eq!(self.0.len(), missing_is_match[RIGHT].len());
        assert_eq!(self.0.len(), intra_matches.len());
        BatchStep2(
            izip!(
                self.0,
                &missing_is_match[LEFT],
                &missing_is_match[RIGHT],
                intra_matches,
            )
            .map(|(step, missing_left, missing_right, intra_matches)| {
                step.step2([missing_left, missing_right], intra_matches)
            })
            .collect_vec(),
        )
    }
}

/// The inner/anti join of one request's search matches, plus per-eye saturation.
///
/// `inner_join` holds vectors that matched on both eyes directly in the search
/// results. `anti_join[side]` holds vectors that matched only on `side`; the
/// other eye is resolved later via `resolve` using the MPC `missing_is_match`.
#[derive(Clone, Debug)]
struct SearchJoin {
    inner_join: VecEdges<(VectorId, BothEyes<bool>)>,
    anti_join: BothEyes<VecEdges<VectorId>>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
}

struct Step1 {
    /// Search matches from the (possibly supermatcher-extended) results.
    join: SearchJoin,
    /// Search matches from the pre-extension results, present only for requests
    /// whose search was extended by the supermatcher. `None` means no extension
    /// happened.
    pre_join: Option<SearchJoin>,
    luc_ids: Vec<VectorId>,
    request_type: RequestType,
}

impl SearchJoin {
    /// Build the inner/anti join by merging match results across all rotations
    /// of both eyes. When `use_pre` is set, the pre-extension matches are used
    /// for any rotation that was extended by the supermatcher (falling back to
    /// the normal matches for rotations that were not extended).
    fn from_rotations(
        search_results: BothEyes<&VecRotations<HawkInsertPlan>>,
        use_pre: bool,
    ) -> SearchJoin {
        let mut full_join: MapEdges<BothEyes<bool>> = HashMap::new();

        let mut saturated = [false, false];
        for (side, rotations) in izip!([LEFT, RIGHT], search_results) {
            // Merge matches from all rotations.
            for rotation in rotations.iter() {
                let matches = if use_pre {
                    rotation
                        .classified
                        .pre_extension
                        .as_ref()
                        .unwrap_or(&rotation.classified.matches)
                } else {
                    &rotation.classified.matches
                };
                if matches.saturated {
                    saturated[side] = true;
                }
                for (vector_id, _) in matches.results.iter() {
                    full_join.entry(*vector_id).or_default()[side] = true;
                }
            }
        }

        let full_join_partial_matches_ordered: Vec<_> = full_join
            .into_iter()
            .filter(|(_, [is_match_l, is_match_r])| *is_match_l || *is_match_r)
            .sorted()
            .collect();

        let mut inner_join = Vec::new();
        let mut anti_join: BothEyes<VecEdges<VectorId>> = [Vec::new(), Vec::new()];
        for (vector_id, is_match_lr) in full_join_partial_matches_ordered {
            match is_match_lr {
                [true, true] => inner_join.push((vector_id, [true, true])),
                [true, false] => anti_join[LEFT].push(vector_id),
                [false, true] => anti_join[RIGHT].push(vector_id),
                [false, false] => {}
            }
        }

        SearchJoin {
            inner_join,
            anti_join,
            saturated,
        }
    }

    /// Resolve anti-join entries into a full join using the MPC-computed
    /// `missing_is_match` results for the opposite eye.
    fn resolve(
        &self,
        missing_is_match: BothEyes<&MapEdges<bool>>,
    ) -> VecEdges<(VectorId, BothEyes<bool>)> {
        let mut full_join = self.inner_join.clone();
        for id in &self.anti_join[LEFT] {
            if let Some(right) = missing_is_match[RIGHT].get(id) {
                full_join.push((*id, [true, *right]));
            }
        }
        for id in &self.anti_join[RIGHT] {
            if let Some(left) = missing_is_match[LEFT].get(id) {
                full_join.push((*id, [*left, true]));
            }
        }
        full_join
    }
}

impl Step1 {
    fn new(
        search_results: BothEyes<&VecRotations<HawkInsertPlan>>,
        luc_ids: Vec<VectorId>,
        request_type: RequestType,
    ) -> Step1 {
        let join = SearchJoin::from_rotations(search_results, false);

        // Only build the pre-extension join when at least one rotation was
        // actually extended by the supermatcher; otherwise it equals `join`.
        let any_pre = search_results.iter().any(|rotations| {
            rotations
                .iter()
                .any(|r| r.classified.pre_extension.is_some())
        });
        let pre_join = any_pre.then(|| SearchJoin::from_rotations(search_results, true));

        Step1 {
            join,
            pre_join,
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

    fn missing_vector_ids(&self, side: usize) -> VecEdges<VectorId> {
        let other_side = 1 - side;
        let anti_join = &self.join.anti_join[other_side];
        // Include the pre-extension anti-join so the pre-extension outcome can be
        // resolved from the same MPC results (its ids are a subset of `join`'s,
        // but include them explicitly to be safe).
        let pre_anti_join = self
            .pre_join
            .iter()
            .flat_map(|j| j.anti_join[other_side].iter());
        // Always add reauth target so is_match is computed even if the search didn't hit it.
        let reauth_id = self.reauth_id().map(|(id, _)| id);

        chain!(anti_join, pre_anti_join, &self.luc_ids, &reauth_id)
            .cloned()
            .unique()
            .collect_vec()
    }

    fn step2(
        self,
        missing_is_match: BothEyes<&MapEdges<bool>>,
        intra_matches: Vec<IntraMatch>,
    ) -> Step2 {
        let luc_results = self
            .luc_ids
            .iter()
            .map(|id| {
                let is_match =
                    [LEFT, RIGHT].map(|side| *missing_is_match[side].get(id).unwrap_or(&false));
                (*id, is_match)
            })
            .collect_vec();

        let reauth_result = self.reauth_id().map(|(id, or_rule)| {
            let is_match =
                [LEFT, RIGHT].map(|side| *missing_is_match[side].get(&id).unwrap_or(&false));
            tracing::info!("Reauth ID: {id}, or_rule: {or_rule}, is_match: {is_match:?}");
            (id, or_rule, is_match)
        });

        let join = ResolvedJoin {
            full_join: self.join.resolve(missing_is_match),
            saturated: self.join.saturated,
        };
        let pre_join = self.pre_join.as_ref().map(|pre| ResolvedJoin {
            full_join: pre.resolve(missing_is_match),
            saturated: pre.saturated,
        });

        Step2 {
            join,
            pre_join,
            luc_results,
            reauth_result,
            intra_matches,
            request_type: self.request_type,
        }
    }
}

/// Results for a batch of requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchStep2(VecRequests<Step2>);

impl BatchStep2 {
    pub fn step3(self, mirror: Self) -> BatchStep3 {
        assert_eq!(self.0.len(), mirror.0.len());
        BatchStep3(
            izip!(self.0, mirror.0)
                .map(|(normal, mirror)| Step3 { normal, mirror })
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
pub struct BatchStep3(VecRequests<Step3>);

/// Evaluate whether a uniqueness request matched, given its selected match ids
/// and the decisions already made for earlier requests in the batch.
///
/// Returns `(is_match, because_supermatch)` where `because_supermatch` is true
/// if a `Supermatch` (saturation) id was the reason a match was found. Note this
/// depends on `Supermatch` being yielded last by `select`, so it is only set
/// when no ordinary match short-circuited the search first.
fn uniqueness_is_match(
    ids: impl Iterator<Item = MatchId>,
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
fn record_extension_metrics(request: &Step3, filter: Filter, prior_decisions: &[Decision]) {
    if !request.has_pre_extension() {
        return;
    }
    let not_supermatch = |id: &MatchId| !matches!(id, Supermatch);
    let (extended_match, _) = uniqueness_is_match(
        request.select(filter).filter(not_supermatch),
        prior_decisions,
    );
    let (pre_match, _) = uniqueness_is_match(
        request.select_pre(filter).filter(not_supermatch),
        prior_decisions,
    );
    match (pre_match, extended_match) {
        (false, true) => {
            metrics::counter!("supermatcher_extended_search_found_new_match").increment(1);
        }
        (true, false) => {
            metrics::counter!("supermatcher_extended_search_lost_match").increment(1);
        }
        _ => {}
    }
}

impl BatchStep3 {
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
                    let (is_match, bsm) = uniqueness_is_match(request.select(filter), &decisions);
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
            .map(|step| step.select(filter).collect_vec())
            .collect_vec()
    }
}

/// Results for one request.
#[derive(Clone, Debug, PartialEq, Eq)]
/// A search join after the missing-side MPC comparisons have been resolved,
/// bundled with its per-eye saturation (supermatcher) flags.
struct ResolvedJoin {
    full_join: VecEdges<(VectorId, BothEyes<bool>)>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Step2 {
    /// Search matches from the (possibly supermatcher-extended) results.
    join: ResolvedJoin,
    /// Search matches from the pre-extension results, present only when this
    /// request's search was extended by the supermatcher. `None` means no
    /// extension happened, in which case the pre-extension outcome equals `join`.
    pre_join: Option<ResolvedJoin>,
    luc_results: VecEdges<(VectorId, BothEyes<bool>)>,
    reauth_result: Option<(VectorId, UseOrRule, BothEyes<bool>)>,
    intra_matches: Vec<IntraMatch>,
    request_type: RequestType,
}

impl Step2 {
    /// The IDs of the vectors that matched this request.
    fn select(&self, filter: Filter) -> impl Iterator<Item = MatchId> + '_ {
        self.select_with(filter, &self.join)
    }

    /// Like `select`, but using the pre-extension search matches. When this
    /// request's search was not extended by the supermatcher, this is identical
    /// to `select`.
    fn select_pre(&self, filter: Filter) -> impl Iterator<Item = MatchId> + '_ {
        self.select_with(filter, self.pre_join.as_ref().unwrap_or(&self.join))
    }

    /// The IDs of the vectors that matched this request, evaluated against a
    /// specific resolved `join` (the luc/reauth/intra contributions are
    /// unaffected by supermatcher extension and always use `self`).
    fn select_with<'a>(
        &'a self,
        filter: Filter,
        join: &'a ResolvedJoin,
    ) -> impl Iterator<Item = MatchId> + 'a {
        let search = join
            .full_join
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
struct Step3 {
    normal: Step2,
    mirror: Step2,
}

impl Step3 {
    /// The IDs of the vectors that matched at least partially.
    fn select(&self, filter: Filter) -> impl Iterator<Item = MatchId> + '_ {
        chain!(
            matches!(filter.orient, Only(Normal) | Both).then_some(self.normal.select(filter)),
            matches!(filter.orient, Only(Mirror) | Both).then_some(self.mirror.select(filter)),
        )
        .flatten()
    }

    /// Like `select`, but using the pre-extension search matches on both
    /// orientations. Identical to `select` for requests that were not extended.
    fn select_pre(&self, filter: Filter) -> impl Iterator<Item = MatchId> + '_ {
        chain!(
            matches!(filter.orient, Only(Normal) | Both).then_some(self.normal.select_pre(filter)),
            matches!(filter.orient, Only(Mirror) | Both).then_some(self.mirror.select_pre(filter)),
        )
        .flatten()
    }

    /// True if this request's search was extended by the supermatcher on either
    /// orientation, so a pre-extension comparison is meaningful.
    fn has_pre_extension(&self) -> bool {
        self.normal.pre_join.is_some() || self.mirror.pre_join.is_some()
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
