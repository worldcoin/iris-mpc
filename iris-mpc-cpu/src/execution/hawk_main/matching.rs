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

struct Step1 {
    inner_join: VecEdges<(VectorId, BothEyes<bool>)>,
    anti_join: BothEyes<VecEdges<VectorId>>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
    luc_ids: Vec<VectorId>,
    request_type: RequestType,
}

impl Step1 {
    fn new(
        search_results: BothEyes<&VecRotations<HawkInsertPlan>>,
        luc_ids: Vec<VectorId>,
        request_type: RequestType,
    ) -> Step1 {
        let mut full_join: MapEdges<BothEyes<bool>> = HashMap::new();

        let mut saturated = [false, false];
        for (side, rotations) in izip!([LEFT, RIGHT], search_results) {
            // Merge matches from all rotations.
            for rotation in rotations.iter() {
                if rotation.classified.matches.saturated {
                    saturated[side] = true;
                }
                for (vector_id, _) in rotation.classified.matches.results.iter() {
                    full_join.entry(*vector_id).or_default()[side] = true;
                }
            }
        }

        let mut step1 = Step1::with_capacity(full_join.len());
        step1.saturated = saturated;
        step1.luc_ids = luc_ids;
        step1.request_type = request_type;

        let full_join_partial_matches_ordered: Vec<_> = full_join
            .into_iter()
            .filter(|(_, [is_match_l, is_match_r])| *is_match_l || *is_match_r)
            .sorted()
            .collect();

        for (vector_id, is_match_lr) in full_join_partial_matches_ordered {
            match is_match_lr {
                [true, true] => step1.inner_join.push((vector_id, [true, true])),
                [true, false] => step1.anti_join[LEFT].push(vector_id),
                [false, true] => step1.anti_join[RIGHT].push(vector_id),
                [false, false] => {}
            }
        }

        step1
    }

    fn with_capacity(capacity: usize) -> Self {
        Step1 {
            inner_join: Vec::with_capacity(capacity),
            anti_join: [
                Vec::with_capacity(capacity / 2),
                Vec::with_capacity(capacity / 2),
            ],
            saturated: [false, false],
            luc_ids: Vec::new(),
            request_type: RequestType::Unsupported,
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
        let anti_join = &self.anti_join[other_side];
        // Always add reauth target so is_match is computed even if the search didn't hit it.
        let reauth_id = self.reauth_id().map(|(id, _)| id);

        chain!(anti_join, &self.luc_ids, &reauth_id)
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
            tracing::info!("Reauth ID: {id}, or_rule: {or_rule}");
            tracing::info!(
                "Left match: {}, missing_is_match[LEFT] {:?}",
                missing_is_match[LEFT].get(&id).unwrap_or(&false),
                missing_is_match[LEFT]
            );
            tracing::info!(
                "Right match: {}, missing_is_match[RIGHT] {:?}",
                missing_is_match[RIGHT].get(&id).unwrap_or(&false),
                missing_is_match[RIGHT]
            );
            let is_match =
                [LEFT, RIGHT].map(|side| *missing_is_match[side].get(&id).unwrap_or(&false));
            (id, or_rule, is_match)
        });

        let mut step2 = Step2 {
            full_join: self.inner_join,
            luc_results,
            reauth_result,
            intra_matches,
            saturated: self.saturated,
            request_type: self.request_type,
        };

        for id in &self.anti_join[LEFT] {
            if let Some(right) = missing_is_match[RIGHT].get(id) {
                step2.full_join.push((*id, [true, *right]));
            }
        }

        for id in &self.anti_join[RIGHT] {
            if let Some(left) = missing_is_match[LEFT].get(id) {
                step2.full_join.push((*id, [*left, true]));
            }
        }

        step2
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
                    let is_match = request.select(filter).any(|id| match id {
                        Search(_) | Luc(_) | Reauth(_) => true,
                        Supermatch => {
                            because_supermatch = true;
                            true
                        }
                        IntraBatch(request_i) => {
                            match decisions.get(request_i) {
                                // If the request we matched with will be inserted or updated,
                                // then we are blocked by this intra-batch match.
                                Some(decision) => decision.is_mutation(),
                                // The request we matched with is after us in the batch, so we are not blocked by it.
                                None => false,
                            }
                        }
                    });
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
struct Step2 {
    full_join: VecEdges<(VectorId, BothEyes<bool>)>,
    luc_results: VecEdges<(VectorId, BothEyes<bool>)>,
    reauth_result: Option<(VectorId, UseOrRule, BothEyes<bool>)>,
    intra_matches: Vec<IntraMatch>,
    /// True per eye if any rotation's match results were saturated (supermatcher).
    saturated: BothEyes<bool>,
    request_type: RequestType,
}

impl Step2 {
    /// The IDs of the vectors that matched this request.
    fn select(&self, filter: Filter) -> impl Iterator<Item = MatchId> + '_ {
        let search = self
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
            .supermatch_rule(self.saturated)
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
