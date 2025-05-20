use super::{
    intra_batch::IntraMatch, rot::VecRots, BothEyes, InsertPlan, MapEdges, Orientation, StoreId,
    UseOrRule, VecEdges, VecRequests, VectorId, LEFT, RIGHT,
};
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
        plans: &BothEyes<VecRequests<VecRots<InsertPlan>>>,
        luc_ids: &VecRequests<Vec<VectorId>>,
        reauth_ids: &VecRequests<Option<(VectorId, UseOrRule)>>,
    ) -> Self {
        // Join the results of both eyes into results per eye pair.
        Self(
            izip!(&plans[LEFT], &plans[RIGHT], luc_ids, reauth_ids)
                .map(|(left, right, luc, rea)| Step1::new([left, right], luc.clone(), *rea))
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
    luc_ids: Vec<VectorId>,
    reauth_id: Option<(VectorId, UseOrRule)>,
}

impl Step1 {
    fn new(
        search_results: BothEyes<&VecRots<InsertPlan>>,
        luc_ids: Vec<VectorId>,
        reauth_id: Option<(VectorId, UseOrRule)>,
    ) -> Step1 {
        let mut full_join: MapEdges<BothEyes<bool>> = HashMap::new();

        for (side, rotations) in izip!([LEFT, RIGHT], search_results) {
            // Merge matches from all rotations.
            for rotation in rotations.iter() {
                for vector_id in rotation.match_ids() {
                    full_join.entry(vector_id).or_default()[side] = true;
                }
            }
        }

        let mut step1 = Step1::with_capacity(full_join.len());
        step1.luc_ids = luc_ids;
        step1.reauth_id = reauth_id;

        for (vector_id, is_match_lr) in full_join {
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
            luc_ids: Vec::new(),
            reauth_id: None,
        }
    }

    fn missing_vector_ids(&self, side: usize) -> VecEdges<VectorId> {
        let other_side = 1 - side;
        let anti_join = &self.anti_join[other_side];
        let reauth_id = self.reauth_id.map(|(id, _)| id);

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

        let reauth_result = self.reauth_id.map(|(id, or_rule)| {
            let is_match =
                [LEFT, RIGHT].map(|side| *missing_is_match[side].get(&id).unwrap_or(&false));
            (id, or_rule, is_match)
        });

        let mut step2 = Step2 {
            full_join: self.inner_join,
            luc_results,
            reauth_result,
            intra_matches,
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
    UniqueReject,
    ReauthUpdate(VectorId),
    ReauthReject,
}
use Decision::*;

impl Decision {
    pub fn is_match(&self) -> bool {
        match self {
            UniqueReject | ReauthUpdate(_) => true,
            UniqueInsert | ReauthReject => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchStep3(VecRequests<Step3>);

impl BatchStep3 {
    /// The final decision of what to do with a request.
    ///
    /// Emulate the behavior of inserting entries one by one. Intra-batch matches
    /// only count if they are being inserted themselves.
    pub fn decisions(&self) -> VecRequests<Decision> {
        use Decision::*;

        let filter = Filter {
            eyes: Both,
            orient: Both,
            intra_batch: true,
        };

        let mut decisions = Vec::with_capacity(self.0.len());

        for request in &self.0 {
            let decision = match request.normal.reauth_result {
                // Uniqueness request.
                None => {
                    let is_match = request.select(filter).any(|id| match id {
                        Search(_) | Luc(_) | Reauth(_) => true,
                        IntraBatch(request_i) => {
                            match decisions.get(request_i) {
                                // The request we matched with will be inserted, so we are blocked by this intra-batch match.
                                Some(UniqueInsert) | Some(ReauthUpdate(_)) => true,
                                // The request we matched with is rejected, so we are not blocked by this intra-batch match.
                                Some(UniqueReject) | Some(ReauthReject) => false,
                                // The request we matched with is after us in the batch, so we are not blocked by it.
                                None => false,
                            }
                        }
                    });
                    if is_match {
                        UniqueReject
                    } else {
                        UniqueInsert
                    }
                }

                // Reauth request.
                Some((reauth_id, or_rule, matches)) => {
                    let is_match = filter.reauth_rule(or_rule, matches);
                    if is_match {
                        ReauthUpdate(reauth_id)
                    } else {
                        ReauthReject
                    }
                }
            };

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

        chain!(search, luc, reauth, intra)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatchId {
    Search(VectorId),
    Luc(VectorId),
    Reauth(VectorId),
    IntraBatch(usize),
}
use MatchId::*;

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
/// Intra-batch *OR* policy: match against requests before this request in the same batch.
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
        match self.eyes {
            Only(Left) => left,
            Only(Right) => right,
            Both if or_rule => left || right,
            Both => left && right,
        }
    }

    fn intra_rule(&self, left: bool, right: bool) -> bool {
        self.intra_batch && self.luc_rule(left, right)
    }
}
