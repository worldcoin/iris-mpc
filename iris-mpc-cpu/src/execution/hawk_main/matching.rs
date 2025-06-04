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
    luc_ids: Vec<VectorId>,
    request_type: RequestType,
}

impl Step1 {
    fn new(
        search_results: BothEyes<&VecRots<InsertPlan>>,
        luc_ids: Vec<VectorId>,
        request_type: RequestType,
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
        step1.request_type = request_type;

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
            let is_match =
                [LEFT, RIGHT].map(|side| *missing_is_match[side].get(&id).unwrap_or(&false));
            (id, or_rule, is_match)
        });

        let mut step2 = Step2 {
            full_join: self.inner_join,
            luc_results,
            reauth_result,
            intra_matches,
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

        let mut decisions = Vec::<Decision>::with_capacity(self.0.len());

        for request in &self.0 {
            let decision = match request.normal.request_type {
                RequestType::Uniqueness(UniquenessRequest { skip_persistence }) => {
                    let is_match = request.select(filter).any(|id| match id {
                        Search(_) | Luc(_) | Reauth(_) => true,
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
                // Reset Check request. Nothing to do.
                RequestType::ResetCheck => NoMutation,
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

// TODO: This could move to `BatchQuery` and maybe use the original types in `smpc_request.rs`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RequestType {
    /// A request to check if a vector is unique.
    Uniqueness(UniquenessRequest),
    /// A request to check if a vector is unique without inserting it.
    ResetCheck,
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

#[cfg(test)]
mod tests {
    use super::VectorId;
    use super::*;
    use std::collections::HashMap;

    const FILTER_BOTH: Filter = Filter {
        eyes: Both,
        orient: Both,
        intra_batch: false,
    };
    const FILTER_LEFT: Filter = Filter {
        eyes: Only(Left),
        orient: Both,
        intra_batch: false,
    };
    const FILTER_RIGHT: Filter = Filter {
        eyes: Only(Right),
        orient: Both,
        intra_batch: false,
    };
    const FILTER_INTRA: Filter = Filter {
        eyes: Both,
        orient: Both,
        intra_batch: true,
    };

    #[test]
    fn test_search_rule() {
        for x in [false, true] {
            // Matching from HNSW search: AND rule
            assert_eq!(FILTER_BOTH.search_rule(true, true), true);
            assert_eq!(FILTER_BOTH.search_rule(x, false), false);
            assert_eq!(FILTER_BOTH.search_rule(false, x), false);
            // Only left
            assert_eq!(FILTER_LEFT.search_rule(true, x), true);
            assert_eq!(FILTER_LEFT.search_rule(false, x), false);
            // Only right
            assert_eq!(FILTER_RIGHT.search_rule(x, true), true);
            assert_eq!(FILTER_RIGHT.search_rule(x, false), false);
        }
    }

    #[test]
    fn test_luc_rule() {
        for x in [false, true] {
            // Matching from LUC results: OR rule
            assert_eq!(FILTER_BOTH.luc_rule(false, false), false);
            assert_eq!(FILTER_BOTH.luc_rule(true, x), true);
            assert_eq!(FILTER_BOTH.luc_rule(x, true), true);
            // Only left
            assert_eq!(FILTER_LEFT.luc_rule(true, x), true);
            assert_eq!(FILTER_LEFT.luc_rule(false, x), false);
            // Only right
            assert_eq!(FILTER_RIGHT.luc_rule(x, true), true);
            assert_eq!(FILTER_RIGHT.luc_rule(x, false), false);
        }
    }

    #[test]
    fn test_reauth_rule() {
        let and_rule = false;
        let or_rule = true;
        for x in [false, true] {
            // Reauth with AND rule
            assert_eq!(FILTER_BOTH.reauth_rule(and_rule, [true, true]), true);
            assert_eq!(FILTER_BOTH.reauth_rule(and_rule, [x, false]), false);
            assert_eq!(FILTER_BOTH.reauth_rule(and_rule, [false, x]), false);
            // Reauth with OR rule
            assert_eq!(FILTER_BOTH.reauth_rule(or_rule, [true, x]), true);
            assert_eq!(FILTER_BOTH.reauth_rule(or_rule, [x, true]), true);
            assert_eq!(FILTER_BOTH.reauth_rule(or_rule, [false, false]), false);

            for either_rule in [and_rule, or_rule] {
                // Only left
                assert_eq!(FILTER_LEFT.reauth_rule(either_rule, [true, x]), true);
                assert_eq!(FILTER_LEFT.reauth_rule(either_rule, [false, x]), false);
                // Only right
                assert_eq!(FILTER_RIGHT.reauth_rule(either_rule, [x, true]), true);
                assert_eq!(FILTER_RIGHT.reauth_rule(either_rule, [x, false]), false);
            }
        }
    }

    #[test]
    fn test_intra_rule() {
        for x in [false, true] {
            // Matching within a batch: OR rule.
            assert_eq!(FILTER_INTRA.intra_rule(true, x), true);
            assert_eq!(FILTER_INTRA.intra_rule(x, true), true);
            assert_eq!(FILTER_INTRA.intra_rule(false, false), false);

            // If intra-batch is not requested, always false.
            for y in [false, true] {
                assert_eq!(FILTER_BOTH.intra_rule(x, y), false);
            }
        }
    }

    #[derive(Clone, Debug)]
    struct TestCase {
        search_match: bool,
        other_side_match: bool,
        expected_decision: Decision,
    }

    #[test]
    fn test_matching() {
        let cases = [
            TestCase {
                search_match: false,
                other_side_match: false,
                expected_decision: Decision::UniqueInsert,
            },
            TestCase {
                search_match: true,
                other_side_match: false,
                expected_decision: Decision::NoMutation,
            },
            TestCase {
                search_match: false,
                other_side_match: true,
                expected_decision: Decision::NoMutation,
            },
            TestCase {
                search_match: true,
                other_side_match: true,
                expected_decision: Decision::NoMutation,
            },
        ];

        for case in &cases {
            let batch_step_3 = run_test_matching(&case);
            let decisions = batch_step_3.decisions();
            assert_eq!(
                decisions,
                vec![case.expected_decision],
                "Failed for case: {case:?}",
            );
        }
    }

    fn run_test_matching(tc: &TestCase) -> BatchStep3 {
        let req_i = 0;

        // Hypothetical search results.
        // Left matches; right was inspected but does not match.
        let both_found = VectorId::from_serial_id(1);
        // Both sides match, when in case `with_search_match`.
        let both_match = VectorId::from_serial_id(2);
        // Only left was inspected and it matches.
        let left_match = VectorId::from_serial_id(3);
        // Only right was inspected and it matches.
        let right_match = VectorId::from_serial_id(4);
        // The request wants us to inspect this ID.
        let luc_requested = VectorId::from_serial_id(5);
        // The request wants us to inspect this ID, and it came up in search too.
        let luc_requested_dup = left_match;

        // Simulate Step1
        let step1s = vec![Step1 {
            inner_join: vec![
                (both_found, [false, true]),
                (both_match, [true, tc.search_match]),
            ],
            anti_join: [vec![left_match], vec![right_match]],
            luc_ids: vec![luc_requested, luc_requested_dup],
            request_type: RequestType::Uniqueness(UniquenessRequest {
                skip_persistence: false,
            }),
        }];
        let batch1 = BatchStep1(step1s);

        // Get the other side of partial search results.
        let missing_ids = batch1.missing_vector_ids();
        assert_eq!(
            missing_ids[LEFT][req_i],
            vec![right_match, luc_requested, luc_requested_dup]
        );
        assert_eq!(missing_ids[RIGHT][req_i], vec![left_match, luc_requested]); // No dup.

        // Simulate `calculate_missing_is_match(..)`.
        // Make it match or not depending on `with_other_side_match`.
        let mut missing_is_match = [vec![HashMap::new()], vec![HashMap::new()]];
        for id in &missing_ids[LEFT][req_i] {
            missing_is_match[LEFT][req_i].insert(*id, tc.other_side_match);
        }
        for id in &missing_ids[RIGHT][req_i] {
            missing_is_match[RIGHT][req_i].insert(*id, false);
        }

        // Simulate `intra_batch_is_match(..)`
        let intra_matches = vec![vec![]];

        let batch2 = batch1.step2(&missing_is_match, intra_matches);

        // Do the same with mirrored matching. Amazingly, we got exactly the same result in this test.
        let batch2_mirror = batch2.clone();

        // Return the final decision for the request.
        batch2.step3(batch2_mirror)
    }
}
