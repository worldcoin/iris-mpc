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
        tracing::info!(
            "Calculating decisions for batch of {} requests",
            self.0.len()
        );
        use Decision::*;

        let filter = Filter {
            eyes: Both,
            orient: Both,
            intra_batch: true,
        };

        let mut decisions = Vec::<Decision>::with_capacity(self.0.len());

        tracing::info!("Decisions: {:?}", decisions);

        for request in &self.0 {
            tracing::info!(
                "Processing request type normal: {:?} mirror {:?}",
                request.normal.request_type,
                request.mirror.request_type,
            );
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
}
#[cfg(test)]
#[allow(clippy::bool_assert_comparison)]
mod tests {
    use crate::execution::hawk_main::rot::Rotations;
    use crate::execution::hawk_main::{HawkResult, InsertPlanV, SearchRotations, VecRotations};
    use crate::hawkers::aby3::aby3_store::Aby3Query;
    use crate::hnsw::SortedNeighborhood;
    use crate::protocol::shared_iris::GaloisRingSharedIris;
    use crate::shares::share::DistanceShare;
    use crate::shares::Share;

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
            // Matching within a batch: AND rule.
            assert_eq!(FILTER_INTRA.intra_rule(false, x), false);
            assert_eq!(FILTER_INTRA.intra_rule(x, false), false);
            assert_eq!(FILTER_INTRA.intra_rule(true, true), true);

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
        reauth_match: bool,
        expected_decision: Decision,
        expected_matches: Vec<MatchId>,
        request_type: RequestType,
    }

    impl Default for TestCase {
        fn default() -> Self {
            Self {
                search_match: false,
                other_side_match: false,
                reauth_match: false,
                expected_decision: NoMutation,
                expected_matches: vec![],
                request_type: RequestType::Uniqueness(UniquenessRequest {
                    skip_persistence: false,
                }),
            }
        }
    }

    #[test]
    fn test_matching() {
        let cases = [
            // ### Uniqueness requests
            TestCase {
                search_match: false,
                other_side_match: false,
                expected_decision: Decision::UniqueInsert,
                expected_matches: vec![],
                ..TestCase::default()
            },
            TestCase {
                search_match: false,
                other_side_match: false,
                request_type: RequestType::Uniqueness(UniquenessRequest {
                    skip_persistence: true,
                }),
                expected_decision: Decision::UniqueInsertSkipped,
                expected_matches: vec![],
                ..TestCase::default()
            },
            TestCase {
                search_match: true,
                other_side_match: false,
                expected_decision: Decision::NoMutation,
                expected_matches: vec![MatchId::Search(BOTH_MATCH)],
                ..TestCase::default()
            },
            TestCase {
                search_match: false,
                other_side_match: true,
                expected_decision: Decision::NoMutation,
                expected_matches: vec![
                    MatchId::Search(RIGHT_MATCH),
                    MatchId::Luc(LUC_REQUESTED),
                    MatchId::Luc(LUC_REQUESTED_DUP),
                ],
                ..TestCase::default()
            },
            TestCase {
                search_match: true,
                other_side_match: true,
                expected_decision: Decision::NoMutation,
                expected_matches: vec![
                    MatchId::Search(BOTH_MATCH),
                    MatchId::Search(RIGHT_MATCH),
                    MatchId::Luc(LUC_REQUESTED),
                    MatchId::Luc(LUC_REQUESTED_DUP),
                ],
                ..TestCase::default()
            },
            // ### Reauth requests
            TestCase {
                request_type: RequestType::Reauth(Some((REAUTH, false as UseOrRule))),
                reauth_match: true,
                expected_decision: Decision::ReauthUpdate(REAUTH),
                expected_matches: vec![MatchId::Reauth(REAUTH)],
                ..TestCase::default()
            },
            TestCase {
                request_type: RequestType::Reauth(Some((REAUTH, false as UseOrRule))),
                reauth_match: false,
                expected_decision: Decision::NoMutation,
                expected_matches: vec![],
                ..TestCase::default()
            },
        ];

        for case in &cases {
            let batch_step_3 = run_test_matching(case);
            let decisions = batch_step_3.decisions();
            assert_eq!(
                decisions,
                vec![case.expected_decision],
                "Failed for case: {case:?}",
            );

            let match_ids = batch_step_3.select(HawkResult::MATCH_IDS_FILTER);
            let [match_ids] = match_ids.try_into().unwrap();
            assert_equal_sets(&match_ids, &case.expected_matches, case);
        }
    }

    // ### Hypothetical search results
    /// Left matches; right was inspected but does not match.
    const BOTH_FOUND: VectorId = VectorId::from_serial_id(1);
    /// Both sides match, when in case `search_match = true`.
    const BOTH_MATCH: VectorId = VectorId::from_serial_id(2);
    /// Only left was inspected and it matches.
    const LEFT_MATCH: VectorId = VectorId::from_serial_id(3);
    /// Only right was inspected and it matches.
    const RIGHT_MATCH: VectorId = VectorId::from_serial_id(4);
    /// The request wants us to inspect this ID.
    const LUC_REQUESTED: VectorId = VectorId::from_serial_id(5);
    /// The request wants us to inspect this ID, and it came up in search too.
    const LUC_REQUESTED_DUP: VectorId = LEFT_MATCH;
    /// The request wants us to reauthenticate this ID.
    const REAUTH: VectorId = VectorId::from_serial_id(6);

    fn run_test_matching(tc: &TestCase) -> BatchStep3 {
        let req_i = 0;
        let distance = || DistanceShare::new(Share::default(), Share::default());

        // Simulate a search. We found different partial matches on each side.
        let (mut match_left, non_match_left) = (vec![LEFT_MATCH, BOTH_FOUND], vec![]);
        let (mut match_right, non_match_right) = (vec![RIGHT_MATCH], vec![BOTH_FOUND]);
        // Make a full left+right match, or not depending on the test case.
        if tc.search_match {
            match_left.push(BOTH_MATCH);
            match_right.push(BOTH_MATCH);
        }

        let search_result = |match_ids: Vec<VectorId>, non_match_ids: Vec<VectorId>| {
            let insert_plan = HawkInsertPlan {
                match_count: match_ids.len(),
                plan: InsertPlanV {
                    query: Aby3Query::new_from_raw(GaloisRingSharedIris::dummy_for_party(0)),
                    links: vec![SortedNeighborhood::from_ascending_vec(
                        chain!(match_ids, non_match_ids)
                            .map(|v| (v, distance()))
                            .collect_vec(),
                    )],
                    set_ep: false,
                },
            };
            VecRotations::from(vec![insert_plan; SearchRotations::N_ROTATIONS])
        };

        let search_results = [
            vec![search_result(match_left, non_match_left)],
            vec![search_result(match_right, non_match_right)],
        ];
        let luc_ids = vec![vec![LUC_REQUESTED, LUC_REQUESTED_DUP]];
        let request_types = vec![tc.request_type];
        let batch1 = BatchStep1::new(&search_results, &luc_ids, request_types);

        let missing_ids = batch1.missing_vector_ids();

        // We will inspect the other side of partial search results.
        let mut expect_left = vec![RIGHT_MATCH, LUC_REQUESTED, LUC_REQUESTED_DUP];
        let mut expect_right = vec![LEFT_MATCH, BOTH_FOUND, LUC_REQUESTED];
        // `LUC_REQUESTED_DUP` is the same as `LEFT_MATCH` and we avoided duplicates.
        // `BOTH_FOUND` is requested because it was not a match. We could have noticed that
        // it was already inspected and optimize it away, but we do not.

        // For a reauth request, we will inspect the reauth target vector.
        if matches!(tc.request_type, RequestType::Reauth(_)) {
            expect_left.push(REAUTH);
            expect_right.push(REAUTH);
        }

        assert_equal_sets(
            &missing_ids[LEFT][req_i],
            &expect_left,
            "Left side missing IDs",
        );
        assert_equal_sets(
            &missing_ids[RIGHT][req_i],
            &expect_right,
            "Right side missing IDs",
        );

        // Simulate `calculate_missing_is_match(..)`.
        // Make it match or not depending on `with_other_side_match`.
        let mut missing_is_match = [vec![HashMap::new()], vec![HashMap::new()]];
        for id in &missing_ids[LEFT][req_i] {
            missing_is_match[LEFT][req_i].insert(*id, tc.other_side_match);
        }
        for id in &missing_ids[RIGHT][req_i] {
            missing_is_match[RIGHT][req_i].insert(*id, false);
        }

        // Make the reauth request match.
        if matches!(tc.request_type, RequestType::Reauth(_)) {
            *missing_is_match[LEFT][req_i].get_mut(&REAUTH).unwrap() = tc.reauth_match;
            *missing_is_match[RIGHT][req_i].get_mut(&REAUTH).unwrap() = tc.reauth_match;
        }

        // Simulate `intra_batch_is_match(..)`
        let intra_matches = vec![vec![]];

        let batch2 = batch1.step2(&missing_is_match, intra_matches);

        // Do the same with mirrored matching. Amazingly, we got exactly the same result in this test.
        let batch2_mirror = batch2.clone();

        // Return the final decision for the request.
        batch2.step3(batch2_mirror)
    }

    /// Assert that two sets are equal, ignoring order, and without duplicates.
    fn assert_equal_sets<T>(left: &[T], right: &[T], msg: impl std::fmt::Debug)
    where
        T: std::hash::Hash + Eq + Clone + std::fmt::Debug,
    {
        let left_set: std::collections::HashSet<_> = left.iter().cloned().collect();
        let right_set: std::collections::HashSet<_> = right.iter().cloned().collect();
        assert_eq!(left_set.len(), left.len(), "{msg:?}");
        assert_eq!(right_set.len(), right.len(), "{msg:?}");
        assert_eq!(left_set, right_set, "{msg:?}");
    }
}
