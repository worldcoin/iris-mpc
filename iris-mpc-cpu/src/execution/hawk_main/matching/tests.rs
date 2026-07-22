use ampc_secret_sharing::shares::DistanceShare;
use ampc_secret_sharing::Share;

use crate::execution::hawk_main::iris_worker::QueryId;
use crate::execution::hawk_main::{
    ClassifiedMatches, HawkResult, InsertPlanV, SaturableMatches, VecRotations,
    HAWK_BASE_ROTATIONS_MASK,
};
use crate::hawkers::aby3::aby3_store::Aby3Query;
use crate::hnsw::graph::UpdateEntryPoint;

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

#[test]
fn test_supermatch_rule() {
    for x in [false, true] {
        // Supermatch uses OR rule: either eye saturated is a supermatch
        assert_eq!(FILTER_BOTH.supermatch_rule([false, false]), false);
        assert_eq!(FILTER_BOTH.supermatch_rule([true, x]), true);
        assert_eq!(FILTER_BOTH.supermatch_rule([x, true]), true);
        // Only left
        assert_eq!(FILTER_LEFT.supermatch_rule([true, x]), true);
        assert_eq!(FILTER_LEFT.supermatch_rule([false, x]), false);
        // Only right
        assert_eq!(FILTER_RIGHT.supermatch_rule([x, true]), true);
        assert_eq!(FILTER_RIGHT.supermatch_rule([x, false]), false);
    }
}

#[derive(Clone, Debug)]
struct TestCase {
    search_match: bool,
    other_side_match: bool,
    reauth_match: bool,
    /// Saturated flags per eye: [left, right]. Simulates super-matcher.
    saturated: BothEyes<bool>,
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
            saturated: [false, false],
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
        // ### Super-matcher requests
        // Left eye saturated, no search match → supermatch rejection
        TestCase {
            saturated: [true, false],
            expected_decision: Decision::NoMutation,
            expected_matches: vec![MatchId::Supermatch],
            ..TestCase::default()
        },
        // Right eye saturated, no search match → supermatch rejection
        TestCase {
            saturated: [false, true],
            expected_decision: Decision::NoMutation,
            expected_matches: vec![MatchId::Supermatch],
            ..TestCase::default()
        },
        // Both eyes saturated → supermatch rejection
        TestCase {
            saturated: [true, true],
            expected_decision: Decision::NoMutation,
            expected_matches: vec![MatchId::Supermatch],
            ..TestCase::default()
        },
        // Saturated but also has a search match → NoMutation (match takes priority)
        TestCase {
            search_match: true,
            saturated: [true, false],
            expected_decision: Decision::NoMutation,
            expected_matches: vec![MatchId::Search(BOTH_MATCH), MatchId::Supermatch],
            ..TestCase::default()
        },
        // Saturated with skip_persistence → still NoMutation (supermatch overrides)
        TestCase {
            request_type: RequestType::Uniqueness(UniquenessRequest {
                skip_persistence: true,
            }),
            saturated: [true, false],
            expected_decision: Decision::NoMutation,
            expected_matches: vec![MatchId::Supermatch],
            ..TestCase::default()
        },
        // Not saturated, no match → normal insertion
        TestCase {
            saturated: [false, false],
            expected_decision: Decision::UniqueInsert,
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

    let saturated = tc.saturated;
    let search_result =
        |match_ids: Vec<VectorId>, non_match_ids: Vec<VectorId>, side_saturated: bool| {
            let links_unstructured = vec![chain!(match_ids.clone(), non_match_ids).collect_vec()];

            let matches: Vec<_> = match_ids.iter().cloned().map(|v| (v, distance())).collect();
            let insert_plan = HawkInsertPlan {
                classified: ClassifiedMatches {
                    anon_stats_matches: SaturableMatches {
                        results: matches.clone(),
                        saturated: side_saturated,
                    },
                    matches: SaturableMatches {
                        results: matches,
                        saturated: side_saturated,
                    },
                },
                plan: InsertPlanV {
                    query: Aby3Query::new(QueryId::new()),
                    links: links_unstructured,
                    update_ep: UpdateEntryPoint::False,
                },
            };
            VecRotations::from(vec![
                insert_plan;
                HAWK_BASE_ROTATIONS_MASK.count_ones() as usize
            ])
        };

    let search_results = [
        vec![search_result(match_left, non_match_left, saturated[LEFT])],
        vec![search_result(
            match_right,
            non_match_right,
            saturated[RIGHT],
        )],
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
