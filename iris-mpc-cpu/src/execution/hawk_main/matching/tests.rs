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

#[test]
fn test_evaluate_uniqueness() {
    let v = VectorId::from_serial_id(1);

    // Nothing matched.
    assert_eq!(
        evaluate_uniqueness([], &[]),
        UniquenessOutcome {
            is_match: false,
            only_supermatch: false,
        },
    );
    // Saturation alone is the only reason this is a match.
    assert_eq!(
        evaluate_uniqueness([Supermatch], &[]),
        UniquenessOutcome {
            is_match: true,
            only_supermatch: true,
        },
    );
    // An ordinary match takes priority, whichever order the ids arrive in. The
    // supermatch-first case is real: `RequestMatches::select` yields every id of the
    // normal orientation, including its supermatch, before any mirror id.
    for ids in [vec![Search(v), Supermatch], vec![Supermatch, Search(v)]] {
        assert_eq!(
            evaluate_uniqueness(ids.clone(), &[]),
            UniquenessOutcome {
                is_match: true,
                only_supermatch: false,
            },
            "for ids {ids:?}",
        );
    }
    // An intra-batch peer blocks us only if it is itself being inserted or updated.
    assert_eq!(
        evaluate_uniqueness([IntraBatch(0)], &[UniqueInsert]),
        UniquenessOutcome {
            is_match: true,
            only_supermatch: false,
        },
    );
    assert_eq!(
        evaluate_uniqueness([IntraBatch(0)], &[NoMutation]),
        UniquenessOutcome {
            is_match: false,
            only_supermatch: false,
        },
    );
    // A peer later in the batch has no decision yet, so it does not block us.
    assert_eq!(
        evaluate_uniqueness([IntraBatch(5)], &[]),
        UniquenessOutcome {
            is_match: false,
            only_supermatch: false,
        },
    );
    // A non-blocking peer does not displace the supermatch attribution.
    assert_eq!(
        evaluate_uniqueness([Supermatch, IntraBatch(0)], &[NoMutation]),
        UniquenessOutcome {
            is_match: true,
            only_supermatch: true,
        },
    );
}

/// One scenario for [`run_test_matching`]: a few knobs describing what the searches and
/// the MPC comparisons found, plus what the pipeline is expected to conclude.
///
/// The fields fall into three groups: what the HNSW searches returned (`search_match`,
/// `saturated`, `pre_extension`), what the follow-up MPC comparisons answered
/// (`other_side_match`, `reauth_match`), and what kind of request this is
/// (`request_type`). `expected_decision` and `expected_matches` are the assertions, and
/// are read only by [`test_matching`]'s loop — tests that assert directly leave them at
/// their [`Default`] values.
///
/// Everything not named in a literal comes from [`TestCase::default`]: a uniqueness
/// request that persists, whose searches found no full match and were not saturated, and
/// whose comparisons all came back negative.
#[derive(Clone, Debug)]
struct TestCase {
    /// Whether both eyes' searches found `BOTH_MATCH`, the fixture's only two-eyed hit.
    /// This is the one way to get a match that needs no MPC comparison to confirm, since
    /// the AND rule over both eyes is already satisfied by the search results alone.
    search_match: bool,

    /// The answer the simulated MPC comparison gives for every id compared against the
    /// **left** eye — that is, for ids the right eye's search hit but the left eye never
    /// inspected. The right eye's comparisons are always negative, so this is the only
    /// knob for turning a one-eyed search hit into a confirmed two-eyed match.
    other_side_match: bool,

    /// The answer the simulated MPC comparison gives for the `REAUTH` target on both
    /// eyes, overriding `other_side_match` for that id. Only meaningful for
    /// [`RequestType::Reauth`] cases, where it decides whether the reauth succeeds.
    reauth_match: bool,

    /// Saturated flags per eye: [left, right]. Simulates super-matcher.
    ///
    /// A saturated search returned nearly as many matches as `ef`, so more probably
    /// exist beyond what was retrieved. That alone counts as a match
    /// ([`MatchId::Supermatch`]) and blocks insertion, even with no concrete match id.
    saturated: BothEyes<bool>,

    /// Simulates a supermatcher extended search: what the original search alone found.
    /// `None` means no extension happened.
    ///
    /// When set, the fixture's search results become the *extended* ones and this becomes
    /// the baseline they are compared against, which is what drives the A/B metrics in
    /// [`extension_outcome`]. Setting it also grows the set of ids sent for MPC
    /// comparison, since baseline one-eyed hits need their other eye resolved too.
    pre_extension: Option<BaselineSearch>,

    /// The [`Decision`] the pipeline should reach for this request.
    expected_decision: Decision,

    /// The match ids the request should report under [`HawkResult::MATCH_IDS_FILTER`],
    /// compared as a set.
    expected_matches: Vec<MatchId>,

    /// Which kind of request this is, selecting the branch [`ResolvedBatch::decide`] takes:
    /// uniqueness (with or without persistence), identity match check, reauth, or
    /// unsupported.
    request_type: RequestType,
}

impl Default for TestCase {
    fn default() -> Self {
        Self {
            search_match: false,
            other_side_match: false,
            reauth_match: false,
            saturated: [false, false],
            pre_extension: None,
            expected_decision: NoMutation,
            expected_matches: vec![],
            request_type: RequestType::Uniqueness {
                skip_persistence: false,
            },
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
            request_type: RequestType::Uniqueness {
                skip_persistence: true,
            },
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
            request_type: RequestType::Reauth {
                target: Some((REAUTH, false as UseOrRule)),
            },
            reauth_match: true,
            expected_decision: Decision::ReauthUpdate(REAUTH),
            expected_matches: vec![MatchId::Reauth(REAUTH)],
            ..TestCase::default()
        },
        TestCase {
            request_type: RequestType::Reauth {
                target: Some((REAUTH, false as UseOrRule)),
            },
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
            request_type: RequestType::Uniqueness {
                skip_persistence: true,
            },
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
        // ### Super-matcher extended search
        // The extended search found BOTH_MATCH, which the original search missed;
        // extending also resolved the left eye's saturation.
        TestCase {
            search_match: true,
            saturated: [false, false],
            pre_extension: Some(BaselineSearch {
                matched: [
                    vec![LEFT_MATCH, BOTH_FOUND, BASELINE_ONLY],
                    vec![RIGHT_MATCH],
                ],
                saturated: [true, false],
            }),
            expected_decision: Decision::NoMutation,
            expected_matches: vec![MatchId::Search(BOTH_MATCH)],
            ..TestCase::default()
        },
        // The extended search *lost* a match the original search had found.
        TestCase {
            search_match: false,
            saturated: [false, false],
            pre_extension: Some(BaselineSearch {
                matched: [
                    vec![LEFT_MATCH, BOTH_FOUND, BOTH_MATCH],
                    vec![RIGHT_MATCH, BOTH_MATCH],
                ],
                saturated: [false, false],
            }),
            expected_decision: Decision::UniqueInsert,
            expected_matches: vec![],
            ..TestCase::default()
        },
    ];

    for case in &cases {
        let results = run_test_matching(case);
        let decisions = results.decisions();
        assert_eq!(
            decisions,
            [case.expected_decision].as_slice(),
            "Failed for case: {case:?}",
        );

        let match_ids = results.select(HawkResult::MATCH_IDS_FILTER);
        let [match_ids] = match_ids.try_into().unwrap();
        assert_equal_sets(&match_ids, &case.expected_matches, case);
    }
}

/// The pre-extension (baseline) view of a request is selected independently of the
/// extended view: here the original search found no full match, only saturation,
/// while the extended search found one.
#[test]
fn test_baseline_selection_differs_from_extended() {
    let case = TestCase {
        search_match: true,
        saturated: [false, false],
        pre_extension: Some(BaselineSearch {
            matched: [
                vec![LEFT_MATCH, BOTH_FOUND, BASELINE_ONLY],
                vec![RIGHT_MATCH],
            ],
            saturated: [true, false],
        }),
        // `expected_decision`/`expected_matches` are omitted: this test asserts
        // directly below rather than through `test_matching`'s table loop.
        ..TestCase::default()
    };
    let batch = run_test_matching(&case);

    let extended = batch.select(HawkResult::MATCH_IDS_FILTER);
    assert_equal_sets(&extended[0], &[MatchId::Search(BOTH_MATCH)], "extended");

    let baseline = baseline_match_ids(&batch, HawkResult::MATCH_IDS_FILTER);
    assert_equal_sets(&baseline, &[MatchId::Supermatch], "baseline");
}

/// A request with no extended search has no separate baseline: the two views agree.
#[test]
fn test_baseline_equals_extended_without_extension() {
    let case = TestCase {
        search_match: true,
        ..TestCase::default()
    };
    let batch = run_test_matching(&case);

    let extended = batch.select(HawkResult::MATCH_IDS_FILTER);
    let baseline = baseline_match_ids(&batch, HawkResult::MATCH_IDS_FILTER);
    assert_equal_sets(&baseline, &extended[0], "baseline equals extended");
}

/// The extended search found a full match the original search had missed.
#[test]
fn test_extension_outcome_found_new_match() {
    let case = TestCase {
        search_match: true,
        saturated: [false, false],
        pre_extension: Some(BaselineSearch {
            matched: [
                vec![LEFT_MATCH, BOTH_FOUND, BASELINE_ONLY],
                vec![RIGHT_MATCH],
            ],
            saturated: [true, false],
        }),
        // `expected_decision` / `expected_matches` are omitted: this test asserts on
        // `extension_outcome` directly and never goes through `test_matching`'s loop.
        ..TestCase::default()
    };
    let results = run_test_matching(&case);
    let request = request_0(&results);

    assert!(request.was_extended());
    assert_eq!(
        extension_outcome(request, DECISION_FILTER, &[]),
        ExtensionOutcome {
            extended_decision: true,
            extended_match: true,
            // Saturation is excluded, so the original search alone found nothing.
            baseline_match: false,
        },
    );
}

/// The extended search lost a full match the original search had found.
#[test]
fn test_extension_outcome_lost_match() {
    let case = TestCase {
        search_match: false,
        saturated: [false, false],
        pre_extension: Some(BaselineSearch {
            matched: [
                vec![LEFT_MATCH, BOTH_FOUND, BOTH_MATCH],
                vec![RIGHT_MATCH, BOTH_MATCH],
            ],
            saturated: [false, false],
        }),
        ..TestCase::default()
    };
    let results = run_test_matching(&case);
    let request = request_0(&results);

    assert!(request.was_extended());
    assert_eq!(
        extension_outcome(request, DECISION_FILTER, &[]),
        ExtensionOutcome {
            extended_decision: false,
            extended_match: false,
            baseline_match: true,
        },
    );
}

/// Without an extension *and* without saturation, all three determinations agree: this
/// fixture has `saturated: [false, false]`, so `extended_decision` (which includes
/// saturation) coincides with `extended_match`/`baseline_match` (which exclude it). A
/// fixture with saturation would not pin this equality.
#[test]
fn test_extension_outcome_without_extension() {
    let case = TestCase {
        search_match: true,
        ..TestCase::default()
    };
    let results = run_test_matching(&case);
    let request = request_0(&results);

    assert!(!request.was_extended());
    let outcome = extension_outcome(request, DECISION_FILTER, &[]);
    assert_eq!(outcome.baseline_match, outcome.extended_match);
    assert_eq!(outcome.baseline_match, outcome.extended_decision);
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
/// Only the pre-extension (baseline) search found this, on the left eye.
const BASELINE_ONLY: VectorId = VectorId::from_serial_id(7);

/// A simulated pre-extension (baseline) search result: what each eye's search found
/// *before* the supermatcher re-searched with a larger `ef`.
#[derive(Clone, Debug)]
struct BaselineSearch {
    /// Matching vector ids per eye: `[left, right]`.
    matched: BothEyes<Vec<VectorId>>,
    /// Saturation flag per eye: `[left, right]`.
    saturated: BothEyes<bool>,
}

/// The single request in a one-request test batch.
fn request_0(batch: &MatchResults) -> &RequestMatches {
    &batch.requests[0]
}

/// The match ids the pre-extension (baseline) search alone would have produced.
fn baseline_match_ids(batch: &MatchResults, filter: Filter) -> Vec<MatchId> {
    request_0(batch)
        .select(filter, SearchVariant::Baseline)
        .collect_vec()
}

/// Drive one [`TestCase`] through the whole matching pipeline and return the result.
///
/// This fabricates a plausible pair of eye searches, then plays the part of the two
/// asynchronous helpers the real pipeline depends on — the MPC other-eye comparisons
/// (`is_match_batch`) and the intra-batch comparisons (`intra_batch_is_match`) — so the
/// matching logic can be tested without any MPC session or network:
///
/// 1. Build a [`HawkInsertPlan`] per eye from the fixture ids below, replicated across
///    every rotation, and hand both to [`PendingBatch::new`].
/// 2. Ask for [`PendingBatch::ids_to_compare`] and **assert the exact set** for each eye.
///    This assertion runs for every case, so each one re-checks which vectors the
///    pipeline decides need an other-eye comparison.
/// 3. Answer those comparisons from the test case: every left-eye comparison returns
///    `other_side_match`, every right-eye one returns `false`, and `REAUTH` returns
///    `reauth_match` on both eyes.
/// 4. Report no intra-batch matches, then resolve via [`PendingBatch::resolve`].
/// 5. Combine orientations with [`ResolvedBatch::decide`], using a clone of the resolved
///    batch as the mirror.
///
/// The searches are the same every time apart from the test case's knobs. Each fixture id
/// exercises one path: `BOTH_MATCH` is the only id both eyes match (when `search_match`);
/// `LEFT_MATCH` and `RIGHT_MATCH` are hit by one eye and need the other compared;
/// `BOTH_FOUND` was seen by both eyes but matched only on the left; `LUC_REQUESTED` is
/// compared on both eyes because the request asked for it, and `LUC_REQUESTED_DUP` is an
/// alias of `LEFT_MATCH` that proves duplicates are collapsed; `REAUTH` is compared even
/// though no search hit it; and `BASELINE_ONLY` appears solely in the pre-extension
/// baseline, so it is reached only through [`TestCase::pre_extension`].
///
/// Two limits are worth highlighting:
///
/// - **One request per batch.** Intra-batch matching is therefore never exercised — a
///   request has no peers to be blocked by.
/// - **The mirror orientation is a clone of the normal one.** No case here can produce a
///   disagreement between orientations, so anything that depends on their asymmetry is
///   invisible.
fn run_test_matching(tc: &TestCase) -> MatchResults {
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

    // Build one eye's search result: the same `HawkInsertPlan` for every rotation, holding
    // the ids this eye matched plus the ids it merely inspected (which reach only the HNSW
    // links, not the `classified` matches the matching logic reads). `side_saturated` is
    // this eye's saturation flag; a `Some` baseline simulates a supermatcher extension,
    // making `match_ids` the extended results and the baseline the pre-extension ones.
    let search_result = |match_ids: Vec<VectorId>,
                         non_match_ids: Vec<VectorId>,
                         side_saturated: bool,
                         baseline: Option<(Vec<VectorId>, bool)>| {
        let links_unstructured = vec![chain!(match_ids.clone(), non_match_ids).collect_vec()];

        let as_results =
            |ids: &[VectorId]| -> Vec<_> { ids.iter().cloned().map(|v| (v, distance())).collect() };

        let matches: Vec<_> = as_results(&match_ids);
        let pre_extension = baseline.map(|(ids, saturated)| SaturableMatches {
            results: as_results(&ids),
            saturated,
        });

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
                pre_extension,
            },
            plan: InsertPlanV {
                query: Aby3Query::new(QueryId::new()),
                links: links_unstructured,
                update_ep: UpdateEntryPoint::False,
                as_of: 0,
            },
        };
        VecRotations::from(vec![
            insert_plan;
            HAWK_BASE_ROTATIONS_MASK.count_ones() as usize
        ])
    };

    let saturated = tc.saturated;
    let baseline_for = |side: usize| {
        tc.pre_extension
            .as_ref()
            .map(|b| (b.matched[side].clone(), b.saturated[side]))
    };

    let search_results = [
        vec![search_result(
            match_left,
            non_match_left,
            saturated[LEFT],
            baseline_for(LEFT),
        )],
        vec![search_result(
            match_right,
            non_match_right,
            saturated[RIGHT],
            baseline_for(RIGHT),
        )],
    ];
    let luc_ids = vec![vec![LUC_REQUESTED, LUC_REQUESTED_DUP]];
    let request_types = vec![tc.request_type];

    let pending = PendingBatch::new(&search_results, &luc_ids, request_types);

    let ids_to_compare = pending.ids_to_compare();

    // We will inspect the other side of partial search results.
    let mut expect_left = vec![RIGHT_MATCH, LUC_REQUESTED, LUC_REQUESTED_DUP];
    let mut expect_right = vec![LEFT_MATCH, BOTH_FOUND, LUC_REQUESTED];
    // `LUC_REQUESTED_DUP` is the same as `LEFT_MATCH` and we avoided duplicates.
    // `BOTH_FOUND` is requested because it was not a match. We could have noticed that
    // it was already inspected and optimize it away, but we do not.

    // For a reauth request, we will inspect the reauth target vector.
    if matches!(tc.request_type, RequestType::Reauth { .. }) {
        expect_left.push(REAUTH);
        expect_right.push(REAUTH);
    }

    // Baseline one-sided matches also need an other-eye comparison, so that the
    // pre-extension outcome can be resolved from the same MPC results.
    if let Some(baseline) = &tc.pre_extension {
        for id in &baseline.matched[LEFT] {
            if !baseline.matched[RIGHT].contains(id) {
                expect_right.push(*id);
            }
        }
        for id in &baseline.matched[RIGHT] {
            if !baseline.matched[LEFT].contains(id) {
                expect_left.push(*id);
            }
        }
    }
    // `assert_equal_sets` rejects duplicates, and a baseline hit may already be expected.
    let expect_left = expect_left.into_iter().unique().collect_vec();
    let expect_right = expect_right.into_iter().unique().collect_vec();

    assert_equal_sets(
        &ids_to_compare[LEFT][req_i],
        &expect_left,
        "Left side ids to compare",
    );
    assert_equal_sets(
        &ids_to_compare[RIGHT][req_i],
        &expect_right,
        "Right side ids to compare",
    );

    // Simulate the caller's `is_match_batch(..)` (in `hawk_main/is_match_batch.rs`),
    // which computes these comparisons and passes them into `PendingBatch::resolve`.
    // Make it match or not depending on `with_other_side_match`.
    let mut comparison_results = [vec![HashMap::new()], vec![HashMap::new()]];
    for id in &ids_to_compare[LEFT][req_i] {
        comparison_results[LEFT][req_i].insert(*id, tc.other_side_match);
    }
    for id in &ids_to_compare[RIGHT][req_i] {
        comparison_results[RIGHT][req_i].insert(*id, false);
    }

    // Make the reauth request match.
    if matches!(tc.request_type, RequestType::Reauth { .. }) {
        *comparison_results[LEFT][req_i].get_mut(&REAUTH).unwrap() = tc.reauth_match;
        *comparison_results[RIGHT][req_i].get_mut(&REAUTH).unwrap() = tc.reauth_match;
    }

    // Simulate `intra_batch_is_match(..)`
    let intra_matches = vec![vec![]];

    let resolved = pending.resolve(&comparison_results, intra_matches);

    // Do the same with mirrored matching. Amazingly, we got exactly the same result in this test.
    let resolved_mirror = resolved.clone();

    // Return the final decision for the request.
    ResolvedBatch::decide(resolved, resolved_mirror)
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
