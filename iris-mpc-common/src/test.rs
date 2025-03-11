use crate::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        inmemory_store::InMemoryStore,
        smpc_request::{REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
        statistics::BucketStatistics,
    },
    iris_db::{
        db::IrisDB,
        iris::{IrisCode, IrisCodeArray},
    },
    job::{BatchQuery, JobSubmissionHandle, ServerJobResult},
    IRIS_CODE_LENGTH,
};
use eyre::Result;
use itertools::izip;
use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};
use uuid::Uuid;

const THRESHOLD_ABSOLUTE: usize = IRIS_CODE_LENGTH * 375 / 1000; // 0.375 * 12800

#[derive(Clone)]
pub struct E2ETemplate {
    left: IrisCode,
    right: IrisCode,
}
impl E2ETemplate {
    fn to_shared_template(&self, is_valid: bool, rng: &mut StdRng) -> E2ESharedTemplate {
        let (left_shared_code, left_shared_mask) = get_shared_template(is_valid, &self.left, rng);
        let (right_shared_code, right_shared_mask) =
            get_shared_template(is_valid, &self.right, rng);
        E2ESharedTemplate {
            left_shared_code,
            left_shared_mask,
            right_shared_code,
            right_shared_mask,
        }
    }
}

fn get_shared_template(
    is_valid: bool,
    template: &IrisCode,
    rng: &mut StdRng,
) -> (
    [GaloisRingIrisCodeShare; 3],
    [GaloisRingTrimmedMaskCodeShare; 3],
) {
    let mut shared_code =
        GaloisRingIrisCodeShare::encode_iris_code(&template.code, &template.mask, rng);

    let shared_mask = GaloisRingIrisCodeShare::encode_mask_code(&template.mask, rng);
    let shared_mask_vector: Vec<GaloisRingTrimmedMaskCodeShare> =
        shared_mask.iter().map(|x| x.clone().into()).collect();

    let mut shared_mask: [GaloisRingTrimmedMaskCodeShare; 3] =
        shared_mask_vector.try_into().unwrap();

    if !is_valid {
        shared_code[0] = GaloisRingIrisCodeShare::default_for_party(1);
        shared_mask[0] = GaloisRingTrimmedMaskCodeShare::default_for_party(1)
    }
    (shared_code, shared_mask)
}

type OrRuleSerialIds = Vec<u32>;

#[derive(Clone)]
pub struct E2ESharedTemplate {
    pub left_shared_code: [GaloisRingIrisCodeShare; 3],
    pub left_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
    pub right_shared_code: [GaloisRingIrisCodeShare; 3],
    pub right_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestCase {
    /// Send an iris code known to be in the database
    Match,
    /// Send an iris code known to be in the database
    /// but skip persistence
    MatchSkipPersistence,
    /// Send an iris code that known not to match any in the database, it
    /// will be inserted
    NonMatch,
    /// Send an iris code that known not to match any in the database, it
    /// will not be inserted
    NonMatchSkipPersistence,
    /// Send an iris code that is close to the threshold of matching another
    /// iris code There will be a slight jitter added around the threshold
    /// so it could produce both a match and non-match
    CloseToThreshold,
    /// Send an iris code that was not in the initial DB, but has been since
    /// inserted
    PreviouslyInserted,
    /// Send an iris code known to have been in the database, but has been
    /// deleted
    PreviouslyDeleted,
    /// Send an iris code that uses the OR rule
    WithOrRuleSet,
    /// Send a reauth request matching target serial id's iris code only
    /// (successful reauth)
    ReauthMatchingTarget,
    /// Send a reauth request not matching target serial id's iris code
    /// (failed reauth)
    ReauthNonMatchingTarget,
    /// Send a reauth request with OR rule matching one side of target
    /// serial id's iris code (successful reauth)
    ReauthOrRuleMatchingTarget,
    /// Send a reauth request with OR rule not matching any sides of target
    /// serial id's iris code (failed reauth)
    ReauthOrRuleNonMatchingTarget,
}

impl TestCase {
    /// Returns the default set of allowed test cases, which is used to filter
    /// the later selection. Should usually be the exhaustive set of all
    /// variants.
    fn default_test_set() -> Vec<TestCase> {
        vec![
            TestCase::Match,
            TestCase::MatchSkipPersistence,
            TestCase::NonMatch,
            TestCase::NonMatchSkipPersistence,
            TestCase::CloseToThreshold,
            TestCase::PreviouslyInserted,
            TestCase::PreviouslyDeleted,
            TestCase::WithOrRuleSet,
            TestCase::ReauthNonMatchingTarget,
            TestCase::ReauthMatchingTarget,
            TestCase::ReauthOrRuleNonMatchingTarget,
            TestCase::ReauthOrRuleMatchingTarget,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatabaseRange {
    /// Use the full database range
    Full,
    /// Use only the first 10% of the database range, which has masks set to
    /// all 1. This is useful for testing values close to the threshold,
    /// since we can rely on bitflips always affecting distance.
    FullMaskOnly,
}

pub struct ExpectedResult {
    /// The returned index of the iris code in the database.
    /// It is None if the iris code is not in the database, and Some(idx) if
    /// there is a match at index idx
    db_index: Option<u32>,
    /// The request is a skip persistence request
    is_skip_persistence_request: bool,
    /// Whether the iris code is expected to be in the batch match
    /// This flag indicates that the iris code is expected to match another
    /// iris code in the current batch
    is_batch_match: bool,
    /// Populated only if the request type is REAUTH.
    /// Indicates whether the expected reauth result is successful.
    is_reauth_successful: Option<bool>,
}

struct BucketStatisticParameters {
    num_gpus: usize,
    num_buckets: usize,
    match_buffer_size: usize,
}

pub struct TestCaseGenerator {
    /// enabled TestCases
    enabled_test_cases: Vec<TestCase>,
    /// initial state of the Iris Database
    initial_db_state: IrisDB,
    /// full_mask_range
    full_mask_range: Range<usize>,
    /// expected results for all of the queries we send
    expected_results: HashMap<String, ExpectedResult>,
    /// mapping from request_id to the index of the target entry to be
    /// matched in reauth
    reauth_target_indices: HashMap<String, u32>,
    /// responses received from the servers, where a new iris code was
    /// inserted. Maps position in the database to the E2ETemplate
    inserted_responses: HashMap<u32, E2ETemplate>,
    /// A buffer of indices that have been deleted, to choose a index from
    /// to send for testing against deletions. Once picked, it is removed
    /// from here
    deleted_indices_buffer: Vec<u32>,
    /// The full set of indices that have been deleted
    deleted_indices: HashSet<u32>,
    /// A list of indices that are not allowed to be queried, to avoid
    /// potential false matches
    disallowed_queries: Vec<u32>,
    /// Expected properties of the bucket statistics (num_gpus, )
    bucket_statistic_parameters: Option<BucketStatisticParameters>,
    /// The rng that is used internally
    rng: StdRng,

    // info for current batch, will be cleared at the start of a new batch
    /// New templates that have been inserted in the current batch.
    /// (position in batch, request_id, template)
    new_templates_in_batch: Vec<(usize, String, IrisCode)>,
    /// skip invalidating requests in the current batch, since we expect
    /// them to be processed
    skip_invalidate: bool,
    /// duplicates in the current batch, used to test the batch
    /// deduplication mechanism
    batch_duplicates: HashMap<String, String>,
    /// indices used in the current batch, to avoid deleting those
    db_indices_used_in_current_batch: HashSet<usize>,
    /// items against which the OR rule is used
    or_rule_matches: Vec<String>,
    is_cpu: bool,
}

impl TestCaseGenerator {
    pub fn new_with_db(db: &mut IrisDB, internal_rng_seed: u64, is_cpu: bool) -> Self {
        // Set the masks to all 1s for the first 10%
        let db_len = db.db.len();
        for i in 0..db_len / 10 {
            db.db[i].mask = IrisCodeArray::ONES;
        }
        let rng = StdRng::seed_from_u64(internal_rng_seed);
        Self {
            enabled_test_cases: TestCase::default_test_set(),
            initial_db_state: db.clone(),
            full_mask_range: 0..db_len / 10,
            expected_results: HashMap::new(),
            reauth_target_indices: HashMap::new(),
            inserted_responses: HashMap::new(),
            deleted_indices_buffer: Vec::new(),
            deleted_indices: HashSet::new(),
            disallowed_queries: Vec::new(),
            rng,
            bucket_statistic_parameters: None,
            new_templates_in_batch: Vec::new(),
            skip_invalidate: false,
            batch_duplicates: HashMap::new(),
            db_indices_used_in_current_batch: HashSet::new(),
            or_rule_matches: Vec::new(),
            is_cpu,
        }
    }

    pub fn new_seeded(
        db_size: usize,
        db_rng_seed: u64,
        internal_rng_seed: u64,
        is_cpu: bool,
    ) -> Self {
        // create a copy of the plain database for the test case generator, this needs
        // to be in sync with `generate_db`
        let mut db = IrisDB::new_random_rng(db_size, &mut StdRng::seed_from_u64(db_rng_seed));
        Self::new_with_db(&mut db, internal_rng_seed, is_cpu)
    }
    pub fn enable_test_case(&mut self, test_case: TestCase) {
        if self.enabled_test_cases.contains(&test_case) {
            return;
        }
        self.enabled_test_cases.push(test_case);
    }
    pub fn disable_test_case(&mut self, test_case: TestCase) {
        self.enabled_test_cases.retain(|x| x != &test_case);
    }

    pub fn enable_bucket_statistic_checks(
        &mut self,
        num_buckets: usize,
        num_gpus_per_party: usize,
        match_distances_buffer_size: usize,
    ) {
        self.bucket_statistic_parameters = Some(BucketStatisticParameters {
            num_gpus: num_gpus_per_party,
            num_buckets,
            match_buffer_size: match_distances_buffer_size,
        });
    }

    fn generate_query_batch(
        &mut self,
        max_batch_size: usize,
        max_deletions_per_batch: usize,
    ) -> Result<([BatchQuery; 3], HashMap<String, E2ETemplate>)> {
        let mut requests: HashMap<String, E2ETemplate> = HashMap::new();
        let mut batch0 = BatchQuery::default();
        let mut batch1 = BatchQuery::default();
        let mut batch2 = BatchQuery::default();
        let batch_size = self.rng.gen_range(1..max_batch_size);

        self.batch_duplicates.clear();
        self.skip_invalidate = false;
        self.new_templates_in_batch.clear();
        self.db_indices_used_in_current_batch.clear();
        self.or_rule_matches.clear();

        for idx in 0..batch_size {
            let (request_id, e2e_template, or_rule_indices, skip_persistence) =
                self.generate_query(idx);

            // Invalidate 10% of the queries, but ignore the batch duplicates
            // TODO: remove the check for cpu once batch deduplication is implemented
            let is_valid = self.is_cpu || self.rng.gen_bool(0.10) || self.skip_invalidate;
            if is_valid {
                requests.insert(request_id.to_string(), e2e_template.clone());
            }

            let maybe_reauth_target_index = self.reauth_target_indices.get(&request_id.to_string());
            let shared_template = e2e_template.to_shared_template(is_valid, &mut self.rng);

            prepare_batch(
                &mut batch0,
                is_valid,
                request_id.to_string(),
                0,
                shared_template.clone(),
                or_rule_indices.clone(),
                maybe_reauth_target_index,
                skip_persistence,
            )?;

            prepare_batch(
                &mut batch1,
                true,
                request_id.to_string(),
                1,
                shared_template.clone(),
                or_rule_indices.clone(),
                maybe_reauth_target_index,
                skip_persistence,
            )?;

            prepare_batch(
                &mut batch2,
                true,
                request_id.to_string(),
                2,
                shared_template,
                or_rule_indices.clone(),
                maybe_reauth_target_index,
                skip_persistence,
            )?;
        }

        // Skip empty batch
        if batch0.request_ids.is_empty() {
            return Ok(([batch0, batch1, batch2], requests));
        }

        // for non-empty batches also add some deletions
        if max_deletions_per_batch > 0 {
            for _ in 0..self.rng.gen_range(0..max_deletions_per_batch) {
                let idx = self.rng.gen_range(0..self.initial_db_state.db.len());
                if self.deleted_indices.contains(&(idx as u32))
                    || self.db_indices_used_in_current_batch.contains(&idx)
                {
                    continue;
                }
                self.deleted_indices_buffer.push(idx as u32);
                self.deleted_indices.insert(idx as u32);
                tracing::info!("Deleting index {}", idx);

                batch0.deletion_requests_indices.push(idx as u32);
                batch1.deletion_requests_indices.push(idx as u32);
                batch2.deletion_requests_indices.push(idx as u32);
            }
        }
        Ok(([batch0, batch1, batch2], requests))
    }

    /// Get an Iris code known to be in the database, and return it and its
    /// index. The `DatabaseRange` parameter is used to chose which portion
    /// of the DB the item is chosen from.
    fn get_iris_code_in_db(&mut self, db_range: DatabaseRange) -> (usize, IrisCode) {
        let mut db_index = None;
        let range = match db_range {
            DatabaseRange::FullMaskOnly => self.full_mask_range.clone(),
            DatabaseRange::Full => 0..self.initial_db_state.db.len(),
        };
        for _ in 0..100 {
            let potential_db_index = self.rng.gen_range(range.clone());
            if self.deleted_indices.contains(&(potential_db_index as u32)) {
                continue;
            }
            if self
                .disallowed_queries
                .contains(&(potential_db_index as u32))
            {
                continue;
            }
            db_index = Some(potential_db_index);
            break;
        }
        let db_index = db_index.expect("could not find a valid DB item in 100 random drawings");
        (db_index, self.initial_db_state.db[db_index].clone())
    }

    fn generate_query(
        &mut self,
        internal_batch_idx: usize,
    ) -> (Uuid, E2ETemplate, OrRuleSerialIds, bool) {
        let request_id = Uuid::new_v4();
        let mut skip_persistence = false;
        // Automatic random tests
        let mut options = vec![
            TestCase::Match,
            TestCase::NonMatch,
            TestCase::CloseToThreshold,
            TestCase::WithOrRuleSet,
            TestCase::ReauthNonMatchingTarget,
            TestCase::ReauthMatchingTarget,
            TestCase::ReauthOrRuleNonMatchingTarget,
            TestCase::ReauthOrRuleMatchingTarget,
            TestCase::MatchSkipPersistence,
            TestCase::NonMatchSkipPersistence,
        ];
        if !self.inserted_responses.is_empty() {
            options.push(TestCase::PreviouslyInserted);
        }
        if !self.deleted_indices_buffer.is_empty() {
            options.push(TestCase::PreviouslyDeleted);
        };

        options.retain(|x| self.enabled_test_cases.contains(x));

        let mut or_rule_indices: Vec<u32> = Vec::new();

        // with a 10% chance we pick a template from the batch, to test the batch
        // deduplication mechanism
        // TODO: remove the check for cpu once batch deduplication is implemented
        let pick_from_batch = !self.is_cpu && self.rng.gen_bool(0.10);
        let e2e_template = if pick_from_batch && !self.new_templates_in_batch.is_empty() {
            let random_idx = self.rng.gen_range(0..self.new_templates_in_batch.len());
            let (batch_idx, duplicate_request_id, template) =
                self.new_templates_in_batch[random_idx].clone();
            self.expected_results.insert(
                request_id.to_string(),
                ExpectedResult {
                    db_index: Some(batch_idx as u32),
                    is_batch_match: true,
                    is_reauth_successful: None,
                    is_skip_persistence_request: false,
                },
            );
            self.batch_duplicates
                .insert(request_id.to_string(), duplicate_request_id);
            self.skip_invalidate = true;
            E2ETemplate {
                left: template.clone(),
                right: template.clone(),
            }
        } else {
            // otherwise we pick from the valid test case options
            let option = options
                .choose(&mut self.rng)
                .expect("we have at least one testcase option");
            tracing::info!("Request {} has type {:?}", request_id, option);
            match &option {
                TestCase::NonMatch => {
                    tracing::info!("Sending new iris code");
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: None,
                            is_batch_match: false,
                            is_reauth_successful: None,
                            is_skip_persistence_request: false,
                        },
                    );
                    let template = IrisCode::random_rng(&mut self.rng);
                    self.new_templates_in_batch.push((
                        internal_batch_idx,
                        request_id.to_string(),
                        template.clone(),
                    ));
                    self.skip_invalidate = true;
                    E2ETemplate {
                        left: template.clone(),
                        right: template.clone(),
                    }
                }
                TestCase::NonMatchSkipPersistence => {
                    tracing::info!("Sending new iris code with skip persistence");
                    skip_persistence = true;
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: None,
                            is_batch_match: false,
                            is_reauth_successful: None,
                            is_skip_persistence_request: true,
                        },
                    );
                    let template = IrisCode::random_rng(&mut self.rng);
                    E2ETemplate {
                        left: template.clone(),
                        right: template.clone(),
                    }
                }
                TestCase::Match => {
                    tracing::info!("Sending iris code from db");
                    let (db_index, template) = self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: Some(db_index as u32),
                            is_batch_match: false,
                            is_reauth_successful: None,
                            is_skip_persistence_request: false,
                        },
                    );
                    E2ETemplate {
                        left: template.clone(),
                        right: template,
                    }
                }
                TestCase::MatchSkipPersistence => {
                    tracing::info!("Sending iris code from db with skip persistence");
                    let (db_index, template) = self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    skip_persistence = true;
                    self.disallowed_queries.push(db_index as u32);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: Some(db_index as u32),
                            is_batch_match: false,
                            is_reauth_successful: None,
                            is_skip_persistence_request: true,
                        },
                    );
                    E2ETemplate {
                        left: template.clone(),
                        right: template,
                    }
                }
                TestCase::CloseToThreshold => {
                    tracing::info!("Sending iris code on the threshold");
                    let (db_index, mut template) =
                        self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    let variation = self.rng.gen_range(-1..=1);
                    self.expected_results.insert(
                        request_id.to_string(),
                        if variation > 0 {
                            // we flip more than the threshold so this should not match
                            // however it would afterwards so we no longer pick it
                            self.disallowed_queries.push(db_index as u32);
                            ExpectedResult {
                                db_index: None,
                                is_batch_match: false,
                                is_reauth_successful: None,
                                is_skip_persistence_request: false,
                            }
                        } else {
                            // we flip less or equal to than the threshold so this should
                            // match
                            ExpectedResult {
                                db_index: Some(db_index as u32),
                                is_batch_match: false,
                                is_reauth_successful: None,
                                is_skip_persistence_request: false,
                            }
                        },
                    );
                    assert_eq!(template.mask, IrisCodeArray::ONES);
                    for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                        template.code.flip_bit(i);
                    }
                    E2ETemplate {
                        left: template.clone(),
                        right: template,
                    }
                }
                TestCase::PreviouslyInserted => {
                    tracing::info!("Sending freshly inserted iris code");
                    let (idx, e2e_template) = self
                        .inserted_responses
                        .iter()
                        .choose(&mut self.rng)
                        .expect("we have at least one response");
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: Some(*idx),
                            is_batch_match: false,
                            is_reauth_successful: None,
                            is_skip_persistence_request: false,
                        },
                    );
                    self.db_indices_used_in_current_batch.insert(*idx as usize);
                    E2ETemplate {
                        left: e2e_template.left.clone(),
                        right: e2e_template.right.clone(),
                    }
                }
                TestCase::PreviouslyDeleted => {
                    tracing::info!("Sending deleted iris code");
                    let idx = self.rng.gen_range(0..self.deleted_indices_buffer.len());
                    let deleted_idx = self.deleted_indices_buffer[idx];

                    self.deleted_indices_buffer.remove(idx);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: None,
                            is_batch_match: false,
                            is_reauth_successful: None,
                            is_skip_persistence_request: false,
                        },
                    );
                    E2ETemplate {
                        right: self.initial_db_state.db[deleted_idx as usize].clone(),
                        left: self.initial_db_state.db[deleted_idx as usize].clone(),
                    }
                }
                TestCase::WithOrRuleSet => {
                    tracing::info!(
                        "Sending iris codes that match on one side but not the other with the OR \
                         rule set"
                    );

                    // use 1 to 10 OR-matching iris codes
                    let n_db_indexes = self.rng.gen_range(1..10);

                    // Remove disallowed queries from the pool
                    let db_indexes = (0..n_db_indexes)
                        .map(|_| loop {
                            let (db_index, _) =
                                self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                            if !self.disallowed_queries.contains(&(db_index as u32)) {
                                return db_index;
                            }
                        })
                        .collect::<Vec<_>>();

                    let db_indexes_copy = db_indexes.clone();

                    // select a random one to use as matching signup
                    let matching_db_index =
                        db_indexes_copy[self.rng.gen_range(0..db_indexes_copy.len())];

                    // comparison against this item will use the OR rule
                    or_rule_indices = db_indexes_copy.iter().map(|&x| x as u32).collect();

                    // apply variation to either right of left code
                    let will_match: bool = self.rng.gen();
                    let flip_right = if will_match {
                        Some(self.rng.gen())
                    } else {
                        None
                    };

                    let template =
                        self.prepare_flipped_codes(matching_db_index, will_match, flip_right);

                    if will_match {
                        self.or_rule_matches.push(request_id.to_string());
                        self.expected_results.insert(
                            request_id.to_string(),
                            ExpectedResult {
                                db_index: Some(matching_db_index as u32),
                                is_batch_match: false,
                                is_reauth_successful: None,
                                is_skip_persistence_request: false,
                            },
                        );
                    } else {
                        self.db_indices_used_in_current_batch
                            .insert(matching_db_index);
                        self.disallowed_queries.push(matching_db_index as u32);
                        self.expected_results.insert(
                            request_id.to_string(),
                            ExpectedResult {
                                db_index: None,
                                is_batch_match: false,
                                is_reauth_successful: None,
                                is_skip_persistence_request: false,
                            },
                        );
                    }
                    template
                }
                TestCase::ReauthMatchingTarget => {
                    tracing::info!(
                        "Sending reauth request with AND rule matching the target index"
                    );
                    let (db_index, template) = self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: Some(db_index as u32),
                            is_batch_match: false,
                            is_reauth_successful: Some(true),
                            is_skip_persistence_request: false,
                        },
                    );
                    E2ETemplate {
                        left: template.clone(),
                        right: template,
                    }
                }
                TestCase::ReauthNonMatchingTarget => {
                    tracing::info!(
                        "Sending reauth request with AND rule non-matching the target index"
                    );
                    let (db_index, _) = self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.disallowed_queries.push(db_index as u32);
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);

                    // prepare a template that matches only on one side
                    // it will end up with a failed reauth with AND rule
                    self.or_rule_matches.push(request_id.to_string());
                    let will_match = true;
                    let flip_right = Some(self.rng.gen());
                    let template = self.prepare_flipped_codes(db_index, will_match, flip_right);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: None,
                            is_batch_match: false,
                            is_reauth_successful: Some(false),
                            is_skip_persistence_request: false,
                        },
                    );
                    template
                }
                TestCase::ReauthOrRuleMatchingTarget => {
                    tracing::info!("Sending reauth request with OR rule matching the target index");
                    let (db_index, _) = self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.disallowed_queries.push(db_index as u32);
                    self.or_rule_matches.push(request_id.to_string());
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);
                    or_rule_indices = vec![db_index as u32];
                    let will_match = true;
                    let flip_right = Some(self.rng.gen());
                    let template = self.prepare_flipped_codes(db_index, will_match, flip_right);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: None,
                            is_batch_match: false,
                            is_reauth_successful: Some(true),
                            is_skip_persistence_request: false,
                        },
                    );
                    template
                }
                TestCase::ReauthOrRuleNonMatchingTarget => {
                    tracing::info!(
                        "Sending reauth request with OR rule non-matching the target index"
                    );
                    let (db_index, _) = self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);
                    or_rule_indices = vec![db_index as u32];
                    let will_match = false;
                    let template = self.prepare_flipped_codes(db_index, will_match, None);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult {
                            db_index: None,
                            is_batch_match: false,
                            is_reauth_successful: Some(false),
                            is_skip_persistence_request: false,
                        },
                    );
                    template
                }
            }
        };
        (request_id, e2e_template, or_rule_indices, skip_persistence)
    }

    /// Returns a template with flipped bits of given `db_index`.
    ///
    /// If `will_match` is false, the template will be flipped to above
    /// threshold on both sides If `flip_right` is true, the right
    /// side will be flipped to above threshold
    fn prepare_flipped_codes(
        &mut self,
        db_index: usize,
        will_match: bool,
        flip_right: Option<bool>,
    ) -> E2ETemplate {
        let mut code_left = self.initial_db_state.db[db_index].clone();
        let mut code_right = self.initial_db_state.db[db_index].clone();

        assert_eq!(code_left.mask, IrisCodeArray::ONES);
        assert_eq!(code_right.mask, IrisCodeArray::ONES);

        let variation = self.rng.gen_range(1..100);
        if will_match {
            let flip_right = flip_right.unwrap_or(self.rng.gen());
            if flip_right {
                // Flip right bits to above threshold - (right) does not match
                for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                    code_right.code.flip_bit(i);
                }
            } else {
                // Flip left bits to above threshold - (left) does not match
                for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                    code_left.code.flip_bit(i);
                }
            }
        } else {
            // Flip both to above threshold - neither match
            for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                code_left.code.flip_bit(i);
                code_right.code.flip_bit(i);
            }
        }
        E2ETemplate {
            left: code_left,
            right: code_right,
        }
    }

    // check a received result against the expected results
    #[allow(clippy::too_many_arguments)]
    fn check_result(
        &mut self,
        req_id: &str,
        idx: u32,
        was_match: bool,
        was_skip_persistence_match: bool,
        matched_batch_req_ids: &[String],
        requests: &HashMap<String, E2ETemplate>,
        was_reauth_success: bool,
    ) {
        tracing::info!(
            "Checking result for request_id: {}, idx: {}, was_match: {}, matched_batch_req_ids: \
             {:?}, was_reauth_success: {}",
            req_id,
            idx,
            was_match,
            matched_batch_req_ids,
            was_reauth_success
        );
        let &ExpectedResult {
            db_index: expected_idx,
            is_batch_match,
            is_reauth_successful,
            is_skip_persistence_request,
        } = self
            .expected_results
            .get(req_id)
            .expect("request id not found");

        // if the request is a reauth, we only check the reauth success
        if let Some(is_reauth_successful) = is_reauth_successful {
            assert_eq!(is_reauth_successful, was_reauth_success);
            return;
        }

        if let Some(expected_idx) = expected_idx {
            assert!(was_match);
            assert!(was_skip_persistence_match);
            if !is_batch_match {
                assert_eq!(expected_idx, idx);
            } else {
                assert!(self.batch_duplicates.contains_key(req_id));
                assert!(matched_batch_req_ids.contains(self.batch_duplicates.get(req_id).unwrap()));
            }
        } else {
            assert!(!was_skip_persistence_match);
            if is_skip_persistence_request {
                assert!(was_match);
            } else {
                assert!(!was_match);
                let request = requests.get(req_id).unwrap().clone();
                self.inserted_responses.insert(idx, request);
            }
        }
    }

    pub async fn run_n_batches(
        &mut self,
        num_batches: usize,
        max_batch_size: usize,
        max_deletions_per_batch: usize,
        handles: [&mut impl JobSubmissionHandle; 3],
    ) -> Result<()> {
        let [handle0, handle1, handle2] = handles;
        for _ in 0..num_batches {
            // Skip empty batch
            let ([batch0, batch1, batch2], requests) =
                self.generate_query_batch(max_batch_size, max_deletions_per_batch)?;
            if batch0.request_ids.is_empty() {
                continue;
            }

            // send batches to servers
            let (res0_fut, res1_fut, res2_fut) = tokio::join!(
                handle0.submit_batch_query(batch0),
                handle1.submit_batch_query(batch1),
                handle2.submit_batch_query(batch2)
            );

            let res0 = res0_fut.await;
            let res1 = res1_fut.await;
            let res2 = res2_fut.await;

            let mut resp_counters = HashMap::new();
            for req in requests.keys() {
                resp_counters.insert(req, 0);
            }

            let results = [&res0, &res1, &res2];
            for res in results.iter() {
                let ServerJobResult {
                    request_ids: thread_request_ids,
                    matches,
                    matches_with_skip_persistence,
                    merged_results,
                    match_ids,
                    partial_match_ids_left,
                    partial_match_ids_right,
                    matched_batch_request_ids,
                    anonymized_bucket_statistics_left,
                    anonymized_bucket_statistics_right,
                    successful_reauths,
                    ..
                } = res;

                if let Some(bucket_statistic_parameters) = &self.bucket_statistic_parameters {
                    check_bucket_statistics(
                        anonymized_bucket_statistics_left,
                        bucket_statistic_parameters.num_gpus,
                        bucket_statistic_parameters.num_buckets,
                        bucket_statistic_parameters.match_buffer_size,
                    )?;
                    check_bucket_statistics(
                        anonymized_bucket_statistics_right,
                        bucket_statistic_parameters.num_gpus,
                        bucket_statistic_parameters.num_buckets,
                        bucket_statistic_parameters.match_buffer_size,
                    )?;
                }

                for (
                    req_id,
                    &was_match,
                    &was_skip_persistence_match,
                    &was_reauth_success,
                    &idx,
                    partial_left,
                    partial_right,
                    match_id,
                    matched_batch_req_ids,
                ) in izip!(
                    thread_request_ids,
                    matches,
                    matches_with_skip_persistence,
                    successful_reauths,
                    merged_results,
                    partial_match_ids_left,
                    partial_match_ids_right,
                    match_ids,
                    matched_batch_request_ids
                ) {
                    assert!(requests.contains_key(req_id));

                    resp_counters.insert(req_id, resp_counters.get(req_id).unwrap() + 1);

                    if !self.or_rule_matches.contains(req_id) {
                        assert_eq!(partial_left, partial_right);
                        assert_eq!(partial_left, match_id);
                    }
                    self.check_result(
                        req_id,
                        idx,
                        was_match,
                        was_skip_persistence_match,
                        matched_batch_req_ids,
                        &requests,
                        was_reauth_success,
                    );
                }
            }

            // Check that we received a response from all actors
            for (&id, &count) in resp_counters.iter() {
                assert_eq!(count, 3, "Received {} responses for {}", count, id);
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_batch(
    batch: &mut BatchQuery,
    is_valid: bool,
    request_id: String,
    batch_idx: usize,
    mut e2e_shared_template: E2ESharedTemplate,
    or_rule_indices: Vec<u32>,
    maybe_reauth_target_index: Option<&u32>,
    skip_persistence: bool,
) -> Result<()> {
    batch.metadata.push(Default::default());
    batch.valid_entries.push(is_valid);
    batch.skip_persistence.push(skip_persistence);
    batch.request_ids.push(request_id.clone());
    match maybe_reauth_target_index {
        Some(target_index) => {
            batch.request_types.push(REAUTH_MESSAGE_TYPE.to_string());
            batch
                .reauth_use_or_rule
                .insert(request_id.clone(), !or_rule_indices.is_empty());
            batch
                .reauth_target_indices
                .insert(request_id.clone(), *target_index);
        }
        None => {
            batch
                .request_types
                .push(UNIQUENESS_MESSAGE_TYPE.to_string());
        }
    }

    batch.or_rule_indices.push(or_rule_indices);

    batch
        .left_iris_requests
        .code
        .push(e2e_shared_template.left_shared_code[batch_idx].clone());
    batch
        .left_iris_requests
        .mask
        .push(e2e_shared_template.left_shared_mask[batch_idx].clone());
    batch
        .right_iris_requests
        .code
        .push(e2e_shared_template.right_shared_code[batch_idx].clone());
    batch
        .right_iris_requests
        .mask
        .push(e2e_shared_template.right_shared_mask[batch_idx].clone());

    batch
        .left_iris_rotated_requests
        .code
        .extend(e2e_shared_template.left_shared_code[batch_idx].all_rotations());
    batch
        .left_iris_rotated_requests
        .mask
        .extend(e2e_shared_template.left_shared_mask[batch_idx].all_rotations());

    batch
        .right_iris_rotated_requests
        .code
        .extend(e2e_shared_template.left_shared_code[batch_idx].all_rotations());
    batch
        .right_iris_rotated_requests
        .mask
        .extend(e2e_shared_template.left_shared_mask[batch_idx].all_rotations());

    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(
        &mut e2e_shared_template.left_shared_code[batch_idx],
    );
    GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(
        &mut e2e_shared_template.left_shared_mask[batch_idx],
    );
    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(
        &mut e2e_shared_template.right_shared_code[batch_idx],
    );
    GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(
        &mut e2e_shared_template.right_shared_mask[batch_idx],
    );
    batch
        .left_iris_interpolated_requests
        .code
        .extend(e2e_shared_template.left_shared_code[batch_idx].all_rotations());
    batch
        .left_iris_interpolated_requests
        .mask
        .extend(e2e_shared_template.left_shared_mask[batch_idx].all_rotations());

    batch
        .right_iris_interpolated_requests
        .code
        .extend(e2e_shared_template.right_shared_code[batch_idx].all_rotations());
    batch
        .right_iris_interpolated_requests
        .mask
        .extend(e2e_shared_template.right_shared_mask[batch_idx].all_rotations());

    Ok(())
}

fn check_bucket_statistics(
    bucket_statistics: &BucketStatistics,
    num_gpus_per_party: usize,
    num_buckets: usize,
    match_distances_buffer_size: usize,
) -> Result<()> {
    if bucket_statistics.is_empty() {
        assert_eq!(bucket_statistics.buckets.len(), 0);
        return Ok(());
    }
    assert_eq!(bucket_statistics.buckets.len(), num_buckets);
    assert!(
        bucket_statistics.end_time_utc_timestamp > Some(bucket_statistics.start_time_utc_timestamp)
    );
    let total_count = bucket_statistics
        .buckets
        .iter()
        .map(|b| b.count)
        .sum::<usize>();
    tracing::info!("Total count for bucket: {}", total_count);
    assert_eq!(
        total_count,
        match_distances_buffer_size * num_gpus_per_party
    );
    Ok(())
}

pub fn generate_test_db(
    party_id: usize,
    db_size: usize,
    db_rng_seed: u64,
) -> Vec<(GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare)> {
    let mut rng = StdRng::seed_from_u64(db_rng_seed);
    let mut db = IrisDB::new_random_par(db_size, &mut rng);

    // Set the masks to all 1s for the first 10%
    for i in 0..db_size / 10 {
        db.db[i].mask = IrisCodeArray::ONES;
    }

    let mut share_rng = StdRng::from_rng(rng).unwrap();

    let mut result = Vec::new();

    for iris in db.db.into_iter() {
        let code =
            GaloisRingIrisCodeShare::encode_iris_code(&iris.code, &iris.mask, &mut share_rng)
                [party_id]
                .clone();
        let mask: GaloisRingTrimmedMaskCodeShare =
            GaloisRingIrisCodeShare::encode_mask_code(&iris.mask, &mut share_rng)[party_id]
                .clone()
                .into();
        result.push((code, mask));
    }

    result
}

pub fn load_test_db(
    party_id: usize,
    db_size: usize,
    db_rng_seed: u64,
    loader: &mut impl InMemoryStore,
) -> Result<()> {
    let iris_shares = generate_test_db(party_id, db_size, db_rng_seed);
    for (idx, (code, mask)) in iris_shares.into_iter().enumerate() {
        loader.load_single_record_from_db(idx, &code.coefs, &mask.coefs, &code.coefs, &mask.coefs);
        loader.increment_db_size(idx);
    }

    Ok(())
}
