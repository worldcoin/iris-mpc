use crate::galois_engine::degree4::FullGaloisRingIrisCodeShare;
use crate::job::{BatchMetadata, GaloisSharesBothSides};
use crate::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    helpers::{
        inmemory_store::InMemoryStore,
        smpc_request::{REAUTH_MESSAGE_TYPE, RESET_CHECK_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
    },
    iris_db::{
        db::IrisDB,
        iris::{IrisCode, IrisCodeArray},
    },
    job::{BatchQuery, JobSubmissionHandle, ServerJobResult},
    vector_id::VectorId,
    IRIS_CODE_LENGTH,
};
use eyre::Result;
use itertools::izip;
use rand::{
    rngs::StdRng,
    seq::{IteratorRandom, SliceRandom},
    Rng, SeedableRng,
};
use std::sync::Arc;
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};
use uuid::Uuid;

const THRESHOLD_ABSOLUTE: usize = IRIS_CODE_LENGTH * 375 / 1000; // 0.375 * 12800

const LEFT: usize = 0;
const RIGHT: usize = 1;

#[derive(Clone)]
pub struct E2ETemplate {
    left: IrisCode,
    right: IrisCode,
}
impl E2ETemplate {
    fn to_shared_template(&self, is_valid: bool, rng: &mut StdRng) -> E2ESharedTemplate {
        let (
            left_shared_code,
            left_shared_mask,
            left_mirrored_shared_code,
            left_mirrored_shared_mask,
        ) = get_shared_template(is_valid, &self.left, rng);
        let (
            right_shared_code,
            right_shared_mask,
            right_mirrored_shared_code,
            right_mirrored_shared_mask,
        ) = get_shared_template(is_valid, &self.right, rng);
        E2ESharedTemplate {
            left_shared_code,
            left_shared_mask,
            right_shared_code,
            right_shared_mask,
            left_mirrored_shared_code,
            left_mirrored_shared_mask,
            right_mirrored_shared_code,
            right_mirrored_shared_mask,
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
    [GaloisRingIrisCodeShare; 3],
    [GaloisRingTrimmedMaskCodeShare; 3],
) {
    let mut shared_code =
        GaloisRingIrisCodeShare::encode_iris_code(&template.code, &template.mask, rng);

    let shared_mask = GaloisRingIrisCodeShare::encode_mask_code(&template.mask, rng);

    // Create mirrored versions of the shares (before trimming for masks)
    let mut mirrored_shared_code = [
        shared_code[0].mirrored_code(),
        shared_code[1].mirrored_code(),
        shared_code[2].mirrored_code(),
    ];

    let mirrored_shared_mask = [
        shared_mask[0].mirrored_mask(),
        shared_mask[1].mirrored_mask(),
        shared_mask[2].mirrored_mask(),
    ];

    // Now trim the masks
    let shared_mask_vector: Vec<GaloisRingTrimmedMaskCodeShare> =
        shared_mask.iter().map(|x| x.clone().into()).collect();

    let mirrored_shared_mask_vector: Vec<GaloisRingTrimmedMaskCodeShare> = mirrored_shared_mask
        .iter()
        .map(|x| x.clone().into())
        .collect();

    let mut shared_mask: [GaloisRingTrimmedMaskCodeShare; 3] =
        shared_mask_vector.try_into().unwrap();

    let mut mirrored_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3] =
        mirrored_shared_mask_vector.try_into().unwrap();

    if !is_valid {
        shared_code[0] = GaloisRingIrisCodeShare::default_for_party(1);
        shared_mask[0] = GaloisRingTrimmedMaskCodeShare::default_for_party(1);
        mirrored_shared_code[0] = GaloisRingIrisCodeShare::default_for_party(1);
        mirrored_shared_mask[0] = GaloisRingTrimmedMaskCodeShare::default_for_party(1);
    }

    (
        shared_code,
        shared_mask,
        mirrored_shared_code,
        mirrored_shared_mask,
    )
}

type OrRuleSerialIds = Vec<u32>;

#[derive(Clone)]
pub struct E2ESharedTemplate {
    pub left_shared_code: [GaloisRingIrisCodeShare; 3],
    pub left_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
    pub right_shared_code: [GaloisRingIrisCodeShare; 3],
    pub right_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
    pub left_mirrored_shared_code: [GaloisRingIrisCodeShare; 3],
    pub left_mirrored_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
    pub right_mirrored_shared_code: [GaloisRingIrisCodeShare; 3],
    pub right_mirrored_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
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
    /// Send a reauth request matching target serial id's iris code only
    /// (successful reauth) but skip persistence.
    ReauthMatchingTargetWithSkipPeristence,
    /// Send a reauth request not matching target serial id's iris code
    /// (failed reauth)
    ReauthNonMatchingTarget,
    /// Send a reauth request with OR rule matching one side of target
    /// serial id's iris code (successful reauth)
    ReauthOrRuleMatchingTarget,
    /// Send a reauth request with OR rule not matching any sides of target
    /// serial id's iris code (failed reauth)
    ReauthOrRuleNonMatchingTarget,
    /// Send a reset check request with an iris code known to be in the database
    /// Similar to MatchSkipPersistence, but using RESET_CHECK_MESSAGE_TYPE
    ResetCheckMatch,
    /// Send a reset check request with an iris code known not to match any in the database
    /// Similar to NonMatchSkipPersistence, but using RESET_CHECK_MESSAGE_TYPE
    ResetCheckNonMatch,
    /// Send an enrollment request using the iris codes used during ResetCheckNonMatch and expect it to be inserted without matches. This will make sure that reset_check did not write into the database.
    EnrollmentAfterResetCheckNonMatch,
    /// Send an enrollment request using the iris codes used during ResetUpdate and expect a match result
    MatchAfterResetUpdate,
    /// Send an enrollment request using the iris codes used during
    /// ReauthMatchingTargetWithSkipPeristence and expect a match result.
    MatchAfterReauthSkipPeristence,
    /// Send an iris code crafted for full face mirror attack detection:
    /// - Normal flow won't match anything in the database
    /// - But when the code is mirrored, it will match(mirrored version will be pre-inserted in the test db)
    FullFaceMirrorAttack,
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
            TestCase::ReauthMatchingTargetWithSkipPeristence,
            TestCase::ReauthOrRuleNonMatchingTarget,
            TestCase::ReauthOrRuleMatchingTarget,
            TestCase::ResetCheckMatch,
            TestCase::ResetCheckNonMatch,
            TestCase::EnrollmentAfterResetCheckNonMatch,
            TestCase::MatchAfterResetUpdate,
            TestCase::MatchAfterReauthSkipPeristence,
            TestCase::FullFaceMirrorAttack,
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
    /// Whether this is a RESET_CHECK_MESSAGE_TYPE request
    is_reset_check: bool,
    /// Whether this is a FULL_FACE_MIRROR_ATTACK request
    is_full_face_mirror_attack: bool,
}

impl ExpectedResult {
    /// Creates a new builder for ExpectedResult with default values
    pub fn builder() -> ExpectedResultBuilder {
        ExpectedResultBuilder::default()
    }
}

/// Builder for ExpectedResult
#[derive(Default)]
pub struct ExpectedResultBuilder {
    db_index: Option<u32>,
    is_skip_persistence_request: bool,
    is_batch_match: bool,
    is_reauth_successful: Option<bool>,
    is_reset_check: bool,
    is_full_face_mirror_attack: bool,
}

impl ExpectedResultBuilder {
    /// Sets the db_index to Some value
    pub fn with_db_index(mut self, db_index: u32) -> Self {
        self.db_index = Some(db_index);
        self
    }

    /// Sets is_skip_persistence_request
    pub fn with_skip_persistence(mut self, is_skip_persistence_request: bool) -> Self {
        self.is_skip_persistence_request = is_skip_persistence_request;
        self
    }

    /// Sets is_batch_match
    pub fn with_batch_match(mut self, is_batch_match: bool) -> Self {
        self.is_batch_match = is_batch_match;
        self
    }

    /// Sets is_reauth_successful to Some value
    pub fn with_reauth_successful(mut self, is_reauth_successful: bool) -> Self {
        self.is_reauth_successful = Some(is_reauth_successful);
        self
    }

    /// Sets is_reset_check
    pub fn with_reset_check(mut self, is_reset_check: bool) -> Self {
        self.is_reset_check = is_reset_check;
        self
    }

    /// Sets is_full_face_mirror_attack
    pub fn with_full_face_mirror_attack(mut self, is_full_face_mirror_attack: bool) -> Self {
        self.is_full_face_mirror_attack = is_full_face_mirror_attack;
        self
    }

    /// Builds the ExpectedResult
    pub fn build(self) -> ExpectedResult {
        ExpectedResult {
            db_index: self.db_index,
            is_skip_persistence_request: self.is_skip_persistence_request,
            is_batch_match: self.is_batch_match,
            is_reauth_successful: self.is_reauth_successful,
            is_reset_check: self.is_reset_check,
            is_full_face_mirror_attack: self.is_full_face_mirror_attack,
        }
    }
}

pub struct TestCaseGenerator {
    /// enabled TestCases
    enabled_test_cases: Vec<TestCase>,
    /// initial state of the Iris Database
    initial_db_state: TestDb,
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
    /// responses used for reset checks, where a new iris code was
    /// checked against the database
    non_match_reset_check_templates: HashMap<String, E2ETemplate>,
    /// templates used after reauth skip persistence, where memory should be unchanged
    reauth_skip_persistence_templates: HashMap<u32, E2ETemplate>,
    /// templates used for reset updates where memory is overridden with these
    reset_update_templates: HashMap<u32, E2ETemplate>,
    /// A buffer of indices that have been deleted, to choose a index from
    /// to send for testing against deletions. Once picked, it is removed
    /// from here
    deleted_indices_buffer: Vec<u32>,
    /// The full set of indices that have been deleted
    deleted_indices: HashSet<u32>,
    /// A list of indices that are not allowed to be queried, to avoid
    /// potential false matches
    disallowed_queries: Vec<u32>,
    /// The rng that is used internally
    rng: StdRng,

    // info for current batch, will be cleared at the start of a new batch
    /// New templates that have been inserted in the current batch.
    /// (position in batch, request_id, template)
    new_templates_in_batch: Vec<(usize, String, E2ETemplate)>,
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
    pub fn new_with_db(db: TestDb, internal_rng_seed: u64, is_cpu: bool) -> Self {
        // Set the masks to all 1s for the first 10%
        let rng = StdRng::seed_from_u64(internal_rng_seed);
        let db_len = db.len();
        Self {
            enabled_test_cases: TestCase::default_test_set(),
            initial_db_state: db,
            full_mask_range: 0..db_len / 10,
            expected_results: HashMap::new(),
            reauth_target_indices: HashMap::new(),
            inserted_responses: HashMap::new(),
            non_match_reset_check_templates: HashMap::new(),
            reauth_skip_persistence_templates: HashMap::new(),
            reset_update_templates: HashMap::new(),
            deleted_indices_buffer: Vec::new(),
            deleted_indices: HashSet::new(),
            disallowed_queries: Vec::new(),
            rng,
            new_templates_in_batch: Vec::new(),
            skip_invalidate: false,
            batch_duplicates: HashMap::new(),
            db_indices_used_in_current_batch: HashSet::new(),
            or_rule_matches: Vec::new(),
            is_cpu,
        }
    }

    pub fn disable_test_case(&mut self, test_case: TestCase) {
        self.enabled_test_cases.retain(|x| x != &test_case);
    }

    fn generate_query_batch(
        &mut self,
        max_batch_size: usize,
        max_deletions_per_batch: usize,
        max_reset_updates_per_batch: usize,
    ) -> Result<([BatchQuery; 3], HashMap<String, E2ETemplate>)> {
        let mut requests: HashMap<String, E2ETemplate> = HashMap::new();
        let mut batch0 = BatchQuery::default();
        let mut batch1 = BatchQuery::default();
        let mut batch2 = BatchQuery::default();
        batch0.full_face_mirror_attacks_detection_enabled = true;
        batch1.full_face_mirror_attacks_detection_enabled = true;
        batch2.full_face_mirror_attacks_detection_enabled = true;
        let batch_size = self.rng.gen_range(1..=max_batch_size);

        self.batch_duplicates.clear();
        self.skip_invalidate = false;
        self.new_templates_in_batch.clear();
        self.db_indices_used_in_current_batch.clear();
        self.or_rule_matches.clear();

        for idx in 0..batch_size {
            let (request_id, e2e_template, or_rule_indices, skip_persistence, message_type) =
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
                message_type.clone(),
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
                message_type.clone(),
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
                message_type,
            )?;
        }

        // Skip empty batch
        if batch0.request_ids.is_empty() {
            return Ok(([batch0, batch1, batch2], requests));
        }

        // for non-empty batches also add some deletions
        if max_deletions_per_batch > 0 {
            for _ in 0..self.rng.gen_range(0..max_deletions_per_batch) {
                let idx = self
                    .rng
                    .gen_range(0..self.initial_db_state.initial_db_len());
                if self.deleted_indices.contains(&(idx as u32))
                    || self.db_indices_used_in_current_batch.contains(&idx)
                    || self.disallowed_queries.contains(&(idx as u32))
                {
                    continue;
                }
                self.deleted_indices_buffer.push(idx as u32);
                self.deleted_indices.insert(idx as u32);
                self.disallowed_queries.push(idx as u32);
                tracing::info!("Deleting index {}", idx);

                for b in [&mut batch0, &mut batch1, &mut batch2] {
                    b.push_deletion_request(
                        "sns_id".to_string(),
                        idx as u32,
                        BatchMetadata::default(),
                    );
                }
            }
        }

        // for non-empty batches also add some reset updates
        if max_reset_updates_per_batch > 0 {
            for _ in 0..self.rng.gen_range(0..max_reset_updates_per_batch) {
                let idx = self
                    .rng
                    .gen_range(0..self.initial_db_state.initial_db_len());
                if self.deleted_indices.contains(&(idx as u32))
                    || self.db_indices_used_in_current_batch.contains(&idx)
                    || self.disallowed_queries.contains(&(idx as u32))
                {
                    continue;
                }
                let code = IrisCode::random_rng(&mut self.rng);
                let template = E2ETemplate {
                    left: code.clone(),
                    right: code,
                };
                let shared_template = template.to_shared_template(true, &mut self.rng);
                let shares0 = GaloisSharesBothSides {
                    code_left: shared_template.left_shared_code[0].clone(),
                    mask_left: shared_template.left_shared_mask[0].clone(),
                    code_right: shared_template.right_shared_code[0].clone(),
                    mask_right: shared_template.right_shared_mask[0].clone(),
                };
                let shares1 = GaloisSharesBothSides {
                    code_left: shared_template.left_shared_code[1].clone(),
                    mask_left: shared_template.left_shared_mask[1].clone(),
                    code_right: shared_template.right_shared_code[1].clone(),
                    mask_right: shared_template.right_shared_mask[1].clone(),
                };
                let shares2 = GaloisSharesBothSides {
                    code_left: shared_template.left_shared_code[2].clone(),
                    mask_left: shared_template.left_shared_mask[2].clone(),
                    code_right: shared_template.right_shared_code[2].clone(),
                    mask_right: shared_template.right_shared_mask[2].clone(),
                };
                self.disallowed_queries.push(idx as u32);
                self.reset_update_templates
                    .insert(idx as u32, template.clone());
                let req_id = Uuid::new_v4().to_string();
                requests.insert(req_id.clone(), template.clone());
                tracing::info!(
                    "Applying reset update to index {} with request id {}",
                    idx,
                    req_id
                );
                let sns_id = || "sns_id".to_string();

                batch0.push_reset_update_request(sns_id(), req_id.clone(), idx as u32, shares0);
                batch1.push_reset_update_request(sns_id(), req_id.clone(), idx as u32, shares1);
                batch2.push_reset_update_request(sns_id(), req_id, idx as u32, shares2);
            }
        }

        Ok(([batch0, batch1, batch2], requests))
    }

    /// Get an Iris code known to be in the database, and return it and its
    /// index. The `DatabaseRange` parameter is used to chose which portion
    /// of the DB the item is chosen from.
    fn get_iris_code_in_db(&mut self, db_range: DatabaseRange) -> (usize, [IrisCode; 2]) {
        let mut db_index = None;
        let range = match db_range {
            DatabaseRange::FullMaskOnly => self.full_mask_range.clone(),
            DatabaseRange::Full => 0..self.initial_db_state.initial_db_len(),
        };
        for _ in 0..100 {
            let potential_db_index = self.rng.gen_range(range.clone());
            if self.deleted_indices.contains(&(potential_db_index as u32)) {
                continue;
            }
            if self
                .db_indices_used_in_current_batch
                .contains(&potential_db_index)
            {
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
        (
            db_index,
            [
                self.initial_db_state.plain_dbs[LEFT].db[db_index].clone(),
                self.initial_db_state.plain_dbs[RIGHT].db[db_index].clone(),
            ],
        )
    }

    fn generate_query(
        &mut self,
        internal_batch_idx: usize,
    ) -> (Uuid, E2ETemplate, OrRuleSerialIds, bool, String) {
        let request_id = Uuid::new_v4();
        let mut skip_persistence = false;
        let mut message_type = UNIQUENESS_MESSAGE_TYPE.to_string();
        // Automatic random tests
        let mut options = vec![
            TestCase::Match,
            TestCase::NonMatch,
            TestCase::CloseToThreshold,
            TestCase::WithOrRuleSet,
            TestCase::ReauthNonMatchingTarget,
            TestCase::ReauthMatchingTarget,
            TestCase::ReauthMatchingTargetWithSkipPeristence,
            TestCase::ReauthOrRuleNonMatchingTarget,
            TestCase::ReauthOrRuleMatchingTarget,
            TestCase::MatchSkipPersistence,
            TestCase::NonMatchSkipPersistence,
            TestCase::ResetCheckMatch,
            TestCase::ResetCheckNonMatch,
            TestCase::FullFaceMirrorAttack,
        ];

        if !self.inserted_responses.is_empty() {
            options.push(TestCase::PreviouslyInserted);
        }
        if !self.deleted_indices_buffer.is_empty() {
            options.push(TestCase::PreviouslyDeleted);
        };
        if !self.non_match_reset_check_templates.is_empty() {
            options.push(TestCase::EnrollmentAfterResetCheckNonMatch);
        }
        if !self.reset_update_templates.is_empty() {
            options.push(TestCase::MatchAfterResetUpdate);
        }
        if !self.reauth_skip_persistence_templates.is_empty() {
            options.push(TestCase::MatchAfterReauthSkipPeristence);
        }

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
                ExpectedResult::builder()
                    .with_db_index(batch_idx as u32)
                    .with_batch_match(true)
                    .build(),
            );
            self.batch_duplicates
                .insert(request_id.to_string(), duplicate_request_id);
            self.skip_invalidate = true;
            template.clone()
        } else {
            // otherwise we pick from the valid test case options
            let option = options
                .choose(&mut self.rng)
                .expect("we have at least one testcase option");
            tracing::info!("Request {} has type {:?}", request_id, option);
            match &option {
                TestCase::NonMatch => {
                    tracing::info!("Sending new iris code");
                    self.expected_results
                        .insert(request_id.to_string(), ExpectedResult::builder().build());
                    let template = IrisCode::random_rng(&mut self.rng);
                    self.skip_invalidate = true;
                    let template = E2ETemplate {
                        left: template.clone(),
                        right: template,
                    };
                    self.new_templates_in_batch.push((
                        internal_batch_idx,
                        request_id.to_string(),
                        template.clone(),
                    ));
                    template
                }
                TestCase::NonMatchSkipPersistence => {
                    tracing::info!("Sending new iris code with skip persistence");
                    skip_persistence = true;
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_skip_persistence(true)
                            .build(),
                    );
                    let template = IrisCode::random_rng(&mut self.rng);
                    E2ETemplate {
                        left: template.clone(),
                        right: template.clone(),
                    }
                }
                TestCase::Match => {
                    tracing::info!("Sending iris code from db");
                    let (db_index, [template_left, template_right]) =
                        self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_db_index(db_index as u32)
                            .build(),
                    );
                    E2ETemplate {
                        left: template_left,
                        right: template_right,
                    }
                }
                TestCase::MatchSkipPersistence => {
                    tracing::info!("Sending iris code from db with skip persistence");
                    let (db_index, [template_left, template_right]) =
                        self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    skip_persistence = true;
                    self.disallowed_queries.push(db_index as u32);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_db_index(db_index as u32)
                            .with_skip_persistence(true)
                            .build(),
                    );
                    E2ETemplate {
                        left: template_left,
                        right: template_right,
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
                            ExpectedResult::builder().build()
                        } else {
                            // we flip less or equal to than the threshold so this should
                            // match
                            self.disallowed_queries.push(db_index as u32);
                            ExpectedResult::builder()
                                .with_db_index(db_index as u32)
                                .build()
                        },
                    );
                    for dir in [LEFT, RIGHT] {
                        assert_eq!(template[dir].mask, IrisCodeArray::ONES);
                        for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                            template[dir].code.flip_bit(i);
                        }
                    }
                    let [template_left, template_right] = template;
                    E2ETemplate {
                        left: template_left,
                        right: template_right,
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
                        ExpectedResult::builder().with_db_index(*idx).build(),
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
                    self.expected_results
                        .insert(request_id.to_string(), ExpectedResult::builder().build());
                    E2ETemplate {
                        left: self.initial_db_state.plain_dbs[LEFT].db[deleted_idx as usize]
                            .clone(),
                        right: self.initial_db_state.plain_dbs[RIGHT].db[deleted_idx as usize]
                            .clone(),
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
                    self.disallowed_queries.push(matching_db_index as u32);
                    // comparison against this item will use the OR rule
                    or_rule_indices = db_indexes_copy.iter().map(|&x| x as u32).collect();

                    // apply variation to either right or left code
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
                            ExpectedResult::builder()
                                .with_db_index(matching_db_index as u32)
                                .with_batch_match(false)
                                .build(),
                        );
                    } else {
                        self.db_indices_used_in_current_batch
                            .insert(matching_db_index);
                        self.disallowed_queries.push(matching_db_index as u32);
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult::builder().build());
                    }
                    template
                }
                TestCase::ReauthMatchingTarget => {
                    tracing::info!(
                        "Sending reauth request with AND rule matching the target index"
                    );
                    message_type = REAUTH_MESSAGE_TYPE.to_string();
                    let (db_index, [template_left, template_right]) =
                        self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_reauth_successful(true)
                            .build(),
                    );
                    E2ETemplate {
                        left: template_left,
                        right: template_right,
                    }
                }
                TestCase::ReauthMatchingTargetWithSkipPeristence => {
                    tracing::info!(
                        "Sending reauth request with AND rule matching the target index with skip persistence"
                    );
                    message_type = REAUTH_MESSAGE_TYPE.to_string();
                    skip_persistence = true;
                    let (db_index, _) = self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.disallowed_queries.push(db_index as u32);
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);
                    let (reauth_template, probe_template) =
                        self.prepare_disjoint_matching_codes(db_index);
                    self.reauth_skip_persistence_templates
                        .insert(db_index as u32, probe_template);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_reauth_successful(true)
                            .build(),
                    );
                    self.skip_invalidate = true;
                    reauth_template
                }
                TestCase::ReauthNonMatchingTarget => {
                    tracing::info!(
                        "Sending reauth request with AND rule non-matching the target index"
                    );
                    message_type = REAUTH_MESSAGE_TYPE.to_string();
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
                        ExpectedResult::builder()
                            .with_reauth_successful(false)
                            .build(),
                    );
                    template
                }
                TestCase::ReauthOrRuleMatchingTarget => {
                    tracing::info!("Sending reauth request with OR rule matching the target index");
                    message_type = REAUTH_MESSAGE_TYPE.to_string();
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
                        ExpectedResult::builder()
                            .with_reauth_successful(true)
                            .build(),
                    );
                    template
                }
                TestCase::ReauthOrRuleNonMatchingTarget => {
                    tracing::info!(
                        "Sending reauth request with OR rule non-matching the target index"
                    );
                    message_type = REAUTH_MESSAGE_TYPE.to_string();
                    let (db_index, _) = self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.reauth_target_indices
                        .insert(request_id.to_string(), db_index as u32);
                    or_rule_indices = vec![db_index as u32];
                    let will_match = false;
                    let template = self.prepare_flipped_codes(db_index, will_match, None);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_reauth_successful(false)
                            .build(),
                    );
                    template
                }
                TestCase::ResetCheckMatch => {
                    tracing::info!("Sending reset check request with an existing iris code");
                    let (db_index, [template_left, template_right]) =
                        self.get_iris_code_in_db(DatabaseRange::Full);
                    self.db_indices_used_in_current_batch.insert(db_index);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_db_index(db_index as u32)
                            .with_reset_check(true)
                            .build(),
                    );
                    message_type = RESET_CHECK_MESSAGE_TYPE.to_string();
                    E2ETemplate {
                        left: template_left,
                        right: template_right,
                    }
                }
                TestCase::ResetCheckNonMatch => {
                    tracing::info!("Sending reset check request with fresh iris code");
                    let template = IrisCode::random_rng(&mut self.rng);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder().with_reset_check(true).build(),
                    );
                    message_type = RESET_CHECK_MESSAGE_TYPE.to_string();
                    E2ETemplate {
                        left: template.clone(),
                        right: template,
                    }
                }
                TestCase::EnrollmentAfterResetCheckNonMatch => {
                    tracing::info!("Sending enrollment request using iris codes used during reset check non match");
                    let req_id = self
                        .non_match_reset_check_templates
                        .keys()
                        .choose(&mut self.rng)
                        .unwrap()
                        .clone();
                    let e2e_template = self
                        .non_match_reset_check_templates
                        .get(&req_id)
                        .unwrap()
                        .clone();
                    self.non_match_reset_check_templates.remove(&req_id);
                    self.expected_results
                        .insert(request_id.to_string(), ExpectedResult::builder().build());
                    self.skip_invalidate = true;
                    e2e_template
                }
                TestCase::FullFaceMirrorAttack => {
                    tracing::info!("Sending iris code crafted for mirror attack detection");

                    // Get an existing template from the database
                    let (db_index, original_template) =
                        self.get_iris_code_in_db(DatabaseRange::Full);
                    tracing::info!("db_index used for the mirror attack: {}", db_index);
                    self.db_indices_used_in_current_batch.insert(db_index);

                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder()
                            .with_full_face_mirror_attack(true)
                            .build(),
                    );

                    // send a mirrored template as our test case
                    // this will ensure that the original template will be mirrored
                    E2ETemplate {
                        // This is swapped on purpose due to the mirror attack flow
                        left: original_template[RIGHT].mirrored(),
                        right: original_template[LEFT].mirrored(),
                    }
                }
                TestCase::MatchAfterResetUpdate => {
                    tracing::info!(
                        "Sending enrollment request using iris codes used during reset update"
                    );
                    let db_idx = *self
                        .reset_update_templates
                        .keys()
                        .choose(&mut self.rng)
                        .unwrap();
                    let e2e_template = self.reset_update_templates.get(&db_idx).unwrap().clone();
                    self.reset_update_templates.remove(&db_idx);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder().with_db_index(db_idx).build(),
                    );
                    self.skip_invalidate = true;
                    e2e_template
                }
                TestCase::MatchAfterReauthSkipPeristence => {
                    tracing::info!(
                        "Sending enrollment request using iris codes used during reauth skip persistence"
                    );
                    let db_idx = *self
                        .reauth_skip_persistence_templates
                        .keys()
                        .choose(&mut self.rng)
                        .unwrap();
                    let e2e_template = self
                        .reauth_skip_persistence_templates
                        .get(&db_idx)
                        .unwrap()
                        .clone();
                    self.reauth_skip_persistence_templates.remove(&db_idx);
                    self.disallowed_queries.retain(|&idx| idx != db_idx);
                    self.db_indices_used_in_current_batch.insert(db_idx as usize);
                    self.expected_results.insert(
                        request_id.to_string(),
                        ExpectedResult::builder().with_db_index(db_idx).build(),
                    );
                    self.skip_invalidate = true;
                    e2e_template
                }
            }
        };
        (
            request_id,
            e2e_template,
            or_rule_indices,
            skip_persistence,
            message_type,
        )
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
        let mut code_left = self.initial_db_state.plain_dbs[LEFT].db[db_index].clone();
        let mut code_right = self.initial_db_state.plain_dbs[RIGHT].db[db_index].clone();

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

    /// Returns two templates that both match the original DB entry, but not each other.
    /// This is used to detect unintended reauth updates when skip_persistence is set.
    fn prepare_disjoint_matching_codes(
        &mut self,
        db_index: usize,
    ) -> (E2ETemplate, E2ETemplate) {
        let original_left = self.initial_db_state.plain_dbs[LEFT].db[db_index].clone();
        let original_right = self.initial_db_state.plain_dbs[RIGHT].db[db_index].clone();

        assert_eq!(original_left.mask, IrisCodeArray::ONES);
        assert_eq!(original_right.mask, IrisCodeArray::ONES);

        let mut reauth_left = original_left.clone();
        let mut reauth_right = original_right.clone();
        let mut probe_left = original_left;
        let mut probe_right = original_right;

        let flip_count = (THRESHOLD_ABSOLUTE / 2) + 1;
        assert!(
            flip_count * 2 < IRIS_CODE_LENGTH,
            "Not enough bits to generate disjoint flip sets"
        );

        for i in 0..flip_count {
            reauth_left.code.flip_bit(i);
            reauth_right.code.flip_bit(i);
        }

        for i in flip_count..(flip_count * 2) {
            probe_left.code.flip_bit(i);
            probe_right.code.flip_bit(i);
        }

        (
            E2ETemplate {
                left: reauth_left,
                right: reauth_right,
            },
            E2ETemplate {
                left: probe_left,
                right: probe_right,
            },
        )
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
        full_face_mirror_attack_detected: bool,
    ) {
        tracing::info!(
            "Checking result for request_id: {}, idx: {}, was_match: {}, matched_batch_req_ids: \
             {:?}, was_reauth_success: {}, was_skip_persistence_match: {}, full_face_mirror_attack: {}",
            req_id,
            idx,
            was_match,
            matched_batch_req_ids,
            was_reauth_success,
            was_skip_persistence_match,
            full_face_mirror_attack_detected
        );
        let &ExpectedResult {
            db_index: expected_idx,
            is_batch_match,
            is_reauth_successful,
            is_skip_persistence_request,
            is_reset_check,
            is_full_face_mirror_attack,
        } = self
            .expected_results
            .get(req_id)
            .expect("request id not found");

        if is_full_face_mirror_attack {
            assert!(was_match);
            assert!(full_face_mirror_attack_detected);
            return;
        }

        if is_reset_check {
            // assert that the reset_check requests are not reported as unique. match fields are only used for enrollment requests
            assert!(was_match);
            assert!(was_skip_persistence_match);
            assert!(!was_reauth_success);
            assert!(!full_face_mirror_attack_detected);

            // assert that we report correct matched indices upon reset_check requests
            if expected_idx.is_some() {
                assert_eq!(idx, expected_idx.unwrap());
            } else {
                assert_eq!(idx, u32::MAX);

                // insert the template to enable testing EnrollmentAfterResetCheckNonMatch case
                tracing::info!("Inserting non_match_reset_check_templates {}", req_id);
                self.non_match_reset_check_templates
                    .insert(req_id.to_string(), requests.get(req_id).unwrap().clone());
            }
            return;
        }

        // if the request is a reauth, we only check the reauth success
        if let Some(is_reauth_successful) = is_reauth_successful {
            assert!(!full_face_mirror_attack_detected);
            assert_eq!(
                is_reauth_successful, was_reauth_success,
                "expected reauth success status to be as expected"
            );
            return;
        }

        if let Some(expected_idx) = expected_idx {
            assert!(!full_face_mirror_attack_detected);
            assert!(
                was_match,
                "expected this request to be a match, but it was not"
            );
            assert!(was_skip_persistence_match);
            if !is_batch_match {
                assert_eq!(
                    expected_idx, idx,
                    "expected matched index to be as expected"
                );
            } else {
                assert!(
                    self.batch_duplicates.contains_key(req_id),
                    "expected this request to be a batch duplicate"
                );
                assert!(
                    matched_batch_req_ids.contains(self.batch_duplicates.get(req_id).unwrap()),
                    "expected the batch match index to be in the batch duplicates"
                );
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
        max_reset_updates_per_batch: usize,
        handles: [&mut impl JobSubmissionHandle; 3],
    ) -> Result<()> {
        let [handle0, handle1, handle2] = handles;
        for _ in 0..num_batches {
            // Skip empty batch
            let ([batch0, batch1, batch2], requests) = self.generate_query_batch(
                max_batch_size,
                max_deletions_per_batch,
                max_reset_updates_per_batch,
            )?;
            if batch0.request_ids.is_empty() {
                continue;
            }

            // send batches to servers
            let (res0_fut, res1_fut, res2_fut) = tokio::join!(
                handle0.submit_batch_query(batch0),
                handle1.submit_batch_query(batch1),
                handle2.submit_batch_query(batch2)
            );

            let res0 = res0_fut.await?;
            let res1 = res1_fut.await?;
            let res2 = res2_fut.await?;

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
                    matched_batch_request_ids,
                    successful_reauths,
                    reset_update_indices,
                    reset_update_request_ids,
                    full_face_mirror_attack_detected,
                    ..
                } = res;

                for (
                    req_id,
                    &was_match,
                    &was_skip_persistence_match,
                    &was_reauth_success,
                    &idx,
                    matched_batch_req_ids,
                    &full_face_mirror_attack_detected,
                ) in izip!(
                    thread_request_ids,
                    matches,
                    matches_with_skip_persistence,
                    successful_reauths,
                    merged_results,
                    matched_batch_request_ids,
                    full_face_mirror_attack_detected,
                ) {
                    assert!(requests.contains_key(req_id));

                    resp_counters.insert(req_id, resp_counters.get(req_id).unwrap() + 1);

                    self.check_result(
                        req_id,
                        idx,
                        was_match,
                        was_skip_persistence_match,
                        matched_batch_req_ids,
                        &requests,
                        was_reauth_success,
                        full_face_mirror_attack_detected,
                    );
                }

                for (req_id, idx) in izip!(reset_update_request_ids, reset_update_indices) {
                    assert!(requests.contains_key(req_id));
                    assert!(self.reset_update_templates.contains_key(idx));
                    resp_counters.insert(req_id, resp_counters.get(req_id).unwrap() + 1);
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
    message_type: String,
) -> Result<()> {
    batch.valid_entries.push(is_valid);

    if message_type == REAUTH_MESSAGE_TYPE {
        let target_index = maybe_reauth_target_index.unwrap();
        batch
            .reauth_use_or_rule
            .insert(request_id.clone(), !or_rule_indices.is_empty());
        batch
            .reauth_target_indices
            .insert(request_id.clone(), *target_index);
    }

    batch.push_matching_request(
        "sns_id".to_string(),
        request_id.clone(),
        &message_type,
        BatchMetadata::default(),
        or_rule_indices,
        skip_persistence,
    );

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
        .extend(e2e_shared_template.right_shared_code[batch_idx].all_rotations());
    batch
        .right_iris_rotated_requests
        .mask
        .extend(e2e_shared_template.right_shared_mask[batch_idx].all_rotations());

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

    // Also preprocess the mirrored codes and masks
    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(
        &mut e2e_shared_template.left_mirrored_shared_code[batch_idx],
    );
    GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(
        &mut e2e_shared_template.left_mirrored_shared_mask[batch_idx],
    );
    GaloisRingIrisCodeShare::preprocess_iris_code_query_share(
        &mut e2e_shared_template.right_mirrored_shared_code[batch_idx],
    );
    GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(
        &mut e2e_shared_template.right_mirrored_shared_mask[batch_idx],
    );

    // Add the mirrored iris data
    batch
        .left_mirrored_iris_interpolated_requests
        .code
        .extend(e2e_shared_template.left_mirrored_shared_code[batch_idx].all_rotations());
    batch
        .left_mirrored_iris_interpolated_requests
        .mask
        .extend(e2e_shared_template.left_mirrored_shared_mask[batch_idx].all_rotations());
    batch
        .right_mirrored_iris_interpolated_requests
        .code
        .extend(e2e_shared_template.right_mirrored_shared_code[batch_idx].all_rotations());
    batch
        .right_mirrored_iris_interpolated_requests
        .mask
        .extend(e2e_shared_template.right_mirrored_shared_mask[batch_idx].all_rotations());

    Ok(())
}

pub struct PartyDb {
    pub party_id: usize,
    pub db_left: Vec<FullGaloisRingIrisCodeShare>,
    pub db_right: Vec<FullGaloisRingIrisCodeShare>,
}

pub struct TestDb {
    /// A plain iris database for LEFT & RIGHT eyes
    plain_dbs: [IrisDB; 2],
    /// A shared iris database for each parties
    shared_dbs: [Arc<PartyDb>; 3],
    /// initial db len
    initial_db_len: usize,
}

impl TestDb {
    pub fn party_db(&self, party_id: usize) -> Arc<PartyDb> {
        Arc::clone(&self.shared_dbs[party_id])
    }
    pub fn plain_dbs(&self, side: usize) -> &IrisDB {
        &self.plain_dbs[side]
    }
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        // sanity check to ensure that the databases are of the same size
        assert!(self.plain_dbs[0].len() == self.plain_dbs[1].len());
        self.plain_dbs[0].len()
    }
    pub fn initial_db_len(&self) -> usize {
        self.initial_db_len
    }
}

pub fn generate_full_test_db(db_size: usize, db_rng_seed: u64, with_pattern: bool) -> TestDb {
    let mut rng = StdRng::seed_from_u64(db_rng_seed);
    let (mut db_left, mut db_right) = if with_pattern {
        (
            IrisDB::new_random_par_with_pattern(db_size, &mut rng),
            IrisDB::new_random_par_with_pattern(db_size, &mut rng),
        )
    } else {
        (
            IrisDB::new_random_par(db_size, &mut rng),
            IrisDB::new_random_par(db_size, &mut rng),
        )
    };

    // Set the masks to all 1s for the first 10%
    for i in 0..db_size / 10 {
        db_left.db[i].mask = IrisCodeArray::ONES;
        db_right.db[i].mask = IrisCodeArray::ONES;
    }

    let mut share_rng = StdRng::from_rng(rng).unwrap();

    let mut party_dbs = [
        PartyDb {
            party_id: 0,
            db_left: Vec::with_capacity(db_size),
            db_right: Vec::with_capacity(db_size),
        },
        PartyDb {
            party_id: 1,
            db_left: Vec::with_capacity(db_size),
            db_right: Vec::with_capacity(db_size),
        },
        PartyDb {
            party_id: 2,
            db_left: Vec::with_capacity(db_size),
            db_right: Vec::with_capacity(db_size),
        },
    ];

    for (left_iris, right_iris) in db_left.db.iter().zip(db_right.db.iter()) {
        let [left0, left1, left2] =
            FullGaloisRingIrisCodeShare::encode_iris_code(left_iris, &mut share_rng);
        let [right0, right1, right2] =
            FullGaloisRingIrisCodeShare::encode_iris_code(right_iris, &mut share_rng);
        party_dbs[0].db_left.push(left0);
        party_dbs[0].db_right.push(right0);
        party_dbs[1].db_left.push(left1);
        party_dbs[1].db_right.push(right1);
        party_dbs[2].db_left.push(left2);
        party_dbs[2].db_right.push(right2);
    }

    TestDb {
        plain_dbs: [db_left, db_right],
        shared_dbs: party_dbs.map(Arc::new),
        initial_db_len: db_size,
    }
}

pub fn load_test_db(party_db: &PartyDb, loader: &mut impl InMemoryStore) {
    for (idx, (left, right)) in party_db
        .db_left
        .iter()
        .zip(party_db.db_right.iter())
        .enumerate()
    {
        loader.load_single_record_from_db(
            idx,
            VectorId::from_0_index(idx as u32),
            &left.code.coefs,
            &left.mask.coefs,
            &right.code.coefs,
            &right.mask.coefs,
        );
        loader.increment_db_size(idx);
    }
}

// NOTE: SimpleAnonStatsTestGenerator and bucket-statistics assertions were used to validate the
// (now-deprecated) in-pipeline anonymized bucket computation. Bucket computation is owned by the
// anon-stats-server, so these helpers have been removed.
