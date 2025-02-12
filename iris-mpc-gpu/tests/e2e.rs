#[cfg(feature = "gpu_dependent")]
mod e2e_test {
    use cudarc::nccl::Id;
    use eyre::Result;
    use iris_mpc_common::{
        galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
        helpers::{
            smpc_request::{REAUTH_MESSAGE_TYPE, UNIQUENESS_MESSAGE_TYPE},
            statistics::BucketStatistics,
        },
        iris_db::{
            db::IrisDB,
            iris::{IrisCode, IrisCodeArray},
        },
        job::{BatchQuery, JobSubmissionHandle, ServerJobResult},
    };
    use iris_mpc_gpu::{helpers::device_manager::DeviceManager, server::ServerActor};
    use itertools::izip;
    use rand::{
        rngs::StdRng,
        seq::{IteratorRandom, SliceRandom},
        Rng, SeedableRng,
    };
    use std::{
        collections::{HashMap, HashSet},
        env,
        sync::Arc,
    };
    use tokio::sync::oneshot;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    use uuid::Uuid;

    const DB_SIZE: usize = 8 * 1000;
    const DB_BUFFER: usize = 8 * 1000;
    const DB_RNG_SEED: u64 = 0xdeadbeef;
    const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
    const NUM_BATCHES: usize = 30;
    const MAX_BATCH_SIZE: usize = 64;
    const N_BUCKETS: usize = 10;
    const MATCH_DISTANCES_BUFFER_SIZE: usize = 1 << 7;
    const MATCH_DISTANCES_BUFFER_SIZE_EXTRA_PERCENT: usize = 100;
    const MAX_DELETIONS_PER_BATCH: usize = 10;
    const THRESHOLD_ABSOLUTE: usize = 4800; // 0.375 * 12800

    #[derive(Clone)]
    pub struct E2ETemplate {
        left:  IrisCode,
        right: IrisCode,
    }

    type OrRuleSerialIds = Vec<u32>;

    #[derive(Clone)]
    pub struct E2ESharedTemplate {
        pub left_shared_code:  [GaloisRingIrisCodeShare; 3],
        pub left_shared_mask:  [GaloisRingTrimmedMaskCodeShare; 3],
        pub right_shared_code: [GaloisRingIrisCodeShare; 3],
        pub right_shared_mask: [GaloisRingTrimmedMaskCodeShare; 3],
    }

    fn generate_db(party_id: usize) -> Result<(Vec<u16>, Vec<u16>)> {
        let mut rng = StdRng::seed_from_u64(DB_RNG_SEED);
        let mut db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        // Set the masks to all 1s for the first 10%
        for i in 0..DB_SIZE / 10 {
            db.db[i].mask = IrisCodeArray::ONES;
        }

        let codes_db = db
            .db
            .iter()
            .flat_map(|iris| {
                GaloisRingIrisCodeShare::encode_iris_code(
                    &iris.code,
                    &iris.mask,
                    &mut StdRng::seed_from_u64(DB_RNG_SEED),
                )[party_id]
                    .coefs
            })
            .collect::<Vec<_>>();

        let masks_db = db
            .db
            .iter()
            .flat_map(|iris| {
                let mask: GaloisRingTrimmedMaskCodeShare =
                    GaloisRingIrisCodeShare::encode_mask_code(
                        &iris.mask,
                        &mut StdRng::seed_from_u64(DB_RNG_SEED),
                    )[party_id]
                        .clone()
                        .into();
                mask.coefs
            })
            .collect::<Vec<_>>();

        Ok((codes_db, masks_db))
    }

    fn install_tracing() {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    #[tokio::test]
    async fn e2e_test() -> Result<()> {
        install_tracing();
        env::set_var("NCCL_P2P_LEVEL", "LOC");
        env::set_var("NCCL_NET", "Socket");

        let db0: (Vec<u16>, Vec<u16>) = generate_db(0)?;
        let db1 = generate_db(1)?;
        let db2 = generate_db(2)?;

        let chacha_seeds0 = ([0u32; 8], [2u32; 8]);
        let chacha_seeds1 = ([1u32; 8], [0u32; 8]);
        let chacha_seeds2 = ([2u32; 8], [1u32; 8]);

        // a bit convoluted, but we need to create the actor on the thread already,
        // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
        // channel
        let (tx0, rx0) = oneshot::channel();
        let (tx1, rx1) = oneshot::channel();
        let (tx2, rx2) = oneshot::channel();

        let device_manager = DeviceManager::init();
        let mut device_managers = device_manager
            .split_into_n_chunks(3)
            .expect("have at least 3 devices");
        let device_manager2 = Arc::new(device_managers.pop().unwrap());
        let device_manager1 = Arc::new(device_managers.pop().unwrap());
        let device_manager0 = Arc::new(device_managers.pop().unwrap());
        let num_devices = device_manager0.devices().len();
        let ids0 = (0..num_devices)
            .map(|_| Id::new().unwrap())
            .collect::<Vec<_>>();
        let ids1 = ids0.clone();
        let ids2 = ids0.clone();

        let actor0_task = tokio::task::spawn_blocking(move || {
            let comms0 = device_manager0
                .instantiate_network_from_ids(0, &ids0)
                .unwrap();
            let actor = match ServerActor::new_with_device_manager_and_comms(
                0,
                chacha_seeds0,
                device_manager0,
                comms0,
                8,
                DB_SIZE + DB_BUFFER,
                MAX_BATCH_SIZE,
                MATCH_DISTANCES_BUFFER_SIZE,
                MATCH_DISTANCES_BUFFER_SIZE_EXTRA_PERCENT,
                N_BUCKETS,
                true,
                false,
                false,
            ) {
                Ok((mut actor, handle)) => {
                    actor.load_full_db(&(&db0.0, &db0.1), &(&db0.0, &db0.1), DB_SIZE);
                    actor.register_host_memory();
                    tx0.send(Ok(handle)).unwrap();
                    actor
                }
                Err(e) => {
                    tx0.send(Err(e)).unwrap();
                    return;
                }
            };
            actor.run();
        });
        let actor1_task = tokio::task::spawn_blocking(move || {
            let comms1 = device_manager1
                .instantiate_network_from_ids(1, &ids1)
                .unwrap();
            let actor = match ServerActor::new_with_device_manager_and_comms(
                1,
                chacha_seeds1,
                device_manager1,
                comms1,
                8,
                DB_SIZE + DB_BUFFER,
                MAX_BATCH_SIZE,
                MATCH_DISTANCES_BUFFER_SIZE,
                MATCH_DISTANCES_BUFFER_SIZE_EXTRA_PERCENT,
                N_BUCKETS,
                true,
                false,
                false,
            ) {
                Ok((mut actor, handle)) => {
                    actor.load_full_db(&(&db1.0, &db1.1), &(&db1.0, &db1.1), DB_SIZE);
                    actor.register_host_memory();
                    tx1.send(Ok(handle)).unwrap();
                    actor
                }
                Err(e) => {
                    tx1.send(Err(e)).unwrap();
                    return;
                }
            };
            actor.run();
        });
        let actor2_task = tokio::task::spawn_blocking(move || {
            let comms2 = device_manager2
                .instantiate_network_from_ids(2, &ids2)
                .unwrap();
            let actor = match ServerActor::new_with_device_manager_and_comms(
                2,
                chacha_seeds2,
                device_manager2,
                comms2,
                8,
                DB_SIZE + DB_BUFFER,
                MAX_BATCH_SIZE,
                MATCH_DISTANCES_BUFFER_SIZE,
                MATCH_DISTANCES_BUFFER_SIZE_EXTRA_PERCENT,
                N_BUCKETS,
                true,
                false,
                false,
            ) {
                Ok((mut actor, handle)) => {
                    actor.load_full_db(&(&db2.0, &db2.1), &(&db2.0, &db2.1), DB_SIZE);
                    actor.register_host_memory();
                    tx2.send(Ok(handle)).unwrap();
                    actor
                }
                Err(e) => {
                    tx2.send(Err(e)).unwrap();
                    return;
                }
            };
            actor.run();
        });
        let mut handle0 = rx0.await??;
        let mut handle1 = rx1.await??;
        let mut handle2 = rx2.await??;

        // create a copy of the plain database for the test case generator, this needs
        // to be in sync with `generate_db`
        let mut db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(DB_RNG_SEED));
        // Set the masks to all 1s for the first 10%
        for i in 0..DB_SIZE / 10 {
            db.db[i].mask = IrisCodeArray::ONES;
        }
        let rng = StdRng::seed_from_u64(INTERNAL_RNG_SEED);
        let mut test_case_generator = TestCaseGenerator::new(db, rng);

        for _ in 0..NUM_BATCHES {
            // Skip empty batch
            let ([batch0, batch1, batch2], requests) =
                test_case_generator.generate_query_batch()?;
            if batch0.request_ids.is_empty() {
                continue;
            }

            // send batches to servers
            let res0_fut = handle0.submit_batch_query(batch0).await;
            let res1_fut = handle1.submit_batch_query(batch1).await;
            let res2_fut = handle2.submit_batch_query(batch2).await;

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

                check_bucket_statistics(anonymized_bucket_statistics_left, num_devices)?;
                check_bucket_statistics(anonymized_bucket_statistics_right, num_devices)?;

                for (
                    req_id,
                    &was_match,
                    &was_reauth_success,
                    &idx,
                    partial_left,
                    partial_right,
                    match_id,
                    matched_batch_req_ids,
                ) in izip!(
                    thread_request_ids,
                    matches,
                    successful_reauths,
                    merged_results,
                    partial_match_ids_left,
                    partial_match_ids_right,
                    match_ids,
                    matched_batch_request_ids
                ) {
                    assert!(requests.contains_key(req_id));

                    resp_counters.insert(req_id, resp_counters.get(req_id).unwrap() + 1);

                    if !test_case_generator.or_rule_matches.contains(req_id) {
                        assert_eq!(partial_left, partial_right);
                        assert_eq!(partial_left, match_id);
                    }
                    test_case_generator.check_result(
                        req_id,
                        idx,
                        was_match,
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

        drop(handle0);
        drop(handle1);
        drop(handle2);

        actor0_task.await.unwrap();
        actor1_task.await.unwrap();
        actor2_task.await.unwrap();

        Ok(())
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
    ) -> Result<()> {
        batch.metadata.push(Default::default());
        batch.valid_entries.push(is_valid);
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
            .store_left
            .code
            .push(e2e_shared_template.left_shared_code[batch_idx].clone());
        batch
            .store_left
            .mask
            .push(e2e_shared_template.left_shared_mask[batch_idx].clone());
        batch
            .store_right
            .code
            .push(e2e_shared_template.right_shared_code[batch_idx].clone());
        batch
            .store_right
            .mask
            .push(e2e_shared_template.right_shared_mask[batch_idx].clone());

        batch
            .db_left
            .code
            .extend(e2e_shared_template.left_shared_code[batch_idx].all_rotations());
        batch
            .db_left
            .mask
            .extend(e2e_shared_template.left_shared_mask[batch_idx].all_rotations());

        batch
            .db_right
            .code
            .extend(e2e_shared_template.left_shared_code[batch_idx].all_rotations());
        batch
            .db_right
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
            .query_left
            .code
            .extend(e2e_shared_template.left_shared_code[batch_idx].all_rotations());
        batch
            .query_left
            .mask
            .extend(e2e_shared_template.left_shared_mask[batch_idx].all_rotations());

        batch
            .query_right
            .code
            .extend(e2e_shared_template.right_shared_code[batch_idx].all_rotations());
        batch
            .query_right
            .mask
            .extend(e2e_shared_template.right_shared_mask[batch_idx].all_rotations());

        Ok(())
    }

    fn check_bucket_statistics(
        bucket_statistics: &BucketStatistics,
        num_gpus_per_party: usize,
    ) -> Result<()> {
        if bucket_statistics.is_empty() {
            assert_eq!(bucket_statistics.buckets.len(), 0);
            return Ok(());
        }
        assert_eq!(bucket_statistics.buckets.len(), N_BUCKETS);
        assert!(
            bucket_statistics.end_time_utc_timestamp
                > Some(bucket_statistics.start_time_utc_timestamp)
        );
        let total_count = bucket_statistics
            .buckets
            .iter()
            .map(|b| b.count)
            .sum::<usize>();
        tracing::info!("Total count for bucket: {}", total_count);
        assert_eq!(
            total_count,
            MATCH_DISTANCES_BUFFER_SIZE * num_gpus_per_party
        );
        Ok(())
    }

    fn to_shared_template(
        is_valid: bool,
        template: &E2ETemplate,
        rng: &mut StdRng,
    ) -> E2ESharedTemplate {
        let (left_shared_code, left_shared_mask) =
            get_shared_template(is_valid, &template.left, rng);
        let (right_shared_code, right_shared_mask) =
            get_shared_template(is_valid, &template.right, rng);
        E2ESharedTemplate {
            left_shared_code,
            left_shared_mask,
            right_shared_code,
            right_shared_mask,
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestCases {
        /// Send an iris code known to be in the database
        Match,
        /// Send an iris code that known not to match any in the database, it
        /// will be inserted
        NonMatch,
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum DatabaseRange {
        /// Use the full database range
        Full,
        /// Use only the first 10% of the database range, which has masks set to
        /// all 1. This is useful for testing values close to the threshold,
        /// since we can rely on bitflips always affecting distance.
        FullMaskOnly,
    }

    struct ExpectedResult {
        /// The returned index of the iris code in the database.
        /// It is None if the iris code is not in the database, and Some(idx) if
        /// there is a match at index idx
        db_index:             Option<u32>,
        /// Whether the iris code is expected to be in the batch match
        /// This flag indicates that the iris code is expected to match another
        /// iris code in the current batch
        is_batch_match:       bool,
        /// Populated only if the request type is REAUTH.
        /// Indicates whether the expected reauth result is successful.
        is_reauth_successful: Option<bool>,
    }

    struct TestCaseGenerator {
        /// initial state of the Iris Database
        initial_db_state:       IrisDB,
        /// expected results for all of the queries we send
        expected_results:       HashMap<String, ExpectedResult>,
        /// mapping from request_id to the index of the target entry to be
        /// matched in reauth
        reauth_target_indices:  HashMap<String, u32>,
        /// responses received from the servers, where a new iris code was
        /// inserted. Maps position in the database to the E2ETemplate
        inserted_responses:     HashMap<u32, E2ETemplate>,
        /// A buffer of indices that have been deleted, to choose a index from
        /// to send for testing against deletions. Once picked, it is removed
        /// from here
        deleted_indices_buffer: Vec<u32>,
        /// The full set of indices that have been deleted
        deleted_indices:        HashSet<u32>,
        /// A list of indices that are not allowed to be queried, to avoid
        /// potential false matches
        disallowed_queries:     Vec<u32>,
        /// The rng that is used internally
        rng:                    StdRng,

        // info for current batch, will be cleared at the start of a new batch
        /// New templates that have been inserted in the current batch.
        /// (position in batch, request_id, template)
        new_templates_in_batch:           Vec<(usize, String, IrisCode)>,
        /// skip invalidating requests in the current batch, since we expect
        /// them to be processed
        skip_invalidate:                  bool,
        /// duplicates in the current batch, used to test the batch
        /// deduplication mechanism
        batch_duplicates:                 HashMap<String, String>,
        /// indices used in the current batch, to avoid deleting those
        db_indices_used_in_current_batch: HashSet<usize>,
        /// items against which the OR rule is used
        or_rule_matches:                  Vec<String>,
    }

    impl TestCaseGenerator {
        fn new(db: IrisDB, rng: StdRng) -> Self {
            Self {
                initial_db_state: db,
                expected_results: HashMap::new(),
                reauth_target_indices: HashMap::new(),
                inserted_responses: HashMap::new(),
                deleted_indices_buffer: Vec::new(),
                deleted_indices: HashSet::new(),
                disallowed_queries: Vec::new(),
                rng,
                new_templates_in_batch: Vec::new(),
                skip_invalidate: false,
                batch_duplicates: HashMap::new(),
                db_indices_used_in_current_batch: HashSet::new(),
                or_rule_matches: Vec::new(),
            }
        }

        fn generate_query_batch(
            &mut self,
        ) -> Result<([BatchQuery; 3], HashMap<String, E2ETemplate>)> {
            let mut requests: HashMap<String, E2ETemplate> = HashMap::new();
            let mut batch0 = BatchQuery::default();
            let mut batch1 = BatchQuery::default();
            let mut batch2 = BatchQuery::default();
            let batch_size = self.rng.gen_range(1..MAX_BATCH_SIZE);

            self.batch_duplicates.clear();
            self.skip_invalidate = false;
            self.new_templates_in_batch.clear();
            self.db_indices_used_in_current_batch.clear();
            self.or_rule_matches.clear();

            for idx in 0..batch_size {
                let (request_id, e2e_template, or_rule_indices) = self.generate_query(idx);

                // Invalidate 10% of the queries, but ignore the batch duplicates
                let is_valid = self.rng.gen_bool(0.10) || self.skip_invalidate;
                if is_valid {
                    requests.insert(request_id.to_string(), e2e_template.clone());
                }

                let maybe_reauth_target_index =
                    self.reauth_target_indices.get(&request_id.to_string());
                let shared_template = to_shared_template(is_valid, &e2e_template, &mut self.rng);

                prepare_batch(
                    &mut batch0,
                    is_valid,
                    request_id.to_string(),
                    0,
                    shared_template.clone(),
                    or_rule_indices.clone(),
                    maybe_reauth_target_index,
                )?;

                prepare_batch(
                    &mut batch1,
                    true,
                    request_id.to_string(),
                    1,
                    shared_template.clone(),
                    or_rule_indices.clone(),
                    maybe_reauth_target_index,
                )?;

                prepare_batch(
                    &mut batch2,
                    true,
                    request_id.to_string(),
                    2,
                    shared_template,
                    or_rule_indices.clone(),
                    maybe_reauth_target_index,
                )?;
            }

            // Skip empty batch
            if batch0.request_ids.is_empty() {
                return Ok(([batch0, batch1, batch2], requests));
            }

            // for non-empty batches also add some deletions
            for _ in 0..self.rng.gen_range(0..MAX_DELETIONS_PER_BATCH) {
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
            Ok(([batch0, batch1, batch2], requests))
        }

        /// Get an Iris code known to be in the database, and return it and its
        /// index. The `DatabaseRange` parameter is used to chose which portion
        /// of the DB the item is chosen from.
        fn get_iris_code_in_db(&mut self, db_range: DatabaseRange) -> (usize, IrisCode) {
            let mut db_index = None;
            let range = match db_range {
                DatabaseRange::FullMaskOnly => 0..DB_SIZE / 10,
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
        ) -> (Uuid, E2ETemplate, OrRuleSerialIds) {
            let request_id = Uuid::new_v4();
            // Automatic random tests
            let mut options = vec![
                TestCases::Match,
                TestCases::NonMatch,
                TestCases::CloseToThreshold,
                TestCases::WithOrRuleSet,
                TestCases::ReauthNonMatchingTarget,
                TestCases::ReauthMatchingTarget,
                TestCases::ReauthOrRuleNonMatchingTarget,
                TestCases::ReauthOrRuleMatchingTarget,
            ];
            if !self.inserted_responses.is_empty() {
                options.push(TestCases::PreviouslyInserted);
            }
            if !self.deleted_indices_buffer.is_empty() {
                options.push(TestCases::PreviouslyDeleted);
            };

            let mut or_rule_indices: Vec<u32> = Vec::new();

            // with a 10% chance we pick a template from the batch, to test the batch
            // deduplication mechanism
            let pick_from_batch = self.rng.gen_bool(0.10);
            let e2e_template = if pick_from_batch && !self.new_templates_in_batch.is_empty() {
                let random_idx = self.rng.gen_range(0..self.new_templates_in_batch.len());
                let (batch_idx, duplicate_request_id, template) =
                    self.new_templates_in_batch[random_idx].clone();
                self.expected_results
                    .insert(request_id.to_string(), ExpectedResult {
                        db_index:             Some(batch_idx as u32),
                        is_batch_match:       true,
                        is_reauth_successful: None,
                    });
                self.batch_duplicates
                    .insert(request_id.to_string(), duplicate_request_id);
                self.skip_invalidate = true;
                E2ETemplate {
                    left:  template.clone(),
                    right: template.clone(),
                }
            } else {
                // otherwise we pick from the valid test case options
                let option = options
                    .choose(&mut self.rng)
                    .expect("we have at least one testcase option");
                tracing::info!("Request {} has type {:?}", request_id, option);
                match &option {
                    TestCases::NonMatch => {
                        tracing::info!("Sending new iris code");
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             None,
                                is_batch_match:       false,
                                is_reauth_successful: None,
                            });
                        let template = IrisCode::random_rng(&mut self.rng);
                        self.new_templates_in_batch.push((
                            internal_batch_idx,
                            request_id.to_string(),
                            template.clone(),
                        ));
                        self.skip_invalidate = true;
                        E2ETemplate {
                            left:  template.clone(),
                            right: template.clone(),
                        }
                    }
                    TestCases::Match => {
                        tracing::info!("Sending iris code from db");
                        let (db_index, template) = self.get_iris_code_in_db(DatabaseRange::Full);
                        self.db_indices_used_in_current_batch.insert(db_index);
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             Some(db_index as u32),
                                is_batch_match:       false,
                                is_reauth_successful: None,
                            });
                        E2ETemplate {
                            left:  template.clone(),
                            right: template,
                        }
                    }
                    TestCases::CloseToThreshold => {
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
                                    db_index:             None,
                                    is_batch_match:       false,
                                    is_reauth_successful: None,
                                }
                            } else {
                                // we flip less or equal to than the threshold so this should
                                // match
                                ExpectedResult {
                                    db_index:             Some(db_index as u32),
                                    is_batch_match:       false,
                                    is_reauth_successful: None,
                                }
                            },
                        );
                        assert_eq!(template.mask, IrisCodeArray::ONES);
                        for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                            template.code.flip_bit(i);
                        }
                        E2ETemplate {
                            left:  template.clone(),
                            right: template,
                        }
                    }
                    TestCases::PreviouslyInserted => {
                        tracing::info!("Sending freshly inserted iris code");
                        let (idx, e2e_template) = self
                            .inserted_responses
                            .iter()
                            .choose(&mut self.rng)
                            .expect("we have at least one response");
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             Some(*idx),
                                is_batch_match:       false,
                                is_reauth_successful: None,
                            });
                        self.db_indices_used_in_current_batch.insert(*idx as usize);
                        E2ETemplate {
                            left:  e2e_template.left.clone(),
                            right: e2e_template.right.clone(),
                        }
                    }
                    TestCases::PreviouslyDeleted => {
                        tracing::info!("Sending deleted iris code");
                        let idx = self.rng.gen_range(0..self.deleted_indices_buffer.len());
                        let deleted_idx = self.deleted_indices_buffer[idx];

                        self.deleted_indices_buffer.remove(idx);
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             None,
                                is_batch_match:       false,
                                is_reauth_successful: None,
                            });
                        E2ETemplate {
                            right: self.initial_db_state.db[deleted_idx as usize].clone(),
                            left:  self.initial_db_state.db[deleted_idx as usize].clone(),
                        }
                    }
                    TestCases::WithOrRuleSet => {
                        tracing::info!(
                            "Sending iris codes that match on one side but not the other with the \
                             OR rule set"
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
                            self.expected_results
                                .insert(request_id.to_string(), ExpectedResult {
                                    db_index:             Some(matching_db_index as u32),
                                    is_batch_match:       false,
                                    is_reauth_successful: None,
                                });
                        } else {
                            self.db_indices_used_in_current_batch
                                .insert(matching_db_index);
                            self.disallowed_queries.push(matching_db_index as u32);
                            self.expected_results
                                .insert(request_id.to_string(), ExpectedResult {
                                    db_index:             None,
                                    is_batch_match:       false,
                                    is_reauth_successful: None,
                                });
                        }
                        template
                    }
                    TestCases::ReauthMatchingTarget => {
                        tracing::info!(
                            "Sending reauth request with AND rule matching the target index"
                        );
                        let (db_index, template) = self.get_iris_code_in_db(DatabaseRange::Full);
                        self.db_indices_used_in_current_batch.insert(db_index);
                        self.reauth_target_indices
                            .insert(request_id.to_string(), db_index as u32);
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             Some(db_index as u32),
                                is_batch_match:       false,
                                is_reauth_successful: Some(true),
                            });
                        E2ETemplate {
                            left:  template.clone(),
                            right: template,
                        }
                    }
                    TestCases::ReauthNonMatchingTarget => {
                        tracing::info!(
                            "Sending reauth request with AND rule non-matching the target index"
                        );
                        let (db_index, _) = self.get_iris_code_in_db(DatabaseRange::FullMaskOnly);
                        self.db_indices_used_in_current_batch.insert(db_index);
                        self.reauth_target_indices
                            .insert(request_id.to_string(), db_index as u32);

                        // prepare a template that matches only on one side
                        // it will end up with a failed reauth with AND rule
                        self.or_rule_matches.push(request_id.to_string());
                        let will_match = true;
                        let flip_right = Some(self.rng.gen());
                        let template = self.prepare_flipped_codes(db_index, will_match, flip_right);
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             None,
                                is_batch_match:       false,
                                is_reauth_successful: Some(false),
                            });
                        template
                    }
                    TestCases::ReauthOrRuleMatchingTarget => {
                        tracing::info!(
                            "Sending reauth request with OR rule matching the target index"
                        );
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
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             None,
                                is_batch_match:       false,
                                is_reauth_successful: Some(true),
                            });
                        template
                    }
                    TestCases::ReauthOrRuleNonMatchingTarget => {
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
                        self.expected_results
                            .insert(request_id.to_string(), ExpectedResult {
                                db_index:             None,
                                is_batch_match:       false,
                                is_reauth_successful: Some(false),
                            });
                        template
                    }
                }
            };
            (request_id, e2e_template, or_rule_indices)
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
                left:  code_left,
                right: code_right,
            }
        }

        // check a received result against the expected results
        fn check_result(
            &mut self,
            req_id: &str,
            idx: u32,
            was_match: bool,
            matched_batch_req_ids: &[String],
            requests: &HashMap<String, E2ETemplate>,
            was_reauth_success: bool,
        ) {
            tracing::info!(
                "Checking result for request_id: {}, idx: {}, was_match: {}, \
                 matched_batch_req_ids: {:?}, was_reauth_success: {}",
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
                if !is_batch_match {
                    assert_eq!(expected_idx, idx);
                } else {
                    assert!(self.batch_duplicates.contains_key(req_id));
                    assert!(
                        matched_batch_req_ids.contains(self.batch_duplicates.get(req_id).unwrap())
                    );
                }
            } else {
                assert!(!was_match);
                let request = requests.get(req_id).unwrap().clone();
                self.inserted_responses.insert(idx, request);
            }
        }
    }
}
