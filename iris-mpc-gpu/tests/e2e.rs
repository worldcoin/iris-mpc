#[cfg(feature = "gpu_dependent")]
mod e2e_test {
    use cudarc::nccl::Id;
    use eyre::Result;
    use iris_mpc_common::{
        galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
        iris_db::{
            db::IrisDB,
            iris::{IrisCode, IrisCodeArray},
        },
    };
    use iris_mpc_gpu::{
        helpers::device_manager::DeviceManager,
        server::{BatchQuery, BatchQueryEntriesPreprocessed, ServerActor, ServerJobResult},
    };
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::{collections::HashMap, env, sync::Arc};
    use tokio::sync::oneshot;
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    use uuid::Uuid;

    const DB_SIZE: usize = 8 * 1000;
    const DB_BUFFER: usize = 8 * 1000;
    const DB_RNG_SEED: u64 = 0xdeadbeef;
    const INTERNAL_RNG_SEED: u64 = 0xdeadbeef;
    const NUM_BATCHES: usize = 10;
    const MAX_BATCH_SIZE: usize = 64;
    const MAX_DELETIONS_PER_BATCH: usize = 10;
    const THRESHOLD_ABSOLUTE: usize = 4800; // 0.375 * 12800

    #[derive(Clone)]
    pub struct E2ETemplate {
        left:  IrisCode,
        right: IrisCode,
    }

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
        use std::collections::HashSet;

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

        // make a test query and send it to server

        let mut db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(DB_RNG_SEED));

        // Set the masks to all 1s for the first 10%
        for i in 0..DB_SIZE / 10 {
            db.db[i].mask = IrisCodeArray::ONES;
        }

        let mut rng = StdRng::seed_from_u64(INTERNAL_RNG_SEED);

        let mut expected_results: HashMap<String, (Option<u32>, bool)> = HashMap::new();
        let mut responses: HashMap<u32, E2ETemplate> = HashMap::new();
        let mut deleted_indices_buffer = vec![];
        let mut deleted_indices: HashSet<u32> = HashSet::new();
        let mut disallowed_queries = Vec::new();

        for _ in 0..NUM_BATCHES {
            let mut requests: HashMap<String, E2ETemplate> = HashMap::new();
            let mut batch0 = BatchQuery::default();
            let mut batch1 = BatchQuery::default();
            let mut batch2 = BatchQuery::default();
            let batch_size = rng.gen_range(1..MAX_BATCH_SIZE);
            let mut new_templates_in_batch: Vec<(usize, String, IrisCode)> = vec![];
            let mut skip_invalidate = false;
            let mut batch_duplicates: HashMap<String, String> = HashMap::new();

            for idx in 0..batch_size {
                let request_id = Uuid::new_v4();
                // Automatic random tests
                let options = if responses.is_empty() {
                    3
                } else if deleted_indices_buffer.is_empty() {
                    4
                } else {
                    5
                };

                let pick_from_batch = rng.gen_range(0..10);
                let e2e_template = if pick_from_batch == 0 && !new_templates_in_batch.is_empty() {
                    let random_idx = rng.gen_range(0..new_templates_in_batch.len());
                    let (batch_idx, duplicate_request_id, template) =
                        new_templates_in_batch[random_idx].clone();
                    expected_results.insert(request_id.to_string(), (Some(batch_idx as u32), true));
                    batch_duplicates.insert(request_id.to_string(), duplicate_request_id);
                    skip_invalidate = true;
                    E2ETemplate {
                        left:  template.clone(),
                        right: template.clone(),
                    }
                } else {
                    let option = rng.gen_range(0..options);
                    match option {
                        0 => {
                            println!("Sending new iris code");
                            expected_results.insert(request_id.to_string(), (None, false));
                            let template = IrisCode::random_rng(&mut rng);
                            new_templates_in_batch.push((
                                idx,
                                request_id.to_string(),
                                template.clone(),
                            ));
                            skip_invalidate = true;
                            E2ETemplate {
                                left:  template.clone(),
                                right: template.clone(),
                            }
                        }
                        1 => {
                            println!("Sending iris code from db");
                            let db_index = rng.gen_range(0..db.db.len());
                            if deleted_indices.contains(&(db_index as u32)) {
                                continue;
                            }
                            expected_results
                                .insert(request_id.to_string(), (Some(db_index as u32), false));
                            E2ETemplate {
                                left:  db.db[db_index].clone(),
                                right: db.db[db_index].clone(),
                            }
                        }
                        2 => {
                            println!("Sending iris code on the threshold!!!!!!");
                            let db_index = loop {
                                let db_index = rng.gen_range(0..DB_SIZE / 10);
                                if !disallowed_queries.contains(&db_index) {
                                    break db_index;
                                }
                            };
                            if deleted_indices.contains(&(db_index as u32)) {
                                continue;
                            }
                            let variation = rng.gen_range(-1..=1);
                            expected_results.insert(
                                request_id.to_string(),
                                if variation > 0 {
                                    // we flip more than the threshold so this should not match
                                    // however it would afterwards so we no longer pick it
                                    disallowed_queries.push(db_index);
                                    (None, false)
                                } else {
                                    // we flip less or equal to than the threshold so this should
                                    // match
                                    (Some(db_index as u32), false)
                                },
                            );
                            let mut code = db.db[db_index].clone();
                            assert_eq!(code.mask, IrisCodeArray::ONES);
                            for i in 0..(THRESHOLD_ABSOLUTE as i32 + variation) as usize {
                                code.code.flip_bit(i);
                            }
                            E2ETemplate {
                                left:  code.clone(),
                                right: code.clone(),
                            }
                        }
                        3 => {
                            println!("Sending freshly inserted iris code");
                            let keys = responses.keys().collect::<Vec<_>>();
                            let idx = rng.gen_range(0..keys.len());
                            let e2e_template = responses.get(keys[idx]).unwrap().clone();
                            expected_results
                                .insert(request_id.to_string(), (Some(*keys[idx]), false));

                            E2ETemplate {
                                left:  e2e_template.left.clone(),
                                right: e2e_template.right.clone(),
                            }
                        }
                        4 => {
                            println!("Sending deleted iris code");
                            let idx = rng.gen_range(0..deleted_indices_buffer.len());
                            let deleted_idx = deleted_indices_buffer[idx];
                            deleted_indices_buffer.remove(idx);
                            expected_results.insert(request_id.to_string(), (None, false));
                            E2ETemplate {
                                left:  db.db[deleted_idx as usize].clone(),
                                right: db.db[deleted_idx as usize].clone(),
                            }
                        }
                        // TODO: implement test that enables testing the bitmap OR rule setting.
                        // This will require setting the left and right iris code to be different,
                        // one of them below the threshold and the other
                        // above the threshold against the other codes. It will
                        // also be required to set the param BatchQuery.threshold_bitmap_or to true
                        // for the specific iris codes to be matched against
                        // 5 => {
                        // println!("Sending iris codes that match on right but not left with the OR
                        // rule set");
                        _ => unreachable!(),
                    }
                };

                // Invalidate 10% of the queries, but ignore the batch duplicates
                let is_valid = rng.gen_range(0..10) != 0 || skip_invalidate;

                if is_valid {
                    requests.insert(request_id.to_string(), e2e_template.clone());
                }

                let shared_template = to_shared_template(is_valid, &e2e_template, &mut rng);

                prepare_batch(
                    &mut batch0,
                    is_valid,
                    request_id.to_string(),
                    0,
                    shared_template.clone(),
                )?;

                prepare_batch(
                    &mut batch1,
                    true,
                    request_id.to_string(),
                    1,
                    shared_template.clone(),
                )?;

                prepare_batch(
                    &mut batch2,
                    true,
                    request_id.to_string(),
                    2,
                    shared_template.clone(),
                )?;
            }

            // Skip empty batch
            if batch0.request_ids.is_empty() {
                continue;
            }

            for _ in 0..rng.gen_range(0..MAX_DELETIONS_PER_BATCH) {
                let idx = rng.gen_range(0..db.db.len());
                if deleted_indices.contains(&(idx as u32)) {
                    continue;
                }
                deleted_indices_buffer.push(idx as u32);
                deleted_indices.insert(idx as u32);
                println!("Deleting index {}", idx);

                batch0.deletion_requests_indices.push(idx as u32);
                batch1.deletion_requests_indices.push(idx as u32);
                batch2.deletion_requests_indices.push(idx as u32);
            }

            // Preprocess the batches
            preprocess_batch(&mut batch0)?;
            preprocess_batch(&mut batch1)?;
            preprocess_batch(&mut batch2)?;

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
                    ..
                } = res;
                for (
                    (((((req_id, was_match), idx), partial_left), partial_right), match_id),
                    matched_batch_req_ids,
                ) in thread_request_ids
                    .iter()
                    .zip(matches.iter())
                    .zip(merged_results.iter())
                    .zip(partial_match_ids_left.iter())
                    .zip(partial_match_ids_right.iter())
                    .zip(match_ids.iter())
                    .zip(matched_batch_request_ids.iter())
                {
                    assert!(requests.contains_key(req_id));

                    resp_counters.insert(req_id, resp_counters.get(req_id).unwrap() + 1);

                    assert_eq!(partial_left, partial_right);
                    assert_eq!(partial_left, match_id);

                    let (expected_idx, is_batch_match) = expected_results.get(req_id).unwrap();

                    if let Some(expected_idx) = expected_idx {
                        assert!(was_match);
                        if !is_batch_match {
                            assert_eq!(expected_idx, idx);
                        } else {
                            assert!(batch_duplicates.contains_key(req_id));
                            assert!(matched_batch_req_ids
                                .contains(batch_duplicates.get(req_id).unwrap()));
                        }
                    } else {
                        assert!(!was_match);
                        let request = requests.get(req_id).unwrap().clone();
                        responses.insert(*idx, request);
                    }
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
    ) -> Result<()> {
        batch.metadata.push(Default::default());
        batch.valid_entries.push(is_valid);
        batch.request_ids.push(request_id);

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

    fn preprocess_batch(batch: &mut BatchQuery) -> Result<()> {
        batch.query_left_preprocessed =
            BatchQueryEntriesPreprocessed::from(batch.query_left.clone());
        batch.query_right_preprocessed =
            BatchQueryEntriesPreprocessed::from(batch.query_right.clone());
        batch.db_left_preprocessed = BatchQueryEntriesPreprocessed::from(batch.db_left.clone());
        batch.db_right_preprocessed = BatchQueryEntriesPreprocessed::from(batch.db_right.clone());
        Ok(())
    }
}
