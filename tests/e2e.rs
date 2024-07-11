use eyre::Result;
use gpu_iris_mpc::{
    config::ServersConfig,
    helpers::{device_manager::DeviceManager, mmap},
    server::{BatchQuery, ServerActor, ServerJobResult},
    setup::{
        galois_engine::degree4::GaloisRingIrisCodeShare,
        iris_db::{db::IrisDB, iris::IrisCode},
    },
};
use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use std::{collections::HashMap, fs, sync::Arc};
use tokio::sync::oneshot;
use uuid::Uuid;

const DB_SIZE: usize = 8 * 1000;
const RNG_SEED: u64 = 0xdeadbeef;
const NUM_BATCHES: usize = 5;
const BATCH_SIZE: usize = 32;

fn generate_or_load_db(party_id: usize) -> Result<(Vec<u16>, Vec<u16>)> {
    let code_db_path = format!("/tmp/code_db{party_id}");
    let mask_db_path = format!("/tmp/mask_db{party_id}");
    if fs::metadata(&code_db_path).is_ok() && fs::metadata(&mask_db_path).is_ok() {
        Ok((
            mmap::read_mmap_file(&code_db_path)?,
            mmap::read_mmap_file(&mask_db_path)?,
        ))
    } else {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let db = IrisDB::new_random_par(DB_SIZE, &mut rng);

        let codes_db = db
            .db
            .iter()
            .flat_map(|iris| {
                GaloisRingIrisCodeShare::encode_iris_code(
                    &iris.code,
                    &iris.mask,
                    &mut StdRng::seed_from_u64(RNG_SEED),
                )[party_id]
                    .coefs
            })
            .collect::<Vec<_>>();

        let masks_db = db
            .db
            .iter()
            .flat_map(|iris| {
                GaloisRingIrisCodeShare::encode_mask_code(
                    &iris.mask,
                    &mut StdRng::seed_from_u64(RNG_SEED),
                )[party_id]
                    .coefs
            })
            .collect::<Vec<_>>();
        mmap::write_mmap_file(&code_db_path, &codes_db)?;
        mmap::write_mmap_file(&mask_db_path, &masks_db)?;

        Ok((codes_db, masks_db))
    }
}

#[tokio::test]
async fn e2e_test() -> Result<()> {
    let db0 = generate_or_load_db(0)?;
    let db1 = generate_or_load_db(1)?;
    let db2 = generate_or_load_db(2)?;

    let config0 = ServersConfig {
        codes_engine_port:       10001,
        masks_engine_port:       10002,
        batch_codes_engine_port: 10003,
        batch_masks_engine_port: 10004,
        phase_2_batch_port:      10005,
        phase_2_port:            10006,
        bootstrap_url:           None,
    };
    let config1 = ServersConfig {
        bootstrap_url: Some("localhost".to_string()),
        ..config0.clone()
    };
    let config2 = config1.clone();

    let chacha_seeds0 = ([0u32; 8], [2u32; 8]);
    let chacha_seeds1 = ([1u32; 8], [0u32; 8]);
    let chacha_seeds2 = ([2u32; 8], [1u32; 8]);

    // a bit convoluted, but we need to create the actor on the thread already,
    // since it blocks a lot and is `!Send`, we get back the handle via the oneshot
    // channel
    let (tx0, rx0) = oneshot::channel();
    let (tx1, rx1) = oneshot::channel();
    let (tx2, rx2) = oneshot::channel();

    let device_manager0 = Arc::new(
        DeviceManager::init_with_device_offset_and_limit(0, 2)
            .map_err(|e| eyre::eyre!("wanted 2 devices starting at 0, only have {e}"))?,
    );
    let device_manager1 = Arc::new(
        DeviceManager::init_with_device_offset_and_limit(2, 2)
            .map_err(|e| eyre::eyre!("wanted 2 devices starting at 0, only have {e}"))?,
    );
    let device_manager2 = Arc::new(
        DeviceManager::init_with_device_offset_and_limit(4, 2)
            .map_err(|e| eyre::eyre!("wanted 2 devices starting at 0, only have {e}"))?,
    );
    let actor0_task = tokio::task::spawn_blocking(move || {
        let actor = match ServerActor::new_with_device_manager(
            0,
            config0,
            chacha_seeds0,
            &db0.0,
            &db0.1,
            device_manager0,
            8,
        ) {
            Ok((actor, handle)) => {
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
        let actor = match ServerActor::new_with_device_manager(
            1,
            config1,
            chacha_seeds1,
            &db1.0,
            &db1.1,
            device_manager1,
            8,
        ) {
            Ok((actor, handle)) => {
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
        let actor = match ServerActor::new_with_device_manager(
            2,
            config2,
            chacha_seeds2,
            &db2.0,
            &db2.1,
            device_manager2,
            8,
        ) {
            Ok((actor, handle)) => {
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

    let db = IrisDB::new_random_par(DB_SIZE, &mut StdRng::seed_from_u64(RNG_SEED));

    let mut choice_rng = thread_rng();
    let mut rng = thread_rng();

    let mut expected_results: HashMap<String, Option<u32>> = HashMap::new();
    let mut requests: HashMap<String, IrisCode> = HashMap::new();
    let mut responses: HashMap<u32, IrisCode> = HashMap::new();

    for _ in 0..NUM_BATCHES {
        let mut batch0 = BatchQuery::default();
        let mut batch1 = BatchQuery::default();
        let mut batch2 = BatchQuery::default();
        for _ in 0..BATCH_SIZE {
            let request_id = Uuid::new_v4();
            // Automatic random tests
            let options = if responses.len() == 0 { 2 } else { 3 };
            let template = match choice_rng.gen_range(0..options) {
                0 => {
                    println!("Sending new iris code");
                    expected_results.insert(request_id.to_string(), None);
                    IrisCode::random_rng(&mut rng)
                }
                1 => {
                    println!("Sending iris code from db");
                    let db_index = rng.gen_range(0..db.db.len());
                    expected_results.insert(request_id.to_string(), Some(db_index as u32));
                    db.db[db_index].clone()
                }
                2 => {
                    println!("Sending freshly inserted iris code");
                    let keys = responses.keys().collect::<Vec<_>>();
                    let idx = rng.gen_range(0..keys.len());
                    let iris_code = responses.get(keys[idx]).unwrap().clone();
                    expected_results.insert(request_id.to_string(), Some(*keys[idx]));
                    iris_code
                }
                _ => unreachable!(),
            };
            requests.insert(request_id.to_string(), template.clone());

            let mut shared_code =
                GaloisRingIrisCodeShare::encode_iris_code(&template.code, &template.mask, &mut rng);
            let mut shared_mask =
                GaloisRingIrisCodeShare::encode_mask_code(&template.mask, &mut rng);
            // batch0
            batch0.request_ids.push(request_id.to_string());
            // for storage
            batch0.store.code.push(shared_code[0].clone());
            batch0.store.mask.push(shared_mask[0].clone());
            // with rotations
            batch0.db.code.extend(shared_code[0].all_rotations());
            batch0.db.mask.extend(shared_mask[0].all_rotations());
            // with rotations
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut shared_code[0]);
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut shared_mask[0]);
            batch0.query.code.extend(shared_code[0].all_rotations());
            batch0.query.mask.extend(shared_mask[0].all_rotations());

            // batch 1
            batch1.request_ids.push(request_id.to_string());
            // for storage
            batch1.store.code.push(shared_code[1].clone());
            batch1.store.mask.push(shared_mask[1].clone());
            // with rotations
            batch1.db.code.extend(shared_code[1].all_rotations());
            batch1.db.mask.extend(shared_mask[1].all_rotations());
            // with rotations
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut shared_code[1]);
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut shared_mask[1]);
            batch1.query.code.extend(shared_code[1].all_rotations());
            batch1.query.mask.extend(shared_mask[1].all_rotations());

            // batch 2
            batch2.request_ids.push(request_id.to_string());
            // for storage
            batch2.store.code.push(shared_code[2].clone());
            batch2.store.mask.push(shared_mask[2].clone());
            // with rotations
            batch2.db.code.extend(shared_code[2].all_rotations());
            batch2.db.mask.extend(shared_mask[2].all_rotations());
            // with rotations
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut shared_code[2]);
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut shared_mask[2]);
            batch2.query.code.extend(shared_code[2].all_rotations());
            batch2.query.mask.extend(shared_mask[2].all_rotations());
        }

        // send batches to servers
        let res0_fut = handle0.submit_batch_query(batch0).await;
        let res1_fut = handle1.submit_batch_query(batch1).await;
        let res2_fut = handle2.submit_batch_query(batch2).await;

        let res0 = res0_fut.await;
        let res1 = res1_fut.await;
        let res2 = res2_fut.await;

        // go over results and check if correct
        for res in [res0, res1, res2].iter() {
            let ServerJobResult {
                thread_request_ids,
                matches,
                merged_results,
                ..
            } = res;
            for ((req_id, &was_match), &idx) in thread_request_ids
                .iter()
                .zip(matches.iter())
                .zip(merged_results.iter())
            {
                let expected_idx = expected_results.get(req_id).unwrap();
                if let Some(expected_idx) = expected_idx {
                    assert!(was_match);
                    assert_eq!(expected_idx, &idx);
                } else {
                    assert!(!was_match);
                    let request = requests.get(req_id).unwrap().clone();
                    responses.insert(idx, request);
                }
            }
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
