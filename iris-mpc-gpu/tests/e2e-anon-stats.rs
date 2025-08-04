#[cfg(feature = "gpu_dependent")]
mod e2e_anon_stats_test {
    use cudarc::nccl::Id;
    use eyre::Result;
    use iris_mpc_common::{
        helpers::inmemory_store::InMemoryStore,
        job::Eye,
        test::{generate_full_test_db, load_test_db, SimpleAnonStatsTestGenerator},
    };
    use iris_mpc_gpu::{helpers::device_manager::DeviceManager, server::ServerActor};
    use rand::random;
    use std::{env, sync::Arc};
    use tokio::sync::oneshot;
    use tracing_subscriber::{fmt::format::FmtSpan, layer::SubscriberExt, util::SubscriberInitExt};

    const DB_SIZE: usize = 8 * 1000;
    const DB_BUFFER: usize = 8 * 1000;
    const NUM_BATCHES: usize = 300;
    const MAX_BATCH_SIZE: usize = 64;
    const N_BUCKETS: usize = 8;
    const MATCH_DISTANCES_BUFFER_SIZE: usize = 1 << 6;
    const MATCH_DISTANCES_BUFFER_SIZE_EXTRA_PERCENT: usize = 5000;

    fn install_tracing() {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
                    .with_target(true),
            )
            .init();
    }

    #[tokio::test]
    async fn e2e_test() -> Result<()> {
        install_tracing();
        env::set_var("NCCL_P2P_LEVEL", "LOC");
        env::set_var("NCCL_NET", "Socket");

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

        let internal_seed = match env::var("INTERNAL_SEED") {
            Ok(seed) => {
                tracing::info!("Internal SEED was passed: {}", seed);
                seed.parse::<u64>()?
            }
            Err(_) => {
                tracing::info!("Internal SEED not set, using random seed");
                random()
            }
        };
        let db_seed = match env::var("DB_SEED") {
            Ok(seed) => {
                tracing::info!("DB SEED was passed: {}", seed);
                seed.parse::<u64>()?
            }
            Err(_) => {
                tracing::info!("DB SEED not set, using random seed");
                random()
            }
        };
        tracing::info!(
            "Seeds for this test run. DB: {}, Internal: {}",
            db_seed,
            internal_seed
        );

        let test_db = generate_full_test_db(DB_SIZE, db_seed, true);
        let party_db0 = test_db.party_db(0);
        let party_db1 = test_db.party_db(1);
        let party_db2 = test_db.party_db(2);

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
                Eye::Left,
                false,
            ) {
                Ok((mut actor, handle)) => {
                    load_test_db(&party_db0, &mut actor);
                    actor.preprocess_db();
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
                Eye::Left,
                false,
            ) {
                Ok((mut actor, handle)) => {
                    load_test_db(&party_db1, &mut actor);
                    actor.preprocess_db();
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
                Eye::Left,
                false,
            ) {
                Ok((mut actor, handle)) => {
                    load_test_db(&party_db2, &mut actor);
                    actor.preprocess_db();
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

        let mut test_case_generator =
            SimpleAnonStatsTestGenerator::new(test_db, internal_seed, N_BUCKETS);

        tracing::info!("Setup done, starting tests");
        test_case_generator
            .run_n_batches(NUM_BATCHES, [&mut handle0, &mut handle1, &mut handle2])
            .await?;

        drop(handle0);
        drop(handle1);
        drop(handle2);

        actor0_task.await.unwrap();
        actor1_task.await.unwrap();
        actor2_task.await.unwrap();

        Ok(())
    }
}
