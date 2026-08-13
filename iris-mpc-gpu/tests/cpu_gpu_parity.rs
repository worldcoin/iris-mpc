#![recursion_limit = "256"]

#[cfg(feature = "gpu_dependent")]
mod parity {
    use ampc_anon_stats::types::Eye;
    use cudarc::nccl::Id;
    use eyre::Result;
    use iris_mpc_common::{
        helpers::inmemory_store::InMemoryStore,
        job::{JobSubmissionHandle, ServerJobResult},
        test::{generate_full_test_db, load_test_db, PartyDb, TestCaseGenerator},
        VectorId,
    };
    use iris_mpc_cpu::{
        execution::{
            hawk_main::{HawkActor, HawkArgs, HawkHandle, HawkMutation},
            local::get_free_local_addresses,
        },
        hawkers::aby3::aby3_store::{Aby3SharedIrises, Aby3Store, FhdOps},
        hnsw::GraphMem,
        protocol::shared_iris::GaloisRingSharedIris,
    };
    use iris_mpc_gpu::{helpers::device_manager::DeviceManager, server::ServerActor};
    use std::{collections::HashMap, env, sync::Arc, time::Instant};
    use tokio::sync::oneshot;
    use tokio_util::sync::CancellationToken;

    const DB_SIZE: usize = 1024;
    const DB_BUFFER: usize = 128;
    const MAX_BATCH_SIZE: usize = 32;
    const PARITY_BATCHES: usize = 3;
    const MATCH_DISTANCES_BUFFER_SIZE: usize = 1 << 15;

    #[test]
    fn cpu_linear_scan_matches_gpu_actor() -> Result<()> {
        std::thread::Builder::new()
            .name("cpu_gpu_parity".to_string())
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .thread_stack_size(64 * 1024 * 1024)
                    .build()
                    .expect("build parity runtime")
                    .block_on(async {
                        // Two actor lifetimes model coordinated restarts with
                        // the opposite resident/full-scan eye selected.
                        run_parity(Eye::Left).await?;
                        run_parity(Eye::Right).await
                    })
            })
            .expect("spawn parity test")
            .join()
            .expect("parity test thread panicked")
    }

    async fn run_parity(full_scan_side: Eye) -> Result<()> {
        env::set_var("NCCL_P2P_LEVEL", "LOC");
        env::set_var("NCCL_NET", "Socket");

        let test_db = generate_full_test_db(DB_SIZE, 0x5eed_600d, false);
        let party_dbs = [
            test_db.party_db(0),
            test_db.party_db(1),
            test_db.party_db(2),
        ];

        let (mut gpu_handles, gpu_tasks) = start_gpu_nodes(&party_dbs, full_scan_side).await?;
        let mut cpu_handles = start_cpu_nodes(&party_dbs, full_scan_side).await?;

        // `is_cpu=true` keeps every share valid. GPU-only invalid-share
        // filtering is tested separately and would otherwise intentionally
        // make the actors receive different effective batches.
        let mut generator = TestCaseGenerator::new_with_db(test_db, 0xc0de_cafe, true);
        for batch_index in 0..PARITY_BATCHES {
            let (mut batches, _requests) = generator.generate_query_batch(MAX_BATCH_SIZE, 4, 4)?;

            // Cover the CUDA actor's no-mirror branch in the final batch. It
            // is last because the generator models mirror-enabled outcomes
            // when preparing its state for subsequent batches.
            if batch_index + 1 == PARITY_BATCHES {
                for batch in &mut batches {
                    batch.full_face_mirror_attacks_detection_enabled = false;
                }
            }

            let gpu_started = Instant::now();
            let gpu_results = submit_gpu(&mut gpu_handles, batches.clone()).await?;
            let gpu_elapsed = gpu_started.elapsed();

            let cpu_started = Instant::now();
            let cpu_results = submit_cpu(&mut cpu_handles, batches).await?;
            let cpu_elapsed = cpu_started.elapsed();

            eprintln!(
                "parity side={full_scan_side} batch {batch_index}: gpu={gpu_elapsed:?}, cpu={cpu_elapsed:?}"
            );
            for party in 0..3 {
                assert_same_public_result(&gpu_results[party], &cpu_results[party]);
            }
        }

        drop(gpu_handles);
        for task in gpu_tasks {
            task.await.expect("GPU actor task panicked");
        }
        drop(cpu_handles);
        Ok(())
    }

    async fn start_gpu_nodes(
        party_dbs: &[Arc<PartyDb>; 3],
        full_scan_side: Eye,
    ) -> Result<(
        [iris_mpc_gpu::server::ServerActorHandle; 3],
        [tokio::task::JoinHandle<()>; 3],
    )> {
        let device_manager = DeviceManager::init();
        let mut managers = device_manager
            .split_into_n_chunks(3)
            .expect("parity test requires at least three GPUs");
        let managers = [
            Arc::new(managers.remove(0)),
            Arc::new(managers.remove(0)),
            Arc::new(managers.remove(0)),
        ];

        let ids0 = (0..managers[0].devices().len())
            .map(|_| Id::new().expect("create NCCL ID"))
            .collect::<Vec<_>>();
        let ids = [ids0.clone(), ids0.clone(), ids0];
        let seeds = [
            ([0u32; 8], [2u32; 8]),
            ([1u32; 8], [0u32; 8]),
            ([2u32; 8], [1u32; 8]),
        ];

        let mut receivers = Vec::with_capacity(3);
        let mut tasks = Vec::with_capacity(3);
        for party in 0..3 {
            let manager = managers[party].clone();
            let party_ids = ids[party].clone();
            let party_db = party_dbs[party].clone();
            let (tx, rx) = oneshot::channel();
            receivers.push(rx);
            tasks.push(tokio::task::spawn_blocking(move || {
                let comms = manager
                    .instantiate_network_from_ids(party, &party_ids)
                    .expect("initialize NCCL communicator");
                let (mut actor, handle) = ServerActor::new_with_device_manager_and_comms(
                    party,
                    seeds[party],
                    manager,
                    comms,
                    8,
                    DB_SIZE + DB_BUFFER,
                    MAX_BATCH_SIZE,
                    MATCH_DISTANCES_BUFFER_SIZE,
                    100,
                    true,
                    false,
                    false,
                    full_scan_side,
                    false,
                    None,
                )
                .expect("initialize GPU actor");
                load_test_db(&party_db, &mut actor);
                actor.preprocess_db();
                tx.send(handle).expect("return GPU actor handle");
                actor.run();
            }));
        }

        let mut handles = Vec::with_capacity(3);
        for receiver in receivers {
            handles.push(receiver.await?);
        }
        Ok((
            handles.try_into().ok().unwrap(),
            tasks.try_into().ok().unwrap(),
        ))
    }

    fn shared_store(party_db: &PartyDb, left: bool) -> Aby3SharedIrises {
        let source = if left {
            &party_db.db_left
        } else {
            &party_db.db_right
        };
        let points = source
            .iter()
            .enumerate()
            .map(|(index, share)| {
                (
                    VectorId::from_0_index(index as u32),
                    Arc::new(GaloisRingSharedIris {
                        code: share.code.clone(),
                        mask: share.mask.clone(),
                    }),
                )
            })
            .collect::<HashMap<_, _>>();
        Aby3Store::<FhdOps>::new_storage(Some(points))
    }

    fn cpu_args(party: usize, addresses: &[String], full_scan_side: Eye) -> HawkArgs {
        HawkArgs {
            party_index: party,
            addresses: addresses.to_vec(),
            outbound_addrs: addresses.to_vec(),
            request_parallelism: 2,
            connection_parallelism: 1,
            hnsw_param_ef_constr: 320,
            hnsw_param_m: 256,
            hnsw_param_ef_search: 256,
            hnsw_param_ef_search_layers_override: None,
            hnsw_param_ef_supermatch: 4000,
            hnsw_param_ef_saturation_margin: 0,
            hnsw_layer_density: None,
            hnsw_min_layer_search_batch_size: None,
            hnsw_prf_key: Some(7),
            disable_persistence: false,
            return_partial_results: true,
            full_scan_side,
            hnsw_disable_memory_persistence: false,
            tls: None,
            numa: false,
        }
    }

    async fn start_cpu_nodes(
        party_dbs: &[Arc<PartyDb>; 3],
        full_scan_side: Eye,
    ) -> Result<[HawkHandle; 3]> {
        let addresses = get_free_local_addresses(3).await?;
        let nodes = (0..3).map(|party| {
            let args = cpu_args(party, &addresses, full_scan_side);
            let stores = [
                shared_store(&party_dbs[party], true),
                shared_store(&party_dbs[party], false),
            ];
            async move {
                let actor = HawkActor::from_cli_with_graph_and_store_linear_scan(
                    &args,
                    CancellationToken::new(),
                    [GraphMem::new(), GraphMem::new()],
                    stores,
                )
                .await?;
                HawkHandle::new(actor).await
            }
        });
        let [node0, node1, node2]: [_; 3] = nodes.collect::<Vec<_>>().try_into().ok().unwrap();
        let (node0, node1, node2) = tokio::join!(node0, node1, node2);
        Ok([node0?, node1?, node2?])
    }

    async fn submit_gpu(
        handles: &mut [iris_mpc_gpu::server::ServerActorHandle; 3],
        batches: [iris_mpc_common::job::BatchQuery; 3],
    ) -> Result<[ServerJobResult; 3]> {
        let [handle0, handle1, handle2] = handles;
        let [batch0, batch1, batch2] = batches;
        let (result0, result1, result2) = tokio::join!(
            handle0.submit_batch_query(batch0),
            handle1.submit_batch_query(batch1),
            handle2.submit_batch_query(batch2),
        );
        Ok([result0.await?, result1.await?, result2.await?])
    }

    async fn submit_cpu(
        handles: &mut [HawkHandle; 3],
        batches: [iris_mpc_common::job::BatchQuery; 3],
    ) -> Result<[ServerJobResult<HawkMutation>; 3]> {
        let [handle0, handle1, handle2] = handles;
        let [batch0, batch1, batch2] = batches;
        let (result0, result1, result2) = tokio::join!(
            handle0.submit_batch_query(batch0),
            handle1.submit_batch_query(batch1),
            handle2.submit_batch_query(batch2),
        );
        Ok([result0.await?, result1.await?, result2.await?])
    }

    fn assert_same_public_result(gpu: &ServerJobResult, cpu: &ServerJobResult<HawkMutation>) {
        for index in 0..gpu.request_ids.len() {
            if gpu.merged_results[index] != cpu.merged_results[index] {
                eprintln!(
                    "merged mismatch at {index}: request_id={}, type={}, gpu={{merged:{}, matches:{}, skip_matches:{}, ids:{:?}, batch_ids:{:?}, mirror_attack:{}}}, cpu={{merged:{}, matches:{}, skip_matches:{}, ids:{:?}, batch_ids:{:?}, mirror_attack:{}}}",
                    gpu.request_ids[index],
                    gpu.request_types[index],
                    gpu.merged_results[index],
                    gpu.matches[index],
                    gpu.matches_with_skip_persistence[index],
                    gpu.match_ids[index],
                    gpu.matched_batch_request_ids[index],
                    gpu.full_face_mirror_attack_detected[index],
                    cpu.merged_results[index],
                    cpu.matches[index],
                    cpu.matches_with_skip_persistence[index],
                    cpu.match_ids[index],
                    cpu.matched_batch_request_ids[index],
                    cpu.full_face_mirror_attack_detected[index],
                );
            }
        }

        macro_rules! same {
            ($($field:ident),+ $(,)?) => {$({
                assert_eq!(gpu.$field, cpu.$field, "mismatch in {}", stringify!($field));
            })+};
        }

        same!(
            merged_results,
            sqs_sequence_numbers,
            request_ids,
            request_types,
            metadata,
            matches,
            matches_with_skip_persistence,
            skip_persistence,
            match_ids,
            full_face_mirror_match_ids,
            partial_match_ids_left,
            partial_match_ids_right,
            partial_match_rotation_indices_left,
            partial_match_rotation_indices_right,
            full_face_mirror_partial_match_ids_left,
            full_face_mirror_partial_match_ids_right,
            partial_match_counters_left,
            partial_match_counters_right,
            full_face_mirror_partial_match_counters_left,
            full_face_mirror_partial_match_counters_right,
            left_iris_requests,
            right_iris_requests,
            deleted_ids,
            matched_batch_request_ids,
            successful_reauths,
            reauth_target_indices,
            reauth_or_rule_used,
            modifications,
            identity_update_indices,
            identity_update_request_ids,
            identity_update_request_types,
            identity_update_shares,
            full_face_mirror_attack_detected,
        );
    }
}
