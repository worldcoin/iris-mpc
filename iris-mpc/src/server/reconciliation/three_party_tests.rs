use super::super::{
    drain::{self, Phase},
    handoff::Coordinator,
    ManagedHandoff,
};
use super::*;
use ampc_anon_stats::types::Eye;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use iris_mpc_common::{
    config::{Config, CpuCellSlot},
    iris_db::iris::IrisCode,
    postgres::{run_migrations, AccessMode, PostgresClient},
    VectorId,
};
use iris_mpc_store::cpu_cell::CpuCellHandoffs;
use rand::{rngs::StdRng, SeedableRng};
use std::sync::Arc;

struct SharedRow {
    id: VectorId,
    eyes: Vec<Vec<GaloisRingSharedIris>>,
}

impl SharedRow {
    fn generate(rng: &mut StdRng, serial: u32, version: i16) -> Self {
        Self {
            id: VectorId::new(serial, version),
            eyes: (0..2)
                .map(|_| {
                    let iris = IrisCode::random_rng(rng);
                    GaloisRingSharedIris::generate_shares_locally(rng, iris).into()
                })
                .collect(),
        }
    }

    async fn write(&self, db: &PostgresClient, party: usize) -> Result<()> {
        let left = &self.eyes[0][party];
        let right = &self.eyes[1][party];
        let mut tx = db.pool.begin().await?;
        let mut explicit = iris_mpc_store::ExplicitVersionToken::enable(&mut tx).await?;
        sqlx::query(
            "INSERT INTO irises (id, version_id, left_code, left_mask, right_code, right_mask) VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (id) DO UPDATE SET version_id = EXCLUDED.version_id,
             left_code = EXCLUDED.left_code, left_mask = EXCLUDED.left_mask,
             right_code = EXCLUDED.right_code, right_mask = EXCLUDED.right_mask",
        )
        .bind(i64::from(self.id.serial_id()))
        .bind(self.id.version_id())
        .bind(bytemuck::cast_slice::<u16, u8>(&left.code.coefs))
        .bind(bytemuck::cast_slice::<u16, u8>(&left.mask.coefs))
        .bind(bytemuck::cast_slice::<u16, u8>(&right.code.coefs))
        .bind(bytemuck::cast_slice::<u16, u8>(&right.mask.coefs))
        .execute(&mut **explicit.tx())
        .await?;
        tx.commit().await?;
        Ok(())
    }
}

struct Party {
    config: Config,
    writer: PostgresClient,
    candidate: CandidateReconciliation,
}

struct Coordinated {
    context: ReconciliationContext,
    coordinators: Vec<Coordinator>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    shutdowns: Vec<tokio_util::sync::CancellationToken>,
    configs: Vec<Config>,
    next_listeners: Vec<tokio::net::TcpListener>,
}

struct Serving {
    writer: PostgresClient,
    physical_config: Config,
    config: Config,
    handle: iris_mpc_cpu::execution::hawk_main::HawkHandle,
    shutdown: Arc<ShutdownHandler>,
    monitor: ampc_server_utils::TaskMonitor,
    coordinator: Coordinator,
    drain: Arc<drain::Drain>,
    ready: Arc<std::sync::atomic::AtomicBool>,
}

async fn preload_parties(rows: &[SharedRow], side: Eye) -> Result<Vec<Party>> {
    let url = std::env::var("CPU_CELL_TEST_DATABASE_URL")?;
    let mut parties = Vec::new();
    for party in 0..3 {
        let mut config: Config = serde_json::from_value(serde_json::json!({
            "startup_mode": "candidate", "party_id": party, "max_batch_size": 1,
            "disable_persistence": false, "full_scan_side_switching_enabled": false,
            "enable_s3_importer": false, "hawk_numa": false,
            "luc_enabled": true, "luc_lookback_records": 3,
            "schema_name": format!("SMPC_test_three_party_{}", Uuid::new_v4().simple()),
            "database": { "url": url, "load_parallelism": 1 }
        }))?;
        config.full_scan_side = side;
        let writer =
            PostgresClient::new(&url, &config.get_gpu_db_schema(), AccessMode::ReadWrite).await?;
        run_migrations(&writer.pool, false).await?;
        for row in rows {
            row.write(&writer, party).await?;
        }
        let candidate = super::super::candidate::preload(
            &config,
            Arc::new(ShutdownHandler::new(0)),
            Uuid::new_v4(),
        )
        .await?;
        parties.push(Party {
            config,
            writer,
            candidate,
        });
    }
    Ok(parties)
}

fn common_context() -> Result<ReconciliationContext> {
    ReconciliationContext::new(Uuid::new_v4(), Some(normalize_sns_sequence_number("42")?))
}

async fn coordinate_drain_and_reconcile_candidates(
    parties: &mut Vec<Party>,
    restart_candidate: bool,
) -> Result<Coordinated> {
    let (drains, tasks) = drain::tests::cohort(Duration::from_secs(30)).await;
    let (id, source, destination) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let candidates: [Uuid; 3] = std::array::from_fn(|i| parties[i].candidate.registry.process_id());
    let mut source_listeners = Vec::new();
    let mut candidate_listeners = Vec::new();
    let mut next_listeners = Vec::new();
    for _ in 0..3 {
        source_listeners.push(tokio::net::TcpListener::bind("127.0.0.1:0").await?);
        candidate_listeners.push(tokio::net::TcpListener::bind("127.0.0.1:0").await?);
        next_listeners.push(tokio::net::TcpListener::bind("127.0.0.1:0").await?);
    }
    let replacement_address = candidate_listeners[2].local_addr()?;
    let source_urls: [String; 3] = std::array::from_fn(|i| {
        format!(
            "http://{}/handoff-status",
            source_listeners[i].local_addr().unwrap()
        )
    });
    let candidate_urls: [String; 3] = std::array::from_fn(|i| {
        format!(
            "http://{}/handoff-status",
            candidate_listeners[i].local_addr().unwrap()
        )
    });
    let mut source_coordinators = Vec::new();
    let next_urls: [String; 3] = std::array::from_fn(|i| {
        format!(
            "http://{}/handoff-status",
            next_listeners[i].local_addr().unwrap()
        )
    });
    let mut candidate_coordinators = Vec::new();
    let mut candidate_configs = Vec::new();
    let mut source_tasks = Vec::new();
    let mut candidate_tasks = Vec::new();
    let mut candidate_shutdowns = Vec::new();
    for (i, ((party, source_listener), candidate_listener)) in parties
        .iter()
        .zip(source_listeners)
        .zip(candidate_listeners)
        .enumerate()
    {
        let mut config = party.config.clone();
        // Isolated test schemas represent the same logical database in each party.
        config.schema_name = "coordinator_fixture".into();
        config.startup_mode = iris_mpc_common::config::StartupMode::Active;
        config.db_backed_ingest = true;
        let mut peers: ampc_server_utils::config::ServerCoordinationConfig =
            serde_json::from_value(serde_json::json!({}))?;
        peers.party_id = i;
        peers.node_hostnames = vec!["127.0.0.1".into(); 3];
        peers.healthcheck_ports = source_urls
            .iter()
            .map(|url| {
                reqwest::Url::parse(url)
                    .unwrap()
                    .port()
                    .unwrap()
                    .to_string()
            })
            .collect();
        config.server_coordination = Some(peers);
        config.cpu_cell = Some(serde_json::from_value(
            serde_json::json!({"deployment_id": source,
                "peers": {"slot": "a", "a": serde_json::to_string(&next_urls)?, "b": serde_json::to_string(&candidate_urls)?}}),
        )?);
        let source_coordinator = Coordinator::new(
            &config,
            Uuid::parse_str(&drains[i].status().process_uuid)?,
            Some(drains[i].clone()),
        )?
        .unwrap();
        source_coordinator
            .mark_ready(Uuid::parse_str(&drains[i].status().process_uuid)?)
            .await;
        let router = source_coordinator.routes();
        source_tasks.push(tokio::spawn(async move {
            axum::serve(source_listener, router).await.unwrap();
        }));
        source_coordinators.push(source_coordinator);
        config.startup_mode = iris_mpc_common::config::StartupMode::Candidate;
        config.cpu_cell = Some(serde_json::from_value(
            serde_json::json!({"deployment_id": destination,
            "peers": {"slot": "b", "a": serde_json::to_string(&next_urls)?, "b": serde_json::to_string(&candidate_urls)?},
            "handoff": {"id": id, "source_deployment_id": source, "source_status_urls": serde_json::to_string(&source_urls)?}}),
        )?);
        let candidate_coordinator = Coordinator::new(&config, candidates[i], None)?.unwrap();
        let router = candidate_coordinator.routes();
        let stop = tokio_util::sync::CancellationToken::new();
        candidate_shutdowns.push(stop.clone());
        candidate_tasks.push(tokio::spawn(async move {
            axum::serve(candidate_listener, router)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        }));
        candidate_coordinators.push(candidate_coordinator);
        candidate_configs.push(config);
    }
    let mut controls = Vec::new();
    let mut source_handoffs = Vec::new();
    for (i, (party, drain)) in parties.iter().zip(&drains).enumerate() {
        let schema = party.config.get_gpu_db_schema();
        let control = PostgresClient::new(
            &party.config.database.as_ref().unwrap().url,
            &schema,
            AccessMode::ExistingSchema,
        )
        .await?;
        let handoffs = CpuCellHandoffs::new(control.pool.clone(), &schema)?;
        // Represent already committed intake; actual SQS/SNS workers are not run here.
        let store = Store::new(&party.writer).await?;
        let sequences = [
            normalize_sns_sequence_number("10")?,
            normalize_sns_sequence_number("42")?,
        ];
        for sequence in &sequences {
            store
                .insert_ingested_request(sequence, "settled fixture request", true)
                .await?;
        }
        let mut tx = store.tx().await?;
        store
            .mark_ingested_requests_persisted_tx(&mut tx, &sequences)
            .await?;
        tx.commit().await?;
        // Completed journal retention differs; pending work above C survives the handoff.
        sqlx::query("DELETE FROM ingested_requests WHERE persisted_at IS NOT NULL AND ($1 = 2 OR ($1 = 1 AND sequence_number = $2))")
            .bind(i as i32).bind(&sequences[0]).execute(&party.writer.pool).await?;
        store
            .insert_ingested_request(
                &normalize_sns_sequence_number("100")?,
                "pending fixture request",
                true,
            )
            .await?;
        candidate_coordinators[i]
            .prepare_candidate(&handoffs)
            .await?;
        handoffs.check_startup(destination, true).await?;
        drain.require_durable_seal(Arc::new(ManagedHandoff {
            handoffs: handoffs.clone(),
            deployment: source,
        }));
        source_handoffs.push(handoffs);
        if i < 2 {
            candidate_coordinators[i].mark_ready(candidates[i]).await;
        }
        drain.ingest_stopped();
        drain.receiver_stopped();
        assert!(drain.settled());
        if i == 0 {
            assert_eq!(drain.status().cut, None);
            assert!(!drain.ready_to_report());
        }
        controls.push(control);
    }
    super::super::handoff::tests::assert_sources_wait_for_preload_then_finish_drain(
        &source_coordinators,
        &source_handoffs,
        &candidate_coordinators[2],
        candidates[2],
    )
    .await?;
    drain::tests::until(|| drains.iter().all(|drain| drain.ready_to_report())).await;
    for (i, (party, drain)) in parties.iter().zip(&drains).enumerate() {
        assert_eq!(drain.status().cut, Some(0));
        let store = Store::new(&party.writer).await?;
        let watermark = store.drain_completed_ingested_sequence_number().await?;
        drain.publish_report(drain::Report {
            watermark,
            last_publication: None,
        });
        assert!(!drain.ready_to_report());
        if i == 0 {
            assert_eq!(drain.status().phase, Phase::Quiescent);
            assert!(drain.persist_completion().await.is_err());
        }
    }
    drain::tests::until(|| {
        drains
            .iter()
            .all(|drain| drain.status().phase == Phase::Paused)
    })
    .await;
    for (drain, control) in drains.iter().zip(controls) {
        drain.persist_completion().await?;
        control.pool.close().await;
    }
    let source_tasks = tasks.into_iter().chain(source_tasks).collect::<Vec<_>>();
    for task in &source_tasks {
        task.abort();
    }
    for task in source_tasks {
        assert!(task.await.unwrap_err().is_cancelled());
    }
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()?;
    for url in source_urls {
        assert!(client.get(url).send().await.is_err());
    }
    // Completion agreement consumes part of its budget; reconciliation gets a fresh one.
    let mut lock = parties[0].writer.pool.begin().await?;
    sqlx::query("LOCK TABLE irises IN ACCESS EXCLUSIVE MODE")
        .execute(&mut *lock)
        .await?;
    let started = Instant::now();
    let release_lock = async move {
        tokio::time::sleep(Duration::from_secs(9)).await;
        lock.commit().await
    };
    // A no longer serves any endpoint. B finishes using only durable records and current B peers.
    let reconcile = futures::future::try_join_all(
        parties
            .iter_mut()
            .zip(&candidate_coordinators)
            .enumerate()
            .map(|(i, (party, coordinator))| async move {
                let reader = PostgresClient::new(
                    &party.config.database.as_ref().unwrap().url,
                    &party.config.get_gpu_db_schema(),
                    AccessMode::EnforcedReadOnly,
                )
                .await?;
                let handoffs =
                    CpuCellHandoffs::new(reader.pool.clone(), &party.config.get_gpu_db_schema())?;
                let store = Store::new(&reader).await?;
                if i == 2 {
                    tokio::time::sleep(Duration::from_secs(4)).await;
                }
                let context = coordinator
                    .reconcile_candidate(
                        &handoffs,
                        &store,
                        &mut party.candidate,
                        Duration::from_secs(8),
                    )
                    .await?;
                let pending: bool = sqlx::query_scalar(
                    "SELECT persisted_at IS NULL FROM ingested_requests WHERE sequence_number=$1",
                )
                .bind(normalize_sns_sequence_number("100")?)
                .fetch_one(&reader.pool)
                .await?;
                assert!(pending, "handoff must preserve pending-above-C work");
                reader.pool.close().await;
                Ok::<_, eyre::Report>(context)
            }),
    );
    let (contexts, released) = tokio::join!(reconcile, release_lock);
    released?;
    let contexts = contexts?;
    assert!(started.elapsed() >= Duration::from_secs(9));
    assert!(contexts.iter().all(|context| context == &contexts[0]));
    if restart_candidate {
        let task = candidate_tasks.pop().unwrap();
        candidate_shutdowns.pop().unwrap().cancel();
        tokio::time::timeout(Duration::from_secs(3), task).await??;
        drop(candidate_coordinators.pop().unwrap());
        let Party {
            config,
            writer,
            candidate,
        } = parties.pop().unwrap();
        let old_process = candidate.registry.process_id();
        assert!(candidate.report_for(&contexts[2]).is_some());
        drop(candidate);

        let process = Uuid::new_v4();
        let coordinator = Coordinator::new(&candidate_configs[2], process, None)?.unwrap();
        let listener = tokio::net::TcpListener::bind(replacement_address).await?;
        let router = coordinator.routes();
        let stop = tokio_util::sync::CancellationToken::new();
        candidate_shutdowns.push(stop.clone());
        candidate_tasks.push(tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        }));
        let status: serde_json::Value = client.get(&candidate_urls[2]).send().await?.json().await?;
        assert_eq!(status["ready"], false);
        assert!(status["completion"].is_null());
        assert!(status["reconciliation"].is_null());

        // Full startup reloads this party while A is gone and B0/B1 retain their reports.
        let mut restart_config = candidate_configs[2].clone();
        restart_config.schema_name = config.schema_name.clone();
        let mut candidate = super::super::candidate::load_or_recover(
            &restart_config,
            Arc::new(ShutdownHandler::new(0)),
            process,
            Some(&coordinator),
        )
        .await?
        .expect("unactivated candidate preloads");
        assert_ne!(candidate.registry.process_id(), old_process);
        assert!(candidate.report_for(&contexts[2]).is_none());
        let reader = PostgresClient::new(
            &config.database.as_ref().unwrap().url,
            &config.get_gpu_db_schema(),
            AccessMode::EnforcedReadOnly,
        )
        .await?;
        let store = Store::new(&reader).await?;
        let handoffs = CpuCellHandoffs::new(reader.pool.clone(), &config.get_gpu_db_schema())?;
        coordinator.mark_ready(process).await;
        let context = coordinator
            .reconcile_candidate(&handoffs, &store, &mut candidate, Duration::from_secs(8))
            .await?;
        assert_eq!(context, contexts[2]);
        assert!(candidate.report_for(&context).is_some());
        let pending: bool = sqlx::query_scalar(
            "SELECT persisted_at IS NULL FROM ingested_requests WHERE sequence_number=$1",
        )
        .bind(normalize_sns_sequence_number("100")?)
        .fetch_one(&reader.pool)
        .await?;
        assert!(
            pending,
            "candidate restart must preserve pending-above-C work"
        );
        reader.pool.close().await;
        parties.push(Party {
            config,
            writer,
            candidate,
        });
        candidate_coordinators.push(coordinator);
    }
    let expected = ReconciliationContext::new(id, Some(normalize_sns_sequence_number("42")?))?;
    assert_eq!(contexts[0], expected);
    Ok(Coordinated {
        context: expected,
        coordinators: candidate_coordinators,
        tasks: candidate_tasks,
        shutdowns: candidate_shutdowns,
        configs: candidate_configs,
        next_listeners,
    })
}

async fn reconcile_parties(
    parties: &mut [Party],
    context: &ReconciliationContext,
) -> Result<[ReconciliationReport; 3]> {
    let mut reports = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    for party in parties {
        let reader = PostgresClient::new(
            &party.config.database.as_ref().unwrap().url,
            &party.config.get_gpu_db_schema(),
            AccessMode::EnforcedReadOnly,
        )
        .await?;
        let store = Store::new(&reader).await?;
        reports.push(
            party
                .candidate
                .reconcile(&store, context.clone(), deadline)
                .await?
                .clone(),
        );
        reader.pool.close().await;
    }
    Ok(reports.try_into().unwrap())
}

async fn cleanup(parties: Vec<Party>) -> Result<()> {
    for party in parties {
        drop(party.candidate);
        iris_mpc_store::test_utils::cleanup(&party.writer, &party.config.get_gpu_db_schema())
            .await?;
        party.writer.pool.close().await;
    }
    Ok(())
}

async fn promote_and_process(
    parties: Vec<Party>,
    coordinated: Coordinated,
    resident: usize,
) -> Result<Vec<Serving>> {
    use super::super::{promote_actor, serving_processes};
    use ampc_server_utils::{
        start_coordination_server_with_extra_routes, wait_for_others_ready,
        wait_for_startup_barriers, TaskMonitor,
    };
    use iris_mpc_common::{
        helpers::sync::SyncState,
        job::{BatchMetadata, BatchQuery, JobSubmissionHandle},
    };
    use iris_mpc_cpu::execution::{
        hawk_main::{HawkHandle, StoreId},
        local::get_free_local_addresses,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    let health = get_free_local_addresses(3).await?;
    let mpc = get_free_local_addresses(3).await?;
    let handoffs = parties
        .iter()
        .map(|party| {
            CpuCellHandoffs::new(party.writer.pool.clone(), &party.config.get_gpu_db_schema())
        })
        .collect::<Result<Vec<_>>>()?;
    super::super::handoff::tests::reject_changed_activation_evidence(
        &coordinated.coordinators,
        &handoffs,
    )
    .await?;
    let activated = Arc::new(AtomicUsize::new(0));
    let changed = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let processing = Arc::new(AtomicBool::new(false));
    let controls = parties
        .iter()
        .map(|party| party.writer.pool.clone())
        .collect::<Vec<_>>();
    let candidate_configs = coordinated.configs.clone();
    let release_last = {
        let activated = activated.clone();
        let changed = changed.clone();
        let release = release.clone();
        let processing = processing.clone();
        async move {
            while activated.load(Ordering::SeqCst) != 2 {
                changed.notified().await;
            }
            assert!(
                !processing.load(Ordering::SeqCst),
                "partial activation must not process work"
            );
            for (party, pool) in controls.iter().enumerate() {
                let phase: String = sqlx::query_scalar(
                    "SELECT phase FROM cpu_cell_handoffs ORDER BY sequence DESC LIMIT 1",
                )
                .fetch_one(pool)
                .await?;
                assert_eq!(phase, if party < 2 { "activated" } else { "drained" });
                let replacement =
                    Coordinator::new(&candidate_configs[party], Uuid::new_v4(), None)?.unwrap();
                let mode = replacement.prepare_candidate(&handoffs[party]).await?;
                use iris_mpc_common::config::StartupMode;
                if party < 2 {
                    assert_eq!(mode, StartupMode::Active);
                    assert!(
                        replacement
                            .wait_cold_cohort(Instant::now() + Duration::from_millis(150))
                            .await
                            .is_err(),
                        "a replacement cannot join a warm activation in flight"
                    );
                } else {
                    assert_eq!(mode, StartupMode::Candidate);
                }
                let pending: (Option<i64>, bool) = sqlx::query_as(
                    "SELECT consumed_batch_id, persisted_at IS NULL FROM ingested_requests WHERE sequence_number=$1",
                )
                .bind(normalize_sns_sequence_number("100")?)
                .fetch_one(pool)
                .await?;
                assert_eq!(pending, (None, true));
            }
            release.notify_one();
            Ok::<_, eyre::Report>(())
        }
    };
    let context = &coordinated.context;
    let promote = futures::future::try_join_all(
        parties
            .into_iter()
            .zip(coordinated.configs)
            .zip(&coordinated.coordinators)
            .enumerate()
            .map(|(party, ((loaded, mut config), coordinator))| {
                let activated = activated.clone();
                let changed = changed.clone();
                let release = release.clone();
                let processing = processing.clone();
                let health = health.clone();
                let mpc = mpc.clone();
                async move {
                    let Party {
                        config: physical_config,
                        writer,
                        candidate,
                    } = loaded;
                    let preload_process = candidate.registry.process_id();
                    let resident_pool = candidate.workers.pools[resident].clone();
                    let workers = candidate.into_workers(context)?;
                    let coord = config.server_coordination.as_mut().unwrap();
                    coord.node_hostnames = vec!["127.0.0.1".into(); 3];
                    coord.healthcheck_ports = health
                        .iter()
                        .map(|address| address.rsplit(':').next().unwrap().to_owned())
                        .collect();
                    config.service_ports = mpc
                        .iter()
                        .map(|address| address.rsplit(':').next().unwrap().parse())
                        .collect::<std::result::Result<_, _>>()?;
                    config.service_outbound_ports = config.service_ports.clone();
                    config.hawk_request_parallelism = 1;
                    config.hawk_connection_parallelism = 1;
                    config.startup_mode = iris_mpc_common::config::StartupMode::Active;
                    let coord_config = config.server_coordination.as_ref().unwrap();
                    let state = SyncState {
                        db_len: workers.registries[0].read().await.size as u64,
                        modifications: vec![],
                        next_sns_sequence_num: None,
                        common_config: iris_mpc_common::config::CommonConfig::from(config.clone()),
                        graph_mutation_bytes: vec![],
                        max_persisted_sequence_number: coordinator
                            .agreed_completion()
                            .await?
                            .watermark,
                        dh_nonce: None,
                    };
                    let shutdown = Arc::new(ShutdownHandler::new(0));
                    let mut monitor = TaskMonitor::new();
                    let drain = drain::Drain::new(party, Duration::from_secs(30), true);
                    let (ready, verified, process) = start_coordination_server_with_extra_routes(
                        coord_config,
                        &mut monitor,
                        &shutdown,
                        &state,
                        None,
                        Some(coordinator.routes().merge(drain.routes())),
                    )
                    .await;
                    let process_id = Uuid::parse_str(&process)?;
                    assert_ne!(preload_process, process_id);
                    drain.initialize(process.clone(), ready.clone());
                    let coordinator = coordinator
                        .bind_serving(&config, process_id, drain.clone())
                        .await?;
                    wait_for_startup_barriers(coord_config, &verified, &process).await?;
                    let actor = promote_actor(&config, workers, &shutdown).await?;
                    assert!(Arc::ptr_eq(
                        &resident_pool,
                        &actor.worker_pool(if resident == 0 {
                            StoreId::Left
                        } else {
                            StoreId::Right
                        })
                    ));
                    let mut handle = HawkHandle::new(actor).await?;
                    let handoff = CpuCellHandoffs::new(
                        writer.pool.clone(),
                        &physical_config.get_gpu_db_schema(),
                    )?;
                    drain.require_durable_seal(Arc::new(ManagedHandoff {
                        handoffs: handoff.clone(),
                        deployment: Uuid::parse_str(
                            &config.cpu_cell.as_ref().unwrap().deployment_id,
                        )?,
                    }));
                    if party == 2 {
                        release.notified().await;
                    }
                    coordinator.activate_candidate(&handoff).await?;
                    if party < 2 {
                        activated.fetch_add(1, Ordering::SeqCst);
                        changed.notify_one();
                    }
                    let verified = verified.lock().await.clone();
                    let expected = serving_processes(coord_config, &process, &verified).await?;
                    let deadline = Instant::now() + Duration::from_secs(20);
                    coordinator
                        .serving_barrier(&expected, context, deadline)
                        .await?;
                    drain.mark_ready();
                    wait_for_others_ready(coord_config, verified.clone()).await?;
                    assert_eq!(
                        serving_processes(coord_config, &process, &verified).await?,
                        expected
                    );
                    coordinator.mark_ready(process_id).await;
                    let drain_task = drain.clone();
                    let drain_config = coord_config.clone();
                    let cohort = expected.iter().map(ToString::to_string).collect();
                    monitor.spawn(async move { drain_task.coordinate(drain_config, cohort).await });
                    assert!(ready.load(Ordering::SeqCst));
                    assert!(
                        handoff
                            .check_startup(handoff.latest().await?.unwrap().destination, true)
                            .await
                            .is_err(),
                        "an activated destination requires coordinated recovery"
                    );
                    let peers = config.cpu_cell.as_ref().unwrap().peers.as_ref().unwrap();
                    let own_urls = match peers.slot {
                        CpuCellSlot::A => &peers.a,
                        CpuCellSlot::B => &peers.b,
                    };
                    let response: serde_json::Value = reqwest::Client::new()
                        .get(&own_urls.0[party])
                        .send()
                        .await?
                        .json()
                        .await?;
                    assert_eq!(response["process"], process_id.to_string());
                    assert_eq!(response["mode"], "active");
                    assert_eq!(response["drain"]["process_uuid"], process);
                    processing.store(true, Ordering::SeqCst);
                    let mut batch = BatchQuery::default();
                    batch.push_deletion_request(
                        "first-after-handoff".into(),
                        0,
                        BatchMetadata::default(),
                    );
                    let result = handle.submit_batch_query(batch).await.await?;
                    assert_eq!(result.deleted_ids, vec![0]);
                    assert!(handoff
                        .check_startup(handoff.latest().await?.unwrap().source, false)
                        .await
                        .is_err());
                    Ok::<_, eyre::Report>(Serving {
                        writer,
                        physical_config,
                        config,
                        handle,
                        shutdown,
                        monitor,
                        coordinator,
                        drain,
                        ready,
                    })
                }
            }),
    );
    let (promoted, released) = tokio::time::timeout(Duration::from_secs(40), async {
        tokio::join!(promote, release_last)
    })
    .await?;
    released?;
    let promoted = promoted?;
    super::super::handoff::tests::reject_changed_serving_evidence(
        &coordinated.coordinators,
        context,
    )
    .await?;
    for (task, stop) in coordinated.tasks.into_iter().zip(coordinated.shutdowns) {
        stop.cancel();
        tokio::time::timeout(Duration::from_secs(3), task).await??;
    }
    Ok(promoted)
}

// Restart the promoted cohort with its original candidate manifest and durable databases.
async fn restart_activated_slot(serving: Vec<Serving>) -> Result<()> {
    let mut listeners = tokio::task::JoinSet::new();
    let mut restarts = Vec::new();
    for party in serving {
        drop(party.handle);
        party.shutdown.get_network_cancellation_token().cancel();
        drop(party.monitor);
        let mut config = party.config;
        config.startup_mode = iris_mpc_common::config::StartupMode::Candidate;
        let peers = config.cpu_cell.as_ref().unwrap().peers.as_ref().unwrap();
        let urls = match peers.slot {
            CpuCellSlot::A => &peers.a,
            CpuCellSlot::B => &peers.b,
        };
        let url = reqwest::Url::parse(&urls.0[config.party_id])?;
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", url.port().unwrap())).await?;
        let process = Uuid::new_v4();
        let coordinator = Coordinator::new(&config, process, None)?.unwrap();
        let routes = coordinator.routes();
        listeners.spawn(async move { axum::serve(listener, routes).await });
        config.schema_name = party.physical_config.schema_name.clone();
        // The successor has requested a handoff on one party, but nobody has sealed.
        if config.party_id == 0 {
            let handoffs =
                CpuCellHandoffs::new(party.writer.pool.clone(), &config.get_gpu_db_schema())?;
            let owner = handoffs.latest().await?.unwrap().destination;
            handoffs
                .request(Uuid::new_v4(), owner, Uuid::new_v4())
                .await?;
        }
        restarts.push(async move {
            let outcome = super::super::candidate::load_or_recover(
                &config,
                Arc::new(ShutdownHandler::new(0)),
                process,
                Some(&coordinator),
            )
            .await?;
            assert!(
                outcome.is_none(),
                "an activated owner must use a full cold start"
            );
            iris_mpc_store::test_utils::cleanup(&party.writer, &config.get_gpu_db_schema()).await?;
            party.writer.pool.close().await;
            Ok::<_, eyre::Report>(())
        });
    }
    tokio::time::timeout(
        Duration::from_secs(15),
        futures::future::try_join_all(restarts),
    )
    .await??;
    Ok(())
}

async fn handoff_to_next_slot(
    mut serving: Vec<Serving>,
    listeners: Vec<tokio::net::TcpListener>,
) -> Result<(Vec<Party>, Coordinated)> {
    use iris_mpc_common::job::{BatchMetadata, BatchQuery, JobSubmissionHandle};
    use std::sync::atomic::Ordering;

    let source = Uuid::parse_str(&serving[0].config.cpu_cell.as_ref().unwrap().deployment_id)?;
    let (id, destination) = (Uuid::new_v4(), Uuid::new_v4());
    let peers = serving[0].config.server_coordination.as_ref().unwrap();
    let source_urls: [String; 3] = std::array::from_fn(|i| {
        format!(
            "http://{}:{}/handoff-status",
            peers.node_hostnames[i], peers.healthcheck_ports[i]
        )
    });
    let mut cell_peers = serving[0]
        .config
        .cpu_cell
        .as_ref()
        .unwrap()
        .peers
        .clone()
        .unwrap();
    assert_eq!(cell_peers.slot, CpuCellSlot::B);
    cell_peers.slot = CpuCellSlot::A;
    let mut coordinated = Coordinated {
        context: ReconciliationContext::new(id, Some(normalize_sns_sequence_number("42")?))?,
        coordinators: Vec::new(),
        tasks: Vec::new(),
        shutdowns: Vec::new(),
        configs: Vec::new(),
        next_listeners: Vec::new(),
    };
    let mut handoffs = Vec::new();
    let mut parties = Vec::new();
    let sources = serving
        .iter()
        .map(|party| party.coordinator.clone())
        .collect::<Vec<_>>();
    let processes = serving
        .iter()
        .map(|party| party.drain.status().process_uuid)
        .collect::<Vec<_>>();
    for (i, (party, listener)) in serving.iter().zip(listeners).enumerate() {
        let mut config = party.config.clone();
        config.startup_mode = iris_mpc_common::config::StartupMode::Candidate;
        config.cpu_cell = Some(serde_json::from_value(serde_json::json!({
            "deployment_id": destination,
            "peers": cell_peers,
            "handoff": {"id": id, "source_deployment_id": source,
                "source_status_urls": serde_json::to_string(&source_urls)?}
        }))?);
        let coordinator = Coordinator::new(&config, Uuid::new_v4(), None)?.unwrap();
        let router = coordinator.routes();
        let stop = tokio_util::sync::CancellationToken::new();
        coordinated.shutdowns.push(stop.clone());
        coordinated.tasks.push(tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(stop.cancelled_owned())
                .await
                .unwrap();
        }));
        let handoff = CpuCellHandoffs::new(
            party.writer.pool.clone(),
            &party.physical_config.get_gpu_db_schema(),
        )?;
        coordinator.prepare_candidate(&handoff).await?;
        handoff.check_startup(destination, true).await?;
        if i < 2 {
            let mut physical_config = config.clone();
            physical_config.schema_name = party.physical_config.schema_name.clone();
            let candidate = super::super::candidate::preload(
                &physical_config,
                Arc::new(ShutdownHandler::new(0)),
                Uuid::new_v4(),
            )
            .await?;
            coordinator
                .mark_ready(candidate.registry.process_id())
                .await;
            parties.push(Party {
                config: physical_config,
                writer: party.writer.clone(),
                candidate,
            });
        }
        coordinated.coordinators.push(coordinator);
        coordinated.configs.push(config);
        handoffs.push(handoff);
    }
    super::super::handoff::tests::reject_incompatible_next_candidate(
        &sources,
        &handoffs,
        &coordinated.coordinators[2],
    )
    .await?;
    // B still processes a real MPC request while the last next-A process has not preloaded.
    futures::future::try_join_all(serving.iter_mut().enumerate().map(|(i, party)| {
        let process = &processes[i];
        async move {
            assert!(party.ready.load(Ordering::SeqCst));
            assert_eq!(&party.drain.status().process_uuid, process);
            let mut batch = BatchQuery::default();
            batch.push_deletion_request(
                "while-next-candidate-loads".into(),
                0,
                BatchMetadata::default(),
            );
            assert_eq!(
                party
                    .handle
                    .submit_batch_query(batch)
                    .await
                    .await?
                    .deleted_ids,
                vec![0]
            );
            Ok::<_, eyre::Report>(())
        }
    }))
    .await?;
    let last = &serving[2];
    let mut physical_config = coordinated.configs[2].clone();
    physical_config.schema_name = last.physical_config.schema_name.clone();
    let candidate = super::super::candidate::preload(
        &physical_config,
        Arc::new(ShutdownHandler::new(0)),
        Uuid::new_v4(),
    )
    .await?;
    let last_process = candidate.registry.process_id();
    parties.push(Party {
        config: physical_config,
        writer: last.writer.clone(),
        candidate,
    });
    super::super::handoff::tests::assert_sources_wait_for_preload_then_finish_drain(
        &sources,
        &handoffs,
        &coordinated.coordinators[2],
        last_process,
    )
    .await?;
    for (party, process) in serving.iter().zip(&processes) {
        assert_eq!(&party.drain.status().process_uuid, process);
        assert!(!party.ready.load(Ordering::SeqCst));
        party.drain.ingest_stopped();
        party.drain.receiver_stopped();
        assert!(party.drain.settled());
    }
    drain::tests::until(|| serving.iter().all(|party| party.drain.ready_to_report())).await;
    for party in &serving {
        party.drain.publish_report(drain::Report {
            watermark: Store::new(&party.writer)
                .await?
                .drain_completed_ingested_sequence_number()
                .await?,
            last_publication: None,
        });
    }
    drain::tests::until(|| {
        serving
            .iter()
            .all(|party| party.drain.status().phase == Phase::Paused)
    })
    .await;
    for (party, handoff) in serving.into_iter().zip(&handoffs) {
        party.drain.persist_completion().await?;
        let record = handoff.latest().await?.unwrap();
        assert_eq!(record.handoff_id, id);
        assert_eq!(record.source, source);
        assert_eq!(record.destination, destination);
        assert_eq!(
            record.phase,
            iris_mpc_store::cpu_cell::HandoffPhase::Drained
        );
        let mut original_manifest = party.config.clone();
        original_manifest.startup_mode = iris_mpc_common::config::StartupMode::Candidate;
        let old_owner = Coordinator::new(&original_manifest, Uuid::new_v4(), None)?.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(3), old_owner.prepare_candidate(handoff))
                .await?
                .is_err()
        );
        drop(party.handle);
        party.shutdown.get_network_cancellation_token().cancel();
        drop(party.monitor);
    }
    let contexts = futures::future::try_join_all(
        parties
            .iter_mut()
            .zip(&coordinated.coordinators)
            .zip(&handoffs)
            .map(|((party, coordinator), handoff)| async move {
                let store = Store::new(&party.writer).await?;
                coordinator
                    .reconcile_candidate(
                        handoff,
                        &store,
                        &mut party.candidate,
                        Duration::from_secs(10),
                    )
                    .await
            }),
    )
    .await?;
    assert!(contexts
        .iter()
        .all(|context| context == &coordinated.context));
    Ok((parties, coordinated))
}

#[tokio::test]
async fn three_party_reconciliation_agrees_and_preserves_each_partys_payloads() -> Result<()> {
    // Sparse left-eye state covers removals; dense right-eye state permits a full restart.
    for (side, resident, restart_candidate) in [(Eye::Left, 0, false), (Eye::Right, 1, true)] {
        let mut rng = StdRng::seed_from_u64(4374);
        let mut expected = (1..=6)
            .map(|id| SharedRow::generate(&mut rng, id, 1))
            .collect::<Vec<_>>();
        let mut parties = preload_parties(&expected, side).await?;
        let resident_pools = parties
            .iter()
            .map(|party| party.candidate.workers.pools[resident].clone())
            .collect::<Vec<_>>();
        for (party_id, party) in parties.iter().enumerate() {
            assert_eq!(party.candidate.workers.passive_resident_side()?, resident);
            let loaded = resident_pools[party_id]
                .fetch_irises(vec![expected[0].id])
                .await?;
            assert_eq!(loaded[0].as_ref(), &expected[0].eyes[resident][party_id]);
        }

        // Simulate A's committed writes after every B candidate has preloaded.
        for (serial, version) in [(1, 3), (4, 9), (5, 2)] {
            let row = SharedRow::generate(&mut rng, serial, version);
            for (party_id, party) in parties.iter().enumerate() {
                row.write(&party.writer, party_id).await?;
            }
            expected[serial as usize - 1] = row;
        }
        let inserted = SharedRow::generate(&mut rng, 7, 0);
        for (party_id, party) in parties.iter().enumerate() {
            if !restart_candidate {
                sqlx::query("DELETE FROM irises WHERE id IN (2, 6)")
                    .execute(&party.writer.pool)
                    .await?;
            }
            inserted.write(&party.writer, party_id).await?;
            for registry in &party.candidate.workers.registries {
                let registry = registry.read().await;
                assert_eq!(registry.get_current_version(1), Some(1));
                assert_eq!(registry.get_current_version(6), Some(1));
                assert_eq!(registry.get_current_version(7), None);
            }
        }
        if !restart_candidate {
            expected.retain(|row| ![2, 6].contains(&row.id.serial_id()));
        }
        expected.push(inserted);

        // Queue intake/publication are outside this fixture; its simulated writers stop here.
        let mut coordinated =
            coordinate_drain_and_reconcile_candidates(&mut parties, restart_candidate).await?;
        let context = coordinated.context.clone();
        let reports: [ReconciliationReport; 3] =
            std::array::from_fn(|i| parties[i].candidate.report_for(&context).unwrap().clone());
        ReconciliationReport::verify_agreement(&context, &reports)?;
        for (party_id, party) in parties.iter().enumerate() {
            let candidate = &party.candidate;
            assert_eq!(candidate.report_for(&context), Some(&reports[party_id]));
            assert_eq!(reports[party_id].registry_count, expected.len() as u64);
            assert_eq!(
                Arc::ptr_eq(
                    &resident_pools[party_id],
                    &candidate.workers.pools[resident]
                ),
                !restart_candidate || party_id != 2,
            );
            for registry in &candidate.workers.registries {
                let registry = registry.read().await;
                assert_eq!(registry.size, expected.len());
                assert_eq!(registry.next_id, 8);
                for row in &expected {
                    assert_eq!(
                        registry.get_current_version(row.id.serial_id()),
                        Some(row.id.version_id())
                    );
                }
                if !restart_candidate {
                    for removed in [2, 6] {
                        assert_eq!(registry.get_current_version(removed), None);
                    }
                }
            }
            // Rows outside the cold LUC exercise database reads.
            for (eye, pool) in candidate.workers.pools.iter().enumerate() {
                let fetched = pool
                    .fetch_irises(expected.iter().map(|row| row.id).collect())
                    .await?;
                assert_eq!(fetched.len(), expected.len());
                for (actual, row) in fetched.iter().zip(&expected) {
                    assert_eq!(actual.as_ref(), &row.eyes[eye][party_id]);
                }
            }
            let mut stale_ids = vec![
                VectorId::new(1, 1),
                VectorId::new(4, 1),
                VectorId::new(5, 1),
            ];
            if !restart_candidate {
                stale_ids.extend([VectorId::new(2, 1), VectorId::new(6, 1)]);
            }
            for stale in stale_ids {
                assert!(candidate.workers.pools[1 - resident]
                    .fetch_irises(vec![stale])
                    .await
                    .is_err());
            }
            if !restart_candidate {
                let empty = resident_pools[party_id]
                    .fetch_irises(vec![VectorId::new(8, 0)])
                    .await?;
                for removed in [2, 6] {
                    let fetched = resident_pools[party_id]
                        .fetch_irises(vec![VectorId::new(removed, 1)])
                        .await?;
                    assert_eq!(fetched[0], empty[0]);
                }
            }
        }
        let next_listeners = std::mem::take(&mut coordinated.next_listeners);
        let mut serving = promote_and_process(parties, coordinated, resident).await?;
        if restart_candidate {
            let (parties, coordinated) = handoff_to_next_slot(serving, next_listeners).await?;
            serving = promote_and_process(parties, coordinated, resident).await?;
            restart_activated_slot(serving).await?;
            continue;
        }
        for party in serving {
            drop(party.handle);
            party.shutdown.get_network_cancellation_token().cancel();
            drop(party.monitor);
            iris_mpc_store::test_utils::cleanup(
                &party.writer,
                &party.physical_config.get_gpu_db_schema(),
            )
            .await?;
            party.writer.pool.close().await;
        }
    }
    Ok(())
}

#[tokio::test]
async fn three_party_reconciliation_rejects_real_divergent_metadata() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(4375);
    let rows = (1..=3)
        .map(|id| SharedRow::generate(&mut rng, id, 1))
        .collect::<Vec<_>>();
    let mut parties = preload_parties(&rows, Eye::Left).await?;
    // A fresh cohort observes equal row counts and maxima but different versions.
    SharedRow::generate(&mut rng, 1, 2)
        .write(&parties[2].writer, 2)
        .await?;
    let context = common_context()?;
    let reports = reconcile_parties(&mut parties, &context).await?;
    assert!(reports.iter().all(|report| report.registry_count == 3));
    assert_eq!(reports[0].registry_digest, reports[1].registry_digest);
    assert_ne!(reports[0].registry_digest, reports[2].registry_digest);
    for (party_id, party) in parties.iter().enumerate() {
        assert_eq!(
            party.candidate.report_for(&context),
            Some(&reports[party_id])
        );
        for registry in &party.candidate.workers.registries {
            let registry = registry.read().await;
            assert_eq!(registry.next_id, 4);
            assert_eq!(
                registry.get_current_version(1),
                Some(if party_id == 2 { 2 } else { 1 })
            );
        }
    }
    let error = ReconciliationReport::verify_agreement(&context, &reports).unwrap_err();
    assert!(error
        .to_string()
        .contains("candidate registry metadata differs across parties"));
    cleanup(parties).await?;
    Ok(())
}
