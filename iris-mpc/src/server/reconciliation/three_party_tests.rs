use super::super::{
    drain::{self, Phase},
    handoff::Coordinator,
    ManagedHandoff,
};
use super::*;
use ampc_anon_stats::types::Eye;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use iris_mpc_common::{
    config::Config,
    iris_db::iris::IrisCode,
    postgres::{run_migrations, AccessMode, PostgresClient},
    VectorId,
};
use iris_mpc_store::cpu_cell::{CpuCellHandoffs, HandoffPhase};
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
) -> Result<ReconciliationContext> {
    let (drains, tasks) = drain::tests::cohort(Duration::from_secs(30)).await;
    let (id, source, destination) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let candidates: [Uuid; 3] = std::array::from_fn(|i| parties[i].candidate.registry.process_id());
    let mut source_listeners = Vec::new();
    let mut candidate_listeners = Vec::new();
    for _ in 0..3 {
        source_listeners.push(tokio::net::TcpListener::bind("127.0.0.1:0").await?);
        candidate_listeners.push(tokio::net::TcpListener::bind("127.0.0.1:0").await?);
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
    let mut candidate_coordinators = Vec::new();
    let mut candidate_configs = Vec::new();
    let mut source_tasks = Vec::new();
    let mut candidate_tasks = Vec::new();
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
            serde_json::json!({"deployment_id": source, "candidate_status_urls": serde_json::to_string(&candidate_urls)?}),
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
            serde_json::json!({"deployment_id": destination, "candidate_status_urls": serde_json::to_string(&candidate_urls)?,
            "handoff": {"id": id, "source_deployment_id": source, "source_status_urls": serde_json::to_string(&source_urls)?}}),
        )?);
        let candidate_coordinator = Coordinator::new(&config, candidates[i], None)?.unwrap();
        let router = candidate_coordinator.routes();
        candidate_tasks.push(tokio::spawn(async move {
            axum::serve(candidate_listener, router).await.unwrap();
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
    super::super::handoff::tests::assert_sources_require_error_free_acknowledgements(
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
    let mut records = Vec::new();
    let mut reports = Vec::new();
    for ((party, drain), control) in parties.iter().zip(&drains).zip(controls) {
        drain.persist_completion().await?;
        let schema = party.config.get_gpu_db_schema();
        let handoffs = CpuCellHandoffs::new(control.pool.clone(), &schema)?;
        let completed = handoffs.get(id).await?;
        drain.persist_completion().await?;
        assert_eq!(handoffs.get(id).await?, completed);
        control.pool.close().await;
        let reopened = PostgresClient::new(
            &party.config.database.as_ref().unwrap().url,
            &schema,
            AccessMode::ExistingSchema,
        )
        .await?;
        let reopened_handoffs = CpuCellHandoffs::new(reopened.pool.clone(), &schema)?;
        let saved = reopened_handoffs.get(id).await?;
        assert_eq!(saved, completed);
        assert_eq!(saved.phase, HandoffPhase::Drained);
        assert_eq!(
            saved.source_process,
            Some(Uuid::parse_str(&drain.status().process_uuid)?)
        );
        assert!(reopened_handoffs
            .check_startup(source, false)
            .await
            .is_err());
        let report: drain::Report = serde_json::from_value(saved.drain_report.clone().unwrap())?;
        assert_eq!(drain.status().report.as_ref(), Some(&report));
        records.push(saved);
        reports.push(report);
        reopened.pool.close().await;
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
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
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
        candidate_tasks.push(tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        }));
        let status: serde_json::Value = client.get(&candidate_urls[2]).send().await?.json().await?;
        assert_eq!(status["ready"], false);
        assert!(status["completion"].is_null());
        assert!(status["reconciliation"].is_null());

        // Full startup reloads this party while A is gone and B0/B1 retain their reports.
        let mut restart_config = candidate_configs[2].clone();
        restart_config.schema_name = config.schema_name.clone();
        let mut candidate = super::super::candidate::preload_with_coordinator(
            &restart_config,
            Arc::new(ShutdownHandler::new(0)),
            process,
            Some(&coordinator),
        )
        .await?;
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
        assert_eq!(
            candidate.report_for(&context).unwrap().process_id(),
            process
        );
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
    }
    for task in &candidate_tasks {
        task.abort();
    }
    for task in candidate_tasks {
        assert!(task.await.unwrap_err().is_cancelled());
    }
    assert!(reports.iter().all(|report| report == &reports[0]));
    assert_eq!(
        reports[0].watermark,
        Some(normalize_sns_sequence_number("42")?)
    );
    let expected = ReconciliationContext::new(records[0].handoff_id, reports[0].watermark.clone())?;
    assert_eq!(contexts[0], expected);
    Ok(expected)
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

#[tokio::test]
async fn three_party_reconciliation_agrees_and_preserves_each_partys_payloads() -> Result<()> {
    for (side, resident, restart_candidate) in [
        (Eye::Left, 0, false),
        (Eye::Right, 1, false),
        (Eye::Left, 0, true),
    ] {
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
        let context =
            coordinate_drain_and_reconcile_candidates(&mut parties, restart_candidate).await?;
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
        cleanup(parties).await?;
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
