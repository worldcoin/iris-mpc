use super::super::{
    drain::{self, Phase},
    ManagedHandoff,
};
use super::*;
use ampc_anon_stats::types::Eye;
use ampc_server_utils::shutdown_handler::ShutdownHandler;
use iris_mpc_common::{
    config::Config,
    iris_db::iris::IrisCode,
    postgres::{check_handoff_startup, run_migrations, AccessMode, PostgresClient},
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

fn common_context(parties: &[Party]) -> Result<ReconciliationContext> {
    ReconciliationContext::new(
        Uuid::new_v4(),
        std::array::from_fn(|_| Uuid::new_v4()),
        std::array::from_fn(|party| parties[party].candidate.registry.process_id()),
        Some(normalize_sns_sequence_number("42")?),
    )
}

async fn drain_and_reload_handoff(parties: &[Party]) -> Result<ReconciliationContext> {
    let (drains, tasks) = drain::tests::cohort(Duration::from_secs(30)).await;
    let (id, source, destination) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let candidates = std::array::from_fn(|i| parties[i].candidate.registry.process_id());
    let mut controls = Vec::new();
    for (i, (party, drain)) in parties.iter().zip(&drains).enumerate() {
        let schema = party.config.get_gpu_db_schema();
        let control =
            PostgresClient::new_control(&party.config.database.as_ref().unwrap().url, &schema)
                .await?;
        let handoffs = CpuCellHandoffs::new(control.pool.clone(), &schema)?;
        // Represent already committed intake; actual SQS/SNS workers are not run here.
        for sequence in ["10", "42"] {
            sqlx::query("INSERT INTO ingested_requests (sequence_number, message_body, is_valid, persisted_at) VALUES ($1, 'settled fixture request', true, now())")
                .bind(normalize_sns_sequence_number(sequence)?).execute(&party.writer.pool).await?;
        }
        handoffs.request(id, source, destination).await?;
        handoffs.request(id, source, destination).await?;
        drain.require_durable_seal(Arc::new(ManagedHandoff {
            handoffs,
            deployment: source,
        }));
        drain.request_managed(id, &candidates).await?;
        drain.request_managed(id, &candidates).await?;
        drain.ingest_stopped();
        drain.receiver_stopped();
        assert!(drain.settled());
        if i == 0 {
            assert_eq!(drain.status().cut, None);
            assert!(!drain.ready_to_report());
        }
        controls.push(control);
    }
    drain::tests::until(|| drains.iter().all(|drain| drain.ready_to_report())).await;
    for (i, (party, drain)) in parties.iter().zip(&drains).enumerate() {
        assert_eq!(drain.status().cut, Some(0));
        let store = Store::new(&party.writer).await?;
        let watermark = store.max_persisted_sequence_number().await?;
        let (persisted_count, persisted_sha256) =
            store.prove_persisted_prefix(watermark.as_deref()).await?;
        drain.publish_report(drain::Report {
            watermark,
            persisted_count,
            persisted_sha256,
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
        let reopened =
            PostgresClient::new_control(&party.config.database.as_ref().unwrap().url, &schema)
                .await?;
        let saved = CpuCellHandoffs::new(reopened.pool.clone(), &schema)?
            .get(id)
            .await?;
        assert_eq!(saved, completed);
        assert_eq!(saved.phase, HandoffPhase::Drained);
        assert_eq!(
            saved.source_process,
            Some(Uuid::parse_str(&drain.status().process_uuid)?)
        );
        assert_eq!(saved.candidate_processes, Some(candidates.to_vec()));
        assert!(
            check_handoff_startup(&reopened.pool, &schema, &source.to_string(), false)
                .await
                .is_err()
        );
        let report: drain::Report = serde_json::from_value(saved.drain_report.clone().unwrap())?;
        assert_eq!(drain.status().report.as_ref(), Some(&report));
        records.push(saved);
        reports.push(report);
        reopened.pool.close().await;
    }
    for task in tasks {
        task.abort();
    }
    assert!(reports.iter().all(|report| report == &reports[0]));
    assert_eq!(reports[0].persisted_count, 2);
    ReconciliationContext::new(
        records[0].handoff_id,
        std::array::from_fn(|i| records[i].source_process.unwrap()),
        records[0]
            .candidate_processes
            .clone()
            .unwrap()
            .try_into()
            .unwrap(),
        reports[0].watermark.clone(),
    )
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
    for (side, resident) in [(Eye::Left, 0), (Eye::Right, 1)] {
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
            sqlx::query("DELETE FROM irises WHERE id IN (2, 6)")
                .execute(&party.writer.pool)
                .await?;
            inserted.write(&party.writer, party_id).await?;
            for registry in &party.candidate.workers.registries {
                let registry = registry.read().await;
                assert_eq!(registry.get_current_version(1), Some(1));
                assert_eq!(registry.get_current_version(6), Some(1));
                assert_eq!(registry.get_current_version(7), None);
            }
        }
        expected.retain(|row| ![2, 6].contains(&row.id.serial_id()));
        expected.push(inserted);

        // Queue intake/publication are outside this fixture; its simulated writers stop here.
        let context = drain_and_reload_handoff(&parties).await?;
        let reports = reconcile_parties(&mut parties, &context).await?;
        ReconciliationReport::verify_agreement(&context, &reports)?;
        for (party_id, party) in parties.iter().enumerate() {
            let candidate = &party.candidate;
            assert_eq!(candidate.report_for(&context), Some(&reports[party_id]));
            assert_eq!(reports[party_id].registry_count, expected.len() as u64);
            assert!(Arc::ptr_eq(
                &resident_pools[party_id],
                &candidate.workers.pools[resident]
            ));
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
                for removed in [2, 6] {
                    assert_eq!(registry.get_current_version(removed), None);
                }
            }
            // The cold LUC contains 5 and 7; older rows exercise sparse DB reads.
            for (eye, pool) in candidate.workers.pools.iter().enumerate() {
                let fetched = pool
                    .fetch_irises(expected.iter().map(|row| row.id).collect())
                    .await?;
                assert_eq!(fetched.len(), expected.len());
                for (actual, row) in fetched.iter().zip(&expected) {
                    assert_eq!(actual.as_ref(), &row.eyes[eye][party_id]);
                }
            }
            for stale in [
                VectorId::new(1, 1),
                VectorId::new(4, 1),
                VectorId::new(5, 1),
                VectorId::new(2, 1),
                VectorId::new(6, 1),
            ] {
                assert!(candidate.workers.pools[1 - resident]
                    .fetch_irises(vec![stale])
                    .await
                    .is_err());
            }
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
    let context = common_context(&parties)?;
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
