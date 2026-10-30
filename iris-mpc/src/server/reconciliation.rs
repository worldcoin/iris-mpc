//! Passive reconciliation evidence. Ownership and authenticated peer exchange are
//! enforced by the handoff coordinator, not by this local primitive.

use super::registry_diff::{CandidateRegistry, RegistryChange};
use eyre::{ensure, Result, WrapErr};
use iris_mpc_common::VectorId;
use iris_mpc_cpu::execution::hawk_main::worker_pool_initializer::InitializedWorkers;
use iris_mpc_cpu::protocol::shared_iris::{ArcIris, GaloisRingSharedIris};
use iris_mpc_store::{normalize_sns_sequence_number, Store};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::{timeout_at, Instant};
use uuid::Uuid;

// Bound transient share payloads independently of total population or delta size.
const APPLY_BATCH: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationContext {
    attempt_id: Uuid,
    source_processes: [Uuid; 3],
    candidate_processes: [Uuid; 3],
    watermark: Option<String>,
}

impl ReconciliationContext {
    pub fn new(
        attempt_id: Uuid,
        source_processes: [Uuid; 3],
        candidate_processes: [Uuid; 3],
        watermark: Option<String>,
    ) -> Result<Self> {
        let context = Self {
            attempt_id,
            source_processes,
            candidate_processes,
            watermark,
        };
        context.validate()?;
        Ok(context)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.attempt_id.is_nil(),
            "reconciliation attempt must not be nil"
        );
        let processes = self
            .source_processes
            .iter()
            .chain(&self.candidate_processes)
            .collect::<Vec<_>>();
        for (index, id) in processes.iter().enumerate() {
            ensure!(
                !id.is_nil() && !processes[..index].contains(id),
                "reconciliation requires distinct non-nil process identities"
            );
        }
        if let Some(watermark) = &self.watermark {
            ensure!(
                normalize_sns_sequence_number(watermark)? == *watermark,
                "reconciliation watermark must be normalized"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationReport {
    context: ReconciliationContext,
    party_id: usize,
    registry_count: u64,
    registry_digest: [u8; 32],
}

impl ReconciliationReport {
    /// Reports must come from authenticated parties and be indexed by party.
    /// Agreement is evidence for this frozen context, never permission to serve.
    pub fn verify_agreement(expected: &ReconciliationContext, reports: &[Self; 3]) -> Result<()> {
        expected.validate()?;
        for (party, report) in reports.iter().enumerate() {
            ensure!(
                report.party_id == party && &report.context == expected,
                "stale or incorrectly bound reconciliation report"
            );
            ensure!(
                report.registry_count == reports[0].registry_count
                    && report.registry_digest == reports[0].registry_digest,
                "candidate registry metadata differs across parties"
            );
        }
        Ok(())
    }
}

enum State {
    Unreconciled,
    // Set before the first await: dropping the future cannot preserve readiness.
    Failed,
    Complete(ReconciliationReport),
}

pub struct CandidateReconciliation {
    pub(super) workers: InitializedWorkers,
    pub(super) registry: CandidateRegistry,
    party_id: usize,
    state: State,
}

impl CandidateReconciliation {
    pub(crate) fn new(
        party_id: usize,
        process_id: Uuid,
        workers: InitializedWorkers,
    ) -> Result<Self> {
        ensure!(party_id < 3, "invalid reconciliation party");
        workers.passive_resident_side()?;
        let registry = CandidateRegistry::new(process_id, workers.registries.clone())?;
        Ok(Self {
            workers,
            registry,
            party_id,
            state: State::Unreconciled,
        })
    }

    /// Superseding an attempt or losing its frozen-writer precondition invalidates
    /// this preload. Partial application/cancellation requires a fresh candidate.
    pub fn invalidate(&mut self) {
        self.state = State::Failed;
    }

    pub fn report_for(&self, current: &ReconciliationContext) -> Option<&ReconciliationReport> {
        match &self.state {
            State::Complete(report) if &report.context == current => Some(report),
            _ => None,
        }
    }

    /// All canonical writers must remain frozen and A sealed until activation.
    /// Exclusively own these workers; never run serving work alongside this call.
    /// This consumes the preload's one reconciliation attempt, including failure
    /// or cancellation. No actor, intake, publication, or ownership is started.
    pub async fn reconcile(
        &mut self,
        store: &Store,
        context: ReconciliationContext,
        deadline: Instant,
    ) -> Result<&ReconciliationReport> {
        let eligible = matches!(self.state, State::Unreconciled);
        self.invalidate();
        ensure!(
            eligible,
            "candidate requires a fresh preload after reconciliation or invalidation"
        );
        context.validate()?;
        ensure!(
            context.candidate_processes[self.party_id] == self.registry.process_id(),
            "reconciliation is bound to another candidate process"
        );
        ensure!(
            Instant::now() < deadline,
            "reconciliation deadline exceeded"
        );
        let started = Instant::now();
        tracing::info!(attempt_id = %context.attempt_id, candidate_process_id = %self.registry.process_id(), "candidate_reconciliation_started");
        let result = timeout_at(deadline, self.apply_and_verify(store, &context, deadline))
            .await
            .wrap_err("candidate reconciliation deadline exceeded")
            .and_then(|result| result);
        let (registry_count, registry_digest, scan_duration, apply_duration, verify_duration) =
            match result {
                Ok(result) if Instant::now() < deadline => result,
                result => {
                    let error = result.err().unwrap_or_else(|| {
                        eyre::eyre!("candidate reconciliation deadline exceeded")
                    });
                    tracing::warn!(attempt_id = %context.attempt_id, candidate_process_id = %self.registry.process_id(), party_id = self.party_id, elapsed_ms = started.elapsed().as_secs_f64() * 1000.0, error = ?error, "candidate_reconciliation_failed");
                    return Err(error);
                }
            };
        tracing::info!(attempt_id = %context.attempt_id, candidate_process_id = %self.registry.process_id(), party_id = self.party_id, registry_count, scan_ms = scan_duration.as_secs_f64() * 1000.0, verify_ms = verify_duration.as_secs_f64() * 1000.0, apply_ms = apply_duration.as_secs_f64() * 1000.0, elapsed_ms = started.elapsed().as_secs_f64() * 1000.0, "candidate_reconciliation_completed");
        self.state = State::Complete(ReconciliationReport {
            context,
            party_id: self.party_id,
            registry_count,
            registry_digest,
        });
        match &self.state {
            State::Complete(report) => Ok(report),
            _ => unreachable!(),
        }
    }

    async fn apply_and_verify(
        &mut self,
        store: &Store,
        context: &ReconciliationContext,
        deadline: Instant,
    ) -> Result<(u64, [u8; 32], Duration, Duration, Duration)> {
        let resident_side = self.workers.passive_resident_side()?;
        let scan_started = Instant::now();
        let mut diff = self
            .registry
            .differences(
                store,
                context.attempt_id,
                context.watermark.clone(),
                deadline,
            )
            .await?;
        let mut scan_duration = scan_started.elapsed();
        let mut apply_duration = Duration::ZERO;
        let mut reader = None;
        loop {
            let scan_started = Instant::now();
            let mut batch = Vec::with_capacity(APPLY_BATCH);
            while batch.len() < APPLY_BATCH {
                match diff.next().await? {
                    Some(change) => batch.push(change),
                    None => break,
                }
            }
            scan_duration += scan_started.elapsed();
            if batch.is_empty() {
                break;
            }
            let started = Instant::now();
            let mut current = Vec::new();
            let mut removed = Vec::new();
            for change in batch {
                match change {
                    RegistryChange::Inserted { current: id }
                    | RegistryChange::Changed { current: id, .. } => current.push(id),
                    RegistryChange::Removed { loaded } => removed.push(loaded),
                }
            }
            let ids = current
                .iter()
                .map(|id| i64::from(id.serial_id()))
                .collect::<Vec<_>>();
            let mut updates: [Vec<(VectorId, ArcIris)>; 2] = [Vec::new(), Vec::new()];
            for (side, updates) in updates.iter_mut().enumerate() {
                if resident_side.is_some_and(|resident| resident != side) {
                    continue;
                }
                let rows = if ids.is_empty() {
                    Vec::new()
                } else {
                    if reader.is_none() {
                        reader = Some(store.iris_payload_reader(deadline).await?);
                    }
                    reader
                        .as_mut()
                        .expect("reader initialized")
                        .read_side(&ids, side)
                        .await?
                };
                ensure!(
                    rows.len() == current.len(),
                    "canonical payload missing during reconciliation"
                );
                for (row, expected) in rows.into_iter().zip(&current) {
                    ensure!(
                        row.vector_id() == *expected,
                        "canonical version changed during reconciliation"
                    );
                    updates.push((
                        *expected,
                        GaloisRingSharedIris::try_from_buffers(
                            self.party_id,
                            row.code(),
                            row.mask(),
                        )?,
                    ));
                }
            }
            self.workers.reconcile_batch(updates, &removed).await?;
            apply_duration += started.elapsed();
        }
        ensure!(diff.completed(), "registry diff did not finish");
        drop(diff);
        drop(reader);
        // Deferred cold initialization rereads only the bounded lookback window,
        // including unchanged rows; the complete cold column stays in Postgres.
        let started = Instant::now();
        self.workers.finish_reconciliation(deadline).await?;
        apply_duration += started.elapsed();
        let verify_started = Instant::now();
        let mut verification = self
            .registry
            .differences(
                store,
                context.attempt_id,
                context.watermark.clone(),
                deadline,
            )
            .await?;
        ensure!(
            verification.next().await?.is_none() && verification.completed(),
            "candidate registry does not match canonical state after reconciliation"
        );
        drop(verification);
        let (count, digest) = self.registry.fingerprint(deadline).await?;
        Ok((
            count,
            digest,
            scan_duration,
            apply_duration,
            verify_started.elapsed(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(process: Uuid) -> ReconciliationContext {
        ReconciliationContext::new(
            Uuid::new_v4(),
            std::array::from_fn(|_| Uuid::new_v4()),
            [process, Uuid::new_v4(), Uuid::new_v4()],
            Some(normalize_sns_sequence_number("42").unwrap()),
        )
        .unwrap()
    }

    #[test]
    fn agreement_requires_exact_current_context_and_matching_metadata() {
        let expected = context(Uuid::new_v4());
        let reports: [ReconciliationReport; 3] =
            std::array::from_fn(|party_id| ReconciliationReport {
                context: expected.clone(),
                party_id,
                registry_count: 3,
                registry_digest: [7; 32],
            });
        ReconciliationReport::verify_agreement(&expected, &reports).unwrap();
        for field in 0..6 {
            let mut bad = reports.clone();
            match field {
                0 => bad[1].context.attempt_id = Uuid::new_v4(),
                1 => bad[1].context.candidate_processes[1] = Uuid::new_v4(),
                2 => bad[1].context.source_processes[0] = Uuid::new_v4(),
                3 => bad[1].context.watermark = None,
                4 => bad[1].registry_count += 1,
                _ => bad[1].registry_digest[0] ^= 1,
            }
            assert!(ReconciliationReport::verify_agreement(&expected, &bad).is_err());
        }
        let mut swapped = reports.clone();
        swapped.swap(0, 1);
        assert!(ReconciliationReport::verify_agreement(&expected, &swapped).is_err());
        let mut invalid = expected.clone();
        invalid.candidate_processes[1] = invalid.candidate_processes[0];
        assert!(invalid.validate().is_err());
    }

    #[tokio::test]
    async fn dropping_reconciliation_invalidates_candidate_before_any_mutation() -> Result<()> {
        use iris_mpc_common::postgres::PostgresClient;
        use iris_mpc_cpu::execution::hawk_main::worker_pool_initializer::{
            LocalWorkerPoolInitializer, WorkerPoolInitializer,
        };
        use iris_mpc_cpu::hawkers::aby3::aby3_store::DistanceMode;
        let workers = Box::new(
            LocalWorkerPoolInitializer::new_empty(0, DistanceMode::Simple, false)
                .defer_cold_storage(),
        )
        .initialize()
        .await?;
        let process = Uuid::new_v4();
        let mut candidate = CandidateReconciliation::new(0, process, workers)?;
        let registry = candidate.workers.registries[0].clone();
        let guard = registry.write().await;
        let db = PostgresClient {
            pool: sqlx::postgres::PgPoolOptions::new()
                .connect_lazy("postgres://unused:unused@localhost/unused")?,
            schema_name: "unused".into(),
        };
        let store = Store::new(&db).await?;
        let expected = context(process);
        {
            let run = candidate.reconcile(
                &store,
                expected.clone(),
                Instant::now() + Duration::from_secs(5),
            );
            futures::pin_mut!(run);
            assert!(futures::poll!(run.as_mut()).is_pending());
        }
        drop(guard);
        assert!(candidate.report_for(&expected).is_none());
        assert!(candidate
            .reconcile(&store, expected, Instant::now() + Duration::from_secs(5))
            .await
            .is_err());
        Ok(())
    }

    #[cfg(feature = "db_dependent")]
    async fn fixture(
        rows: i64,
    ) -> Result<(
        iris_mpc_common::config::Config,
        iris_mpc_common::postgres::PostgresClient,
    )> {
        use iris_mpc_common::{
            postgres::{AccessMode, PostgresClient},
            IRIS_CODE_LENGTH, MASK_CODE_LENGTH,
        };
        let url = iris_mpc_store::test_utils::test_db_url()?;
        let config: iris_mpc_common::config::Config = serde_json::from_value(serde_json::json!({
            "startup_mode": "candidate", "party_id": 0, "max_batch_size": 1,
            "disable_persistence": false, "full_scan_side_switching_enabled": false,
            "enable_s3_importer": false, "hawk_numa": false,
            "luc_enabled": true, "luc_lookback_records": 3,
            "schema_name": format!("SMPC_test_reconcile_{}", Uuid::new_v4().simple()),
            "database": { "url": url, "load_parallelism": 1 }
        }))?;
        let db =
            PostgresClient::new(&url, &config.get_gpu_db_schema(), AccessMode::ReadWrite).await?;
        sqlx::query("CREATE TABLE irises (id bigint PRIMARY KEY, version_id smallint NOT NULL, left_code bytea NOT NULL, left_mask bytea NOT NULL, right_code bytea NOT NULL, right_mask bytea NOT NULL)").execute(&db.pool).await?;
        sqlx::query(
            "INSERT INTO irises SELECT id, 1, $1, $2, $1, $2 FROM generate_series(1, $3) AS id",
        )
        .bind(bytemuck::cast_slice::<u16, u8>(&vec![7; IRIS_CODE_LENGTH]))
        .bind(bytemuck::cast_slice::<u16, u8>(&vec![3; MASK_CODE_LENGTH]))
        .bind(rows)
        .execute(&db.pool)
        .await?;
        Ok((config, db))
    }

    #[cfg(feature = "db_dependent")]
    async fn preload(config: &iris_mpc_common::config::Config) -> Result<CandidateReconciliation> {
        super::super::candidate::preload(
            config,
            std::sync::Arc::new(ampc_server_utils::shutdown_handler::ShutdownHandler::new(0)),
            Uuid::new_v4(),
        )
        .await
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn reconciles_existing_pools_and_initializes_current_cold_window() -> Result<()> {
        use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
        let (config, db) = fixture(5).await?;
        let mut candidate = preload(&config).await?;
        let resident = match config.full_scan_side {
            ampc_anon_stats::types::Eye::Left => 0,
            ampc_anon_stats::types::Eye::Right => 1,
        };
        let resident_pool = candidate.workers.pools[resident].clone();
        sqlx::query("UPDATE irises SET version_id = 2 WHERE id = 4")
            .execute(&db.pool)
            .await?;
        sqlx::query(
            "UPDATE irises SET version_id = 9, left_code = $1, right_code = $1 WHERE id = 4",
        )
        .bind(bytemuck::cast_slice::<u16, u8>(&vec![11; IRIS_CODE_LENGTH]))
        .execute(&db.pool)
        .await?;
        sqlx::query("DELETE FROM irises WHERE id IN (2, 5)")
            .execute(&db.pool)
            .await?;
        sqlx::query("INSERT INTO irises VALUES (6, 0, $1, $2, $1, $2)")
            .bind(bytemuck::cast_slice::<u16, u8>(&vec![13; IRIS_CODE_LENGTH]))
            .bind(bytemuck::cast_slice::<u16, u8>(&vec![3; MASK_CODE_LENGTH]))
            .execute(&db.pool)
            .await?;
        let store = Store::new(&db).await?;
        let expected = context(candidate.registry.process_id());
        let report = candidate
            .reconcile(
                &store,
                expected.clone(),
                Instant::now() + Duration::from_secs(20),
            )
            .await?
            .clone();
        assert_eq!(report.registry_count, 4);
        assert!(std::sync::Arc::ptr_eq(
            &resident_pool,
            &candidate.workers.pools[resident]
        ));
        for registry in &candidate.workers.registries {
            let registry = registry.read().await;
            assert_eq!(registry.get_current_version(2), None);
            assert_eq!(registry.get_current_version(5), None);
            assert_eq!(registry.get_current_version(4), Some(9));
            assert_eq!(registry.get_current_version(6), Some(0));
        }
        let current = GaloisRingSharedIris::try_from_buffers(
            0,
            &vec![11; IRIS_CODE_LENGTH],
            &vec![3; MASK_CODE_LENGTH],
        )?;
        for pool in &candidate.workers.pools {
            assert_eq!(
                pool.fetch_irises(vec![VectorId::new(4, 9)]).await?[0],
                current
            );
        }
        assert!(candidate.workers.pools[1 - resident]
            .fetch_irises(vec![VectorId::new(5, 1)])
            .await
            .is_err());
        assert!(candidate.report_for(&expected).is_some());
        let mut superseded = expected.clone();
        superseded.attempt_id = Uuid::new_v4();
        assert!(candidate.report_for(&superseded).is_none());
        candidate.invalidate();
        assert!(candidate.report_for(&expected).is_none());
        drop(candidate);
        iris_mpc_store::test_utils::cleanup(&db, &config.get_gpu_db_schema()).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn tail_removal_matches_fresh_preload_frontier_and_partial_failure_is_terminal(
    ) -> Result<()> {
        use iris_mpc_common::IRIS_CODE_LENGTH;
        let (config, db) = fixture(129).await?;
        let store = Store::new(&db).await?;
        let mut failed = preload(&config).await?;
        sqlx::query("UPDATE irises SET version_id = 2")
            .execute(&db.pool)
            .await?;
        sqlx::query("UPDATE irises SET left_code = $1, right_code = $1 WHERE id = 129")
            .bind(vec![0_u8; 2])
            .execute(&db.pool)
            .await?;
        let failed_context = context(failed.registry.process_id());
        assert!(failed
            .reconcile(
                &store,
                failed_context.clone(),
                Instant::now() + Duration::from_secs(20)
            )
            .await
            .is_err());
        assert_eq!(
            failed.workers.registries[0]
                .read()
                .await
                .get_current_version(1),
            Some(2)
        );
        assert_eq!(
            failed.workers.registries[0]
                .read()
                .await
                .get_current_version(129),
            Some(1)
        );
        assert!(failed.report_for(&failed_context).is_none());
        assert!(failed
            .reconcile(
                &store,
                failed_context,
                Instant::now() + Duration::from_secs(20)
            )
            .await
            .is_err());
        drop(failed);
        sqlx::query("UPDATE irises SET left_code = $1, right_code = $1 WHERE id = 129")
            .bind(bytemuck::cast_slice::<u16, u8>(&vec![7; IRIS_CODE_LENGTH]))
            .execute(&db.pool)
            .await?;
        let mut old = preload(&config).await?;
        sqlx::query("DELETE FROM irises WHERE id >= 128")
            .execute(&db.pool)
            .await?;
        let mut fresh = preload(&config).await?;
        let old_context = context(old.registry.process_id());
        let fresh_context = context(fresh.registry.process_id());
        let old_report = old
            .reconcile(
                &store,
                old_context,
                Instant::now() + Duration::from_secs(20),
            )
            .await?
            .clone();
        let fresh_report = fresh
            .reconcile(
                &store,
                fresh_context,
                Instant::now() + Duration::from_secs(20),
            )
            .await?
            .clone();
        assert_eq!(old.workers.registries[0].read().await.next_id, 128);
        assert_eq!(old_report.registry_digest, fresh_report.registry_digest);
        assert_eq!(old_report.registry_count, fresh_report.registry_count);
        drop((old, fresh));
        iris_mpc_store::test_utils::cleanup(&db, &config.get_gpu_db_schema()).await?;
        Ok(())
    }
}
