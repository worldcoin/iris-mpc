//! Metadata-only differences against a frozen canonical database.

use eyre::{ensure, Result, WrapErr};
use futures::{stream::BoxStream, StreamExt, TryStreamExt};
use iris_mpc_common::VectorId;
use iris_mpc_cpu::hawkers::{aby3::aby3_store::VectorIdRegistryRef, shared_irises::SharedIrises};
use iris_mpc_store::{normalize_sns_sequence_number, Store};
use std::cmp::Ordering;
use tokio::time::{timeout_at, Instant};
use uuid::Uuid;

/// The identity and watermark for which a completed diff is valid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffContext {
    attempt_id: Uuid,
    candidate_process_id: Uuid,
    watermark: Option<String>,
}

impl DiffContext {
    pub fn attempt_id(&self) -> Uuid {
        self.attempt_id
    }
    pub fn candidate_process_id(&self) -> Uuid {
        self.candidate_process_id
    }
    pub fn watermark(&self) -> Option<&str> {
        self.watermark.as_deref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryChange {
    Inserted { current: VectorId },
    Changed { loaded: VectorId, current: VectorId },
    Removed { loaded: VectorId },
}

/// Associates the actual preload registries with their existing process identity.
/// Cloning registry references does not copy the loaded metadata or shares.
pub struct CandidateRegistry {
    process_id: Uuid,
    registries: [VectorIdRegistryRef; 2],
}

impl CandidateRegistry {
    pub(crate) fn new(process_id: Uuid, registries: [VectorIdRegistryRef; 2]) -> Result<Self> {
        ensure!(
            !process_id.is_nil(),
            "candidate process identity must not be nil"
        );
        Ok(Self {
            process_id,
            registries,
        })
    }

    pub fn process_id(&self) -> Uuid {
        self.process_id
    }

    /// The caller must keep canonical writers frozen for this attempt through
    /// reconciliation. W and this handle do not establish ownership or authorize
    /// a handoff. The caller must exclusively own candidate reconciliation:
    /// between calls to `next`, it may apply only changes already returned by
    /// this scan. Other mutations or overlapping attempts invalidate the diff.
    /// Version-preserving changes cannot be detected by this scan.
    pub async fn differences<'a>(
        &'a mut self,
        store: &'a Store,
        attempt_id: Uuid,
        watermark: Option<String>,
        deadline: Instant,
    ) -> Result<RegistryDiff<'a>> {
        self.scan(
            store.stream_all_iris_ids(deadline).boxed(),
            attempt_id,
            watermark,
            deadline,
        )
        .await
    }

    async fn scan<'a>(
        &'a mut self,
        rows: BoxStream<'a, sqlx::Result<(i64, i16)>>,
        attempt_id: Uuid,
        watermark: Option<String>,
        deadline: Instant,
    ) -> Result<RegistryDiff<'a>> {
        ensure!(
            !attempt_id.is_nil(),
            "diff attempt identity must not be nil"
        );
        if let Some(watermark) = &watermark {
            ensure!(
                normalize_sns_sequence_number(watermark)? == *watermark,
                "diff watermark must be normalized"
            );
        }
        ensure!(Instant::now() < deadline, "registry diff deadline exceeded");
        let loaded_len = timeout_at(deadline, async {
            let left = self.registries[0].read().await;
            let right = self.registries[1].read().await;
            left.get_points().len().max(right.get_points().len())
        })
        .await
        .wrap_err("registry diff deadline exceeded acquiring registries")?;
        let started = Instant::now();
        tracing::info!(%attempt_id, candidate_process_id = %self.process_id, "registry_diff_started");
        Ok(RegistryDiff {
            context: DiffContext {
                attempt_id,
                candidate_process_id: self.process_id,
                watermark,
            },
            state: ScanState::Active(Cursor {
                registries: &self.registries,
                loaded_len,
                rows,
                next_index: 0,
                loaded: None,
                canonical: None,
                previous_db_id: 0,
                db_ended: false,
                rows_read: 0,
            }),
            deadline,
            started,
            inserted: 0,
            changed: 0,
            removed: 0,
        })
    }
}

/// A lazy ordered diff with constant additional metadata memory.
///
/// Only successful exhaustion establishes a complete diff. On failure, discard
/// partial output; it must not establish reconciliation readiness. Candidate
/// mutations already applied require a new scan before readiness can be proved.
/// Registry locks are held only during `next`, so returned changes can be applied
/// without buffering the complete diff. Exhaustion/failure/drop closes the scan's
/// connection; a server statement timeout bounds outstanding query cleanup.
#[must_use]
pub struct RegistryDiff<'a> {
    context: DiffContext,
    state: ScanState<'a>,
    deadline: Instant,
    started: Instant,
    inserted: u64,
    changed: u64,
    removed: u64,
}

impl RegistryDiff<'_> {
    pub fn context(&self) -> &DiffContext {
        &self.context
    }
    pub fn completed(&self) -> bool {
        matches!(self.state, ScanState::Complete)
    }

    pub async fn next(&mut self) -> Result<Option<RegistryChange>> {
        let cursor = match &mut self.state {
            ScanState::Active(cursor) => cursor,
            ScanState::Complete => return Ok(None),
            ScanState::Failed => eyre::bail!("registry diff previously failed; start a new scan"),
        };
        let result = timeout_at(self.deadline, cursor.next(self.deadline))
            .await
            .wrap_err("registry diff deadline exceeded")
            .and_then(|result| result)
            .and_then(|change| {
                ensure!(
                    Instant::now() < self.deadline,
                    "registry diff deadline exceeded"
                );
                Ok(change)
            });
        match &result {
            Ok(None) => {
                self.log_outcome("registry_diff_completed", None);
                self.state = ScanState::Complete;
            }
            Err(error) => {
                self.log_outcome("registry_diff_failed", Some(error));
                self.state = ScanState::Failed;
            }
            Ok(Some(RegistryChange::Inserted { .. })) => self.inserted += 1,
            Ok(Some(RegistryChange::Changed { .. })) => self.changed += 1,
            Ok(Some(RegistryChange::Removed { .. })) => self.removed += 1,
        }
        result
    }

    fn log_outcome(&self, event: &str, error: Option<&eyre::Report>) {
        let ScanState::Active(cursor) = &self.state else {
            return;
        };
        if let Some(error) = error {
            tracing::warn!(
                attempt_id = %self.context.attempt_id,
                candidate_process_id = %self.context.candidate_process_id,
                elapsed_ms = self.started.elapsed().as_secs_f64() * 1000.0,
                canonical_rows_read = cursor.rows_read,
                inserted = self.inserted,
                changed = self.changed,
                removed = self.removed,
                error = ?error,
                "{event}"
            );
        } else {
            tracing::info!(
                attempt_id = %self.context.attempt_id,
                candidate_process_id = %self.context.candidate_process_id,
                elapsed_ms = self.started.elapsed().as_secs_f64() * 1000.0,
                canonical_rows_read = cursor.rows_read,
                inserted = self.inserted,
                changed = self.changed,
                removed = self.removed,
                "{event}"
            );
        }
    }
}

impl Drop for RegistryDiff<'_> {
    fn drop(&mut self) {
        self.log_outcome("registry_diff_abandoned", None);
    }
}

enum ScanState<'a> {
    Active(Cursor<'a>),
    Complete,
    Failed,
}

struct Cursor<'a> {
    registries: &'a [VectorIdRegistryRef; 2],
    loaded_len: usize,
    rows: BoxStream<'a, sqlx::Result<(i64, i16)>>,
    next_index: usize,
    loaded: Option<VectorId>,
    canonical: Option<VectorId>,
    previous_db_id: i64,
    db_ended: bool,
    rows_read: u64,
}

impl Cursor<'_> {
    async fn next_loaded(
        &mut self,
        left: &SharedIrises<()>,
        right: &SharedIrises<()>,
        deadline: Instant,
    ) -> Result<Option<VectorId>> {
        let left = left.get_points();
        let right = right.get_points();
        while self.next_index < self.loaded_len {
            ensure!(Instant::now() < deadline, "registry diff deadline exceeded");
            let index = self.next_index;
            if index.is_multiple_of(1024) {
                tokio::task::yield_now().await;
            }
            self.next_index += 1;
            let left = left.get(index).and_then(|entry| *entry);
            let right = right.get(index).and_then(|entry| *entry);
            ensure!(
                left == right,
                "candidate eye registries disagree at serial ID {index}"
            );
            if let Some((version, ())) = left {
                let id = u32::try_from(index).wrap_err("candidate serial ID exceeds u32")?;
                ensure!(
                    id > 0 && version >= 0,
                    "invalid candidate registry metadata at serial ID {index}"
                );
                return Ok(Some(VectorId::new(id, version)));
            }
        }
        Ok(None)
    }

    async fn next(&mut self, deadline: Instant) -> Result<Option<RegistryChange>> {
        let registries = self.registries;
        let left = registries[0].read().await;
        let right = registries[1].read().await;
        loop {
            ensure!(Instant::now() < deadline, "registry diff deadline exceeded");
            if self.loaded.is_none() {
                self.loaded = self.next_loaded(&left, &right, deadline).await?;
            }
            if self.canonical.is_none() && !self.db_ended {
                self.canonical = match self
                    .rows
                    .try_next()
                    .await
                    .wrap_err("read canonical iris metadata")?
                {
                    Some((id, version)) => {
                        self.rows_read += 1;
                        ensure!(
                            id > self.previous_db_id,
                            "canonical iris IDs must be positive and strictly increasing"
                        );
                        let serial = u32::try_from(id).wrap_err("canonical iris ID exceeds u32")?;
                        ensure!(
                            version >= 0,
                            "negative canonical iris version for serial ID {id}"
                        );
                        self.previous_db_id = id;
                        Some(VectorId::new(serial, version))
                    }
                    None => {
                        self.db_ended = true;
                        None
                    }
                };
            }
            match (self.loaded, self.canonical) {
                (None, None) => return Ok(None),
                (Some(loaded), None) => {
                    self.loaded = None;
                    return Ok(Some(RegistryChange::Removed { loaded }));
                }
                (None, Some(current)) => {
                    self.canonical = None;
                    return Ok(Some(RegistryChange::Inserted { current }));
                }
                (Some(loaded), Some(current)) => match loaded.serial_id().cmp(&current.serial_id())
                {
                    Ordering::Less => {
                        self.loaded = None;
                        return Ok(Some(RegistryChange::Removed { loaded }));
                    }
                    Ordering::Greater => {
                        self.canonical = None;
                        return Ok(Some(RegistryChange::Inserted { current }));
                    }
                    Ordering::Equal => {
                        self.loaded = None;
                        self.canonical = None;
                        if loaded != current {
                            return Ok(Some(RegistryChange::Changed { loaded, current }));
                        }
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering as AtomicOrdering},
            Arc,
        },
        time::Duration,
    };

    fn candidate(points: &[(u32, i16)]) -> CandidateRegistry {
        let mut registry = SharedIrises::default();
        for (id, version) in points {
            registry.insert(VectorId::new(*id, *version), ());
        }
        CandidateRegistry::new(
            Uuid::new_v4(),
            [registry.clone().to_arc(), registry.to_arc()],
        )
        .unwrap()
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    async fn fixture<'a>(
        candidate: &'a mut CandidateRegistry,
        rows: Vec<(i64, i16)>,
    ) -> RegistryDiff<'a> {
        candidate
            .scan(
                stream::iter(rows.into_iter().map(Ok)).boxed(),
                Uuid::new_v4(),
                None,
                deadline(),
            )
            .await
            .unwrap()
    }

    async fn collect(scan: &mut RegistryDiff<'_>) -> Result<Vec<RegistryChange>> {
        let mut result = Vec::new();
        while let Some(change) = scan.next().await? {
            result.push(change);
        }
        ensure!(scan.completed(), "scan must have completed");
        Ok(result)
    }

    #[tokio::test]
    async fn unchanged_and_empty_registries_complete_without_changes() -> Result<()> {
        for points in [vec![], vec![(1, 7), (3, 2)]] {
            let mut candidate = candidate(&points);
            let registries = candidate.registries.clone();
            let mut scan = fixture(
                &mut candidate,
                points
                    .iter()
                    .map(|(id, version)| (i64::from(*id), *version))
                    .collect(),
            )
            .await;
            assert!(!scan.completed());
            assert!(collect(&mut scan).await?.is_empty());
            // The finished scan does not retain read guards.
            let _write =
                tokio::time::timeout(Duration::from_millis(50), registries[0].write()).await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn streams_inserts_updates_rollbacks_and_removals_in_id_order() -> Result<()> {
        let mut candidate = candidate(&[(1, 1), (3, 7), (5, 8), (9, 1)]);
        let mut scan = fixture(
            &mut candidate,
            vec![(1, 1), (2, 0), (3, 10), (5, 2), (12, 3)],
        )
        .await;
        assert_eq!(
            collect(&mut scan).await?,
            vec![
                RegistryChange::Inserted {
                    current: VectorId::new(2, 0)
                },
                RegistryChange::Changed {
                    loaded: VectorId::new(3, 7),
                    current: VectorId::new(3, 10)
                },
                RegistryChange::Changed {
                    loaded: VectorId::new(5, 8),
                    current: VectorId::new(5, 2)
                },
                RegistryChange::Removed {
                    loaded: VectorId::new(9, 1)
                },
                RegistryChange::Inserted {
                    current: VectorId::new(12, 3)
                },
            ]
        );
        assert_eq!((scan.inserted, scan.changed, scan.removed), (2, 2, 1));
        Ok(())
    }

    #[tokio::test]
    async fn handles_empty_sides_and_sparse_tail() -> Result<()> {
        let mut empty = candidate(&[]);
        let mut inserted = fixture(&mut empty, vec![(4, 2)]).await;
        assert_eq!(
            collect(&mut inserted).await?,
            vec![RegistryChange::Inserted {
                current: VectorId::new(4, 2)
            }]
        );
        let mut sparse = candidate(&[(4097, 2)]);
        let mut removed = fixture(&mut sparse, vec![]).await;
        assert_eq!(
            collect(&mut removed).await?,
            vec![RegistryChange::Removed {
                loaded: VectorId::new(4097, 2)
            }]
        );
        Ok(())
    }

    #[tokio::test]
    async fn consumers_can_apply_each_change_without_buffering_the_diff() -> Result<()> {
        let cases = [
            (vec![], vec![(2, 0), (5, 1)]),
            (
                vec![(1, 1), (3, 7), (5, 8), (9, 1)],
                vec![(1, 1), (2, 0), (3, 10), (5, 2), (12, 3)],
            ),
            (vec![(1, 1), (9, 0)], vec![(1, 1), (3, 0), (4, 0)]),
        ];
        for (loaded, canonical) in cases {
            let mut candidate = candidate(&loaded);
            let registries = candidate.registries.clone();
            let mut scan = fixture(&mut candidate, canonical.clone()).await;
            let mut emitted = 0;
            while let Some(change) = scan.next().await? {
                emitted += 1;
                ensure!(
                    emitted <= loaded.len() + canonical.len(),
                    "diff revisited applied entries"
                );
                for reference in &registries {
                    let mut registry =
                        tokio::time::timeout(Duration::from_millis(100), reference.write()).await?;
                    match change {
                        RegistryChange::Inserted { current }
                        | RegistryChange::Changed { current, .. } => {
                            registry.insert(current, ());
                        }
                        RegistryChange::Removed { loaded } => {
                            let retained = registry
                                .get_points()
                                .iter()
                                .enumerate()
                                .filter_map(|(id, row)| {
                                    row.filter(|_| id != loaded.serial_id() as usize).map(
                                        |(version, ())| (VectorId::new(id as u32, version), ()),
                                    )
                                })
                                .collect();
                            *registry = SharedIrises::new(retained, ());
                        }
                    }
                }
            }
            assert!(scan.completed());
            for registry in &registries {
                let actual: Vec<_> = registry
                    .read()
                    .await
                    .get_points()
                    .iter()
                    .enumerate()
                    .filter_map(|(id, entry)| entry.map(|(version, ())| (id as i64, version)))
                    .collect();
                assert_eq!(actual, canonical);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn binds_context_and_consumes_metadata_lazily() -> Result<()> {
        let mut candidate = candidate(&[]);
        let attempt = Uuid::new_v4();
        let process_id = candidate.process_id();
        let registries = candidate.registries.clone();
        let watermark = normalize_sns_sequence_number("100")?;
        let reads = Arc::new(AtomicUsize::new(0));
        let seen = reads.clone();
        let rows = stream::iter((1..=10000).map(move |id| {
            seen.fetch_add(1, AtomicOrdering::SeqCst);
            Ok((id, 0))
        }));
        let mut scan = candidate
            .scan(rows.boxed(), attempt, Some(watermark.clone()), deadline())
            .await?;
        assert_eq!(reads.load(AtomicOrdering::SeqCst), 0);
        assert_eq!(scan.context().candidate_process_id(), process_id);
        assert_eq!(scan.context().attempt_id(), attempt);
        assert_eq!(scan.context().watermark(), Some(watermark.as_str()));
        assert!(scan.next().await?.is_some());
        assert_eq!(reads.load(AtomicOrdering::SeqCst), 1);
        let ScanState::Active(cursor) = &scan.state else {
            panic!("partial scan must remain active");
        };
        assert_eq!(cursor.rows_read, 1);
        assert_eq!((scan.inserted, scan.changed, scan.removed), (1, 0, 0));
        assert!(!scan.completed());
        drop(scan);
        let _write = tokio::time::timeout(Duration::from_millis(50), registries[0].write()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn errors_after_partial_output_remain_terminal() -> Result<()> {
        let mut candidate = candidate(&[]);
        let registries = candidate.registries.clone();
        let rows = stream::iter([
            Ok((1, 0)),
            Err(sqlx::Error::Protocol("test scan failure".into())),
        ]);
        let mut scan = candidate
            .scan(rows.boxed(), Uuid::new_v4(), None, deadline())
            .await?;
        assert!(scan.next().await?.is_some());
        assert!(scan.next().await.is_err());
        assert!(!scan.completed());
        assert!(scan.next().await.is_err());
        assert_eq!((scan.inserted, scan.changed, scan.removed), (1, 0, 0));
        let _write = tokio::time::timeout(Duration::from_millis(50), registries[0].write()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn rejects_invalid_or_unordered_canonical_metadata() {
        let mut candidate = candidate(&[]);
        for rows in [
            vec![(0, 1)],
            vec![(-1, 1)],
            vec![(i64::from(u32::MAX) + 1, 1)],
            vec![(1, -1)],
            vec![(1, 1), (1, 2)],
            vec![(2, 0), (1, 0)],
        ] {
            let mut scan = fixture(&mut candidate, rows).await;
            assert!(collect(&mut scan).await.is_err());
            assert!(!scan.completed());
        }
    }

    #[tokio::test]
    async fn rejects_invalid_loaded_metadata_and_eye_disagreement() {
        for points in [vec![(0, 1)], vec![(1, -1)]] {
            let mut candidate = candidate(&points);
            let mut scan = fixture(&mut candidate, vec![]).await;
            assert!(scan.next().await.is_err());
        }
        let mut candidate = candidate(&[(1, 1)]);
        candidate.registries[1]
            .write()
            .await
            .insert(VectorId::new(1, 2), ());
        let mut scan = fixture(&mut candidate, vec![(1, 1)]).await;
        assert!(scan.next().await.is_err());
        assert!(!scan.completed());
    }

    #[tokio::test]
    async fn cancelling_a_pending_next_preserves_the_loaded_entry() -> Result<()> {
        let mut candidate = candidate(&[(1024, 7)]);
        let mut scan = fixture(&mut candidate, vec![]).await;
        // Skip the initial cooperative yield; stop at the yield for serial 1024.
        {
            let next = scan.next();
            futures::pin_mut!(next);
            assert!(futures::poll!(next.as_mut()).is_pending());
            assert!(futures::poll!(next.as_mut()).is_pending());
        }
        assert_eq!(
            scan.next().await?,
            Some(RegistryChange::Removed {
                loaded: VectorId::new(1024, 7)
            })
        );
        assert_eq!(scan.next().await?, None);
        assert!(scan.completed());
        assert_eq!((scan.inserted, scan.changed, scan.removed), (0, 0, 1));
        Ok(())
    }

    #[tokio::test]
    async fn rejects_unbound_context() {
        let empty = [
            SharedIrises::default().to_arc(),
            SharedIrises::default().to_arc(),
        ];
        assert!(CandidateRegistry::new(Uuid::nil(), empty).is_err());
        let mut candidate = candidate(&[]);
        assert!(candidate
            .scan(stream::empty().boxed(), Uuid::nil(), None, deadline())
            .await
            .is_err());
        assert!(candidate
            .scan(
                stream::empty().boxed(),
                Uuid::new_v4(),
                Some("100".into()),
                deadline()
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn deadline_bounds_locks_reads_and_cpu_scan() -> Result<()> {
        let mut candidate = candidate(&[]);
        let registries = candidate.registries.clone();
        let write = registries[0].write().await;
        assert!(candidate
            .scan(
                stream::empty().boxed(),
                Uuid::new_v4(),
                None,
                Instant::now() + Duration::from_millis(10)
            )
            .await
            .is_err());
        drop(write);
        let mut scan = candidate
            .scan(
                stream::pending().boxed(),
                Uuid::new_v4(),
                None,
                Instant::now() + Duration::from_millis(10),
            )
            .await?;
        assert!(scan.next().await.is_err());
        assert!(scan.next().await.is_err());
        assert!(!scan.completed());
        let mut candidate = self::candidate(&[(100_000, 0)]);
        let mut scan = fixture(&mut candidate, vec![]).await;
        scan.deadline = Instant::now();
        assert!(scan.next().await.is_err());
        assert!(!scan.completed());
        Ok(())
    }

    #[cfg(feature = "db_dependent")]
    async fn wait_for_backend_exit(pool: &sqlx::PgPool, backend_pid: i32) -> Result<()> {
        timeout_at(Instant::now() + Duration::from_secs(2), async {
            loop {
                let present: bool = sqlx::query_scalar(
                    "SELECT EXISTS (SELECT 1 FROM pg_stat_activity WHERE pid = $1)",
                )
                .bind(backend_pid)
                .fetch_one(pool)
                .await?;
                if !present {
                    return Ok::<_, sqlx::Error>(());
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .wrap_err("scan backend remained alive after cleanup deadline")??;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn postgres_diff_deadline_releases_backend_and_pool_while_locked() -> Result<()> {
        use iris_mpc_common::postgres::{AccessMode, PostgresClient};
        use sqlx::postgres::PgPoolOptions;

        let url = iris_mpc_store::test_utils::test_db_url()?;
        let schema = iris_mpc_store::test_utils::temporary_name();
        let admin = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
        sqlx::query("CREATE TABLE irises (id bigint PRIMARY KEY, version_id smallint NOT NULL)")
            .execute(&admin.pool)
            .await?;
        sqlx::query("INSERT INTO irises VALUES (1, 0), (2, 0)")
            .execute(&admin.pool)
            .await?;
        // A shorter server policy must remain effective, and neither policy may
        // leave a blocked SELECT occupying the only connection after failure.
        for initial_timeout_ms in [0_i64, 25] {
            let search_path = format!("\"{schema}\"");
            let pool = PgPoolOptions::new()
                .max_connections(1)
                .after_connect(move |connection, _| {
                    let search_path = search_path.clone();
                    Box::pin(async move {
                        sqlx::query("SELECT set_config('search_path', $1, false), set_config('statement_timeout', $2, false)")
                            .bind(search_path)
                            .bind(initial_timeout_ms.to_string())
                            .execute(connection)
                            .await?;
                        Ok(())
                    })
                })
                .connect(&url)
                .await?;
            let backend_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
                .fetch_one(&pool)
                .await?;
            let store = Store::new(&PostgresClient {
                pool: pool.clone(),
                schema_name: schema.clone(),
            })
            .await?;
            let mut blocker = admin.pool.begin().await?;
            sqlx::query("LOCK TABLE irises IN ACCESS EXCLUSIVE MODE")
                .execute(&mut *blocker)
                .await?;
            let result: Result<()> = async {
                let mut candidate = candidate(&[]);
                let budget = if initial_timeout_ms == 0 { 100 } else { 500 };
                let mut scan = candidate
                    .differences(
                        &store,
                        Uuid::new_v4(),
                        None,
                        Instant::now() + Duration::from_millis(budget),
                    )
                    .await?;
                let error = scan.next().await.expect_err("locked query must fail");
                ensure!(!scan.completed(), "failed scan must not be complete");
                if initial_timeout_ms != 0 {
                    let database_error = error
                        .downcast_ref::<sqlx::Error>()
                        .and_then(sqlx::Error::as_database_error);
                    ensure!(
                        database_error.and_then(|error| error.code()).as_deref() == Some("57014"),
                        "the shorter server timeout must cancel the statement: {error:?}"
                    );
                }
                wait_for_backend_exit(&admin.pool, backend_pid).await?;
                let restored_timeout: i64 = timeout_at(
                    Instant::now() + Duration::from_secs(2),
                    sqlx::query_scalar(
                        "SELECT setting::bigint FROM pg_settings WHERE name = 'statement_timeout'",
                    )
                    .fetch_one(&pool),
                )
                .await
                .wrap_err("scan kept the only pool slot occupied")??;
                ensure!(
                    restored_timeout == initial_timeout_ms,
                    "scan timeout leaked to a later pool borrower"
                );
                Ok(())
            }
            .await;
            blocker.rollback().await?;
            result?;

            // Cancellation between rows and successful exhaustion must release
            // the same lease, even when no deadline has expired.
            for exhaust in [false, true] {
                let backend_pid: i32 = sqlx::query_scalar("SELECT pg_backend_pid()")
                    .fetch_one(&pool)
                    .await?;
                let mut candidate = candidate(&[]);
                let mut scan = candidate
                    .differences(&store, Uuid::new_v4(), None, deadline())
                    .await?;
                ensure!(scan.next().await?.is_some(), "expected the first row");
                if exhaust {
                    collect(&mut scan).await?;
                } else {
                    ensure!(!scan.completed(), "partial scan cannot be complete");
                }
                drop(scan);
                wait_for_backend_exit(&admin.pool, backend_pid).await?;
            }
            pool.close().await;
        }
        iris_mpc_store::test_utils::cleanup(&admin, &schema).await?;
        Ok(())
    }

    #[tokio::test]
    #[cfg(feature = "db_dependent")]
    async fn postgres_diff_reads_only_metadata_and_sees_latest_versions() -> Result<()> {
        use iris_mpc_common::postgres::{AccessMode, PostgresClient};
        let schema = iris_mpc_store::test_utils::temporary_name();
        let db = PostgresClient::new(
            &iris_mpc_store::test_utils::test_db_url()?,
            &schema,
            AccessMode::ReadWrite,
        )
        .await?;
        // No share columns exist, so fetching payloads would fail this test.
        sqlx::query("CREATE TABLE irises (id bigint PRIMARY KEY, version_id smallint NOT NULL)")
            .execute(&db.pool)
            .await?;
        sqlx::query("INSERT INTO irises VALUES (1, 7), (2, 1)")
            .execute(&db.pool)
            .await?;
        let mut candidate = candidate(&[(1, 7), (2, 1)]);
        sqlx::query("UPDATE irises SET version_id = 8 WHERE id = 1")
            .execute(&db.pool)
            .await?;
        sqlx::query("UPDATE irises SET version_id = 9 WHERE id = 1")
            .execute(&db.pool)
            .await?;
        sqlx::query("DELETE FROM irises WHERE id = 2")
            .execute(&db.pool)
            .await?;
        sqlx::query("INSERT INTO irises VALUES (5, 0)")
            .execute(&db.pool)
            .await?;
        let store = Store::new(&db).await?;
        let mut scan = candidate
            .differences(&store, Uuid::new_v4(), None, deadline())
            .await?;
        assert_eq!(
            collect(&mut scan).await?,
            vec![
                RegistryChange::Changed {
                    loaded: VectorId::new(1, 7),
                    current: VectorId::new(1, 9)
                },
                RegistryChange::Removed {
                    loaded: VectorId::new(2, 1)
                },
                RegistryChange::Inserted {
                    current: VectorId::new(5, 0)
                },
            ]
        );
        iris_mpc_store::test_utils::cleanup(&db, &schema).await?;
        Ok(())
    }
}
