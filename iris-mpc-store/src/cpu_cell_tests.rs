use super::*;
use iris_mpc_common::postgres::{check_handoff_startup, AccessMode, PostgresClient};
use sqlx::Executor;

struct Fixture {
    url: String,
    schema: String,
    db: CpuCellHandoffs,
    id: Uuid,
    source: Uuid,
    destination: Uuid,
    process: Uuid,
    candidates: [Uuid; 3],
}

impl Fixture {
    async fn new() -> Result<Self> {
        let url = std::env::var("CPU_CELL_TEST_DATABASE_URL")
            .wrap_err("CPU_CELL_TEST_DATABASE_URL must identify disposable PostgreSQL")?;
        let schema = format!("handoff_{:x}", rand::random::<u64>());
        let client = PostgresClient::new(
            &url,
            &schema,
            iris_mpc_common::postgres::AccessMode::ReadWrite,
        )
        .await?;
        iris_mpc_common::postgres::run_migrations(&client.pool, false).await?;
        client.pool.close().await;
        let client = PostgresClient::new_control(&url, &schema).await?;
        Ok(Self {
            url,
            schema: schema.clone(),
            db: CpuCellHandoffs::new(client.pool, &schema)?,
            id: Uuid::new_v4(),
            source: Uuid::new_v4(),
            destination: Uuid::new_v4(),
            process: Uuid::new_v4(),
            candidates: std::array::from_fn(|_| Uuid::new_v4()),
        })
    }
    async fn request(&self) -> Result<()> {
        self.db
            .request(self.id, self.source, self.destination)
            .await
    }
    async fn drain(&self) -> Result<()> {
        self.db
            .start_drain(self.id, self.source, self.process, &self.candidates)
            .await
    }
    async fn cleanup(self) -> Result<()> {
        self.db
            .pool
            .execute(format!("DROP SCHEMA {} CASCADE", self.schema).as_str())
            .await?;
        self.db.pool.close().await;
        Ok(())
    }
}

fn report() -> serde_json::Value {
    serde_json::json!({"watermark": null, "persisted_count": 0, "persisted_sha256": "0".repeat(64), "last_publication": null})
}

#[tokio::test]
async fn handoff_transitions_preserve_exact_report_and_history() -> Result<()> {
    let f = Fixture::new().await?;
    f.request().await?;
    f.request().await?;
    assert!(f.db.request(f.id, f.source, Uuid::new_v4()).await.is_err());
    assert!(f
        .db
        .complete_drain(f.id, f.source, f.process, &report())
        .await
        .is_err());
    f.drain().await?;
    f.drain().await?;
    assert!(f
        .db
        .start_drain(f.id, f.source, Uuid::new_v4(), &f.candidates)
        .await
        .is_err());
    f.db.complete_drain(f.id, f.source, f.process, &report())
        .await?;
    f.db.complete_drain(f.id, f.source, f.process, &report())
        .await?;
    let different = serde_json::json!({"different": true});
    assert!(f
        .db
        .complete_drain(f.id, f.source, f.process, &different)
        .await
        .is_err());
    assert!(f
        .db
        .activate(f.id, f.destination, &f.candidates, &different)
        .await
        .is_err());
    let mut replaced = f.candidates;
    replaced[0] = Uuid::new_v4();
    assert!(f
        .db
        .activate(f.id, f.destination, &replaced, &report())
        .await
        .is_err());
    f.db.activate(f.id, f.destination, &f.candidates, &report())
        .await?;
    f.db.activate(f.id, f.destination, &f.candidates, &report())
        .await?;
    f.db.complete_drain(f.id, f.source, f.process, &report())
        .await?;
    assert_eq!(f.db.get(f.id).await?.phase, HandoffPhase::Activated);
    // Swapping slots uses a fresh deployment identity, never a previous release.
    assert!(f
        .db
        .request(Uuid::new_v4(), f.destination, f.source)
        .await
        .is_err());
    f.db.request(Uuid::new_v4(), f.destination, Uuid::new_v4())
        .await?;
    assert!(f
        .db
        .activate(f.id, f.destination, &f.candidates, &report())
        .await
        .is_err());
    f.request().await?; // An old deployment retry cannot reset the table.
    assert_eq!(f.db.get(f.id).await?.drain_report, Some(report()));
    assert!(
        check_handoff_startup(&f.db.pool, &f.schema, &f.source.to_string(), false)
            .await
            .is_err()
    );
    f.cleanup().await
}

#[tokio::test]
async fn competing_deployments_and_drain_calls_do_not_both_win() -> Result<()> {
    let f = Fixture::new().await?;
    let other = Uuid::new_v4();
    let (first, second) = tokio::join!(f.request(), f.db.request(other, f.source, Uuid::new_v4()));
    assert_ne!(first.is_ok(), second.is_ok());
    let winner = if first.is_ok() { f.id } else { other };
    let competing = Uuid::new_v4();
    let (first, second) = tokio::join!(
        f.db.start_drain(winner, f.source, f.process, &f.candidates),
        f.db.start_drain(winner, f.source, competing, &f.candidates),
    );
    assert_ne!(first.is_ok(), second.is_ok());
    let stored = f.db.get(winner).await?;
    assert_eq!(stored.phase, HandoffPhase::Draining);
    assert!(stored.source_process == Some(f.process) || stored.source_process == Some(competing));
    f.cleanup().await
}

#[tokio::test]
async fn startup_denies_restarted_source_and_unconfigured_writers() -> Result<()> {
    let f = Fixture::new().await?;
    assert!(
        check_handoff_startup(&f.db.pool, &f.schema, &f.source.to_string(), false)
            .await
            .is_err()
    );
    f.request().await?;
    check_handoff_startup(&f.db.pool, &f.schema, &f.source.to_string(), false).await?;
    check_handoff_startup(&f.db.pool, &f.schema, &f.destination.to_string(), true).await?;
    assert!(
        check_handoff_startup(&f.db.pool, &f.schema, &f.destination.to_string(), false)
            .await
            .is_err()
    );
    assert!(
        check_handoff_startup(&f.db.pool, &f.schema, &Uuid::new_v4().to_string(), true)
            .await
            .is_err()
    );
    for mode in [AccessMode::ReadWrite, AccessMode::ExistingSchema] {
        assert!(PostgresClient::new(&f.url, &f.schema, mode).await.is_err());
    }
    let candidate = PostgresClient::new(&f.url, &f.schema, AccessMode::EnforcedReadOnly).await?;
    assert!(candidate
        .pool
        .execute("CREATE TABLE must_not_exist (id int)")
        .await
        .is_err());
    candidate.pool.close().await;
    let source =
        PostgresClient::new_handoff_source(&f.url, &f.schema, &f.source.to_string()).await?;
    f.drain().await?;
    assert!(
        PostgresClient::new_handoff_source(&f.url, &f.schema, &f.source.to_string())
            .await
            .is_err()
    );
    // Existing A may open a replacement connection while settling its drain.
    let connection = source.pool.acquire().await?;
    connection.close().await?;
    assert_eq!(
        sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&source.pool)
            .await?,
        1
    );
    source.pool.close().await;
    f.db.complete_drain(f.id, f.source, f.process, &report())
        .await?;
    check_handoff_startup(&f.db.pool, &f.schema, &f.destination.to_string(), true).await?;
    f.db.activate(f.id, f.destination, &f.candidates, &report())
        .await?;
    assert!(
        check_handoff_startup(&f.db.pool, &f.schema, &f.destination.to_string(), true)
            .await
            .is_err()
    );
    assert!(
        check_handoff_startup(&f.db.pool, &f.schema, &f.destination.to_string(), false)
            .await
            .is_err()
    );
    f.cleanup().await
}

#[tokio::test]
async fn startup_waits_for_inflight_drain_commit() -> Result<()> {
    let f = Fixture::new().await?;
    f.request().await?;
    let mut tx = f.db.pool.begin().await?;
    sqlx::query(&format!("UPDATE {} SET phase='draining', source_process=$1, candidate_processes=$2 WHERE handoff_id=$3", f.db.table))
        .bind(f.process).bind(f.candidates.as_slice()).bind(f.id).execute(&mut *tx).await?;
    let pool = f.db.pool.clone();
    let schema = f.schema.clone();
    let source = f.source.to_string();
    let check =
        tokio::spawn(async move { check_handoff_startup(&pool, &schema, &source, false).await });
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !check.is_finished(),
        "startup must wait for the drain transaction"
    );
    tx.commit().await?;
    assert!(
        check.await?.is_err(),
        "committed drain must deny replacement startup"
    );
    f.cleanup().await
}
