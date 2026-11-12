use super::*;
use iris_mpc_common::postgres::{run_migrations, AccessMode, PostgresClient};
use sqlx::{postgres::PgPoolOptions, Executor};

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
        let client = PostgresClient::new(&url, &schema, AccessMode::ExistingSchema).await?;
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
            .record_drain_start(self.id, self.source, self.process, &self.candidates)
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
    assert!(f.db.request(f.id, f.source, Uuid::new_v4()).await.is_err());
    assert!(f
        .db
        .complete_drain(f.id, f.source, f.process, &report())
        .await
        .is_err());
    f.drain().await?;
    assert!(f
        .db
        .record_drain_start(f.id, f.source, Uuid::new_v4(), &f.candidates)
        .await
        .is_err());
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
    assert!(f.db.check_startup(f.source, false).await.is_err());
    f.cleanup().await
}

#[tokio::test]
async fn startup_denies_restarted_source_but_keeps_existing_connections_usable() -> Result<()> {
    let f = Fixture::new().await?;
    assert!(f.db.check_startup(f.source, false).await.is_err());
    f.request().await?;
    f.db.check_startup(f.source, false).await?;
    f.db.check_startup(f.destination, true).await?;
    assert!(f.db.check_startup(f.destination, false).await.is_err());
    assert!(f.db.check_startup(Uuid::new_v4(), true).await.is_err());
    let candidate = PostgresClient::new(&f.url, &f.schema, AccessMode::EnforcedReadOnly).await?;
    assert!(candidate
        .pool
        .execute("CREATE TABLE must_not_exist (id int)")
        .await
        .is_err());
    candidate.pool.close().await;
    let source = PostgresClient::new(&f.url, &f.schema, AccessMode::ReadWrite).await?;
    f.drain().await?;
    assert!(CpuCellHandoffs::new(source.pool.clone(), &f.schema)?
        .check_startup(f.source, false)
        .await
        .is_err());
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
    f.db.check_startup(f.destination, true).await?;
    f.db.activate(f.id, f.destination, &f.candidates, &report())
        .await?;
    assert!(f.db.check_startup(f.destination, true).await.is_err());
    assert!(f.db.check_startup(f.destination, false).await.is_err());
    f.cleanup().await
}

#[tokio::test]
async fn startup_waits_for_inflight_drain_commit() -> Result<()> {
    let f = Fixture::new().await?;
    f.request().await?;
    let mut tx = f.db.pool.begin().await?;
    sqlx::query(&format!("UPDATE {} SET phase='draining', source_process=$1, candidate_processes=$2 WHERE handoff_id=$3", f.db.table))
        .bind(f.process).bind(f.candidates.as_slice()).bind(f.id).execute(&mut *tx).await?;
    let handoffs = f.db.clone();
    let source = f.source;
    let check = tokio::spawn(async move { handoffs.check_startup(source, false).await });
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

#[tokio::test]
async fn handoff_migration_is_inert_until_requested_and_retains_history() -> Result<()> {
    let url = std::env::var("CPU_CELL_TEST_DATABASE_URL")?;
    let schema = format!("upgrade_{}", uuid::Uuid::new_v4().simple());
    let client = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
    let handoffs = CpuCellHandoffs::new(client.pool.clone(), &schema)?;
    run_migrations(&client.pool, false).await?;
    run_migrations(&client.pool, false).await?;
    assert!(!handoffs.is_managed().await?);
    let ordinary = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
    run_migrations(&ordinary.pool, false).await?;
    ordinary.pool.close().await;
    sqlx::query(
        "INSERT INTO cpu_cell_handoffs (handoff_id, source, destination) VALUES ($1,$2,$3)",
    )
    .bind(uuid::Uuid::new_v4())
    .bind(uuid::Uuid::new_v4())
    .bind(uuid::Uuid::new_v4())
    .execute(&client.pool)
    .await?;
    assert!(handoffs.is_managed().await?);
    let ordinary = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
    run_migrations(&ordinary.pool, false).await?;
    ordinary.pool.close().await;
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM cpu_cell_handoffs")
        .fetch_one(&client.pool)
        .await?;
    assert_eq!(count, 1);
    client
        .pool
        .execute(format!("DROP SCHEMA {schema} CASCADE").as_str())
        .await?;
    client.pool.close().await;
    Ok(())
}

#[tokio::test]
async fn handoff_lookup_failure_is_not_legacy_mode() -> Result<()> {
    let pool = PgPoolOptions::new().connect_lazy("postgres://unused:unused@127.0.0.1:1/unused")?;
    pool.close().await;
    assert!(CpuCellHandoffs::new(pool, "canonical")?
        .is_managed()
        .await
        .is_err());
    assert!(PostgresClient::new(
        "postgres://unused:unused@127.0.0.1:1/unused",
        "canonical",
        AccessMode::ReadWrite
    )
    .await
    .is_err());
    Ok(())
}
