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
        })
    }
    async fn request(&self) -> Result<()> {
        self.db
            .request(self.id, self.source, self.destination)
            .await
    }
    async fn drain(&self) -> Result<()> {
        self.db
            .record_drain_start(self.id, self.source, self.process)
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
    serde_json::json!({"watermark": null, "last_publication": null})
}

#[tokio::test]
async fn handoff_transitions_preserve_exact_report_and_history() -> Result<()> {
    let f = Fixture::new().await?;
    f.request().await?;
    assert!(f.db.request(f.id, f.source, Uuid::new_v4()).await.is_err());
    assert!(f
        .db
        .record_drain_completion(f.id, f.source, f.process, &report())
        .await
        .is_err());
    f.drain().await?;
    assert!(f.db.check_startup(f.destination, false).await.is_err());
    assert!(f
        .db
        .record_drain_start(f.id, f.source, Uuid::new_v4())
        .await
        .is_err());
    assert!(f
        .db
        .record_drain_completion(f.id, f.source, Uuid::new_v4(), &report())
        .await
        .is_err());
    f.db.record_drain_completion(f.id, f.source, f.process, &report())
        .await?;
    let different = serde_json::json!({"different": true});
    assert!(f
        .db
        .record_drain_completion(f.id, f.source, f.process, &different)
        .await
        .is_err());
    assert!(f
        .db
        .activate(f.id, f.destination, &different)
        .await
        .is_err());
    f.db.activate(f.id, f.destination, &report()).await?;
    f.db.activate(f.id, f.destination, &report()).await?;
    f.db.record_drain_completion(f.id, f.source, f.process, &report())
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
    assert!(f.db.activate(f.id, f.destination, &report()).await.is_err());
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
    f.db.record_drain_completion(f.id, f.source, f.process, &report())
        .await?;
    f.db.check_startup(f.destination, true).await?;
    assert!(f.db.check_startup(f.destination, false).await.is_err());
    f.db.activate(f.id, f.destination, &report()).await?;
    assert!(f.db.check_startup(f.destination, true).await.is_err());
    f.db.check_startup(f.destination, false).await?;
    assert!(f.db.check_startup(f.source, false).await.is_err());
    assert!(f.db.check_startup(Uuid::new_v4(), false).await.is_err());
    f.cleanup().await
}

#[tokio::test]
async fn startup_waits_for_inflight_drain_commit() -> Result<()> {
    let f = Fixture::new().await?;
    f.request().await?;
    let mut tx = f.db.pool.begin().await?;
    sqlx::query(&format!(
        "UPDATE {} SET phase='draining', source_process=$1 WHERE handoff_id=$2",
        f.db.table
    ))
    .bind(f.process)
    .bind(f.id)
    .execute(&mut *tx)
    .await?;
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
    let mut previous = sqlx::migrate!("./../migrations");
    previous.migrations = previous
        .iter()
        .filter(|migration| migration.version < 20261113000000)
        .cloned()
        .collect::<Vec<_>>()
        .into();
    previous.run(&client.pool).await?;
    previous.run(&client.pool).await?;
    assert!(!handoffs.is_managed().await?);
    let ordinary = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
    previous.run(&ordinary.pool).await?;
    ordinary.pool.close().await;
    let (id, source, destination, process) = (
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
        Uuid::new_v4(),
    );
    let candidates: Vec<_> = (0..3).map(|_| Uuid::new_v4()).collect();
    sqlx::query("INSERT INTO cpu_cell_handoffs (handoff_id, source, destination, phase, source_process, candidate_processes, drain_report) VALUES ($1,$2,$3,'drained',$4,$5,$6)")
    .bind(id).bind(source).bind(destination).bind(process).bind(candidates).bind(report())
    .execute(&client.pool)
    .await?;
    assert!(handoffs.is_managed().await?);
    let completed = handoffs.get(id).await?;
    client.pool.close().await;
    drop(handoffs);
    let ordinary = PostgresClient::new(&url, &schema, AccessMode::ReadWrite).await?;
    run_migrations(&ordinary.pool, false).await?;
    run_migrations(&ordinary.pool, false).await?;
    ordinary.pool.close().await;
    let client = PostgresClient::new(&url, &schema, AccessMode::ExistingSchema).await?;
    let handoffs = CpuCellHandoffs::new(client.pool.clone(), &schema)?;
    assert_eq!(handoffs.get(id).await?, completed);
    let candidate_columns: i64 = sqlx::query_scalar("SELECT count(*) FROM information_schema.columns WHERE table_schema=$1 AND table_name='cpu_cell_handoffs' AND column_name='candidate_processes'")
        .bind(&schema)
        .fetch_one(&client.pool)
        .await?;
    assert_eq!(candidate_columns, 0);
    for invalid in [
        "UPDATE cpu_cell_handoffs SET source_process=NULL",
        "UPDATE cpu_cell_handoffs SET drain_report=NULL",
        "UPDATE cpu_cell_handoffs SET phase='unknown'",
    ] {
        assert!(client.pool.execute(invalid).await.is_err());
    }
    // A replacement B uses the saved completion; no original B process is required.
    let replacement = PostgresClient::new(&url, &schema, AccessMode::ExistingSchema).await?;
    let replacement_handoffs = CpuCellHandoffs::new(replacement.pool.clone(), &schema)?;
    replacement_handoffs
        .check_startup(destination, true)
        .await?;
    replacement_handoffs
        .activate(id, destination, &report())
        .await?;
    assert_eq!(handoffs.get(id).await?.phase, HandoffPhase::Activated);
    assert_eq!(handoffs.get(id).await?.source_process, Some(process));
    replacement.pool.close().await;
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
