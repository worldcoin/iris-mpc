//! Passive exact-scan preload. This path never enters active coordination.

use super::*;
use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use iris_mpc_cpu::execution::hawk_main::worker_pool_initializer::InitializedWorkers;
use serde::Serialize;
use tokio::net::TcpListener;
use uuid::Uuid;

const PRELOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);

#[derive(Clone, Serialize)]
struct CandidateStatus {
    process_id: Uuid,
    party_id: usize,
    state: &'static str,
    loaded_rows: usize,
    serving: bool,
}

type Status = Arc<RwLock<CandidateStatus>>;

// The cold-eye pool is deliberately uninitialized; this is not a serving actor.
struct CandidatePreload {
    workers: InitializedWorkers,
}

pub(super) fn validate_config(config: &Config, mode: HawkSearchMode) -> Result<()> {
    if mode != HawkSearchMode::LinearScan {
        bail!("candidate startup requires CPU linear scan");
    }
    let db = config
        .database
        .as_ref()
        .ok_or_else(|| eyre!("candidate requires database"))?;
    if config.party_id > 2 || db.load_parallelism == 0 {
        bail!("candidate requires party_id within 0..=2 and positive database.load_parallelism");
    }
    if config.fake_db_size > 0 || config.init_db_size > 0 || config.clear_db_before_init {
        bail!("candidate cannot seed, clear, or generate database contents");
    }
    if config.enable_s3_importer {
        bail!("candidate preload currently requires enable_s3_importer=false");
    }
    process_config(config, mode)
}

fn routes(status: Status) -> Router {
    Router::new()
        .route(
            "/candidate-status",
            get(|State(status): State<Status>| async move { Json(status.read().await.clone()) }),
        )
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/ready", get(|| async { StatusCode::SERVICE_UNAVAILABLE }))
        .route(
            "/startup",
            get(|State(status): State<Status>| async move {
                if status.read().await.state == "loaded" {
                    StatusCode::OK
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }),
        )
        .with_state(status)
}

pub(super) async fn run(config: Config, shutdown: Arc<ShutdownHandler>) -> Result<()> {
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, config.candidate_status_port))
        .await
        .wrap_err("bind candidate operator status")?;
    let process_id = Uuid::new_v4();
    let status = Arc::new(RwLock::new(CandidateStatus {
        process_id,
        party_id: config.party_id,
        state: "loading",
        loaded_rows: 0,
        serving: false,
    }));
    tracing::info!(party_id = config.party_id, process_id = %process_id,
        address = %listener.local_addr()?, state = "loading", "CPU candidate preload");
    let status_server = axum::serve(listener, routes(status.clone()));
    tokio::select! {
        result = status_server => result.wrap_err("candidate status server stopped"),
        result = preload_and_wait(&config, shutdown, status) => result,
    }
}

async fn preload_and_wait(
    config: &Config,
    shutdown: Arc<ShutdownHandler>,
    status: Status,
) -> Result<()> {
    let started = Instant::now();
    let loaded = tokio::select! {
        _ = shutdown.wait_for_shutdown() => return Ok(()),
        result = timeout(PRELOAD_TIMEOUT, preload(config, shutdown.clone())) => {
            result.wrap_err("candidate preload exceeded one hour").and_then(|loaded| loaded)
        }
    };
    let workers = match loaded {
        Ok(workers) => workers,
        Err(error) => {
            status.write().await.state = "failed";
            tracing::error!(state = "failed", error = ?error, "CPU candidate preload failed");
            return Err(error);
        }
    };
    let count = workers.workers.registries[0].read().await.size;
    {
        let mut state = status.write().await;
        state.loaded_rows = count;
        state.state = "loaded";
    }
    tracing::info!(
        state = "loaded",
        loaded_rows = count,
        duration_seconds = started.elapsed().as_secs_f64(),
        "CPU candidate remains passive and unreconciled"
    );
    shutdown.wait_for_shutdown().await;
    // Keep the pools (and their resident shares and observed versions) alive
    // until shutdown. A later handoff must reconcile before building an actor.
    drop(workers);
    Ok(())
}

async fn preload(config: &Config, shutdown: Arc<ShutdownHandler>) -> Result<CandidatePreload> {
    let db = config
        .database
        .as_ref()
        .ok_or_else(|| eyre!("candidate requires database"))?;
    let postgres = PostgresClient::new(
        &db.url,
        &config.get_gpu_db_schema(),
        AccessMode::EnforcedReadOnly,
    )
    .await
    .wrap_err("connect candidate read-only database")?;
    let store = Store::new(&postgres).await?;
    // The shared loader requires dense serial IDs. Fail explicitly rather
    // than letting COUNT(*) silently truncate an unexpected sparse database.
    let (count, max): (i64, i64) =
        sqlx::query_as("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM irises")
            .fetch_one(&store.pool)
            .await?;
    if count != max {
        bail!("candidate preload requires dense serial IDs");
    }
    let max_serial_id = usize::try_from(max)?;
    let workers = Box::new(
        LocalWorkerPoolInitializer::new_load_from_db(
            config.party_id,
            HAWK_DISTANCE_MODE,
            config.hawk_numa,
            DbLoadParams {
                store,
                config: Arc::new(config.clone()),
                max_serial_id,
                parallelism: db.load_parallelism,
                s3_max_serial_id: None,
                shutdown_handler: shutdown,
                resident_side: Some(match config.full_scan_side {
                    ampc_anon_stats::types::Eye::Left => 0,
                    ampc_anon_stats::types::Eye::Right => 1,
                }),
            },
        )
        .with_resident_layout(HawkActor::resident_layout_for(HawkSearchMode::LinearScan))
        .defer_cold_storage(),
    )
    .initialize()
    .await?;
    Ok(CandidatePreload { workers })
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris_mpc_common::{config::DbConfig, VectorId, IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
    use sqlx::Executor;

    fn config() -> Config {
        serde_json::from_value(serde_json::json!({
            "cpu_startup_mode": "candidate",
            "max_batch_size": 1,
            "disable_persistence": false,
            "full_scan_side_switching_enabled": false,
            "enable_s3_importer": false,
            "hawk_numa": false,
            "luc_enabled": true,
            "luc_lookback_records": 10,
            "database": {"url": "unused", "load_parallelism": 2}
        }))
        .unwrap()
    }

    #[test]
    fn candidate_config_is_explicit_and_rejects_unsupported_modes() {
        let mut config = config();
        validate_config(&config, HawkSearchMode::LinearScan).unwrap();
        assert!(validate_config(&config, HawkSearchMode::Hnsw).is_err());
        config.enable_s3_importer = true;
        assert!(validate_config(&config, HawkSearchMode::LinearScan).is_err());
        config.enable_s3_importer = false;
        config.clear_db_before_init = true;
        assert!(validate_config(&config, HawkSearchMode::LinearScan).is_err());
        let ordinary: Config = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(ordinary.cpu_startup_mode, CpuStartupMode::Active);
        assert!(serde_json::from_value::<Config>(
            serde_json::json!({"cpu_startup_mode": "canidate"})
        )
        .is_err());
    }

    #[tokio::test]
    #[ignore = "requires POP4370_TEST_DATABASE_URL pointing to disposable Postgres"]
    async fn candidate_preloads_read_only_and_never_enters_live_startup() -> Result<()> {
        tokio::task::LocalSet::new()
            .run_until(check_candidate_startup())
            .await
    }

    async fn check_candidate_startup() -> Result<()> {
        let mut config = config();
        config.schema_name = format!("candidate_{}", Uuid::new_v4().simple());
        config.database = Some(DbConfig {
            url: std::env::var("POP4370_TEST_DATABASE_URL")?,
            load_parallelism: 2,
            ..Default::default()
        });
        let schema = config.get_gpu_db_schema();
        let writer = PostgresClient::new(
            &config.database.as_ref().unwrap().url,
            &schema,
            AccessMode::ReadWrite,
        )
        .await?;
        // Only the share table exists: migrations, claim resets, replay or
        // graph initialization would fail. AWS/peer config is absent too.
        writer.pool.execute("CREATE TABLE irises (id bigint PRIMARY KEY, version_id smallint NOT NULL, left_code bytea NOT NULL, left_mask bytea NOT NULL, right_code bytea NOT NULL, right_mask bytea NOT NULL)").await?;
        let code = vec![0_u8; IRIS_CODE_LENGTH * 2];
        let mask = vec![0_u8; MASK_CODE_LENGTH * 2];
        sqlx::query("INSERT INTO irises VALUES (1, 7, $1, $2, $1, $2)")
            .bind(&code)
            .bind(&mask)
            .execute(&writer.pool)
            .await?;

        let reader = PostgresClient::new(
            &config.database.as_ref().unwrap().url,
            &schema,
            AccessMode::EnforcedReadOnly,
        )
        .await?;
        let mut first = reader.pool.acquire().await?;
        let mut second = reader.pool.acquire().await?;
        for connection in [&mut first, &mut second] {
            for statement in ["DELETE FROM irises", "CREATE TABLE must_not_exist (id int)"] {
                let error = sqlx::query(statement)
                    .execute(&mut **connection)
                    .await
                    .unwrap_err();
                assert_eq!(
                    error.as_database_error().and_then(|e| e.code()).as_deref(),
                    Some("25006")
                );
            }
        }
        drop(first);
        drop(second);

        let shutdown = Arc::new(ShutdownHandler::new(0));
        let candidate = preload(&config, shutdown.clone()).await?;
        for registry in &candidate.workers.registries {
            assert_eq!(registry.read().await.get_current_version(1), Some(7));
            assert_eq!(registry.read().await.size, 1);
        }
        // A continues writing. Candidate retains the versions it actually
        // observed, without trying to refresh its cold cache from live state.
        sqlx::query("UPDATE irises SET version_id = 8 WHERE id = 1")
            .execute(&writer.pool)
            .await?;
        assert_eq!(
            candidate.workers.registries[0]
                .read()
                .await
                .get_current_version(1),
            Some(7)
        );
        assert!(candidate.workers.registries[0]
            .read()
            .await
            .get_vector(&VectorId::new(1, 7))
            .is_some());
        drop(candidate);

        let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
        config.candidate_status_port = listener.local_addr()?.port();
        drop(listener);
        let base = format!("http://127.0.0.1:{}", config.candidate_status_port);
        let mut lock = writer.pool.begin().await?;
        sqlx::query("LOCK TABLE irises IN ACCESS EXCLUSIVE MODE")
            .execute(&mut *lock)
            .await?;
        let task = tokio::task::spawn_local(super::super::linear_scan_server_main(config.clone()));
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()?;
        timeout(Duration::from_secs(10), async {
            loop {
                match client.get(format!("{base}/candidate-status")).send().await {
                    Ok(response) => {
                        let status: serde_json::Value = response.json().await?;
                        assert_eq!(status["state"], "loading");
                        break Ok::<_, Report>(());
                    }
                    Err(error) if error.is_connect() => {}
                    Err(error) => return Err(error.into()),
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await??;
        assert_eq!(
            client.get(format!("{base}/startup")).send().await?.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            client.get(format!("{base}/health")).send().await?.status(),
            StatusCode::OK
        );
        lock.commit().await?;
        let loaded = timeout(Duration::from_secs(20), async {
            loop {
                match client.get(format!("{base}/candidate-status")).send().await {
                    Ok(response) => {
                        let status: serde_json::Value = response.json().await?;
                        if status["state"] == "loaded" {
                            break Ok::<_, Report>(status);
                        }
                    }
                    Err(error) if error.is_connect() => {}
                    Err(error) => return Err(error.into()),
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        })
        .await??;
        Uuid::parse_str(loaded["process_id"].as_str().unwrap())?;
        let again: serde_json::Value = client
            .get(format!("{base}/candidate-status"))
            .send()
            .await?
            .json()
            .await?;
        assert_eq!(again["process_id"], loaded["process_id"]);
        assert_eq!(loaded["loaded_rows"], 1);
        assert_eq!(loaded["serving"], false);
        assert!(!task.is_finished(), "candidate must retain its memory");
        assert_eq!(
            client.get(format!("{base}/ready")).send().await?.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            client.get(format!("{base}/startup")).send().await?.status(),
            StatusCode::OK
        );
        assert_eq!(
            client.get(format!("{base}/health")).send().await?.status(),
            StatusCode::OK
        );
        assert_eq!(
            client.get(format!("{base}/sync")).send().await?.status(),
            StatusCode::NOT_FOUND
        );
        let version: (i16,) = sqlx::query_as("SELECT version_id FROM irises WHERE id = 1")
            .fetch_one(&writer.pool)
            .await?;
        assert_eq!(version.0, 8);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());

        let shutdown = Arc::new(ShutdownHandler::new(0));
        let test_shutdown = shutdown.clone();
        let run_task = tokio::spawn(run(config.clone(), shutdown));
        test_shutdown.trigger_manual_shutdown();
        timeout(Duration::from_secs(5), run_task).await???;
        sqlx::query("UPDATE irises SET id = 2")
            .execute(&writer.pool)
            .await?;
        let error = preload(&config, Arc::new(ShutdownHandler::new(0)))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("dense serial IDs"));
        // A missing table produces a returned startup failure, not a silent
        // live fallback or a process that appears loaded.
        writer.pool.execute("DROP TABLE irises").await?;
        assert!(run(config, Arc::new(ShutdownHandler::new(0)))
            .await
            .is_err());
        writer
            .pool
            .execute(format!("DROP SCHEMA \"{schema}\" CASCADE").as_str())
            .await?;
        reader.pool.close().await;
        writer.pool.close().await;
        Ok(())
    }
}
