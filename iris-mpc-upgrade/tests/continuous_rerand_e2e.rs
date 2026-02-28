#![cfg(feature = "db_dependent")]

mod test_utils;

use eyre::Result;
use iris_mpc_store::rerand as rerand_store;
use serde_json::json;
use std::sync::Mutex;
use test_utils::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const STACK_SIZE: usize = 16 * 1024 * 1024;

/// Tests share 3 Postgres instances and a global advisory lock constant, so
/// they must run sequentially. This mutex enforces that even without
/// `--test-threads=1`.
static SERIAL: Mutex<()> = Mutex::new(());

fn run_async(f: impl std::future::Future<Output = Result<()>> + Send + 'static) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let result = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .name("e2e".into())
        .spawn(move || {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(4)
                .thread_stack_size(STACK_SIZE)
                .enable_all()
                .build()
                .unwrap()
                .block_on(f)
        })
        .unwrap()
        .join()
        .unwrap();
    result.unwrap();
}

async fn set_live_applied_chunk(pool: &sqlx::PgPool, epoch: i32, max_chunk: i32) -> Result<()> {
    for chunk in 0..=max_chunk {
        rerand_store::upsert_rerand_progress(pool, epoch, chunk).await?;
        sqlx::query(
            "UPDATE rerand_progress SET live_applied = TRUE WHERE epoch = $1 AND chunk_id = $2",
        )
        .bind(epoch)
        .bind(chunk)
        .execute(pool)
        .await?;
    }
    Ok(())
}

fn spawn_checking_worker(pool: sqlx::PgPool) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok(true) = rerand_store::check_and_handle_freeze(&pool, None).await {}
    })
}

async fn simulate_server_startup_with_freeze(
    pool: &sqlx::PgPool,
    peer_addrs: &[(&str, usize)],
) -> Result<()> {
    rerand_store::freeze_and_verify_watermarks(pool, peer_addrs).await?;

    // Mimic startup DB load behind apply lock.
    let startup_lock = rerand_store::acquire_apply_lock(pool).await?;
    let _: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM irises")
        .fetch_one(pool)
        .await?;
    rerand_store::release_apply_lock(startup_lock).await?;

    Ok(())
}

async fn start_peer_watermark_server(
    pool: &sqlx::PgPool,
) -> Result<(usize, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port() as usize;
    let pool = pool.clone();
    let handle = tokio::spawn(async move {
        loop {
            let (mut socket, _) = match listener.accept().await {
                Ok(value) => value,
                Err(_) => return,
            };

            let pool = pool.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 2048];
                let _ = socket.read(&mut buf).await;

                let wm = match rerand_store::get_applied_watermark_from_pool(&pool).await {
                    Ok(Some((epoch, chunk_id))) => json!({
                        "epoch": epoch,
                        "max_applied_chunk": chunk_id,
                    })
                    .to_string(),
                    Ok(None) => "null".to_string(),
                    Err(e) => {
                        let body = format!("{{\"error\":\"{}\"}}", e);
                        let response = format!(
                            "HTTP/1.1 500 Internal Server Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                            body.len(),
                            body
                        );
                        let _ = socket.write_all(response.as_bytes()).await;
                        return;
                    }
                };

                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    wm.len(),
                    wm
                );
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    });

    Ok((port, handle))
}

// ============================================================================
// Phase 1: Clean epoch -- run one full epoch, verify crypto correctness
// ============================================================================

#[test]
fn phase1_clean_epoch() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 1] Clean epoch...");

        let (h, t) = env.spawn_all();
        wait_epoch_done(&env.harness, 0).await?;
        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness, &[]).await?;
        assert!(ep >= 1, "Expected rerand_epoch >= 1, got {}", ep);
        verify_fingerprints(&env.harness, &env.fingerprints, &[]).await?;
        println!("[phase 1] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 2: Kill-and-resume -- kill mid-epoch, restart, verify recovery
// ============================================================================

#[test]
fn phase2_kill_and_resume() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 2] Kill-and-resume...");

        // Run epoch 0, let 2 chunks stage, then kill
        let (h, t) = env.spawn_all();
        wait_chunks_staged(&env.harness, 0, 2).await?;
        println!("[phase 2]   killing after 2 chunks staged");
        stop_all(t, h).await;

        // Restart -- should resume from where it left off
        println!("[phase 2]   restarting...");
        let (h, t) = env.spawn_all();
        wait_epoch_done(&env.harness, 0).await?;
        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness, &[]).await?;
        assert!(ep >= 1);
        verify_fingerprints(&env.harness, &env.fingerprints, &[]).await?;
        println!("[phase 2] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 3: Concurrent modifications -- bump version_id mid-epoch, verify
//           optimistic lock skips those rows
// ============================================================================

#[test]
fn phase3_concurrent_modifications() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        let modified_ids: Vec<i64> = vec![5, 10, 15];
        println!("[phase 3] Concurrent modifications...");

        let (h, t) = env.spawn_all();
        wait_chunks_staged(&env.harness, 0, 1).await?;

        // Bump version_id on a few rows (simulates a reauth)
        for &id in &modified_ids {
            for party in &env.harness.parties {
                let (before,): (i16,) =
                    sqlx::query_as("SELECT version_id FROM irises WHERE id = $1")
                        .bind(id)
                        .fetch_one(&party.store.pool)
                        .await?;

                // Flip one byte so the `increment_version_id` trigger fires.
                sqlx::query(
                    r#"
                    UPDATE irises
                    SET left_code = set_byte(left_code, 0, get_byte(left_code, 0) # 1)
                    WHERE id = $1
                    "#,
                )
                .bind(id)
                .execute(&party.store.pool)
                .await?;

                let (after,): (i16,) =
                    sqlx::query_as("SELECT version_id FROM irises WHERE id = $1")
                        .bind(id)
                        .fetch_one(&party.store.pool)
                        .await?;
                eyre::ensure!(
                    after > before,
                    "Expected version_id to increase for id={id}"
                );
            }
        }
        println!("[phase 3]   bumped version_id on {:?}", modified_ids);

        wait_epoch_done(&env.harness, 0).await?;
        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness, &modified_ids).await?;
        assert!(ep >= 1);
        verify_fingerprints(&env.harness, &env.fingerprints, &modified_ids).await?;
        println!("[phase 3] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 4: Server restart during rerand -- simulate main server startup while
//           rerand is running, verify advisory lock serializes access
// ============================================================================

#[test]
fn phase4_server_restart_during_rerand() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 4] Server restart during rerand...");

        let (h, t) = env.spawn_all();
        wait_chunks_staged(&env.harness, 0, 1).await?;

        for p in 0..NUM_PARTIES {
            let r = simulate_server_startup(&env.harness, p).await;
            println!("[phase 4]   party {} server startup: {:?}", p, r.is_ok());
        }

        wait_epoch_done(&env.harness, 0).await?;
        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness, &[]).await?;
        assert!(ep >= 1);
        verify_fingerprints(&env.harness, &env.fingerprints, &[]).await?;
        println!("[phase 4] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 5: Staggered restart -- kill one party mid-epoch, restart it, verify
//           it catches up and the epoch completes
// ============================================================================

#[test]
fn phase5_staggered_restart() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 5] Staggered restart...");

        let (h, t) = env.spawn_all();
        wait_chunks_staged(&env.harness, 0, 2).await?;

        // Kill party 0
        println!("[phase 5]   killing party 0 after 2 chunks");
        t[0].cancel();
        h[0].abort();

        // Immediately restart party 0
        println!("[phase 5]   restarting party 0");
        let (h0, t0) = env.spawn_rerand(0);

        wait_epoch_done(&env.harness, 0).await?;

        t0.cancel();
        h0.abort();
        let _ = h0.await;
        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness, &[]).await?;
        assert!(ep >= 1);
        verify_fingerprints(&env.harness, &env.fingerprints, &[]).await?;
        println!("[phase 5] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 6: Multiple Epochs -- let the system run continuously across multiple
//           epochs, verify seamless transition and correct rerandomization
// ============================================================================

#[test]
fn phase6_multiple_epochs() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 6] Multiple epochs...");

        let (h, t) = env.spawn_all();

        // Wait for epoch 0 to finish
        wait_epoch_done(&env.harness, 0).await?;
        println!("[phase 6]   epoch 0 completed");

        // The continuous rerand servers should automatically move to epoch 1
        wait_epoch_done(&env.harness, 1).await?;
        println!("[phase 6]   epoch 1 completed");

        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness, &[]).await?;
        assert!(ep >= 2, "Expected rerand_epoch >= 2, got {}", ep);
        verify_fingerprints(&env.harness, &env.fingerprints, &[]).await?;
        println!("[phase 6] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 7: Startup validation rejects fatal desync and accepts in-sync state.
// ============================================================================

#[test]
fn phase7_startup_validation() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 7] Startup validation...");

        // Fatal desync (gap > 1) → immediate bail
        for p in 0..NUM_PARTIES {
            let pool = &env.harness.parties[p].store.pool;
            sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (0, 0, TRUE, TRUE, TRUE)")
                .execute(pool).await.unwrap();
        }
        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (2, 0, TRUE, TRUE, TRUE)")
            .execute(&env.harness.parties[0].store.pool).await.unwrap();

        let r_fatal = simulate_server_startup_with_rerand_validation(&env.harness, 1).await;
        assert!(r_fatal.is_err(), "Fatal epoch gap should bail immediately");

        // In-sync → startup succeeds immediately
        for p in 0..NUM_PARTIES {
            let pool = &env.harness.parties[p].store.pool;
            sqlx::query("DELETE FROM rerand_progress")
                .execute(pool)
                .await
                .unwrap();
            sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (0, 0, TRUE, TRUE, TRUE)")
                .execute(pool).await.unwrap();
        }

        let r_ok = simulate_server_startup_with_rerand_validation(&env.harness, 0).await;
        assert!(r_ok.is_ok(), "In-sync startup should succeed");

        println!("[phase 7] PASSED");

        env.teardown().await
    });
}

// ============================================================================
// Phase 8: Disallow loading mismatched peers
// ============================================================================

#[test]
fn phase8_reject_desync() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 8] Reject desync...");

        // Setup the exact boundary desync state in DB manually
        // P1 is on Epoch 0 (has max epoch 0)
        // P0 and P2 are on Epoch 2 (have max epoch 2)
        // If a peer is *more than 1 epoch ahead*, we should panic/reject
        for p in 0..NUM_PARTIES {
            let pool = &env.harness.parties[p].store.pool;
            sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (0, 0, TRUE, TRUE, TRUE)")
            .execute(pool).await.unwrap();
        }

        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (2, 0, TRUE, TRUE, FALSE)")
    .execute(&env.harness.parties[0].store.pool).await.unwrap();
        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (2, 0, TRUE, TRUE, FALSE)")
.execute(&env.harness.parties[2].store.pool).await.unwrap();

        let r1 = simulate_server_startup_with_rerand_validation(&env.harness, 1).await;
        assert!(
            r1.is_err(),
            "P1 startup should have failed due to large epoch gap"
        );

        // Now test the new chunk gap logic
        // P1 has chunk 0 confirmed, P0 has chunk 2 confirmed (gap > 1) in the same epoch
        for p in 0..NUM_PARTIES {
            let pool = &env.harness.parties[p].store.pool;
            sqlx::query("DELETE FROM rerand_progress")
                .execute(pool)
                .await
                .unwrap();
        }

        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (3, 0, TRUE, TRUE, TRUE)")
.execute(&env.harness.parties[1].store.pool).await.unwrap();

        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (3, 2, TRUE, TRUE, FALSE)")
.execute(&env.harness.parties[0].store.pool).await.unwrap();

        let r1_chunk_desync = simulate_server_startup_with_rerand_validation(&env.harness, 1).await;
        assert!(
            r1_chunk_desync.is_err(),
            "P1 startup should have failed due to large chunk gap"
        );

        println!("[phase 8] PASSED");

        env.teardown().await
    });
}

// ============================================================================
// Phase 9: Asymmetric modification — a modification landing on only one
//          party's DB must NOT cause cross-party share divergence.
//          The modification fence (version-map exchange + skip-set union)
//          detects the asymmetry and excludes the affected row.
// ============================================================================

#[test]
fn phase9_asymmetric_modification_consistency() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        let target_id: i64 = 20;
        println!("[phase 9] Asymmetric modification consistency...");

        // Modify iris on P0 ONLY — simulates a reauth that propagated to
        // P0 via SQS but hasn't reached P1/P2 yet.
        sqlx::query(
            r#"
            UPDATE irises
            SET left_code = set_byte(left_code, 0, get_byte(left_code, 0) # 1)
            WHERE id = $1
            "#,
        )
        .bind(target_id)
        .execute(&env.harness.parties[0].store.pool)
        .await?;
        println!("[phase 9]   modified id={} on P0 only", target_id);

        // Run a full epoch across all 3 parties.
        let (h, t) = env.spawn_all();
        wait_epoch_done(&env.harness, 0).await?;
        stop_all(t, h).await;

        // The modification fence should have detected the asymmetric
        // version_id and excluded id=20 from the apply on all parties.
        // Non-modified rows should still be rerandomized consistently.
        let ep = assert_consistent_rerand_epoch(&env.harness, &[target_id]).await?;
        assert!(ep >= 1);

        // The modified ID should have been skipped (rerand_epoch stays 0)
        // on ALL parties, OR applied consistently. Either way shares must
        // reconstruct.
        let epochs = get_rerand_epochs_for_id(&env.harness, target_id).await?;
        let epochs_consistent = epochs[0] == epochs[1] && epochs[1] == epochs[2];
        println!(
            "[phase 9]   rerand_epochs for id={}: {:?} (consistent={})",
            target_id, epochs, epochs_consistent
        );
        assert!(
            epochs_consistent,
            "rerand_epoch diverged for id={}: {:?}",
            target_id, epochs
        );

        // Verify non-modified rows reconstruct correctly.
        verify_fingerprints(&env.harness, &env.fingerprints, &[target_id]).await?;

        println!("[phase 9] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 10: Startup freeze catchup path — local party is behind peers and
// advances while freeze is released and re-acquired.
// ============================================================================

#[test]
fn phase10_startup_freeze_local_catchup() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 10] Startup freeze catchup...");

        let p0_pool = &env.harness.parties[0].store.pool;
        let p1_pool = &env.harness.parties[1].store.pool;
        let p2_pool = &env.harness.parties[2].store.pool;

        // Local is behind peers in this epoch.
        set_live_applied_chunk(p0_pool, 0, 0).await?;
        set_live_applied_chunk(p1_pool, 0, 4).await?;
        set_live_applied_chunk(p2_pool, 0, 4).await?;

        let (p1_port, p1_server) = start_peer_watermark_server(p1_pool).await?;
        let (p2_port, p2_server) = start_peer_watermark_server(p2_pool).await?;
        let worker = spawn_checking_worker(p0_pool.clone());

        // Simulate a main-server startup sequence where this party releases freeze
        // so catchup can happen, then re-enters freeze logic.
        let catchup = tokio::spawn({
            let p0_pool = p0_pool.clone();
            async move {
                loop {
                    let (freeze_requested,): (bool,) =
                        sqlx::query_as("SELECT freeze_requested FROM rerand_control WHERE id = 1")
                            .fetch_one(&p0_pool)
                            .await?;
                    if freeze_requested {
                        set_live_applied_chunk(&p0_pool, 0, 4).await?;
                        return Ok::<_, eyre::Report>(());
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                }
            }
        });

        let startup = tokio::time::timeout(
            std::time::Duration::from_secs(25),
            simulate_server_startup_with_freeze(
                p0_pool,
                &[("127.0.0.1", p1_port), ("127.0.0.1", p2_port)],
            ),
        )
        .await;
        assert!(startup.is_ok(), "startup freeze converge timed out");
        startup.unwrap()?;

        assert_eq!(
            rerand_store::get_applied_watermark_from_pool(p0_pool).await?,
            Some((0, 4))
        );
        rerand_store::release_rerand_freeze(p0_pool).await?;
        catchup.await?.unwrap();

        let control = sqlx::query_as::<_, (bool, Option<String>)>(
            "SELECT freeze_requested, freeze_generation FROM rerand_control WHERE id = 1",
        )
        .fetch_one(p0_pool)
        .await?;
        assert!(
            !control.0,
            "freeze should be released after startup converge"
        );
        assert!(
            control.1.is_none(),
            "stale freeze generation should be cleared"
        );

        worker.abort();
        p1_server.abort();
        p2_server.abort();
        env.teardown().await
    });
}

// ============================================================================
// Phase 11: Startup freeze wait path — local party is at max and peers catch up.
// ============================================================================

#[test]
fn phase11_startup_freeze_waits_for_peers() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 11] Startup freeze peer catchup...");

        let p0_pool = &env.harness.parties[0].store.pool;
        let p1_pool = &env.harness.parties[1].store.pool;
        let p2_pool = &env.harness.parties[2].store.pool;

        // Local is fully caught up initially; peers lag at chunk 0.
        set_live_applied_chunk(p0_pool, 0, 4).await?;
        set_live_applied_chunk(p1_pool, 0, 0).await?;
        set_live_applied_chunk(p2_pool, 0, 0).await?;

        let (p1_port, p1_server) = start_peer_watermark_server(p1_pool).await?;
        let (p2_port, p2_server) = start_peer_watermark_server(p2_pool).await?;
        let worker = spawn_checking_worker(p0_pool.clone());

        let advance_peers = tokio::spawn({
            let p1_pool = p1_pool.clone();
            let p2_pool = p2_pool.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                set_live_applied_chunk(&p1_pool, 0, 4).await?;
                set_live_applied_chunk(&p2_pool, 0, 4).await?;
                Result::<(), eyre::Report>::Ok(())
            }
        });

        let startup = tokio::time::timeout(
            std::time::Duration::from_secs(25),
            simulate_server_startup_with_freeze(
                p0_pool,
                &[("127.0.0.1", p1_port), ("127.0.0.1", p2_port)],
            ),
        )
        .await;
        assert!(startup.is_ok(), "startup freeze converge timed out");
        startup.unwrap()?;

        assert_eq!(
            rerand_store::get_applied_watermark_from_pool(p0_pool).await?,
            Some((0, 4))
        );
        rerand_store::release_rerand_freeze(p0_pool).await?;
        advance_peers.await??;

        let control = sqlx::query_as::<_, (bool, Option<String>)>(
            "SELECT freeze_requested, freeze_generation FROM rerand_control WHERE id = 1",
        )
        .fetch_one(p0_pool)
        .await?;
        assert!(
            !control.0,
            "freeze should be released after startup converge"
        );
        assert!(
            control.1.is_none(),
            "stale freeze generation should be cleared"
        );

        worker.abort();
        p1_server.abort();
        p2_server.abort();
        env.teardown().await
    });
}
