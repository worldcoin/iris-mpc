#![cfg(feature = "db_dependent")]

mod test_utils;

use eyre::Result;
use std::sync::Mutex;
use test_utils::*;

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

        let ep = assert_consistent_rerand_epoch(&env.harness).await?;
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

        let ep = assert_consistent_rerand_epoch(&env.harness).await?;
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
                sqlx::query("UPDATE irises SET left_code = left_code WHERE id = $1")
                    .bind(id)
                    .execute(&party.store.pool)
                    .await?;
            }
        }
        println!("[phase 3]   bumped version_id on {:?}", modified_ids);

        wait_epoch_done(&env.harness, 0).await?;
        stop_all(t, h).await;

        let ep = assert_consistent_rerand_epoch(&env.harness).await?;
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

        let ep = assert_consistent_rerand_epoch(&env.harness).await?;
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

        let ep = assert_consistent_rerand_epoch(&env.harness).await?;
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

        let ep = assert_consistent_rerand_epoch(&env.harness).await?;
        assert!(ep >= 2, "Expected rerand_epoch >= 2, got {}", ep);
        verify_fingerprints(&env.harness, &env.fingerprints, &[]).await?;
        println!("[phase 6] PASSED (epoch={})", ep);

        env.teardown().await
    });
}

// ============================================================================
// Phase 7: Epoch boundary desync -- simulate epoch mismatch
// ============================================================================

#[test]
fn phase7_epoch_boundary_desync() {
    run_async(async {
        let _ = tracing_subscriber::fmt::try_init();
        let env = TestEnv::setup().await?;
        println!("[phase 7] Epoch boundary desync...");

        // Setup the exact boundary desync state in DB manually to test catch-up logic
        // P1 is on Epoch 0 (has max epoch 0)
        // P0 and P2 are on Epoch 1 (have max epoch 1)
        for p in 0..NUM_PARTIES {
            let pool = &env.harness.parties[p].store.pool;
            // Everyone completes Epoch 0
            sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (0, 0, TRUE, TRUE, TRUE)")
                .execute(pool).await.unwrap();
        }

        // P0 and P2 move to Epoch 1
        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (1, 0, TRUE, TRUE, FALSE)")
            .execute(&env.harness.parties[0].store.pool).await.unwrap();
        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (1, 0, TRUE, TRUE, FALSE)")
            .execute(&env.harness.parties[2].store.pool).await.unwrap();

        // Now simulate P1 main server startup (P1 is behind on Epoch 0)
        // Should catch up using safe_up_to = i32::MAX
        let r1 = simulate_server_startup(&env.harness, 1).await;
        assert!(r1.is_ok(), "P1 startup failed during epoch mismatch");

        // Now simulate P0 main server startup (P0 is ahead on Epoch 1)
        // Should catch up using safe_up_to = -1 (nobody confirmed Epoch 1 yet since P1 hasn't started it)
        let r0 = simulate_server_startup(&env.harness, 0).await;
        assert!(r0.is_ok(), "P0 startup failed during epoch mismatch");

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

        let r1 = simulate_server_startup(&env.harness, 1).await;
        assert!(r1.is_err(), "P1 startup should have failed due to large epoch gap");

        // Now test the new chunk gap logic
        // P1 has chunk 0 confirmed, P0 has chunk 2 confirmed (gap > 1) in the same epoch
        for p in 0..NUM_PARTIES {
            let pool = &env.harness.parties[p].store.pool;
            sqlx::query("DELETE FROM rerand_progress").execute(pool).await.unwrap();
        }
        
        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (3, 0, TRUE, TRUE, TRUE)")
            .execute(&env.harness.parties[1].store.pool).await.unwrap();
            
        sqlx::query("INSERT INTO rerand_progress (epoch, chunk_id, staging_written, all_confirmed, live_applied) VALUES (3, 2, TRUE, TRUE, FALSE)")
            .execute(&env.harness.parties[0].store.pool).await.unwrap();
            
        let r1_chunk_desync = simulate_server_startup(&env.harness, 1).await;
        assert!(r1_chunk_desync.is_err(), "P1 startup should have failed due to large chunk gap");

        println!("[phase 8] PASSED");

        env.teardown().await
    });
}
