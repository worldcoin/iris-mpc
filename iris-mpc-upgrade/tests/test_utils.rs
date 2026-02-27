#![allow(dead_code)]

use eyre::Result;
use iris_mpc_common::{
    config::CommonConfig,
    galois_engine::degree4::FullGaloisRingIrisCodeShare,
    helpers::sync::{SyncResult, SyncState},
    iris_db::iris::IrisCode,
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_store::rerand::{self as rerand_store};
use iris_mpc_store::{Store, StoredIrisRef};
use iris_mpc_upgrade::config::RerandomizeContinuousConfig;
use iris_mpc_upgrade::continuous_rerand::run_continuous_rerand;
use iris_mpc_upgrade::rerandomization::reconstruct_shares;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub const NUM_PARTIES: usize = 3;
pub const DB_SIZE: usize = 50;
pub const CHUNK_SIZE: u64 = 25;

fn db_urls() -> Vec<String> {
    (0..3)
        .map(|i| format!("postgres://postgres:postgres@localhost:{}", 6200 + i))
        .collect()
}

pub struct PartyDb {
    pub store: Store,
    pub schema_name: String,
}

pub struct TestHarness {
    pub parties: Vec<PartyDb>,
}

impl TestHarness {
    pub async fn new(db_urls: &[&str], schema_prefix: &str) -> Result<Self> {
        let mut parties = Vec::new();
        for (i, url) in db_urls.iter().enumerate() {
            let schema = format!("{}_{}", schema_prefix, i);
            let pg = PostgresClient::new(url, &schema, AccessMode::ReadWrite).await?;
            let store = Store::new(&pg).await?;
            rerand_store::ensure_staging_schema(
                &store.pool,
                &rerand_store::staging_schema_name(&schema),
            )
            .await?;
            parties.push(PartyDb {
                store,
                schema_name: schema,
            });
        }
        Ok(Self { parties })
    }

    pub fn store(&self, party: usize) -> &Store {
        &self.parties[party].store
    }
}

/// Full test environment: harness + AWS clients + unique prefix + unique S3 bucket.
pub struct TestEnv {
    pub harness: TestHarness,
    pub s3: aws_sdk_s3::Client,
    pub sm: aws_sdk_secretsmanager::Client,
    pub prefix: String,
    pub bucket: String,
    pub fingerprints: PlaintextFingerprints,
}

impl TestEnv {
    pub async fn setup() -> Result<Self> {
        let id = rand::random::<u32>();
        let prefix = format!("SMPC_e2e_{}", id);
        let bucket = format!("rerand-e2e-{}", id);
        let urls = db_urls();
        let url_refs: Vec<&str> = urls.iter().map(|s| s.as_str()).collect();
        let harness = TestHarness::new(&url_refs, &prefix).await?;

        let sdk = aws_config::from_env().load().await;
        let s3 = aws_sdk_s3::Client::new(&sdk);
        let sm = aws_sdk_secretsmanager::Client::new(&sdk);

        s3.create_bucket().bucket(&bucket).send().await
            .map_err(|e| eyre::eyre!("Failed to create bucket {}: {}", bucket, e))?;

        println!("  [setup] Seeding {} irises (prefix={}, bucket={})", DB_SIZE, prefix, bucket);
        seed_three_party_db(&harness, DB_SIZE).await?;
        let fingerprints = snapshot_all_fingerprints(&harness).await?;

        Ok(Self { harness, s3, sm, prefix, bucket, fingerprints })
    }

    pub async fn teardown(&self) -> Result<()> {
        cleanup(&self.harness).await?;
        // Delete all objects in the bucket then delete the bucket
        let mut token = None;
        loop {
            let mut req = self.s3.list_objects_v2().bucket(&self.bucket);
            if let Some(t) = &token { req = req.continuation_token(t); }
            let resp = req.send().await?;
            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    self.s3.delete_object().bucket(&self.bucket).key(key).send().await?;
                }
            }
            if resp.is_truncated() == Some(true) {
                token = resp.next_continuation_token().map(|s| s.to_string());
            } else { break; }
        }
        let _ = self.s3.delete_bucket().bucket(&self.bucket).send().await;
        Ok(())
    }

    pub fn make_config(&self, party_id: u8) -> RerandomizeContinuousConfig {
        RerandomizeContinuousConfig {
            party_id,
            db_url: format!(
                "postgres://postgres:postgres@localhost:{}",
                6200 + party_id as u16
            ),
            env: "testing".to_string(),
            s3_bucket: self.bucket.clone(),
            schema_name: format!("{}_{}", self.prefix, party_id),
            chunk_size: CHUNK_SIZE,
            chunk_delay_secs: 0,
            safety_buffer_ids: 0,
            s3_poll_interval_ms: 200,
            healthcheck_port: 3020 + party_id as usize,
        }
    }

    pub fn spawn_rerand(&self, party_id: u8) -> (tokio::task::JoinHandle<Result<()>>, CancellationToken) {
        let config = self.make_config(party_id);
        let s3 = self.s3.clone();
        let sm = self.sm.clone();
        let store = self.harness.store(party_id as usize).clone();
        let token = CancellationToken::new();
        let tc = token.clone();
        let h = tokio::spawn(async move {
            run_continuous_rerand(&config, &s3, &sm, &store, Some(&tc)).await
        });
        (h, token)
    }

    pub fn spawn_all(&self) -> (Vec<tokio::task::JoinHandle<Result<()>>>, Vec<CancellationToken>) {
        let mut handles = Vec::new();
        let mut tokens = Vec::new();
        for p in 0u8..3 {
            let (h, t) = self.spawn_rerand(p);
            handles.push(h);
            tokens.push(t);
        }
        (handles, tokens)
    }
}

pub async fn stop_all(
    tokens: Vec<CancellationToken>,
    handles: Vec<tokio::task::JoinHandle<Result<()>>>,
) {
    for t in &tokens { t.cancel(); }
    for h in &handles { h.abort(); }
    for h in handles { let _ = h.await; }
}

// ---- DB seeding ----

pub async fn seed_three_party_db(harness: &TestHarness, count: usize) -> Result<()> {
    let mut rng = StdRng::seed_from_u64(42);
    for chunk_start in (1..=count).step_by(100) {
        let chunk_end = std::cmp::min(chunk_start + 100, count + 1);

        struct S { id: i64, lc: Vec<u16>, lm: Vec<u16>, rc: Vec<u16>, rm: Vec<u16> }

        let mut party_data: Vec<Vec<S>> = (0..NUM_PARTIES).map(|_| Vec::new()).collect();
        for serial_id in chunk_start..chunk_end {
            let il = IrisCode::random_rng(&mut rng);
            let ir = IrisCode::random_rng(&mut rng);
            let [l0, l1, l2] = FullGaloisRingIrisCodeShare::encode_iris_code(&il, &mut rng);
            let [r0, r1, r2] = FullGaloisRingIrisCodeShare::encode_iris_code(&ir, &mut rng);
            for (pi, (left, right)) in [(l0, r0), (l1, r1), (l2, r2)].into_iter().enumerate() {
                party_data[pi].push(S {
                    id: serial_id as i64,
                    lc: left.code.coefs.to_vec(), lm: left.mask.coefs.to_vec(),
                    rc: right.code.coefs.to_vec(), rm: right.mask.coefs.to_vec(),
                });
            }
        }
        for (pi, shares) in party_data.iter().enumerate() {
            let refs: Vec<StoredIrisRef> = shares.iter().map(|s| StoredIrisRef {
                id: s.id, left_code: &s.lc, left_mask: &s.lm,
                right_code: &s.rc, right_mask: &s.rm,
            }).collect();
            let store = harness.store(pi);
            let mut tx = store.tx().await?;
            store.insert_irises_overriding(&mut tx, &refs).await?;
            tx.commit().await?;
        }
    }
    Ok(())
}

// ---- Fingerprint verification ----

/// blake3 hash of the concatenated reconstructed plaintext (left_code ++ left_mask
/// ++ right_code ++ right_mask) for every iris ID.
pub type PlaintextFingerprints = HashMap<i64, [u8; 32]>;

/// Compute a fingerprint for every iris in the DB by reconstructing shares
/// from all 3 parties.
pub async fn snapshot_all_fingerprints(harness: &TestHarness) -> Result<PlaintextFingerprints> {
    let ids: Vec<(i64,)> = sqlx::query_as("SELECT id FROM irises ORDER BY id")
        .fetch_all(&harness.store(0).pool)
        .await?;

    let mut fps = PlaintextFingerprints::new();
    for (id,) in ids {
        let mut shares = Vec::new();
        for party in 0..NUM_PARTIES {
            shares.push(harness.store(party).get_iris_data_by_id(id).await?);
        }
        let mut hasher = blake3::Hasher::new();
        let fields: Vec<[&[u16]; 3]> = vec![
            [shares[0].left_code(), shares[1].left_code(), shares[2].left_code()],
            [shares[0].left_mask(), shares[1].left_mask(), shares[2].left_mask()],
            [shares[0].right_code(), shares[1].right_code(), shares[2].right_code()],
            [shares[0].right_mask(), shares[1].right_mask(), shares[2].right_mask()],
        ];
        for [s0, s1, s2] in &fields {
            let recon = reconstruct_shares(s0, s1, s2);
            hasher.update(bytemuck::cast_slice::<u16, u8>(&recon));
        }
        fps.insert(id, *hasher.finalize().as_bytes());
    }
    Ok(fps)
}

/// Verify that current shares reconstruct to the same plaintexts as the
/// snapshot. `skip_ids` are excluded (modified during test).
pub async fn verify_fingerprints(
    harness: &TestHarness,
    expected: &PlaintextFingerprints,
    skip_ids: &[i64],
) -> Result<()> {
    let current = snapshot_all_fingerprints(harness).await?;
    let mut checked = 0;
    for (id, exp) in expected {
        if skip_ids.contains(id) {
            continue;
        }
        let cur = current
            .get(id)
            .unwrap_or_else(|| panic!("ID {} missing from current DB", id));
        assert_eq!(exp, cur, "Plaintext fingerprint mismatch for id {}", id);
        checked += 1;
    }
    println!("  verified {}/{} iris fingerprints", checked, expected.len());
    Ok(())
}

// ---- Polling helpers ----

pub async fn wait_epoch_done(harness: &TestHarness, epoch: i32) -> Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(120);
    let start = std::time::Instant::now();
    let mut last_print = start;
    loop {
        if tokio::time::Instant::now() > deadline {
            eyre::bail!("Timeout waiting for epoch {}", epoch);
        }
        let mut done = true;
        let mut applied = [0usize; 3];
        for (i, party) in harness.parties.iter().enumerate() {
            let rows: Vec<(bool,)> = sqlx::query_as(
                "SELECT live_applied FROM rerand_progress WHERE epoch = $1",
            ).bind(epoch).fetch_all(&party.store.pool).await?;
            applied[i] = rows.iter().filter(|(a,)| *a).count();
            if rows.is_empty() || !rows.iter().all(|(a,)| *a) { done = false; }
        }
        if done {
            println!("  epoch {} done in {:.1}s", epoch, start.elapsed().as_secs_f64());
            return Ok(());
        }
        if last_print.elapsed() > Duration::from_secs(5) {
            println!("  waiting epoch {}: applied {:?} ({:.0}s)", epoch, applied, start.elapsed().as_secs_f64());
            last_print = std::time::Instant::now();
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

pub async fn wait_chunks_staged(harness: &TestHarness, epoch: i32, n: i32) -> Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let start = std::time::Instant::now();
    loop {
        if tokio::time::Instant::now() > deadline {
            eyre::bail!("Timeout waiting for {} chunks staged in epoch {}", n, epoch);
        }
        let mut max_count = 0i64;
        for party in &harness.parties {
            let (count,): (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM rerand_progress WHERE epoch = $1 AND staging_written = TRUE",
            ).bind(epoch).fetch_one(&party.store.pool).await?;
            max_count = max_count.max(count);
        }
        if max_count >= n as i64 {
            println!("  {} chunks staged for epoch {} in {:.1}s", max_count, epoch, start.elapsed().as_secs_f64());
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

// ---- Server simulation ----

pub async fn simulate_server_startup(harness: &TestHarness, party: usize) -> Result<()> {
    let sync_result = build_test_sync_result(harness, party).await?;
    let pool = &harness.parties[party].store.pool;
    let schema = &harness.parties[party].schema_name;
    let lock_conn = rerand_store::rerand_catchup_and_lock(pool, schema, &sync_result).await?;
    let _count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM irises").fetch_one(pool).await?;
    rerand_store::release_rerand_lock(lock_conn).await?;
    Ok(())
}

async fn build_test_sync_result(harness: &TestHarness, party: usize) -> Result<SyncResult> {
    let mut all_states = Vec::new();
    for p in &harness.parties {
        let rerand_state = rerand_store::build_rerand_sync_state(&p.store.pool).await.ok();
        all_states.push(SyncState {
            db_len: p.store.count_irises().await? as u64,
            modifications: vec![],
            next_sns_sequence_num: None,
            common_config: CommonConfig::default(),
            rerand_state,
        });
    }
    let my_state = all_states[party].clone();
    Ok(SyncResult { my_state, all_states })
}

pub async fn assert_consistent_rerand_epoch(harness: &TestHarness) -> Result<i32> {
    let mut all: Vec<Vec<(i64, i32)>> = Vec::new();
    for party in &harness.parties {
        all.push(sqlx::query_as("SELECT id, rerand_epoch FROM irises ORDER BY id")
            .fetch_all(&party.store.pool).await?);
    }
    assert_eq!(all[0].len(), all[1].len());
    assert_eq!(all[1].len(), all[2].len());
    for i in 0..all[0].len() {
        assert_eq!(all[0][i].1, all[1][i].1, "epoch mismatch id {} p0 vs p1", all[0][i].0);
        assert_eq!(all[0][i].1, all[2][i].1, "epoch mismatch id {} p0 vs p2", all[0][i].0);
    }
    Ok(all[0].first().map(|(_, e)| *e).unwrap_or(0))
}

async fn cleanup(harness: &TestHarness) -> Result<()> {
    for party in &harness.parties {
        let staging = rerand_store::staging_schema_name(&party.schema_name);
        let _ = sqlx::query(&format!(r#"DROP SCHEMA IF EXISTS "{}" CASCADE"#, staging))
            .execute(&party.store.pool).await;
        let _ = sqlx::query(&format!(r#"DROP SCHEMA IF EXISTS "{}" CASCADE"#, party.schema_name))
            .execute(&party.store.pool).await;
    }
    Ok(())
}
