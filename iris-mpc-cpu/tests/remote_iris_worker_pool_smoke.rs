//! Smoke test for `RemoteIrisWorkerPool` driving the strawman worker.
//!
//! Production-shaped configuration: `DistanceMode::MinRotation`, party 0,
//! single shard over loopback. Exercises the full trait through the wire:
//! `cache_queries → insert_irises → cache_queries → compute_dot_products
//! → fetch_irises → evict_queries`.
//!
//! This is the closest thing to a conformance test we have until the
//! production worker arrives — when that happens, the same flow can run
//! against the real worker by changing only the binary that's spawned.

use ampc_actor_utils::{
    execution::player::Identity,
    network::workpool::leader::{build_leader, LeaderArgs},
};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{
        iris_worker::{IrisWorkerPool, QueryId, QuerySpec, RemoteIrisWorkerPool, CENTER_ROTATION},
        HAWK_DISTANCE_MODE, HAWK_MIN_DIST_ROTATIONS,
    },
    protocol::shared_iris::{ArcIris, GaloisRingSharedIris},
    strawman_worker::{run_local, StrawmanWorkerArgs},
};
use std::{sync::Arc, time::Duration};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const PARTY_ID: usize = 0;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn find_free_ports(n: usize) -> Vec<u16> {
    let listeners: Vec<std::net::TcpListener> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    listeners
        .iter()
        .map(|l| l.local_addr().unwrap().port())
        .collect()
}

fn fresh_iris() -> ArcIris {
    Arc::new(GaloisRingSharedIris {
        code: GaloisRingIrisCodeShare::default_for_party(PARTY_ID),
        mask: GaloisRingTrimmedMaskCodeShare::default_for_party(PARTY_ID),
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_pool_full_trait_roundtrip_min_rotation() {
    let _ = tracing_subscriber::fmt::try_init();

    let ports = find_free_ports(2);
    let leader_addr = format!("127.0.0.1:{}", ports[0]);
    let worker_addr = format!("127.0.0.1:{}", ports[1]);

    let leader_id = Identity("leader".to_string());
    // Workpool naming convention: workers must announce `{leader_id}-w-{idx}`.
    let worker_id = "leader-w-0".to_string();

    let shutdown = CancellationToken::new();

    // Spawn the strawman worker against an empty in-memory store.
    let strawman_args = StrawmanWorkerArgs {
        worker_id: worker_id.clone(),
        worker_address: worker_addr.clone(),
        leader_id: leader_id.0.clone(),
        leader_address: leader_addr.clone(),
        party_id: PARTY_ID,
        // Production HawkActor uses MinRotation; test the realistic path.
        distance_mode: HAWK_DISTANCE_MODE,
        numa: false,
        shard_index: 0,
    };
    let worker_shutdown = shutdown.clone();
    let worker_task = tokio::spawn(async move {
        run_local(strawman_args, worker_shutdown).await.unwrap();
    });

    // Build leader, gate dispatch on TCP-up.
    let mut leader = build_leader(
        LeaderArgs {
            leader_id,
            leader_address: leader_addr,
            worker_addresses: vec![worker_addr],
            tls: None,
        },
        shutdown.clone(),
    )
    .await
    .expect("failed to build leader");
    leader
        .wait_for_all_connections(Some(CONNECT_TIMEOUT))
        .await
        .expect("worker failed to connect");

    let pool: Arc<dyn IrisWorkerPool> = Arc::new(RemoteIrisWorkerPool::new(
        Arc::new(leader),
        1, // num_shards
        PARTY_ID,
    ));

    // ── 1. cache + insert ────────────────────────────────────────────────
    let query_a = QueryId(1);
    let iris_a = fresh_iris();
    pool.cache_queries(vec![(query_a, iris_a.clone())])
        .await
        .expect("cache_queries(a) failed");

    let vid_0 = VectorId::from_serial_id(0);
    let checksum_after_insert = pool
        .insert_irises(vec![(query_a, vid_0)])
        .await
        .expect("insert_irises failed");
    // We can't assert a specific checksum value (depends on SetHash), but
    // it must be stable across redundant calls — `insert(vid, iris)` is
    // idempotent at the store level. Re-inserting the same content
    // should leave the checksum unchanged.
    let checksum_after_redundant_insert = pool
        .insert_irises(vec![(query_a, vid_0)])
        .await
        .expect("redundant insert_irises failed");
    assert_eq!(
        checksum_after_insert, checksum_after_redundant_insert,
        "redundant insert should not change set_hash checksum"
    );

    // ── 2. cache another query, compute distance against the inserted iris
    let query_b = QueryId(2);
    pool.cache_queries(vec![(query_b, fresh_iris())])
        .await
        .expect("cache_queries(b) failed");

    let spec_b = QuerySpec {
        query_id: query_b,
        rotation: CENTER_ROTATION,
        mirrored: false,
    };
    let dp = pool
        .compute_dot_products(vec![(spec_b, vec![vid_0])])
        .await
        .expect("compute_dot_products failed");
    assert_eq!(dp.len(), 1, "one batch in, one batch out");
    // MinRotation packs `2 * HAWK_MIN_DIST_ROTATIONS` elements per target.
    assert_eq!(dp[0].len(), 2 * HAWK_MIN_DIST_ROTATIONS);

    // ── 3. fetch the inserted iris back, byte-equality with the original
    let fetched = pool
        .fetch_irises(vec![vid_0])
        .await
        .expect("fetch_irises failed");
    assert_eq!(fetched.len(), 1);
    assert_eq!(
        *fetched[0], *iris_a,
        "fetch should return the iris we inserted, byte-for-byte"
    );

    // ── 4. evict, then assert dot product against the now-evicted query fails
    pool.evict_queries(vec![query_b])
        .await
        .expect("evict_queries failed");

    let after_evict = pool.compute_dot_products(vec![(spec_b, vec![vid_0])]).await;
    assert!(
        after_evict.is_err(),
        "compute_dot_products should fail after the query is evicted, got {after_evict:?}"
    );

    // ── shutdown ─────────────────────────────────────────────────────────
    shutdown.cancel();
    let _ = timeout(Duration::from_secs(2), worker_task).await;
}
