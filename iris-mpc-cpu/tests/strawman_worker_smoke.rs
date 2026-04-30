//! End-to-end smoke test for the strawman worker.
//!
//! Spins up one strawman worker against a fresh `LocalIrisWorkerPool` over
//! localhost TCP, drives a `LeaderHandle` against it, and verifies that
//! `cache_queries` and `compute_dot_products` round-trip through the wire
//! protocol successfully. This is the minimum proof that the wire types
//! and the strawman dispatch agree end-to-end. The full conformance suite
//! (running the existing `LocalIrisWorkerPool` test fixtures through
//! `RemoteIrisWorkerPool`) lands with the remote pool itself.

use ampc_actor_utils::{
    execution::player::Identity,
    network::workpool::leader::{build_leader, LeaderArgs},
};
use bytes::Bytes;
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    vector_id::VectorId,
};
use iris_mpc_cpu::{
    hawkers::aby3::aby3_store::DistanceMode,
    strawman_worker::{run_local, StrawmanWorkerArgs},
};
use iris_mpc_worker_protocol::{
    decode_response, encode_request, IrisShare, QueryId, QuerySpec, WorkerRequest, WorkerResponse,
    CENTER_ROTATION,
};
use std::time::Duration;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;

const PARTY_ID: usize = 0;
const JOB_TIMEOUT: Duration = Duration::from_secs(10);

fn find_free_ports(n: usize) -> Vec<u16> {
    let listeners: Vec<std::net::TcpListener> = (0..n)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    listeners
        .iter()
        .map(|l| l.local_addr().unwrap().port())
        .collect()
}

fn sample_iris() -> IrisShare {
    IrisShare {
        code: GaloisRingIrisCodeShare::default_for_party(PARTY_ID),
        mask: GaloisRingTrimmedMaskCodeShare::default_for_party(PARTY_ID),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn strawman_cache_and_dot_product_roundtrip() {
    let _ = tracing_subscriber::fmt::try_init();

    let ports = find_free_ports(2);
    let leader_port = ports[0];
    let worker_port = ports[1];
    let leader_addr = format!("127.0.0.1:{leader_port}");
    let worker_addr = format!("127.0.0.1:{worker_port}");

    let leader_id = Identity("leader".to_string());
    // Workpool requires worker identities to follow `{leader_id}-w-{idx}`.
    let worker_id = "leader-w-0".to_string();

    let shutdown = CancellationToken::new();

    // Start the strawman worker. It builds an empty `LocalIrisWorkerPool`
    // and serves over loopback.
    let strawman_args = StrawmanWorkerArgs {
        worker_id: worker_id.clone(),
        worker_address: worker_addr.clone(),
        leader_id: leader_id.0.clone(),
        leader_address: leader_addr.clone(),
        party_id: PARTY_ID,
        // Match HAWK_DISTANCE_MODE — the production HawkActor uses
        // MinRotation, so the wire path needs to handle it.
        distance_mode: DistanceMode::MinRotation,
        numa: false,
        shard_index: 0,
    };
    let worker_shutdown = shutdown.clone();
    let worker_task = tokio::spawn(async move {
        run_local(strawman_args, worker_shutdown).await.unwrap();
    });

    // Build the leader. wait_for_all_connections gates dispatch.
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
        .wait_for_all_connections(Some(JOB_TIMEOUT))
        .await
        .expect("worker failed to connect");

    // ── 1. cache_queries ─────────────────────────────────────────────────
    let req = WorkerRequest::CacheQueries {
        queries: vec![(QueryId(1), sample_iris())],
    };
    let bytes: Bytes = encode_request(&req).unwrap().into();

    let job = leader
        .broadcast(bytes)
        .await
        .expect("broadcast submit failed");
    let responses = timeout(JOB_TIMEOUT, job)
        .await
        .expect("cache_queries timed out")
        .expect("cache_queries job failed");

    assert_eq!(responses.len(), 1);
    let payload = responses
        .into_iter()
        .next()
        .unwrap()
        .payload
        .expect("worker returned error");
    let response = decode_response(&payload.to_bytes()).expect("decode_response failed");
    match response {
        WorkerResponse::CacheQueries(Ok(())) => {}
        other => panic!("expected CacheQueries(Ok), got {other:?}"),
    }

    // ── 2. compute_dot_products ──────────────────────────────────────────
    // The store is empty, so `compute_dot_products` against any VectorId
    // should fall through `get_vector_or_empty` and return one
    // RingElement per target. We don't assert the value (it's a function
    // of the empty iris and the cached query), only the shape.
    let req = WorkerRequest::ComputeDotProducts {
        batches: vec![(
            QuerySpec {
                query_id: QueryId(1),
                rotation: CENTER_ROTATION,
                mirrored: false,
            },
            vec![VectorId::from_serial_id(0), VectorId::from_serial_id(1)],
        )],
    };
    let bytes: Bytes = encode_request(&req).unwrap().into();

    let job = leader
        .broadcast(bytes)
        .await
        .expect("broadcast submit failed");
    let responses = timeout(JOB_TIMEOUT, job)
        .await
        .expect("dot_products timed out")
        .expect("dot_products job failed");
    let payload = responses
        .into_iter()
        .next()
        .unwrap()
        .payload
        .expect("worker returned error");
    let response = decode_response(&payload.to_bytes()).expect("decode_response failed");
    match response {
        WorkerResponse::ComputeDotProducts(Ok(batches)) => {
            assert_eq!(batches.len(), 1);
            // MinRotation mode packs `2 * HAWK_MIN_DIST_ROTATIONS` elements
            // per target — code & mask shares for each rotation. With 2
            // targets that's `2 * 11 * 2 = 44`. See
            // `protocol::ops::rotation_aware_pairwise_distance_rowmajor`.
            assert_eq!(batches[0].len(), 44);
        }
        other => panic!("expected ComputeDotProducts(Ok), got {other:?}"),
    }

    // ── shutdown ─────────────────────────────────────────────────────────
    shutdown.cancel();
    let _ = timeout(Duration::from_secs(2), worker_task).await;
}
