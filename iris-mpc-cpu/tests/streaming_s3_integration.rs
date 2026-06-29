//! Integration test for `stream_serialize_and_upload_with` against an
//! S3-compatible endpoint (localstack).
//!
//! Set `S3_TEST_ENDPOINT=http://localhost:4566` to enable; otherwise both
//! tests are skipped. To run locally:
//!
//! ```sh
//! docker run --rm -d --name localstack -p 4566:4566 -e SERVICES=s3 \
//!     public.ecr.aws/localstack/localstack:4.9
//! S3_TEST_ENDPOINT=http://localhost:4566 \
//!     cargo test -p iris-mpc-cpu --test streaming_s3_integration
//! ```

use std::collections::HashMap;

use aws_sdk_s3::{
    config::{BehaviorVersion, Credentials, Region},
    Client as S3Client, Config,
};
use eyre::{eyre, Result};
use iris_mpc_cpu::{
    graph_checkpoint::{
        stream_download_and_deserialize_graph_pair, stream_download_and_deserialize_with,
        stream_serialize_and_upload_with,
    },
    hnsw::graph::layered_graph::GraphMem,
    utils::serialization::{
        graph::GraphFormat,
        types::graph_v3::{EdgeIds, EntryPoint, GraphV3, Layer as V3Layer, VectorId as V3Vid},
        types::graph_v4::{
            EdgeIds as V4EdgeIds, EntryPoint as V4EntryPoint, GraphV4, Layer as V4Layer,
            VectorId as V4Vid,
        },
    },
};

fn s3_test_endpoint() -> Option<String> {
    std::env::var("S3_TEST_ENDPOINT").ok()
}

fn make_test_client(endpoint: &str) -> S3Client {
    let creds = Credentials::new("test", "test", None, None, "test");
    let cfg = Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(creds)
        .endpoint_url(endpoint)
        .force_path_style(true)
        .build();
    S3Client::from_conf(cfg)
}

async fn create_bucket(client: &S3Client, bucket: &str) -> Result<()> {
    client
        .create_bucket()
        .bucket(bucket)
        .send()
        .await
        .map_err(|e| eyre!("create_bucket failed: {e:?}"))?;
    Ok(())
}

async fn cleanup_bucket(client: &S3Client, bucket: &str) {
    if let Ok(list) = client.list_objects_v2().bucket(bucket).send().await {
        for obj in list.contents() {
            if let Some(k) = obj.key() {
                let _ = client.delete_object().bucket(bucket).key(k).send().await;
            }
        }
    }
    let _ = client.delete_bucket().bucket(bucket).send().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streaming_upload_round_trip_multipart() -> Result<()> {
    let Some(endpoint) = s3_test_endpoint() else {
        eprintln!("S3_TEST_ENDPOINT not set; skipping");
        return Ok(());
    };
    let client = make_test_client(&endpoint);
    let bucket = format!("streaming-test-mp-{}", uuid::Uuid::new_v4());
    let key = "round-trip.bin";
    create_bucket(&client, &bucket).await?;

    // ~12 MiB payload so the upload spans 3 parts at 5 MiB each (last ~2 MiB).
    // S3's 5 MiB minimum applies to all-but-last, so this exercises the
    // multi-part path including back-pressure and ordered re-assembly.
    let payload: Vec<u64> = (0..1_500_000u64).collect();
    let buffered = bincode::serialize(&payload).map_err(|e| eyre!("bincode: {e}"))?;
    assert!(buffered.len() > 10 * 1024 * 1024);

    let payload_for_closure = payload;
    let upload = stream_serialize_and_upload_with(
        &client,
        &bucket,
        key,
        move |w| {
            bincode::serialize_into(w, &payload_for_closure).map_err(|e| eyre!("bincode: {e}"))
        },
        5 * 1024 * 1024,
        4,
    )
    .await;
    if let Err(e) = upload {
        cleanup_bucket(&client, &bucket).await;
        return Err(e);
    }

    let downloaded = client
        .get_object()
        .bucket(&bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| eyre!("get_object: {e:?}"))?
        .body
        .collect()
        .await
        .map_err(|e| eyre!("collect body: {e:?}"))?
        .to_vec();

    cleanup_bucket(&client, &bucket).await;

    assert_eq!(downloaded.len(), buffered.len(), "size mismatch");
    assert_eq!(downloaded, buffered, "content mismatch");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_upload_aborts_on_serializer_error() -> Result<()> {
    let Some(endpoint) = s3_test_endpoint() else {
        eprintln!("S3_TEST_ENDPOINT not set; skipping");
        return Ok(());
    };
    let client = make_test_client(&endpoint);
    let bucket = format!("streaming-test-err-{}", uuid::Uuid::new_v4());
    let key = "should-not-exist.bin";
    create_bucket(&client, &bucket).await?;

    let result = stream_serialize_and_upload_with(
        &client,
        &bucket,
        key,
        |_w| Err(eyre!("intentional serializer failure")),
        5 * 1024 * 1024,
        2,
    )
    .await;

    let err = result.expect_err("upload should fail");
    assert!(
        format!("{err}").contains("intentional serializer failure"),
        "unexpected error: {err}"
    );

    // Object must not exist after a failed/aborted multipart upload.
    let head = client.head_object().bucket(&bucket).key(key).send().await;
    assert!(
        head.is_err(),
        "object should not exist after aborted upload"
    );

    cleanup_bucket(&client, &bucket).await;
    Ok(())
}

/// End-to-end: upload via `stream_serialize_and_upload_with`, download via
/// `stream_download_and_deserialize_with`, assert byte-identical value and
/// blake3 hash. Exercises the S3 → `ByteStream::into_async_read` → tee →
/// SyncIoBridge → bincode path that the unit tests bypass.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn streaming_download_round_trip() -> Result<()> {
    let Some(endpoint) = s3_test_endpoint() else {
        eprintln!("S3_TEST_ENDPOINT not set; skipping");
        return Ok(());
    };
    let client = make_test_client(&endpoint);
    let bucket = format!("streaming-test-dl-{}", uuid::Uuid::new_v4());
    let key = "round-trip.bin";
    create_bucket(&client, &bucket).await?;

    // ~12 MiB payload so the upload spans 3 parts; 3 MiB range_size forces
    // the download to issue multiple sequential GetObject(Range) calls and
    // exercises mid-stream back-pressure (pipe capacity 4 MiB << payload).
    let payload: Vec<u64> = (0..1_500_000u64).collect();
    let buffered = bincode::serialize(&payload).map_err(|e| eyre!("bincode: {e}"))?;
    let expected_hash = *blake3::hash(&buffered).as_bytes();
    assert!(buffered.len() > 10 * 1024 * 1024);

    let payload_for_closure = payload.clone();
    let upload = stream_serialize_and_upload_with(
        &client,
        &bucket,
        key,
        move |w| {
            bincode::serialize_into(w, &payload_for_closure).map_err(|e| eyre!("bincode: {e}"))
        },
        5 * 1024 * 1024,
        4,
    )
    .await;
    if let Err(e) = upload {
        cleanup_bucket(&client, &bucket).await;
        return Err(e);
    }

    let download: Result<(Vec<u64>, [u8; 32])> = stream_download_and_deserialize_with(
        &client,
        &bucket,
        key,
        4 * 1024 * 1024,
        3 * 1024 * 1024,
    )
    .await;
    cleanup_bucket(&client, &bucket).await;
    let (got, got_hash) = download?;

    assert_eq!(got, payload, "deserialized value mismatch");
    assert_eq!(got_hash, expected_hash, "hash mismatch");
    Ok(())
}

/// Build a deterministic, non-trivial `[GraphV3; 2]`.
///
/// Each graph has two layers.  `base` offsets the node IDs so the two
/// graphs in the pair have disjoint node sets, making cross-graph
/// contamination visible in assertions.
///
/// The stored `set_hash` values are arbitrary — the streaming reader
/// discards them and lets `Layer::set_links` recompute from scratch,
/// which is exactly what the test exercises.
fn make_v3_pair() -> [GraphV3; 2] {
    let make_graph = |base: u32| {
        let vid = |n: u32| V3Vid { id: n, version: 1 };
        GraphV3 {
            entry_point: vec![EntryPoint {
                point: vid(base),
                layer: 1,
            }],
            layers: vec![
                // Layer 0 — fully connected triangle
                V3Layer {
                    links: HashMap::from([
                        (vid(base), EdgeIds(vec![vid(base + 1), vid(base + 2)])),
                        (vid(base + 1), EdgeIds(vec![vid(base), vid(base + 2)])),
                        (vid(base + 2), EdgeIds(vec![vid(base), vid(base + 1)])),
                    ]),
                    set_hash: 0xdead_beef_u64, // discarded on read
                },
                // Layer 1 — single edge
                V3Layer {
                    links: HashMap::from([(vid(base), EdgeIds(vec![vid(base + 1)]))]),
                    set_hash: 0xcafe_babe_u64, // discarded on read
                },
            ],
        }
    };
    [make_graph(100), make_graph(200)]
}

/// Upload a `[GraphV3; 2]` to localstack S3, stream-download it via
/// `stream_download_and_deserialize_graph_pair(…, GraphFormat::V3)`, and
/// assert the result matches the reference `.into()` conversion.
///
/// Specifically verifies:
/// - `entry_points` and `layers` (including recomputed `set_hash`) match
///   the standard `GraphV3 → GraphMem` path.
/// - `last_update_seq_no` is 0 for both graphs (V3 has no seq_no field).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v3_graph_pair_streams_to_graphmem() -> Result<()> {
    let Some(endpoint) = s3_test_endpoint() else {
        eprintln!("S3_TEST_ENDPOINT not set; skipping");
        return Ok(());
    };

    let client = make_test_client(&endpoint);
    let bucket = format!("streaming-v3-{}", uuid::Uuid::new_v4());
    let key = "v3-pair.bin";

    create_bucket(&client, &bucket).await?;

    let pair = make_v3_pair();

    // Upload as `[GraphV3; 2]` bincode.
    let pair_for_upload = pair.clone();
    let upload = stream_serialize_and_upload_with(
        &client,
        &bucket,
        key,
        move |w| {
            bincode::serialize_into(w, &pair_for_upload)
                .map_err(|e| eyre!("bincode serialize: {e}"))
        },
        5 * 1024 * 1024,
        4,
    )
    .await;
    if let Err(e) = upload {
        cleanup_bucket(&client, &bucket).await;
        return Err(e);
    }

    // Compute the expected BLAKE3 hash from the raw serialized bytes — the
    // same bytes the upload wrote and the download read.
    let expected_hash = {
        let bytes = bincode::serialize(&pair).map_err(|e| eyre!("bincode: {e}"))?;
        *blake3::hash(&bytes).as_bytes()
    };

    // Stream-download V3 bytes → `[GraphMem; 2]`.
    let download =
        stream_download_and_deserialize_graph_pair(&client, &bucket, key, GraphFormat::V3, None)
            .await;
    cleanup_bucket(&client, &bucket).await;
    let (graphs, hash) = download?;

    // Hash must equal BLAKE3 of the on-wire bytes.
    assert_eq!(hash, expected_hash, "BLAKE3 hash mismatch");

    // Reference: standard `.into()` conversion on the original in-memory pair.
    let expected: [GraphMem; 2] = pair.map(|g| g.into());

    for i in 0..2 {
        assert_eq!(
            graphs[i].entry_points, expected[i].entry_points,
            "graph[{i}] entry_points mismatch"
        );
        assert_eq!(
            graphs[i].layers, expected[i].layers,
            "graph[{i}] layers mismatch (links or set_hash diverged)"
        );
        // V3 has no seq_no field; streaming path must yield 0.
        assert_eq!(
            graphs[i].last_update_seq_no, 0,
            "graph[{i}] last_update_seq_no must be 0 for V3"
        );
    }

    Ok(())
}

/// Build a deterministic, non-trivial `[GraphV4; 2]`.
///
/// Mirror of `make_v3_pair` but for V4: same topology, `last_update_seq_no`
/// set to a non-zero sentinel so the test can verify it survives the round-trip.
fn make_v4_pair() -> [GraphV4; 2] {
    let make_graph = |base: u32, seq_no: u64| {
        let vid = |n: u32| V4Vid { id: n, version: 1 };
        GraphV4 {
            entry_points: vec![V4EntryPoint {
                point: vid(base),
                layer: 1,
            }],
            layers: vec![
                // Layer 0 — fully connected triangle
                V4Layer {
                    links: HashMap::from([
                        (vid(base), V4EdgeIds(vec![vid(base + 1), vid(base + 2)])),
                        (vid(base + 1), V4EdgeIds(vec![vid(base), vid(base + 2)])),
                        (vid(base + 2), V4EdgeIds(vec![vid(base), vid(base + 1)])),
                    ]),
                    set_hash: 0xdead_beef_u64, // discarded on read
                },
                // Layer 1 — single edge
                V4Layer {
                    links: HashMap::from([(vid(base), V4EdgeIds(vec![vid(base + 1)]))]),
                    set_hash: 0xcafe_babe_u64, // discarded on read
                },
            ],
            last_update_seq_no: seq_no,
        }
    };
    [make_graph(100, 42), make_graph(200, 99)]
}

/// Upload a `[GraphV4; 2]` to localstack S3, stream-download it via
/// `stream_download_and_deserialize_graph_pair(…, GraphFormat::V4)`, and
/// assert the result matches the reference `.into()` conversion.
///
/// Specifically verifies:
/// - `entry_points` and `layers` (including recomputed `set_hash`) match
///   the standard `GraphV4 → GraphMem` path.
/// - `last_update_seq_no` is preserved (V4 carries it; non-zero sentinel used).
/// - The returned BLAKE3 hash equals `blake3::hash(bincode_bytes)`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v4_graph_pair_streams_to_graphmem_seq_no_preserved() -> Result<()> {
    let Some(endpoint) = s3_test_endpoint() else {
        eprintln!("S3_TEST_ENDPOINT not set; skipping");
        return Ok(());
    };

    let client = make_test_client(&endpoint);
    let bucket = format!("streaming-v4-{}", uuid::Uuid::new_v4());
    let key = "v4-pair.bin";

    create_bucket(&client, &bucket).await?;

    let pair = make_v4_pair();

    // Compute the expected BLAKE3 hash before moving `pair` into the upload.
    let expected_hash = {
        let bytes = bincode::serialize(&pair).map_err(|e| eyre!("bincode: {e}"))?;
        *blake3::hash(&bytes).as_bytes()
    };

    // Upload as `[GraphV4; 2]` bincode.
    let pair_for_upload = pair.clone();
    let upload = stream_serialize_and_upload_with(
        &client,
        &bucket,
        key,
        move |w| {
            bincode::serialize_into(w, &pair_for_upload)
                .map_err(|e| eyre!("bincode serialize: {e}"))
        },
        5 * 1024 * 1024,
        4,
    )
    .await;
    if let Err(e) = upload {
        cleanup_bucket(&client, &bucket).await;
        return Err(e);
    }

    // Stream-download V4 bytes → `[GraphMem; 2]`.
    let download =
        stream_download_and_deserialize_graph_pair(&client, &bucket, key, GraphFormat::V4, None)
            .await;
    cleanup_bucket(&client, &bucket).await;
    let (graphs, hash) = download?;

    // Hash must equal BLAKE3 of the on-wire bytes.
    assert_eq!(hash, expected_hash, "BLAKE3 hash mismatch");

    // Reference: standard `.into()` conversion on the original in-memory pair.
    let expected: [GraphMem; 2] = pair.map(|g| g.into());

    for i in 0..2 {
        assert_eq!(
            graphs[i].entry_points, expected[i].entry_points,
            "graph[{i}] entry_points mismatch"
        );
        assert_eq!(
            graphs[i].layers, expected[i].layers,
            "graph[{i}] layers mismatch (links or set_hash diverged)"
        );
        // V4 carries last_update_seq_no; it must survive the round-trip.
        assert_eq!(
            graphs[i].last_update_seq_no, expected[i].last_update_seq_no,
            "graph[{i}] last_update_seq_no mismatch"
        );
    }

    Ok(())
}
