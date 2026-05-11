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

use aws_sdk_s3::{
    config::{BehaviorVersion, Credentials, Region},
    Client as S3Client, Config,
};
use eyre::{eyre, Result};
use iris_mpc_cpu::graph_checkpoint::stream_serialize_and_upload_with;

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
async fn streaming_upload_round_trip_single_part() -> Result<()> {
    let Some(endpoint) = s3_test_endpoint() else {
        eprintln!("S3_TEST_ENDPOINT not set; skipping");
        return Ok(());
    };
    let client = make_test_client(&endpoint);
    let bucket = format!("streaming-test-sp-{}", uuid::Uuid::new_v4());
    let key = "single-part.bin";
    create_bucket(&client, &bucket).await?;

    // < part_size: payload fits in a single part. That part is by definition
    // the last, so the 5 MiB minimum doesn't apply.
    let payload: Vec<u32> = (0..500u32).collect();
    let buffered = bincode::serialize(&payload).map_err(|e| eyre!("bincode: {e}"))?;

    let payload_for_closure = payload;
    let upload = stream_serialize_and_upload_with(
        &client,
        &bucket,
        key,
        move |w| {
            bincode::serialize_into(w, &payload_for_closure).map_err(|e| eyre!("bincode: {e}"))
        },
        5 * 1024 * 1024,
        2,
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

    assert_eq!(downloaded, buffered);
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
