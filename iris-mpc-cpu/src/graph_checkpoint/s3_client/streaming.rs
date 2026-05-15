//! Streaming serialize + S3 multipart upload.
//!
//! [`stream_serialize_and_upload_with`] serializes a value directly into an
//! S3 multipart upload without materializing the full byte buffer in memory.
//! The serializer runs on a blocking thread and writes into a bounded async
//! duplex pipe; the read side spawns up to `parallelism` concurrent
//! `UploadPart` tasks as full chunks become available. Outside of the
//! serializer's own working set, peak memory is roughly
//! `(parallelism + 1) * part_size` plus the AWS SDK's request buffers.
//!
//! Compare to [`super::multipart::upload_graph`], which requires the caller
//! to pass a fully buffered `Bytes` payload (doubling memory for large
//! graphs).

use std::{io::Write, sync::Arc, time::Duration};

use aws_sdk_s3::{
    types::{CompletedMultipartUpload, CompletedPart},
    Client as S3Client,
};
use bytes::Bytes;
use eyre::{eyre, Result};
use tokio::{
    io::{AsyncReadExt, DuplexStream},
    sync::Semaphore,
    task::JoinSet,
    time::sleep,
};
use tokio_util::io::SyncIoBridge;

const UPLOAD_PART_MAX_RETRIES: u32 = 3;
const UPLOAD_PART_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Suggested part size; also the size of the duplex pipe, which bounds
/// back-pressure on the serializer.
pub const DEFAULT_STREAMING_PART_SIZE: usize = 100 * 1024 * 1024;

/// Suggested cap on concurrent in-flight `UploadPart` tasks.
pub const DEFAULT_STREAMING_PARALLELISM: usize = 8;

/// Serialize directly into an S3 multipart upload without buffering the full
/// payload in memory.
///
/// `serialize` runs on a `spawn_blocking` thread, so it must be `Send +
/// 'static`; callers typically move owned data or an `Arc`/owned guard into
/// the closure. The closure receives a `&mut dyn Write` and **must** stream
/// its output incrementally (e.g. `bincode::serialize_into(w, &v)`); writing
/// a fully buffered `Vec<u8>` to `w` defeats the streaming intent and
/// reintroduces a full-size second copy.
///
/// On any failure (serializer error, S3 error, task panic), the multipart
/// upload is aborted before the function returns. On success,
/// `complete_multipart_upload` is called and the object is durable.
///
/// Callers without a reason to override should pass
/// [`DEFAULT_STREAMING_PART_SIZE`] and [`DEFAULT_STREAMING_PARALLELISM`].
pub async fn stream_serialize_and_upload_with<F>(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    serialize: F,
    part_size: usize,
    parallelism: usize,
) -> Result<()>
where
    F: FnOnce(&mut dyn Write) -> Result<()> + Send + 'static,
{
    tracing::info!(
        "Streaming serialize + upload: bucket={bucket}, key={key}, \
         part_size={part_size}, parallelism={parallelism}"
    );

    let init = s3_client
        .create_multipart_upload()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| eyre!("create_multipart_upload failed: {e:?}"))?;
    let upload_id = init
        .upload_id()
        .ok_or_else(|| eyre!("create_multipart_upload returned no upload_id"))?
        .to_string();

    match drive_upload(
        s3_client,
        bucket,
        key,
        &upload_id,
        serialize,
        part_size,
        parallelism,
    )
    .await
    {
        Ok(parts) => {
            s3_client
                .complete_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(&upload_id)
                .multipart_upload(
                    CompletedMultipartUpload::builder()
                        .set_parts(Some(parts))
                        .build(),
                )
                .send()
                .await
                .map_err(|e| eyre!("complete_multipart_upload failed: {e:?}"))?;
            tracing::info!("Streaming upload complete: key={key}");
            Ok(())
        }
        Err(e) => {
            if let Err(abort_err) = s3_client
                .abort_multipart_upload()
                .bucket(bucket)
                .key(key)
                .upload_id(&upload_id)
                .send()
                .await
            {
                tracing::warn!("abort_multipart_upload after failure also failed: {abort_err:?}");
            }
            Err(e)
        }
    }
}

async fn drive_upload<F>(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    upload_id: &str,
    serialize: F,
    part_size: usize,
    parallelism: usize,
) -> Result<Vec<CompletedPart>>
where
    F: FnOnce(&mut dyn Write) -> Result<()> + Send + 'static,
{
    let (reader, writer) = tokio::io::duplex(part_size);

    let serialize_handle = tokio::task::spawn_blocking(move || -> Result<()> {
        let mut sync_writer = SyncIoBridge::new(writer);
        serialize(&mut sync_writer)?;
        sync_writer
            .flush()
            .map_err(|e| eyre!("serializer flush failed: {e:?}"))?;
        // Drop closes the underlying writer half, signalling EOF to the reader.
        drop(sync_writer);
        Ok(())
    });

    let upload_result = run_upload_loop(
        s3_client.clone(),
        bucket.to_string(),
        key.to_string(),
        upload_id.to_string(),
        reader,
        part_size,
        parallelism,
    )
    .await;

    let serialize_result = serialize_handle
        .await
        .map_err(|e| eyre!("serialize task panicked or was cancelled: {e:?}"))?;

    // Prefer upload errors when both fail: when `run_upload_loop` bails on a
    // part failure it drops the reader, which surfaces as a broken-pipe error
    // in the serializer. The upload error is the root cause.
    match (upload_result, serialize_result) {
        (Ok(parts), Ok(())) => Ok(parts),
        (Err(upload_err), _) => Err(upload_err),
        (Ok(_), Err(serialize_err)) => Err(serialize_err),
    }
}

async fn run_upload_loop(
    s3_client: S3Client,
    bucket: String,
    key: String,
    upload_id: String,
    mut reader: DuplexStream,
    part_size: usize,
    parallelism: usize,
) -> Result<Vec<CompletedPart>> {
    let semaphore = Arc::new(Semaphore::new(parallelism));
    let mut join_set: JoinSet<Result<CompletedPart>> = JoinSet::new();
    let mut parts: Vec<CompletedPart> = Vec::new();
    let mut part_number: i32 = 1;

    loop {
        // Surface any part failure eagerly so we don't keep serializing and
        // spawning uploads after the upload has already gone wrong.
        while let Some(res) = join_set.try_join_next() {
            match res {
                Ok(Ok(part)) => parts.push(part),
                Ok(Err(e)) => {
                    join_set.abort_all();
                    return Err(e);
                }
                Err(e) => {
                    join_set.abort_all();
                    return Err(eyre!("upload task join error: {e:?}"));
                }
            }
        }

        let mut buf = vec![0u8; part_size];
        let mut filled = 0usize;

        while filled < part_size {
            let n = reader
                .read(&mut buf[filled..])
                .await
                .map_err(|e| eyre!("pipe read failed: {e:?}"))?;
            if n == 0 {
                break;
            }
            filled += n;
        }

        if filled == 0 {
            break;
        }
        buf.truncate(filled);

        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| eyre!("semaphore acquire: {e}"))?;

        let pn = part_number;
        let client = s3_client.clone();
        let bucket = bucket.clone();
        let key = key.clone();
        let upload_id = upload_id.clone();
        // `Bytes` so retries clone cheaply (Arc bump) instead of copying.
        let body = Bytes::from(buf);

        join_set.spawn(async move {
            let _permit = permit;
            let mut attempts: u32 = 0;
            loop {
                match client
                    .upload_part()
                    .bucket(&bucket)
                    .key(&key)
                    .upload_id(&upload_id)
                    .part_number(pn)
                    .body(body.clone().into())
                    .send()
                    .await
                {
                    Ok(res) => {
                        let etag = res
                            .e_tag()
                            .ok_or_else(|| eyre!("upload_part {pn} returned no e_tag"))?
                            .to_string();
                        tracing::debug!("uploaded part {pn} (etag={etag})");
                        return Ok(CompletedPart::builder().e_tag(etag).part_number(pn).build());
                    }
                    Err(_) if attempts < UPLOAD_PART_MAX_RETRIES => {
                        attempts += 1;
                        tracing::warn!("Retry {attempts} for part {pn}");
                        sleep(UPLOAD_PART_RETRY_DELAY).await;
                    }
                    Err(e) => {
                        return Err(eyre!(
                            "upload_part {pn} failed after {attempts} retries: {e:?}"
                        ));
                    }
                }
            }
        });

        part_number += 1;
    }

    let mut first_error: Option<eyre::Report> = None;
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(Ok(part)) => parts.push(part),
            Ok(Err(e)) => {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(eyre!("upload task join error: {e:?}"));
                }
            }
        }
    }

    if let Some(e) = first_error {
        join_set.abort_all();
        return Err(e);
    }

    parts.sort_by_key(|p| p.part_number);
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirror of `drive_upload` that consumes chunks into a single `Vec<u8>`
    /// instead of dispatching them to S3. Lets the tests verify byte-equality
    /// with buffered `bincode::serialize` without standing up a mock S3.
    async fn collect_streamed_bytes<F>(serialize: F, part_size: usize) -> Result<Vec<u8>>
    where
        F: FnOnce(&mut dyn Write) -> Result<()> + Send + 'static,
    {
        let (mut reader, writer) = tokio::io::duplex(part_size);

        let serialize_handle = tokio::task::spawn_blocking(move || -> Result<()> {
            let mut sync_writer = SyncIoBridge::new(writer);
            serialize(&mut sync_writer)?;
            sync_writer.flush().map_err(|e| eyre!("flush: {e:?}"))?;
            Ok(())
        });

        let mut bytes = Vec::new();
        let mut buf = vec![0u8; 8192];
        loop {
            let n = reader
                .read(&mut buf)
                .await
                .map_err(|e| eyre!("read: {e:?}"))?;
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..n]);
        }

        serialize_handle.await.map_err(|e| eyre!("join: {e:?}"))??;

        Ok(bytes)
    }

    #[derive(serde::Serialize)]
    struct Sample {
        name: String,
        values: Vec<u32>,
        nested: Vec<Vec<i64>>,
    }

    fn fixture() -> Sample {
        Sample {
            name: "checkpoint-streaming-test".into(),
            values: (0..10_000u32).collect(),
            nested: (0..50)
                .map(|i| (0..200i64).map(|j| (i as i64) * j).collect())
                .collect(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_matches_buffered_bincode() -> Result<()> {
        let data = fixture();

        let buffered = bincode::serialize(&data).map_err(|e| eyre!("bincode: {e}"))?;

        let data_for_closure = Arc::new(data);
        let streamed = collect_streamed_bytes(
            move |w| {
                bincode::serialize_into(w, &*data_for_closure).map_err(|e| eyre!("bincode: {e}"))
            },
            64 * 1024,
        )
        .await?;

        assert_eq!(buffered, streamed);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_handles_payload_larger_than_part_size() -> Result<()> {
        // Payload is much larger than the pipe buffer, forcing the serializer
        // to block on back-pressure until the reader drains.
        let payload: Vec<u64> = (0..200_000u64).collect();
        let buffered = bincode::serialize(&payload).map_err(|e| eyre!("bincode: {e}"))?;

        let streamed = collect_streamed_bytes(
            move |w| bincode::serialize_into(w, &payload).map_err(|e| eyre!("bincode: {e}")),
            8 * 1024,
        )
        .await?;

        assert_eq!(buffered, streamed);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serializer_error_is_propagated() {
        let result =
            collect_streamed_bytes(|_w| Err(eyre!("intentional serializer failure")), 1024).await;

        let err = result.expect_err("should propagate serializer error");
        assert!(format!("{err}").contains("intentional serializer failure"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serializer_panic_is_propagated() {
        let result = collect_streamed_bytes(|_w| -> Result<()> { panic!("boom") }, 1024).await;

        assert!(result.is_err());
    }
}
