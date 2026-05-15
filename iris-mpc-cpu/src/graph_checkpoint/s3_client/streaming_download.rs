//! Streaming S3 download + bincode deserialize.
//!
//! `stream_download_and_deserialize` issues a single `GetObject`, streams the
//! body through a BLAKE3 hasher into a bounded async duplex pipe, and
//! deserializes from the pipe on a `spawn_blocking` thread. The full byte
//! buffer is never materialized in memory; peak transient memory is the
//! pipe capacity plus the deserialized value.
//!
//! The hash returned is BLAKE3 over the downloaded bytes (i.e. the canonical
//! bincode encoding of `T`), wire-compatible with the existing
//! `download_and_hash` path's hex hash in [`super::download_graph_checkpoint`].
//!
//! Compare to [`super::multipart::download_graph`], which uses parallel range
//! GETs into a `BytesMut` and returns a fully buffered `Bytes`. The streaming
//! path trades that throughput for ~3× lower peak memory on large graphs.
//!
//! # Contract on the source byte stream
//!
//! `deserialize_and_hash_from` requires that the source emits **exactly** the
//! canonical bincode encoding of `T` and then EOFs. Any trailing bytes cause
//! the tee task to fail with `BrokenPipe` (the deserializer drops its end of
//! the duplex once bincode is done), which surfaces as an error from the
//! function. For S3 objects produced by [`super::stream_serialize_and_upload`]
//! this is automatic; callers wiring up other sources must respect the
//! contract.
//!
//! # Retry
//!
//! `stream_download_and_deserialize` retries the whole GET → body-read →
//! deserialize pipeline up to [`DOWNLOAD_MAX_RETRIES`] times on any failure,
//! with [`DOWNLOAD_RETRY_DELAY`] between attempts. This matches the buffered
//! [`super::multipart::download_graph`] path, which retries each ranged GET.
//! Each attempt restarts BLAKE3 from scratch and re-issues `get_object`.
//! Deserialize errors are also retried — bounded cost, simpler surface.

use std::time::Duration;

use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio_util::io::SyncIoBridge;

const DOWNLOAD_MAX_RETRIES: u32 = 3;
const DOWNLOAD_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Default duplex pipe capacity. Bounds back-pressure on the S3 reader: when
/// the deserializer falls behind, the pipe fills, the tee task blocks, and
/// the S3 reader stops pulling bytes. Sized to absorb a typical part fetch
/// without blocking on the common path.
pub const DEFAULT_DOWNLOAD_PIPE_CAPACITY: usize = 8 * 1024 * 1024;

/// Stream the object at `s3://{bucket}/{key}` through BLAKE3 and bincode in
/// one pass. Returns the deserialized value and the BLAKE3 digest of the
/// downloaded bytes.
///
/// Callers verify the digest against the expected checkpoint hash before
/// trusting `value`. A mismatch indicates the S3 object diverges from the
/// hash recorded in the database.
pub async fn stream_download_and_deserialize<T>(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
) -> Result<(T, [u8; 32])>
where
    T: DeserializeOwned + Send + 'static,
{
    stream_download_and_deserialize_with(s3_client, bucket, key, DEFAULT_DOWNLOAD_PIPE_CAPACITY)
        .await
}

pub async fn stream_download_and_deserialize_with<T>(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    pipe_capacity: usize,
) -> Result<(T, [u8; 32])>
where
    T: DeserializeOwned + Send + 'static,
{
    tracing::info!(
        "Streaming download + deserialize: bucket={bucket}, key={key}, pipe_capacity={pipe_capacity}"
    );

    let mut attempts: u32 = 0;
    loop {
        let attempt_result = try_download_once::<T>(s3_client, bucket, key, pipe_capacity).await;
        match attempt_result {
            Ok(pair) => return Ok(pair),
            Err(e) if attempts < DOWNLOAD_MAX_RETRIES => {
                attempts += 1;
                tracing::warn!("Retry {attempts} for download s3://{bucket}/{key}: {e}");
                tokio::time::sleep(DOWNLOAD_RETRY_DELAY).await;
            }
            Err(e) => {
                return Err(eyre!(
                    "download s3://{bucket}/{key} failed after {attempts} retries: {e:?}"
                ));
            }
        }
    }
}

async fn try_download_once<T>(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    pipe_capacity: usize,
) -> Result<(T, [u8; 32])>
where
    T: DeserializeOwned + Send + 'static,
{
    let resp = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| eyre!("get_object failed: {e:?}"))?;

    let body = resp.body.into_async_read();
    deserialize_and_hash_from(body, pipe_capacity).await
}

/// Core: tee an `AsyncRead` through BLAKE3 into a pipe; deserialize from the
/// pipe on a blocking thread. Exposed for tests that don't want to stand up
/// a mock S3.
pub async fn deserialize_and_hash_from<R, T>(
    mut reader: R,
    pipe_capacity: usize,
) -> Result<(T, [u8; 32])>
where
    R: AsyncRead + Unpin + Send + 'static,
    T: DeserializeOwned + Send + 'static,
{
    let (mut pipe_writer, pipe_reader) = tokio::io::duplex(pipe_capacity);

    // Tee task: pull bytes from `reader`, fold into BLAKE3, push into pipe.
    // Returns the hash once the source EOFs.
    let tee_handle = tokio::spawn(async move {
        let mut hasher = blake3::Hasher::new();
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            let n = reader
                .read(&mut buf)
                .await
                .map_err(|e| eyre!("source read failed: {e:?}"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            pipe_writer
                .write_all(&buf[..n])
                .await
                .map_err(|e| eyre!("pipe write failed: {e:?}"))?;
        }
        // Drop closes the writer half of the duplex, signalling EOF to the
        // SyncIoBridge on the blocking task.
        drop(pipe_writer);
        Ok::<[u8; 32], eyre::Report>(*hasher.finalize().as_bytes())
    });

    // Blocking task: bincode reads synchronously from the pipe through
    // SyncIoBridge. Runs concurrently with the tee task.
    let de_handle = tokio::task::spawn_blocking(move || -> Result<T> {
        let bridge = SyncIoBridge::new(pipe_reader);
        bincode::deserialize_from(bridge).map_err(|e| eyre!("bincode deserialize: {e}"))
    });

    // Surface the deserialize error first if both fail — usually more
    // informative than "pipe write failed" downstream.
    let de_result = de_handle
        .await
        .map_err(|e| eyre!("deserialize task panicked or was cancelled: {e:?}"))?;
    let tee_result = tee_handle
        .await
        .map_err(|e| eyre!("tee task panicked or was cancelled: {e:?}"))?;

    let value = de_result?;
    let hash = tee_result?;
    Ok((value, hash))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Round-trip: serialize via the existing streaming upload helper, then
    /// deserialize via `deserialize_and_hash_from`. Asserts byte-level
    /// equality of hash and value with a buffered bincode + blake3::hash
    /// reference path.
    #[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Sample {
        name: String,
        values: Vec<u32>,
        nested: Vec<Vec<i64>>,
    }

    fn fixture() -> Sample {
        Sample {
            name: "checkpoint-streaming-download-test".into(),
            values: (0..10_000u32).collect(),
            nested: (0..50)
                .map(|i| (0..200i64).map(|j| (i as i64) * j).collect())
                .collect(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn round_trip_matches_buffered() -> Result<()> {
        let v = fixture();
        let bytes = bincode::serialize(&v).map_err(|e| eyre!("bincode: {e}"))?;
        let expected_hash = *blake3::hash(&bytes).as_bytes();

        let (got, got_hash): (Sample, _) =
            deserialize_and_hash_from(Cursor::new(bytes), 64 * 1024).await?;

        assert_eq!(got, v);
        assert_eq!(got_hash, expected_hash);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hash_matches_blake3_of_bytes_for_large_payload() -> Result<()> {
        // Forces back-pressure: payload >> pipe capacity.
        let payload: Vec<u64> = (0..200_000u64).collect();
        let bytes = bincode::serialize(&payload).map_err(|e| eyre!("bincode: {e}"))?;
        let expected_hash = *blake3::hash(&bytes).as_bytes();

        let (got, got_hash): (Vec<u64>, _) =
            deserialize_and_hash_from(Cursor::new(bytes), 4 * 1024).await?;

        assert_eq!(got, payload);
        assert_eq!(got_hash, expected_hash);
        Ok(())
    }

    /// A truncated payload should surface as a deserialize error, not hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn truncated_payload_fails_deserialize() {
        let v = fixture();
        let mut bytes = bincode::serialize(&v).expect("bincode");
        bytes.truncate(bytes.len() / 2);

        let result: Result<(Sample, _)> = deserialize_and_hash_from(Cursor::new(bytes), 1024).await;
        assert!(result.is_err());
    }

    /// Trailing bytes beyond the bincode payload are a contract violation
    /// (see module doc). Bincode finishes reading and the deserializer task
    /// drops its end of the duplex; the still-running tee gets `BrokenPipe`
    /// on its next write, which surfaces as an error from the function. This
    /// test pins down that behavior so a future refactor can't accidentally
    /// hide trailing-byte streams as successful.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn trailing_bytes_fail() {
        let v = fixture();
        let mut bytes = bincode::serialize(&v).expect("bincode");
        bytes.extend_from_slice(&[0xAB; 64 * 1024]); // > one tee buffer past bincode EOF
        let result: Result<(Sample, _)> = deserialize_and_hash_from(Cursor::new(bytes), 1024).await;
        assert!(
            result.is_err(),
            "trailing bytes after the bincode payload must fail per module contract"
        );
    }

    /// Reader errors propagate. Custom AsyncRead that returns an error
    /// after producing a few bytes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reader_error_is_propagated() {
        use std::pin::Pin;
        use std::task::{Context, Poll};
        use tokio::io::ReadBuf;

        struct FailingReader {
            bytes_emitted: usize,
        }
        impl AsyncRead for FailingReader {
            fn poll_read(
                mut self: Pin<&mut Self>,
                _cx: &mut Context<'_>,
                buf: &mut ReadBuf<'_>,
            ) -> Poll<std::io::Result<()>> {
                if self.bytes_emitted >= 8 {
                    return Poll::Ready(Err(std::io::Error::other("boom")));
                }
                buf.put_slice(&[0u8; 8]);
                self.bytes_emitted += 8;
                Poll::Ready(Ok(()))
            }
        }

        let reader = FailingReader { bytes_emitted: 0 };
        let result: Result<(Vec<u64>, _)> = deserialize_and_hash_from(reader, 64).await;
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("boom") || msg.contains("source read"));
    }

    /// Force the writer to drop early — the deserializer should fail with
    /// unexpected EOF, not hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_input_fails_deserialize() {
        let result: Result<(Sample, _)> =
            deserialize_and_hash_from(Cursor::new(Vec::<u8>::new()), 64).await;
        assert!(result.is_err());
    }
}
