//! Streaming S3 download + bincode deserialize.
//!
//! `stream_download_and_deserialize` fetches the object via a sequence of
//! HTTP range GETs, tees the concatenated bytes through a BLAKE3 hasher
//! into a bounded async duplex pipe, and deserializes from the pipe on a
//! `spawn_blocking` thread. The full byte buffer is never materialized in
//! memory; peak transient memory is roughly `range_size + pipe_capacity +
//! deserialized value`.
//!
//! The hash returned is BLAKE3 over the downloaded bytes (i.e. the
//! canonical bincode encoding of `T`), wire-compatible with the existing
//! `download_and_hash` path's hex hash in
//! [`super::download_graph_checkpoint`].
//!
//! Compare to [`super::multipart::download_graph`], which uses parallel
//! range GETs into a `BytesMut` and returns a fully buffered `Bytes`. The
//! streaming path takes one range at a time — slower in aggregate, but the
//! deserialized value never coexists in memory with a fully buffered copy
//! of the bytes.
//!
//! # Range cadence and retry
//!
//! The download is split into `⌈size / range_size⌉` sequential range GETs
//! (`Range: bytes=START-END`). Each range is fetched independently and
//! retried up to [`RANGE_MAX_RETRIES`] times with [`RANGE_RETRY_DELAY`]
//! between attempts before it bubbles up as a fatal error. Transient
//! network errors thus cost at most one range re-fetch, not a full
//! restart of the download — which is the property the buffered
//! parallel-range path also has, kept here without paying for an
//! out-of-order reassembly buffer.
//!
//! The `head_object` size probe is subject to the same per-attempt retry.
//!
//! # Contract on the byte stream
//!
//! `deserialize_and_hash_from` requires that the source emits **exactly**
//! the canonical bincode encoding of `T` and then EOFs. Any trailing bytes
//! cause the tee task to fail with `BrokenPipe` (the deserializer drops
//! its end of the duplex once bincode is done), which surfaces as an
//! error from the function. For S3 objects produced by
//! `stream_serialize_and_upload` this is automatic; callers
//! wiring up other sources must respect the contract.

use std::time::Duration;

use aws_sdk_s3::Client as S3Client;
use bytes::Bytes;
use eyre::{eyre, Result};
use futures::stream::{self, Stream};
use serde::de::DeserializeOwned;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};
use tokio::time::sleep;
use tokio_util::io::{StreamReader, SyncIoBridge};

use crate::{
    hnsw::graph::layered_graph::GraphMem,
    utils::serialization::graph::{read_graph_pair_streaming, GraphFormat},
};

const RANGE_MAX_RETRIES: u32 = 3;
const RANGE_RETRY_DELAY: Duration = Duration::from_secs(2);

/// Default duplex pipe capacity. Bounds back-pressure on the range
/// reader: when the deserializer falls behind, the pipe fills, the tee
/// task blocks, and the range stream stops being polled, which stops the
/// next `GetObject` from being issued.
pub const DEFAULT_DOWNLOAD_PIPE_CAPACITY: usize = 8 * 1024 * 1024;

/// Default per-range fetch size. Each range becomes one `GetObject` with
/// a `Range` header. Larger ⇒ fewer requests, more work to redo on a
/// flaky range; smaller ⇒ more requests, finer-grained retry.
pub const DEFAULT_DOWNLOAD_RANGE_SIZE: usize = 64 * 1024 * 1024;

/// Stream the object at `s3://{bucket}/{key}` through BLAKE3 and bincode
/// in one pass. Returns the deserialized value and the BLAKE3 digest of
/// the downloaded bytes.
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
    stream_download_and_deserialize_with(
        s3_client,
        bucket,
        key,
        DEFAULT_DOWNLOAD_PIPE_CAPACITY,
        DEFAULT_DOWNLOAD_RANGE_SIZE,
    )
    .await
}

pub async fn stream_download_and_deserialize_with<T>(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    pipe_capacity: usize,
    range_size: usize,
) -> Result<(T, [u8; 32])>
where
    T: DeserializeOwned + Send + 'static,
{
    tracing::info!(
        "Streaming download + deserialize: bucket={bucket}, key={key}, \
         pipe_capacity={pipe_capacity}, range_size={range_size}"
    );

    if range_size == 0 {
        return Err(eyre!("range_size must be > 0"));
    }

    let total_size = head_object_size_with_retry(s3_client, bucket, key).await?;

    // `stream::unfold`'s state machine is `!Unpin`; `StreamReader` needs
    // `Unpin`. Pin the boxed stream once and feed it through.
    let stream = Box::pin(range_stream(
        s3_client.clone(),
        bucket.to_string(),
        key.to_string(),
        total_size,
        range_size as u64,
    ));
    let reader = StreamReader::new(stream);
    deserialize_and_hash_from(reader, pipe_capacity).await
}

/// Stream the object at `s3://{bucket}/{key}` through BLAKE3 and deserialize
/// into a `[GraphMem; 2]`. Bytes are fed to the decoder incrementally as ranges
/// arrive; the decode itself ([`read_graph_pair_streaming`]) is the standard
/// derived path.
pub async fn stream_download_and_deserialize_graph_pair(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    format: GraphFormat,
) -> Result<([GraphMem; 2], [u8; 32])> {
    stream_download_and_deserialize_graph_pair_with(
        s3_client,
        bucket,
        key,
        format,
        DEFAULT_DOWNLOAD_PIPE_CAPACITY,
        DEFAULT_DOWNLOAD_RANGE_SIZE,
    )
    .await
}

/// Like [`stream_download_and_deserialize_graph_pair`] but with explicit
/// `pipe_capacity` and `range_size` knobs.
pub async fn stream_download_and_deserialize_graph_pair_with(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    format: GraphFormat,
    pipe_capacity: usize,
    range_size: usize,
) -> Result<([GraphMem; 2], [u8; 32])> {
    tracing::info!(
        "Streaming download + deserialize graph pair: bucket={bucket}, key={key}, \
         format={format}, pipe_capacity={pipe_capacity}, range_size={range_size}"
    );

    if range_size == 0 {
        return Err(eyre!("range_size must be > 0"));
    }

    if pipe_capacity == 0 {
        return Err(eyre!("pipe_capacity must be > 0"));
    }

    let total_size = head_object_size_with_retry(s3_client, bucket, key).await?;
    let stream = Box::pin(range_stream(
        s3_client.clone(),
        bucket.to_string(),
        key.to_string(),
        total_size,
        range_size as u64,
    ));
    let reader = StreamReader::new(stream);
    deserialize_and_hash_from_fn(reader, pipe_capacity, move |r| {
        read_graph_pair_streaming(r, format)
    })
    .await
}

/// `HeadObject` for `content_length`, retried per [`RANGE_MAX_RETRIES`].
async fn head_object_size_with_retry(s3_client: &S3Client, bucket: &str, key: &str) -> Result<u64> {
    let mut attempts: u32 = 0;
    loop {
        match s3_client.head_object().bucket(bucket).key(key).send().await {
            Ok(out) => {
                let len = out
                    .content_length()
                    .ok_or_else(|| eyre!("head_object {bucket}/{key}: missing content_length"))?;
                if len < 0 {
                    return Err(eyre!("head_object {bucket}/{key}: negative content_length"));
                }
                return Ok(len as u64);
            }
            Err(e) if attempts < RANGE_MAX_RETRIES => {
                attempts += 1;
                tracing::warn!("Retry {attempts} for head_object s3://{bucket}/{key}: {e:?}");
                sleep(RANGE_RETRY_DELAY).await;
            }
            Err(e) => {
                return Err(eyre!(
                    "head_object s3://{bucket}/{key} failed after {attempts} retries: {e:?}"
                ));
            }
        }
    }
}

/// Produce ranges of the object in ascending offset order, one at a time.
/// Each yielded `Bytes` is the complete body of one `GetObject(Range)`;
/// the next range is not requested until the consumer pulls again, so the
/// duplex pipe's back-pressure naturally throttles fetch cadence.
///
/// Per-range retry is internal to each `unfold` step. After
/// [`RANGE_MAX_RETRIES`] failures on a single range, the stream emits an
/// `Err` and ends — the downstream `StreamReader` will then surface the
/// error to its reader.
fn range_stream(
    s3_client: S3Client,
    bucket: String,
    key: String,
    total_size: u64,
    range_size: u64,
) -> impl Stream<Item = std::io::Result<Bytes>> {
    stream::unfold(0u64, move |offset| {
        let s3_client = s3_client.clone();
        let bucket = bucket.clone();
        let key = key.clone();
        async move {
            if offset >= total_size {
                return None;
            }
            let end_inclusive = std::cmp::min(offset + range_size, total_size) - 1;
            let range = format!("bytes={offset}-{end_inclusive}");
            let next_offset = end_inclusive + 1;

            match fetch_range(&s3_client, &bucket, &key, &range).await {
                Ok(bytes) => Some((Ok(bytes), next_offset)),
                Err(e) => Some((
                    Err(std::io::Error::other(format!(
                        "range {range} of s3://{bucket}/{key}: {e}"
                    ))),
                    // Offset value is irrelevant — stream terminates after
                    // yielding the error since StreamReader fuses on first
                    // Err. We pass `total_size` so a buggy continuation
                    // would short-circuit immediately.
                    total_size,
                )),
            }
        }
    })
}

/// Fetch one S3 range. Buffers the response body in memory so a mid-body
/// network failure cleanly maps to a retry of the same range — no partial
/// bytes leak into the downstream tee.
async fn fetch_range(s3_client: &S3Client, bucket: &str, key: &str, range: &str) -> Result<Bytes> {
    let mut attempts: u32 = 0;
    loop {
        let attempt = async {
            let out = s3_client
                .get_object()
                .bucket(bucket)
                .key(key)
                .range(range)
                .send()
                .await
                .map_err(|e| eyre!("get_object: {e:?}"))?;
            let agg = out
                .body
                .collect()
                .await
                .map_err(|e| eyre!("body collect: {e:?}"))?;
            Ok::<_, eyre::Report>(agg.into_bytes())
        }
        .await;

        match attempt {
            Ok(bytes) => return Ok(bytes),
            Err(e) if attempts < RANGE_MAX_RETRIES => {
                attempts += 1;
                tracing::warn!("Retry {attempts} for range {range}: {e}");
                sleep(RANGE_RETRY_DELAY).await;
            }
            Err(e) => {
                return Err(eyre!("range {range} failed after {attempts} retries: {e}"));
            }
        }
    }
}

/// Core: tee an `AsyncRead` through BLAKE3 into a pipe; deserialize from
/// the pipe on a blocking thread using `bincode::deserialize_from`.
///
/// Delegates to [`deserialize_and_hash_from_fn`] with the standard bincode
/// deserializer closure.
async fn deserialize_and_hash_from<R, T>(reader: R, pipe_capacity: usize) -> Result<(T, [u8; 32])>
where
    R: AsyncRead + Unpin + Send + 'static,
    T: DeserializeOwned + Send + 'static,
{
    deserialize_and_hash_from_fn(reader, pipe_capacity, |r| {
        bincode::deserialize_from(r).map_err(|e| eyre!("bincode deserialize: {e}"))
    })
    .await
}

/// Core: tee an `AsyncRead` through BLAKE3 into a pipe; call `deser_fn` on
/// the pipe from a blocking thread.
///
/// `deser_fn` receives a `&mut dyn Read` over the pipe and must consume
/// **exactly** the bytes that represent the serialised value — the same
/// contract as [`deserialize_and_hash_from`].  Any unread bytes after
/// `deser_fn` returns will cause the tee task to see `BrokenPipe` and
/// surface an error.  Any attempt to read past EOF will be surfaced as an
/// IO error from `deser_fn`.
async fn deserialize_and_hash_from_fn<R, T, F>(
    mut reader: R,
    pipe_capacity: usize,
    deser_fn: F,
) -> Result<(T, [u8; 32])>
where
    R: AsyncRead + Unpin + Send + 'static,
    T: Send + 'static,
    F: FnOnce(&mut dyn std::io::Read) -> Result<T> + Send + 'static,
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

    // Blocking task: `deser_fn` reads synchronously from the pipe through
    // SyncIoBridge. Runs concurrently with the tee task.
    let de_handle = tokio::task::spawn_blocking(move || -> Result<T> {
        let mut bridge = SyncIoBridge::new(pipe_reader);
        deser_fn(&mut bridge)
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

    /// Force the writer to drop early — the deserializer should fail with
    /// unexpected EOF, not hang.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_input_fails_deserialize() {
        let result: Result<(Sample, _)> =
            deserialize_and_hash_from(Cursor::new(Vec::<u8>::new()), 64).await;
        assert!(result.is_err());
    }
}
