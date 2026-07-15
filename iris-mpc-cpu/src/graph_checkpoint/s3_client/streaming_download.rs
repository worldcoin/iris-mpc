//! Streaming object-store download + bincode deserialize.
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
//! streaming path fans out the same way (see [`DEFAULT_DOWNLOAD_PARALLELISM`])
//! but emits ranges in order into a bounded pipe, so the deserialized value
//! never coexists in memory with a fully buffered copy of the bytes.
//!
//! # Range cadence and retry
//!
//! The download is split into `⌈size / range_size⌉` range GETs
//! (`Range: bytes=START-END`), up to [`DEFAULT_DOWNLOAD_PARALLELISM`] in
//! flight at once and emitted in ascending offset order. Each range is
//! fetched independently and retried up to [`RANGE_MAX_RETRIES`] times with
//! [`RANGE_RETRY_DELAY`] between attempts before it bubbles up as a fatal
//! error. Transient network errors thus cost at most one range re-fetch,
//! not a full restart of the download.
//!
//! The `head_object` size probe is subject to the same per-attempt retry.
//!
//! # Contract on the byte stream
//!
//! `deserialize_and_hash_from` requires that the source emits **exactly**
//! the canonical bincode encoding of `T` and then EOFs. Any trailing bytes
//! cause the tee task to fail with `BrokenPipe` (the deserializer drops
//! its end of the duplex once bincode is done), which surfaces as an
//! error from the function. For objects produced by
//! `stream_serialize_and_upload` this is automatic; callers
//! wiring up other sources must respect the contract.

use std::time::Duration;

use bytes::Bytes;
use eyre::{eyre, Result};
use futures::stream::{self, Stream, StreamExt};
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt, ObjectStoreRef};
use object_store::path::Path;
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
/// flaky range; smaller ⇒ more requests, finer-grained retry. With
/// [`DEFAULT_DOWNLOAD_PARALLELISM`] in-flight, peak download buffering is
/// roughly `parallelism * range_size`.
pub const DEFAULT_DOWNLOAD_RANGE_SIZE: usize = 16 * 1024 * 1024;

/// Default number of range GETs kept in flight concurrently. Results are
/// emitted to the deserializer in offset order, so this bounds throughput
/// without unbounding memory: peak buffered bytes ≈ `parallelism *
/// range_size` (here ~512 MB). Mirrors the buffered path's 32-way fan-out,
/// which sustains ~1.2 GB/s.
pub const DEFAULT_DOWNLOAD_PARALLELISM: usize = 32;

/// Buffer placed between the bincode graph decoder and the `SyncIoBridge`
/// over the duplex pipe. bincode issues many small reads per node; without
/// this each one would block across the async bridge individually.
const GRAPH_DECODE_BUFFER: usize = 1024 * 1024;

/// Stream an object through BLAKE3 and bincode
/// in one pass. Returns the deserialized value and the BLAKE3 digest of
/// the downloaded bytes.
///
/// Callers verify the digest against the expected checkpoint hash before
/// trusting `value`. A mismatch indicates the stored object diverges from the
/// hash recorded in the database.
pub async fn stream_download_and_deserialize<T>(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
) -> Result<(T, [u8; 32])>
where
    T: DeserializeOwned + Send + 'static,
{
    stream_download_and_deserialize_with(
        client,
        store_location,
        key,
        DEFAULT_DOWNLOAD_PIPE_CAPACITY,
        DEFAULT_DOWNLOAD_RANGE_SIZE,
        DEFAULT_DOWNLOAD_PARALLELISM,
    )
    .await
}

pub async fn stream_download_and_deserialize_with<T>(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
    pipe_capacity: usize,
    range_size: usize,
    parallelism: usize,
) -> Result<(T, [u8; 32])>
where
    T: DeserializeOwned + Send + 'static,
{
    tracing::info!(
        "Streaming download + deserialize: store={store_location}, key={key}, \
         pipe_capacity={pipe_capacity}, range_size={range_size}, parallelism={parallelism}"
    );

    if range_size == 0 {
        return Err(eyre!("range_size must be > 0"));
    }

    let store = client.store(store_location)?;
    let location = path(key)?;
    let total_size = head_object_size_with_retry(&store, &location).await?;

    let stream = Box::pin(range_stream(
        store,
        location,
        total_size,
        range_size as u64,
        parallelism,
    ));
    let reader = StreamReader::new(stream);
    deserialize_and_hash_from(reader, pipe_capacity).await
}

/// Stream an object through BLAKE3 and deserialize
/// into a `[GraphMem; 2]`. Bytes are fed to the decoder incrementally as ranges
/// arrive; the decode itself ([`read_graph_pair_streaming`]) is the standard
/// derived path.
pub async fn stream_download_and_deserialize_graph_pair(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
    format: GraphFormat,
) -> Result<([GraphMem; 2], [u8; 32])> {
    stream_download_and_deserialize_graph_pair_with(
        client,
        store_location,
        key,
        format,
        DEFAULT_DOWNLOAD_PIPE_CAPACITY,
        DEFAULT_DOWNLOAD_RANGE_SIZE,
        DEFAULT_DOWNLOAD_PARALLELISM,
    )
    .await
}

/// Like [`stream_download_and_deserialize_graph_pair`] but with explicit
/// `pipe_capacity`, `range_size`, and `parallelism` knobs.
pub async fn stream_download_and_deserialize_graph_pair_with(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
    format: GraphFormat,
    pipe_capacity: usize,
    range_size: usize,
    parallelism: usize,
) -> Result<([GraphMem; 2], [u8; 32])> {
    tracing::info!(
        "Streaming download + deserialize graph pair: store={store_location}, key={key}, \
         format={format}, pipe_capacity={pipe_capacity}, range_size={range_size}, \
         parallelism={parallelism}"
    );

    if range_size == 0 {
        return Err(eyre!("range_size must be > 0"));
    }

    if pipe_capacity == 0 {
        return Err(eyre!("pipe_capacity must be > 0"));
    }

    let store = client.store(store_location)?;
    let location = path(key)?;
    let total_size = head_object_size_with_retry(&store, &location).await?;
    let stream = Box::pin(range_stream(
        store,
        location,
        total_size,
        range_size as u64,
        parallelism,
    ));
    let reader = StreamReader::new(stream);
    deserialize_and_hash_from_fn(reader, pipe_capacity, move |r| {
        // bincode reads the graph field-by-field (count, then per-entry
        // VectorId + edge Vec); without buffering each tiny read blocks
        // across the duplex via SyncIoBridge — ~10x slower at prod scale.
        let mut buf = std::io::BufReader::with_capacity(GRAPH_DECODE_BUFFER, r);
        read_graph_pair_streaming(&mut buf, format)
    })
    .await
}

/// Object metadata size probe, retried per [`RANGE_MAX_RETRIES`].
async fn head_object_size_with_retry(store: &ObjectStoreRef, key: &Path) -> Result<u64> {
    let mut attempts: u32 = 0;
    loop {
        match store.head(key).await {
            Ok(out) => return Ok(out.size),
            Err(e) if attempts < RANGE_MAX_RETRIES => {
                attempts += 1;
                tracing::warn!("Retry {attempts} for object metadata {key}: {e:?}");
                sleep(RANGE_RETRY_DELAY).await;
            }
            Err(e) => {
                return Err(eyre!(
                    "object metadata {key} failed after {attempts} retries: {e:?}"
                ));
            }
        }
    }
}

/// Produce the object's ranges in ascending offset order, keeping up to
/// `parallelism` `GetObject(Range)` requests in flight at once. `buffered`
/// preserves emission order regardless of completion order, so the
/// downstream `StreamReader` sees a contiguous byte stream while up to
/// `parallelism * range_size` bytes are buffered in flight.
///
/// Per-range retry is internal to [`fetch_range`]. After [`RANGE_MAX_RETRIES`]
/// failures on a single range, that range yields an `Err`; the downstream
/// `StreamReader` fuses on the first `Err` and surfaces it to its reader,
/// at which point the dropped stream cancels any still-in-flight fetches.
fn range_stream(
    store: ObjectStoreRef,
    key: Path,
    total_size: u64,
    range_size: u64,
    parallelism: usize,
) -> impl Stream<Item = std::io::Result<Bytes>> {
    let mut ranges = Vec::new();
    let mut offset = 0u64;
    while offset < total_size {
        let end_inclusive = std::cmp::min(offset + range_size, total_size) - 1;
        ranges.push((offset, end_inclusive));
        offset = end_inclusive + 1;
    }

    stream::iter(ranges)
        .map(move |(start, end_inclusive)| {
            let store = store.clone();
            let key = key.clone();
            async move {
                fetch_range(&store, &key, start..end_inclusive + 1)
                    .await
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "range {start}-{end_inclusive} of {key}: {e}"
                        ))
                    })
            }
        })
        .buffered(parallelism.max(1))
}

/// Fetch one object range. Buffers the response body in memory so a mid-body
/// network failure cleanly maps to a retry of the same range — no partial
/// bytes leak into the downstream tee.
async fn fetch_range(
    store: &ObjectStoreRef,
    key: &Path,
    range: std::ops::Range<u64>,
) -> Result<Bytes> {
    let mut attempts: u32 = 0;
    loop {
        let attempt = store
            .get_range(key, range.clone())
            .await
            .map_err(|e| eyre!("get range: {e:?}"));

        match attempt {
            Ok(bytes) => return Ok(bytes),
            Err(e) if attempts < RANGE_MAX_RETRIES => {
                attempts += 1;
                tracing::warn!("Retry {attempts} for range {range:?}: {e}");
                sleep(RANGE_RETRY_DELAY).await;
            }
            Err(e) => {
                return Err(eyre!(
                    "range {range:?} failed after {attempts} retries: {e}"
                ));
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
