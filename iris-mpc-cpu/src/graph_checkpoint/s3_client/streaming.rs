//! Streaming serialize + multipart object-store upload.
//!
//! [`stream_serialize_and_upload_with`] serializes a value directly into a
//! multipart object-store upload without materializing the full byte buffer in memory.
//! The serializer runs on a blocking thread and writes into a bounded async
//! duplex pipe; the read side spawns up to `parallelism` concurrent
//! `UploadPart` tasks as full chunks become available. Outside of the
//! serializer's own working set, peak memory is roughly
//! `(parallelism + 1) * part_size` plus the backend's request buffers.
//!
//! Compare to [`super::multipart::upload_graph`], which requires the caller
//! to pass a fully buffered `Bytes` payload (doubling memory for large
//! graphs).

use std::{
    io::{BufWriter, Write},
    sync::Arc,
};

use bytes::Bytes;
use eyre::{eyre, Result};
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt};
use object_store::MultipartUpload;
use tokio::{
    io::{AsyncReadExt, DuplexStream},
    sync::Semaphore,
    task::JoinSet,
};
use tokio_util::io::SyncIoBridge;

/// Suggested part size; also the size of the duplex pipe, which bounds
/// back-pressure on the serializer.
pub const DEFAULT_STREAMING_PART_SIZE: usize = 100 * 1024 * 1024;

/// Suggested cap on concurrent in-flight `UploadPart` tasks.
pub const DEFAULT_STREAMING_PARALLELISM: usize = 8;

/// Buffer between the bincode serializer and the `SyncIoBridge` over the
/// duplex pipe. bincode emits many small writes per node; without this each
/// one would block across the async bridge individually. Mirrors
/// `GRAPH_DECODE_BUFFER` on the download path.
const GRAPH_ENCODE_BUFFER: usize = 1024 * 1024;

/// `Write` adapter that forwards every byte to an inner writer while
/// folding the same bytes into a `blake3::Hasher`. Lets callers compute
/// the BLAKE3 digest of their upload payload in one streaming pass, with
/// no intermediate `Vec<u8>`.
///
/// Pairs with [`stream_serialize_and_upload_with`] when the caller needs
/// the canonical-bytes hash (e.g. to record into a verification field)
///
/// Kept deliberately outside the upload primitive itself so the primitive
/// stays content-agnostic: callers that don't need a hash (or want a
/// different algorithm) can write straight to the underlying `&mut dyn
/// Write` instead.
pub struct BlakeTeeWriter<W> {
    inner: W,
    hasher: blake3::Hasher,
}

impl<W: Write> BlakeTeeWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
        }
    }

    /// Consume the tee and return the BLAKE3 digest of every byte that
    /// passed through `Write::write`.
    pub fn finalize(self) -> [u8; 32] {
        *self.hasher.finalize().as_bytes()
    }
}

impl<W: Write> Write for BlakeTeeWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Serialize directly into a multipart object-store upload without buffering the full
/// payload in memory.
///
/// `serialize` runs on a `spawn_blocking` thread, so it must be `Send +
/// 'static`; callers typically move owned data or an `Arc`/owned guard into
/// the closure. The closure receives a `&mut dyn Write` and **must** stream
/// its output incrementally (e.g. `bincode::serialize_into(w, &v)`); writing
/// a fully buffered `Vec<u8>` to `w` defeats the streaming intent and
/// reintroduces a full-size second copy.
///
/// On any failure (serializer error, object-store error, task panic), the multipart
/// upload is aborted before the function returns. On success,
/// `complete_multipart_upload` is called and the object is durable.
///
/// Callers without a reason to override should pass
/// [`DEFAULT_STREAMING_PART_SIZE`] and [`DEFAULT_STREAMING_PARALLELISM`].
pub async fn stream_serialize_and_upload_with<F>(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
    serialize: F,
    part_size: usize,
    parallelism: usize,
) -> Result<()>
where
    F: FnOnce(&mut dyn Write) -> Result<()> + Send + 'static,
{
    tracing::info!(
        "Streaming serialize + upload: store={store_location}, key={key}, \
         part_size={part_size}, parallelism={parallelism}"
    );

    let store = client.store(store_location)?;
    let location = path(key)?;
    let mut upload = store
        .put_multipart(&location)
        .await
        .map_err(|e| eyre!("multipart upload creation failed: {e:?}"))?;

    match drive_upload(&mut *upload, serialize, part_size, parallelism).await {
        Ok(()) => {
            upload
                .complete()
                .await
                .map_err(|e| eyre!("multipart upload completion failed: {e:?}"))?;
            tracing::info!("Streaming upload complete: key={key}");
            Ok(())
        }
        Err(e) => {
            if let Err(abort_err) = upload.abort().await {
                tracing::warn!("multipart upload abort after failure also failed: {abort_err:?}");
            }
            Err(e)
        }
    }
}

async fn drive_upload<F>(
    upload: &mut dyn MultipartUpload,
    serialize: F,
    part_size: usize,
    parallelism: usize,
) -> Result<()>
where
    F: FnOnce(&mut dyn Write) -> Result<()> + Send + 'static,
{
    let (reader, writer) = tokio::io::duplex(part_size);

    let serialize_handle = tokio::task::spawn_blocking(move || -> Result<()> {
        // Coalesce bincode's many small writes so they don't each block across
        // the async bridge individually.
        let mut buf_writer =
            BufWriter::with_capacity(GRAPH_ENCODE_BUFFER, SyncIoBridge::new(writer));
        serialize(&mut buf_writer)?;
        buf_writer
            .flush()
            .map_err(|e| eyre!("serializer flush failed: {e:?}"))?;
        // Drop closes the underlying writer half, signalling EOF to the reader.
        drop(buf_writer);
        Ok(())
    });

    let upload_result = run_upload_loop(upload, reader, part_size, parallelism).await;

    let serialize_result = serialize_handle
        .await
        .map_err(|e| eyre!("serialize task panicked or was cancelled: {e:?}"))?;

    // Prefer upload errors when both fail: when `run_upload_loop` bails on a
    // part failure it drops the reader, which surfaces as a broken-pipe error
    // in the serializer. The upload error is the root cause.
    match (upload_result, serialize_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(upload_err), _) => Err(upload_err),
        (Ok(_), Err(serialize_err)) => Err(serialize_err),
    }
}

async fn run_upload_loop(
    upload: &mut dyn MultipartUpload,
    mut reader: DuplexStream,
    part_size: usize,
    parallelism: usize,
) -> Result<()> {
    let semaphore = Arc::new(Semaphore::new(parallelism));
    let mut join_set: JoinSet<Result<i32>> = JoinSet::new();
    let mut part_number: i32 = 1;

    loop {
        // Surface any part failure eagerly so we don't keep serializing and
        // spawning uploads after the upload has already gone wrong.
        while let Some(res) = join_set.try_join_next() {
            match res {
                Ok(Ok(part_number)) => tracing::debug!("uploaded part {part_number}"),
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
        let body = Bytes::from(buf);
        let upload_part = upload.put_part(body.into());

        join_set.spawn(async move {
            let _permit = permit;
            upload_part
                .await
                .map_err(|e| eyre!("upload part {pn} failed: {e}"))?;
            Ok(pn)
        });

        part_number += 1;
    }

    // Fail fast: break on the first part failure rather than draining the
    // JoinSet to completion. `abort_all` only has an effect while tasks
    // are still running, so the cancellation of in-flight `UploadPart`s
    // has to happen before `join_next` is exhausted.
    let mut first_error: Option<eyre::Report> = None;
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(Ok(part_number)) => tracing::debug!("uploaded part {part_number}"),
            Ok(Err(e)) => {
                first_error = Some(e);
                break;
            }
            Err(e) => {
                first_error = Some(eyre!("upload task join error: {e:?}"));
                break;
            }
        }
    }

    if let Some(e) = first_error {
        join_set.abort_all();
        return Err(e);
    }

    Ok(())
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

    /// Bytes that pass through `BlakeTeeWriter` reach the inner sink
    /// unchanged, and `finalize` returns `blake3::hash(written_bytes)`.
    #[test]
    fn blake_tee_writer_hashes_what_it_forwards() {
        let mut sink: Vec<u8> = Vec::new();
        let mut tee = BlakeTeeWriter::new(&mut sink);

        let payload = b"the quick brown fox jumps over the lazy dog";
        tee.write_all(payload).unwrap();
        let hash = tee.finalize();

        assert_eq!(sink, payload);
        assert_eq!(hash, *blake3::hash(payload).as_bytes());
    }
}
