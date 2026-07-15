//! Backend-independent graph checkpoint upload and download helpers.

use bytes::{Bytes, BytesMut};
use eyre::{eyre, Result};
use futures::{stream, StreamExt, TryStreamExt};
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt};
use std::{future::Future, time::Duration};
use tokio::time::sleep;

pub const DEFAULT_CHECKPOINT_CHUNK_SIZE: usize = 100 * 1024 * 1024;
pub const DEFAULT_CHECKPOINT_PARALLELISM: usize = 32;
const MULTIPART_THRESHOLD: usize = 5 * 1024 * 1024;
const MAX_RETRIES: usize = 3;

pub async fn upload_graph(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
    data: Bytes,
) -> Result<()> {
    tracing::info!(
        "Uploading graph checkpoint: store={}, key={}, size={}",
        store_location,
        key,
        data.len()
    );

    let store = client.store(store_location)?;
    let location = path(key)?;
    if data.len() < MULTIPART_THRESHOLD {
        return upload_graph_simple(&store, &location, data).await;
    }

    let mut upload = store
        .put_multipart(&location)
        .await
        .map_err(|e| eyre!("Failed to initiate multipart upload: {e}"))?;
    let chunk_size = DEFAULT_CHECKPOINT_CHUNK_SIZE.max(MULTIPART_THRESHOLD);
    let part_futures: Vec<_> = (0..data.len())
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(data.len());
            upload.put_part(data.slice(start..end).into())
        })
        .collect();

    let result: object_store::Result<Vec<_>> = stream::iter(part_futures)
        .buffer_unordered(DEFAULT_CHECKPOINT_PARALLELISM)
        .try_collect()
        .await;
    if let Err(error) = result {
        if let Err(abort_error) = upload.abort().await {
            tracing::warn!("Failed to abort multipart upload: {abort_error}");
        }
        return Err(eyre!("Multipart upload failed: {error}"));
    }

    upload
        .complete()
        .await
        .map_err(|e| eyre!("Failed to complete multipart upload: {e}"))?;
    tracing::info!("Successfully uploaded graph checkpoint: key={key}");
    Ok(())
}

pub async fn download_graph(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
) -> Result<Bytes> {
    tracing::info!(
        "Downloading graph checkpoint: store={}, key={}",
        store_location,
        key
    );
    let store = client.store(store_location)?;
    let location = path(key)?;
    let total_size: usize = store.head(&location).await?.size.try_into()?;
    if total_size == 0 {
        return Ok(Bytes::new());
    }

    let ranges: Vec<_> = (0..total_size)
        .step_by(DEFAULT_CHECKPOINT_CHUNK_SIZE)
        .map(|start| start..(start + DEFAULT_CHECKPOINT_CHUNK_SIZE).min(total_size))
        .collect();
    let chunks: Vec<(usize, Bytes)> = stream::iter(ranges)
        .map(|range| {
            let store = store.clone();
            let location = location.clone();
            async move {
                let start = range.start;
                let range = (range.start as u64)..(range.end as u64);
                let bytes = retry("range download", || {
                    store.get_range(&location, range.clone())
                })
                .await?;
                Ok::<_, eyre::Report>((start, bytes))
            }
        })
        .buffer_unordered(DEFAULT_CHECKPOINT_PARALLELISM)
        .try_collect()
        .await?;

    let mut result = BytesMut::zeroed(total_size);
    for (offset, chunk) in chunks {
        result[offset..offset + chunk.len()].copy_from_slice(&chunk);
    }
    Ok(result.freeze())
}

pub async fn delete_graph(
    client: &ObjectStoreClient,
    store_location: &str,
    key: &str,
) -> Result<()> {
    let store = client.store(store_location)?;
    store.delete(&path(key)?).await?;
    Ok(())
}

async fn upload_graph_simple(
    store: &iris_mpc_common::object_store::ObjectStoreRef,
    location: &object_store::path::Path,
    data: Bytes,
) -> Result<()> {
    retry("object upload", || store.put(location, data.clone().into()))
        .await
        .map(|_| ())
}

async fn retry<T, F, Fut>(operation: &str, mut f: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = object_store::Result<T>>,
{
    let mut attempts = 0;
    loop {
        match f().await {
            Ok(value) => return Ok(value),
            Err(_) if attempts < MAX_RETRIES => {
                attempts += 1;
                tracing::warn!("Retry {attempts} for {operation}");
                sleep(Duration::from_secs(2)).await;
            }
            Err(error) => {
                return Err(eyre!(
                    "{operation} failed after {attempts} retries: {error}"
                ));
            }
        }
    }
}
