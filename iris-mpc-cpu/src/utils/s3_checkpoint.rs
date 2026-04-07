//! S3 Graph Checkpoint Module
//!
//! This module provides functionality for storing and loading graph checkpoints
//! needed by genesis and hawk.

use std::{io::Cursor, sync::Arc, time::Duration};

use aws_sdk_s3::{
    types::{CompletedMultipartUpload, CompletedPart},
    Client as S3Client,
};
use bytes::{Bytes, BytesMut};
use eyre::{eyre, Result};
use iris_mpc_common::IrisVectorId;
use tokio::{sync::Semaphore, task::JoinSet, time::sleep};

use crate::{
    execution::hawk_main::BothEyes,
    hnsw::graph::layered_graph::GraphMem,
    utils::serialization::graph::{read_graph_pair, write_graph_pair_current, ALL_CONCRETE_GRAPH_FORMATS},
};

pub const DEFAULT_CHECKPOINT_CHUNK_SIZE: usize = 100 * 1024 * 1024; // 100 MB chunks
pub const DEFAULT_CHECKPOINT_PARALLELISM: usize = 32;
const MULTIPART_THRESHOLD: usize = 5 * 1024 * 1024; // 5MB - S3 multipart minimum part size

/// Uploads checkpoint data to S3.
/// Uses simple PUT for files under 5MB, multipart upload for larger files.
pub async fn upload_graph(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    data: Bytes,
) -> Result<()> {
    tracing::info!(
        "Uploading graph checkpoint to S3: bucket={}, key={}, size={}",
        bucket,
        key,
        data.len()
    );

    let chunk_size = DEFAULT_CHECKPOINT_CHUNK_SIZE;
    let upload_parallelism = DEFAULT_CHECKPOINT_PARALLELISM;

    if data.len() < MULTIPART_THRESHOLD {
        return upload_graph_simple(s3_client, bucket, key, data).await;
    }

    let mut join_set = JoinSet::new();
    let semaphore = Arc::new(Semaphore::new(upload_parallelism));

    // Initiate Multipart Upload
    let multipart_res = s3_client
        .create_multipart_upload()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| eyre!("Failed to initiate: {:?}", e))?;

    let upload_id = multipart_res
        .upload_id()
        .ok_or_else(|| eyre!("S3 did not return an upload ID"))?;

    // Build chunks, merging last chunk if it's under 5MB (S3 minimum part size)
    let data_len = data.len();
    let chunk_size = std::cmp::max(chunk_size, MULTIPART_THRESHOLD);
    let mut chunks: Vec<(usize, usize)> = (0..data_len)
        .step_by(chunk_size)
        .map(|start| {
            let end = (start + chunk_size).min(data_len);
            (start, end)
        })
        .collect();
    if chunks.len() >= 2 {
        if let Some((last_start, last_end)) = chunks.last().copied() {
            if last_end - last_start < MULTIPART_THRESHOLD {
                // Merge last two chunks by adjusting the second-to-last to include the remainder
                let last_two_start = chunks[chunks.len() - 2].0;
                chunks.pop();
                chunks.pop();
                chunks.push((last_two_start, data_len));
            }
        }
    }

    // Spawn Workers for Chunks
    for (i, (start, end)) in chunks.into_iter().enumerate() {
        let part_number = (i + 1) as i32;
        let client = s3_client.clone();
        let bucket = bucket.to_string();
        let key = key.to_string();
        let upload_id = upload_id.to_string();
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| eyre!("failed to acquire semaphore: {}", e))?;
        let body = data.slice(start..end);

        join_set.spawn(async move {
            let mut attempts = 0;
            let max_retries = 3;

            loop {
                match client
                    .upload_part()
                    .bucket(&bucket)
                    .key(&key)
                    .upload_id(&upload_id)
                    .part_number(part_number)
                    .body(body.clone().into())
                    .send()
                    .await
                {
                    Ok(res) => {
                        let etag = res.e_tag().map(|s| s.to_string());
                        tracing::debug!("part {} uploaded: e_tag={:?}", part_number, etag);

                        let etag = etag.ok_or_else(|| {
                            eyre!("s3 didn't return ETag for part {}", part_number)
                        })?;

                        drop(permit);
                        return Ok(CompletedPart::builder()
                            .e_tag(etag)
                            .part_number(part_number)
                            .build());
                    }
                    Err(_e) if attempts < max_retries => {
                        attempts += 1;
                        tracing::warn!("Retry {} for part {}", attempts, part_number);
                        sleep(Duration::from_secs(2)).await;
                    }
                    Err(e) => {
                        drop(permit);
                        return Err(eyre!("Part {} failed: {:?}", part_number, e));
                    }
                }
            }
        });
    }

    // Collect & Sort ETags
    let mut completed_parts = Vec::new();
    let mut error: Option<eyre::Report> = None;
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok(part)) => completed_parts.push(part),
            Ok(Err(e)) => {
                error.replace(e);
                break;
            }
            Err(e) => {
                error.replace(eyre!("Join error: {:?}", e));
                break;
            }
        }
    }

    if let Some(e) = error {
        join_set.abort_all();
        let _ = s3_client
            .abort_multipart_upload()
            .bucket(bucket)
            .key(key)
            .upload_id(upload_id)
            .send()
            .await;
        return Err(e);
    }

    completed_parts.sort_by_key(|p| p.part_number);

    // Complete Upload
    s3_client
        .complete_multipart_upload()
        .bucket(bucket)
        .key(key)
        .upload_id(upload_id)
        .multipart_upload(
            CompletedMultipartUpload::builder()
                .set_parts(Some(completed_parts))
                .build(),
        )
        .send()
        .await
        .map_err(|e| eyre!("Failed to complete upload: {:?}", e))?;

    tracing::info!("Successfully uploaded graph checkpoint to S3: key={}", key);

    Ok(())
}

/// Downloads the graph from s3
pub async fn download_graph(s3_client: &S3Client, bucket: &str, key: &str) -> Result<Bytes> {
    tracing::info!(
        "Downloading graph checkpoint from S3: bucket={}, key={}",
        bucket,
        key
    );

    let chunk_size = DEFAULT_CHECKPOINT_CHUNK_SIZE;
    let download_parallelism = DEFAULT_CHECKPOINT_PARALLELISM;

    // Get object metadata to find total size
    let head = s3_client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| {
            eyre!(
                "failed to get s3 checkpoint metadata  for bucket {}:, key: {}, error: {}",
                bucket,
                key,
                e
            )
        })?;
    let total_size = head
        .content_length()
        .ok_or_else(|| eyre!("Missing content length"))?
        .try_into()?;

    let mut final_data = BytesMut::zeroed(total_size);
    let semaphore = Arc::new(Semaphore::new(download_parallelism));
    let mut join_set = JoinSet::new();

    tracing::info!("Starting parallel download: {} bytes", total_size);

    // Spawn range-request workers
    for start in (0..total_size).step_by(chunk_size) {
        let end = std::cmp::min(start + chunk_size - 1, total_size - 1);
        let client = s3_client.clone();
        let (b, k) = (bucket.to_string(), key.to_string());
        let permit = semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| eyre!("failed to acquire semaphore: {e}"))?;

        join_set.spawn(async move {
            let mut attempts = 0;
            let range = format!("bytes={}-{}", start, end);

            loop {
                let res = client
                    .get_object()
                    .bucket(&b)
                    .key(&k)
                    .range(&range)
                    .send()
                    .await;

                match res {
                    Ok(output) => {
                        let data = output
                            .body
                            .collect()
                            .await
                            .map_err(|e| eyre!("Body collect error: {:?}", e))?;

                        drop(permit);
                        return Ok((start, data.into_bytes()));
                    }
                    Err(_e) if attempts < 3 => {
                        attempts += 1;
                        tracing::warn!("Retry {} for range {}", attempts, range);
                        sleep(Duration::from_secs(2)).await;
                    }
                    Err(e) => {
                        drop(permit);
                        return Err(eyre!("Range {} failed: {:?}", range, e));
                    }
                }
            }
        });
    }

    // Assemble the pieces
    while let Some(result) = join_set.join_next().await {
        match result {
            Ok(Ok((offset, chunk_data))) => {
                let len = chunk_data.len();
                final_data[offset..offset + len].copy_from_slice(&chunk_data);
            }
            Ok(Err(e)) => {
                join_set.abort_all();
                return Err(e);
            }
            Err(e) => {
                join_set.abort_all();
                return Err(eyre!("Join error: {:?}", e));
            }
        }
    }

    tracing::info!(
        "Successfully downloaded graph checkpoint from S3: key={}, size={}",
        key,
        final_data.len()
    );

    Ok(final_data.freeze())
}

pub async fn delete_graph(s3_client: &S3Client, bucket: &str, key: &str) -> Result<()> {
    tracing::info!("Deleting graph checkpoint: bucket={}, key={}", bucket, key,);
    s3_client
        .delete_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| {
            tracing::error!("Failed to delete graph from S3: {:?}", e);
            e
        })?;
    Ok(())
}

/// Serialize a graph pair for S3 upload using the stable GraphV3 format.
pub fn serialize_both_eyes(
    both_eyes: &BothEyes<&GraphMem<IrisVectorId>>,
) -> Result<Bytes> {
    let data: [GraphMem<IrisVectorId>; 2] = [both_eyes[0].clone(), both_eyes[1].clone()];
    let mut buffer = Vec::new();
    write_graph_pair_current(&mut buffer, data)?;
    Ok(Bytes::from(buffer))
}

/// Deserialize graph pair retrieved from S3.
/// Tries the current stable format first, then falls back to older formats with a warning.
pub fn deserialize_both_eyes(
    data: &[u8],
) -> Result<BothEyes<GraphMem<IrisVectorId>>> {
    // Try current format first (GraphV3)
    let mut cursor = Cursor::new(data);
    if let Ok(graphs) = read_graph_pair(&mut cursor, ALL_CONCRETE_GRAPH_FORMATS[0]) {
        return Ok(graphs);
    }

    // Fallback to older formats with warning
    for format in &ALL_CONCRETE_GRAPH_FORMATS[1..] {
        let mut cursor = Cursor::new(data);
        if let Ok(graphs) = read_graph_pair(&mut cursor, *format) {
            tracing::warn!(
                "S3 checkpoint deserialized using legacy format {:?}. Consider re-uploading with current format.",
                format
            );
            return Ok(graphs);
        }
    }

    Err(eyre!("Unable to deserialize graph pair from S3 checkpoint using any known format"))
}

/// Simple PUT upload for small files (under 5MB).
async fn upload_graph_simple(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    data: Bytes,
) -> Result<()> {
    tracing::info!(
        "Using simple PUT upload for small file: bucket={}, key={}, size={}",
        bucket,
        key,
        data.len()
    );

    let mut attempts = 0;
    let max_retries = 3;

    loop {
        match s3_client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(data.to_vec().into())
            .send()
            .await
        {
            Ok(res) => {
                tracing::info!("Simple PUT upload completed: e_tag={:?}", res.e_tag());
                return Ok(());
            }
            Err(_e) if attempts < max_retries => {
                attempts += 1;
                tracing::warn!("Retry {} for simple PUT upload", attempts);
                sleep(Duration::from_secs(2)).await;
            }
            Err(e) => {
                return Err(eyre!("Simple PUT upload failed: {:?}", e));
            }
        }
    }
}
