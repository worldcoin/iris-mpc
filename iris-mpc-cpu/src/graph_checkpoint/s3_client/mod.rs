mod multipart;
use std::{fmt::Display, io::Cursor, str::FromStr, time::Instant};

use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use bytes::Bytes;
use eyre::{bail, eyre, Result};

use crate::{
    execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT},
    hawkers::plaintext_store::PlaintextVectorRef,
    hnsw::{
        graph::{
            graph_store::{self, GraphPg},
            layered_graph::GraphMem,
        },
        vector_store::Ref,
        VectorStore,
    },
    utils::serialization::graph::{graph_format_to_i32, GraphFormat},
};

use crate::graph_checkpoint::data::*;
use iris_mpc_common::IrisSerialId;
pub use multipart::*;

/// Creates an S3 graph checkpoint.
pub async fn upload_graph_checkpoint(
    bucket: &str,
    party_id: usize,
    graph_mem: &BothEyes<GraphRef>,
    s3_client: &S3Client,
    last_indexed_iris_id: IrisSerialId,
    last_indexed_modification_id: i64,
    is_archival: bool,
) -> Result<GraphCheckpointState> {
    let start = Instant::now();
    tracing::info!(
        "Creating S3 graph checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}, is_archival={}",
        last_indexed_iris_id,
        last_indexed_modification_id,
        is_archival,
    );

    let left_graph = graph_mem[LEFT].read().await;
    let right_graph = graph_mem[RIGHT].read().await;
    let data = Bytes::from(bincode::serialize(&[&*left_graph, &*right_graph])?);
    let data_len = data.len();
    tracing::info!(
        "Serialized graphs to {} bytes in {:?}",
        data_len,
        start.elapsed()
    );

    // Compute BLAKE3 hash before upload
    let hash_start = Instant::now();
    let blake3_hash = blake3::hash(&data).to_hex().to_string();
    tracing::info!(
        "Computed BLAKE3 hash: {} in {:?}",
        blake3_hash,
        hash_start.elapsed()
    );

    let s3_key = format!(
        "genesis/{}/checkpoint_{}.bin",
        party_id,
        uuid::Uuid::new_v4()
    );

    upload_graph(s3_client, bucket, &s3_key, data).await?;

    let checkpoint = GraphCheckpointState {
        s3_key: s3_key.clone(),
        last_indexed_iris_id,
        last_indexed_modification_id,
        blake3_hash,
        graph_version: graph_format_to_i32(GraphFormat::Current),
        is_archival,
    };

    tracing::info!(
        "S3 graph checkpoint created successfully: s3_key={}, is_archival={}, duration={:?}",
        s3_key,
        is_archival,
        start.elapsed()
    );

    metrics::histogram!("graph_checkpoint_upload_duration").record(start.elapsed().as_secs_f64());
    metrics::gauge!("graph_checkpoint_size_bytes").set(data_len as f64);
    metrics::gauge!("graph_checkpoint_last_indexed_id").set(last_indexed_iris_id as f64);
    metrics::gauge!("graph_checkpoint_last_modification_id")
        .set(last_indexed_modification_id as f64);

    Ok(checkpoint)
}

pub async fn download_graph_checkpoint<T: Ref + Display + FromStr + Ord>(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<BothEyes<GraphMem<T>>> {
    if state.graph_version != graph_format_to_i32(GraphFormat::Current) {
        bail!("unexpected graph version: {}", state.graph_version);
    }

    let start = Instant::now();
    let binary_graph = download_and_hash(s3_client, bucket, state).await?;

    // todo: deserialize in a way that does not require holding 2 graphs in memory at once.
    // currently binary_graph and graphs make two graphs in RAM at once
    let mut cursor = Cursor::new(&binary_graph);
    let graphs: BothEyes<GraphMem<_>> = bincode::deserialize_from(&mut cursor)?;
    Ok(graphs)
}

// this is used for the genesis integration tests.
// it does not convert between graph types
pub async fn download_genesis_checkpoint_plaintext(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<BothEyes<GraphMem<PlaintextVectorRef>>> {
    if state.graph_version != graph_format_to_i32(GraphFormat::Current) {
        bail!("unexpected graph version: {}", state.graph_version);
    }
    let binary_graph = download_and_hash(s3_client, bucket, state).await?;
    let mut cursor = Cursor::new(&binary_graph);
    let graphs: BothEyes<GraphMem<_>> = bincode::deserialize_from(&mut cursor)?;
    Ok(graphs)
}

/// Gets the latest checkpoint from genesis_graph_checkpoint table.
pub async fn get_latest_checkpoint_state<V: VectorStore>(
    graph_store: &GraphPg<V>,
) -> Result<Option<GraphCheckpointState>> {
    tracing::info!("Retrieving latest graph checkpoint metadata from genesis_graph_checkpoint");

    let row = graph_store.get_latest_genesis_graph_checkpoint().await?;
    let metadata: Option<GraphCheckpointState> = row.map(|row| row.try_into()).transpose()?;

    if let Some(m) = &metadata {
        tracing::info!(
            "Found existing checkpoint: s3_key={}, last_indexed_iris_id={}, is_archival={}",
            m.s3_key,
            m.last_indexed_iris_id,
            m.is_archival,
        );
    } else {
        tracing::info!("No existing checkpoint found");
    }

    Ok(metadata)
}

/// stores checkpoint in genesis_graph_checkpoint table.
pub async fn save_checkpoint_state<V: VectorStore>(
    tx: graph_store::GraphTx<'_, V>,
    state: &GraphCheckpointState,
) -> Result<()> {
    tracing::info!(
        "Persisting graph checkpoint state: s3_key={}, last_indexed_iris_id={}, is_archival={}",
        state.s3_key,
        state.last_indexed_iris_id,
        state.is_archival,
    );

    let mut tx = tx.tx;
    GraphPg::<V>::insert_genesis_graph_checkpoint(
        &mut tx,
        &state.s3_key,
        i64::from(state.last_indexed_iris_id),
        state.last_indexed_modification_id,
        &state.blake3_hash,
        state.is_archival,
        state.graph_version,
    )
    .await
    .map_err(|e| eyre!("Failed to persist checkpoint state: {:?}", e))?;
    tx.commit().await?;
    Ok(())
}

/// DANGER: requires that the 3 parties have
/// reached consensus on a common checkpoint.
/// if the parties have not reached agreement, then
/// calling this function could make rollback impossible
pub async fn cleanup_checkpoints<V: VectorStore>(
    bucket: &str,
    s3_client: &S3Client,
    current_state: &GraphCheckpointState,
    graph_store: &GraphPg<V>,
    pruning_mode: PruningMode,
) -> Result<()> {
    tracing::info!(
        "cleaning up old genesis graph checkpoints (mode: {})",
        pruning_mode
    );

    if pruning_mode == PruningMode::None {
        tracing::info!("pruning mode is 'none', skipping cleanup");
        return Ok(());
    }

    let all_checkpoints = graph_store.get_genesis_graph_checkpoints().await?;
    if !all_checkpoints
        .iter()
        .any(|x| x.s3_key == current_state.s3_key)
    {
        bail!("current checkpoint not found in the db");
    }

    for checkpoint in all_checkpoints
        .into_iter()
        .filter(|x| x.s3_key != current_state.s3_key)
        .filter(|x| match pruning_mode {
            PruningMode::AllOlder => true,
            PruningMode::OlderNonArchival => !x.is_archival,
            PruningMode::None => unreachable!(),
        })
    {
        delete_graph(s3_client, bucket, &checkpoint.s3_key).await?;
        graph_store.delete_genesis_checkpoint(checkpoint.id).await?;
    }
    Ok(())
}

/// Verifies that the S3 client has read, write, and delete access to the
/// checkpoint bucket. Uploads a small sentinel object, reads it back, and
/// deletes it. This catches misconfigured buckets/regions/IAM before any
/// mutations occur.
pub async fn verify_s3_checkpoint_access(
    s3_client: &S3Client,
    bucket: &str,
    party_id: usize,
) -> Result<()> {
    let key = format!("genesis/{party_id}/_access_check");
    let body = b"access_check";

    // Write
    s3_client
        .put_object()
        .bucket(bucket)
        .key(&key)
        .body(ByteStream::from_static(body))
        .send()
        .await
        .map_err(|e| eyre!("S3 checkpoint bucket write check failed: {e}"))?;

    // Read
    let resp = s3_client
        .get_object()
        .bucket(bucket)
        .key(&key)
        .send()
        .await
        .map_err(|e| eyre!("S3 checkpoint bucket read check failed: {e}"))?;
    let data = resp
        .body
        .collect()
        .await
        .map_err(|e| eyre!("S3 checkpoint bucket read check failed to collect body: {e}"))?;
    if data.into_bytes().as_ref() != body {
        bail!("S3 checkpoint bucket read check returned unexpected content");
    }

    // Delete
    if let Err(e) = s3_client
        .delete_object()
        .bucket(bucket)
        .key(&key)
        .send()
        .await
    {
        tracing::warn!("S3 checkpoint bucket delete check failed: {e}");
    }

    Ok(())
}

/// Checks whether a given key exists in an S3 bucket using a `HeadObject`
/// request. Returns `true` if the key is present, `false` if it does not
/// exist (HTTP 404 / `NoSuchKey`), or an error for any other failure.
///
/// # Arguments
///
/// * `s3_client` - Authenticated S3 client.
/// * `bucket`    - Name of the S3 bucket to query.
/// * `key`       - Object key to check for existence.
pub async fn s3_key_exists(s3_client: &S3Client, bucket: &str, key: &str) -> Result<bool> {
    match s3_client.head_object().bucket(bucket).key(key).send().await {
        Ok(_) => Ok(true),
        Err(e) => {
            // `head_object` returns a 404 when the key does not exist.
            // The SDK surfaces this as a `NotFound` service error.
            if e.as_service_error()
                .map(|se| se.is_not_found())
                .unwrap_or(false)
            {
                Ok(false)
            } else {
                Err(eyre!("S3 head_object failed for s3://{bucket}/{key}: {e}"))
            }
        }
    }
}

async fn download_and_hash(
    s3_client: &S3Client,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<Bytes> {
    let start = Instant::now();

    let binary_graph = download_graph(s3_client, bucket, &state.s3_key).await?;

    metrics::histogram!("genesis_checkpoint_download_duration")
        .record(start.elapsed().as_secs_f64());

    // Verify BLAKE3 hash after download
    let computed_hash = blake3::hash(&binary_graph).to_hex().to_string();
    if computed_hash != state.blake3_hash {
        return Err(eyre!(
            "BLAKE3 hash mismatch: expected {}, got {}",
            state.blake3_hash,
            computed_hash
        ));
    }
    tracing::info!("BLAKE3 hash verified successfully: {}", computed_hash);
    Ok(binary_graph)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_blake3_hash_to_string() {
        let bytes = b"ABCDEFGHIJKLMNOP";
        let hash1 = blake3::hash(bytes);
        let hash_str1 = hash1.to_hex().to_string();

        // now go back
        let hash2 = blake3::Hash::from_hex(hash_str1.as_bytes()).unwrap();
        assert_eq!(hash1.as_bytes(), hash2.as_bytes());
    }

    #[test]
    fn test_pruning_mode_from_str() {
        use super::PruningMode;
        use std::str::FromStr;

        assert_eq!(PruningMode::from_str("none").unwrap(), PruningMode::None);
        assert_eq!(
            PruningMode::from_str("older-non-archival").unwrap(),
            PruningMode::OlderNonArchival
        );
        assert_eq!(
            PruningMode::from_str("all-older").unwrap(),
            PruningMode::AllOlder
        );
        assert!(PruningMode::from_str("invalid").is_err());
    }

    #[test]
    fn test_pruning_mode_display() {
        use super::PruningMode;

        assert_eq!(PruningMode::None.to_string(), "none");
        assert_eq!(
            PruningMode::OlderNonArchival.to_string(),
            "older-non-archival"
        );
        assert_eq!(PruningMode::AllOlder.to_string(), "all-older");
    }
}
