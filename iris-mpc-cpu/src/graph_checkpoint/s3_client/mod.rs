mod multipart;
mod streaming;
mod streaming_download;
use std::{io::Cursor, time::Instant};

use bytes::Bytes;
use eyre::{bail, eyre, Result};
use iris_mpc_common::object_store::{path, ObjectStoreClient, ObjectStoreExt};

use crate::{
    execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT},
    hnsw::{
        graph::{
            graph_store::{self, GraphPg},
            layered_graph::GraphMem,
        },
        VectorStore,
    },
    utils::serialization::graph::GraphFormat,
};

use crate::graph_checkpoint::data::*;
use iris_mpc_common::SerialId;
pub use multipart::*;
pub use streaming::*;
pub use streaming_download::*;

/// Creates an S3 graph checkpoint.
#[allow(clippy::too_many_arguments)]
pub async fn upload_graph_checkpoint(
    bucket: &str,
    party_id: usize,
    graph_mem: &BothEyes<GraphRef>,
    s3_client: &ObjectStoreClient,
    last_indexed_iris_id: SerialId,
    last_indexed_modification_id: i64,
    graph_mutation_id: Option<i64>,
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
    _upload_graph_checkpoint(
        bucket,
        party_id,
        s3_client,
        last_indexed_iris_id,
        last_indexed_modification_id,
        graph_mutation_id,
        is_archival,
        data,
        "genesis",
        "graph",
        start,
    )
    .await
}

/// Creates an S3 graph checkpoint from plaintext graphs (for testing).
#[allow(clippy::too_many_arguments)]
pub async fn upload_graph_checkpoint_plaintext(
    bucket: &str,
    party_id: usize,
    graph_mem: &BothEyes<GraphMem>,
    s3_client: &ObjectStoreClient,
    last_indexed_iris_id: SerialId,
    last_indexed_modification_id: i64,
    graph_mutation_id: Option<i64>,
    is_archival: bool,
) -> Result<GraphCheckpointState> {
    let start = Instant::now();
    tracing::info!(
        "Creating S3 plaintext graph checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}, is_archival={}",
        last_indexed_iris_id,
        last_indexed_modification_id,
        is_archival,
    );

    let data = Bytes::from(bincode::serialize(graph_mem)?);
    _upload_graph_checkpoint(
        bucket,
        party_id,
        s3_client,
        last_indexed_iris_id,
        last_indexed_modification_id,
        graph_mutation_id,
        is_archival,
        data,
        "plaintext",
        "plaintext graph",
        start,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn _upload_graph_checkpoint(
    bucket: &str,
    party_id: usize,
    s3_client: &ObjectStoreClient,
    last_indexed_iris_id: SerialId,
    last_indexed_modification_id: i64,
    graph_mutation_id: Option<i64>,
    is_archival: bool,
    data: Bytes,
    s3_prefix: &str,
    log_label: &str,
    start: Instant,
) -> Result<GraphCheckpointState> {
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
        "{}/{}/checkpoint_{}.bin",
        s3_prefix,
        party_id,
        uuid::Uuid::new_v4()
    );

    upload_graph(s3_client, bucket, &s3_key, data).await?;

    let checkpoint = GraphCheckpointState {
        s3_key: s3_key.clone(),
        last_indexed_iris_id,
        last_indexed_modification_id,
        graph_mutation_id,
        blake3_hash,
        graph_version: GraphFormat::Current.version(),
        is_archival,
    };

    tracing::info!(
        "S3 {} checkpoint created successfully: s3_key={}, is_archival={}, duration={:?}",
        log_label,
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

pub async fn download_graph_checkpoint(
    s3_client: &ObjectStoreClient,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<BothEyes<GraphMem>> {
    let format = GraphFormat::try_from(state.graph_version)?;
    if format == GraphFormat::Raw {
        bail!("Unexpected graph checkpoint format: Raw");
    }
    let start = Instant::now();
    let (graphs, hash_bytes) =
        stream_download_and_deserialize_graph_pair(s3_client, bucket, &state.s3_key, format)
            .await?;
    metrics::histogram!("genesis_checkpoint_download_duration")
        .record(start.elapsed().as_secs_f64());

    // Verify BLAKE3 hash after download
    let computed_hash = blake3::Hash::from_bytes(hash_bytes).to_hex().to_string();
    if computed_hash != state.blake3_hash {
        return Err(eyre!(
            "BLAKE3 hash mismatch: expected {}, got {}",
            state.blake3_hash,
            computed_hash
        ));
    }
    tracing::info!("BLAKE3 hash verified successfully: {}", computed_hash);
    Ok(graphs)
}

// this is used for the genesis integration tests.
// it does not convert between graph types
pub async fn download_genesis_checkpoint_plaintext(
    s3_client: &ObjectStoreClient,
    bucket: &str,
    state: &GraphCheckpointState,
) -> Result<BothEyes<GraphMem>> {
    if state.graph_version != GraphFormat::Current.version() {
        bail!("unexpected graph version: {}", state.graph_version);
    }
    let binary_graph = download_and_hash(s3_client, bucket, state).await?;
    let mut cursor = Cursor::new(&binary_graph);
    let graphs: BothEyes<GraphMem> = bincode::deserialize_from(&mut cursor)?;
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
        state.graph_mutation_id,
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
///
/// `retain_from_id`: local row-id watermark; rows with `id >= retain_from_id`
/// are kept regardless of pruning mode. Callers pruning right after recording
/// a checkpoint that peers have NOT yet confirmed durable (the sidecar's
/// `UploadAndRecord`) must pass the cycle's agreed base id here: the base was
/// advertised by every party in Phase 1, so retaining it guarantees a common
/// checkpoint survives even if a peer crashes before recording the new one.
/// `None` is only safe when `current_state` itself is known to be durably
/// recorded by all parties (the startup-agreed common checkpoint).
pub async fn cleanup_checkpoints<V: VectorStore>(
    bucket: &str,
    s3_client: &ObjectStoreClient,
    current_state: &GraphCheckpointState,
    retain_from_id: Option<i64>,
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
        .filter(|x| retain_from_id.is_none_or(|min_id| x.id < min_id))
        .filter(|x| match pruning_mode {
            PruningMode::AllOlder => true,
            PruningMode::OlderNonArchival => !x.is_archival,
            PruningMode::None => unreachable!(),
        })
    {
        graph_store.delete_genesis_checkpoint(checkpoint.id).await?;
        delete_graph(s3_client, bucket, &checkpoint.s3_key).await?;
    }
    Ok(())
}

/// Verifies read, write, and delete access to the checkpoint store. Uploads a
/// small sentinel object, reads it back, and
/// deletes it. This catches misconfigured buckets/regions/IAM before any
/// mutations occur.
pub async fn verify_s3_checkpoint_access(
    s3_client: &ObjectStoreClient,
    bucket: &str,
    party_id: usize,
) -> Result<()> {
    let key = format!("genesis/{party_id}/_access_check");
    let body = b"access_check";

    let store = s3_client.store(bucket)?;
    let location = path(&key)?;

    // Write
    store
        .put(&location, Bytes::from_static(body).into())
        .await
        .map_err(|e| eyre!("Checkpoint object-store write check failed: {e}"))?;

    // Read
    let data = store
        .get(&location)
        .await
        .map_err(|e| eyre!("Checkpoint object-store read check failed: {e}"))?
        .bytes()
        .await
        .map_err(|e| eyre!("Checkpoint object-store body read failed: {e}"))?;
    if data.as_ref() != body {
        bail!("Checkpoint object-store read check returned unexpected content");
    }

    // Delete
    if let Err(e) = store.delete(&location).await {
        tracing::warn!("Checkpoint object-store delete check failed: {e}");
    }

    Ok(())
}

/// Checks whether a given key exists in an S3 bucket using a `HeadObject`
/// request. Returns `true` if the key is present, `false` if it does not
/// exist (HTTP 404 / `NoSuchKey`), or an error for any other failure.
///
/// # Arguments
///
/// * `s3_client` - Configured object-store client.
/// * `bucket`    - Object-store location to query.
/// * `key`       - Object key to check for existence.
pub async fn s3_key_exists(s3_client: &ObjectStoreClient, bucket: &str, key: &str) -> Result<bool> {
    let store = s3_client.store(bucket)?;
    let location = path(key)?;
    match store.head(&location).await {
        Ok(_) => Ok(true),
        Err(object_store::Error::NotFound { .. }) => Ok(false),
        Err(e) => Err(eyre!("Object metadata failed for {bucket}/{key}: {e}")),
    }
}

async fn download_and_hash(
    s3_client: &ObjectStoreClient,
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
