use std::{fmt::Display, io::Cursor, str::FromStr, time::Instant};

use aws_sdk_s3::Client as S3Client;
use bytes::Bytes;
use eyre::{bail, eyre, Result};
use iris_mpc_common::{IrisSerialId, IrisVectorId};
use serde::{Deserialize, Serialize};

use crate::{
    execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT},
    hawkers::plaintext_store::PlaintextVectorRef,
    hnsw::{
        graph::{
            graph_store::{self, GenesisGraphCheckpointRow, GraphPg},
            layered_graph::GraphMem,
        },
        VectorStore,
    },
    utils::{s3_checkpoint::*, serialization::graph::GRAPH_VERSION},
};

/// Controls which older checkpoints are deleted during cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningMode {
    /// Do not prune any checkpoints.
    None,
    /// Prune older checkpoints that are not marked archival (default).
    OlderNonArchival,
    /// Prune all older checkpoints regardless of archival flag.
    AllOlder,
}

impl Display for PruningMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PruningMode::None => write!(f, "none"),
            PruningMode::OlderNonArchival => write!(f, "older-non-archival"),
            PruningMode::AllOlder => write!(f, "all-older"),
        }
    }
}

impl FromStr for PruningMode {
    type Err = eyre::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(PruningMode::None),
            "older-non-archival" => Ok(PruningMode::OlderNonArchival),
            "all-older" => Ok(PruningMode::AllOlder),
            _ => Err(eyre!(
                "invalid pruning mode: '{}', expected one of: none, older-non-archival, all-older",
                s
            )),
        }
    }
}

/// Metadata stored in genesis_graph_checkpoint table for graph checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenesisCheckpointState {
    /// S3 key where the checkpoint is stored
    pub s3_key: String,
    /// Last iris serial ID included in this checkpoint
    pub last_indexed_iris_id: IrisSerialId,
    /// Last modification ID included in this checkpoint
    pub last_indexed_modification_id: i64,
    /// BLAKE3 hash of the checkpoint data for integrity verification
    pub blake3_hash: String,
    /// Corresponds to the GraphFormat enum
    pub graph_version: i32,
    /// Whether this checkpoint is archival (i.e. should be retained by pruning).
    pub is_archival: bool,
}

impl TryFrom<GenesisGraphCheckpointRow> for GenesisCheckpointState {
    type Error = eyre::Error;
    fn try_from(value: GenesisGraphCheckpointRow) -> Result<Self, Self::Error> {
        let last_indexed_iris_id: IrisSerialId =
            value.last_indexed_iris_id.try_into().map_err(|_| {
                eyre!(
                    "Invalid last_indexed_iris_id for checkpoint: {}",
                    value.last_indexed_iris_id
                )
            })?;

        Ok(Self {
            s3_key: value.s3_key,
            last_indexed_iris_id,
            last_indexed_modification_id: value.last_indexed_modification_id,
            blake3_hash: value.blake3_hash,
            graph_version: value.graph_version,
            is_archival: value.is_archival,
        })
    }
}

/// Creates an S3 graph checkpoint.
pub async fn upload_genesis_checkpoint(
    bucket: &str,
    party_id: usize,
    graph_mem: &BothEyes<GraphRef>,
    s3_client: &S3Client,
    last_indexed_iris_id: IrisSerialId,
    last_indexed_modification_id: i64,
    is_archival: bool,
) -> Result<GenesisCheckpointState> {
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

    let checkpoint = GenesisCheckpointState {
        s3_key: s3_key.clone(),
        last_indexed_iris_id,
        last_indexed_modification_id,
        blake3_hash,
        graph_version: GRAPH_VERSION,
        is_archival,
    };

    tracing::info!(
        "S3 graph checkpoint created successfully: s3_key={}, is_archival={}, duration={:?}",
        s3_key,
        is_archival,
        start.elapsed()
    );

    metrics::histogram!("genesis_checkpoint_upload_duration").record(start.elapsed().as_secs_f64());
    metrics::gauge!("genesis_checkpoint_size_bytes").set(data_len as f64);
    metrics::gauge!("genesis_checkpoint_last_indexed_id").set(last_indexed_iris_id as f64);
    metrics::gauge!("genesis_checkpoint_last_modification_id")
        .set(last_indexed_modification_id as f64);

    Ok(checkpoint)
}

// this function will eventually convert between graph types if we switch to a golomb-rice encoding.
// It can not use a generic for the VectorRef if we want to convert from one graph type to another
pub async fn download_genesis_checkpoint(
    s3_client: &S3Client,
    bucket: &str,
    state: &GenesisCheckpointState,
) -> Result<BothEyes<GraphMem<IrisVectorId>>> {
    // this is disallowed until there are multiple valid graph versions to choose from
    if state.graph_version != GRAPH_VERSION {
        bail!("unexpected graph version: {}", state.graph_version);
    }

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
    state: &GenesisCheckpointState,
) -> Result<BothEyes<GraphMem<PlaintextVectorRef>>> {
    if state.graph_version != GRAPH_VERSION {
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
) -> Result<Option<GenesisCheckpointState>> {
    tracing::info!("Retrieving latest graph checkpoint metadata from genesis_graph_checkpoint");

    let row = graph_store.get_latest_genesis_graph_checkpoint().await?;
    let metadata: Option<GenesisCheckpointState> = row.map(|row| row.try_into()).transpose()?;

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
    state: &GenesisCheckpointState,
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
    current_state: &GenesisCheckpointState,
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

async fn download_and_hash(
    s3_client: &S3Client,
    bucket: &str,
    state: &GenesisCheckpointState,
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
