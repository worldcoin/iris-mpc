use std::{fmt::Display, str::FromStr, time::Instant};

use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use iris_mpc_common::{config::Config, IrisSerialId};
use serde::{Deserialize, Serialize};

use crate::{
    execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT},
    hnsw::{
        graph::{
            graph_store::{self, GraphPg},
            layered_graph::GraphMem,
        },
        vector_store::Ref,
        VectorStore,
    },
    utils::s3_checkpoint::*,
};

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
}

/// Creates an S3 graph checkpoint.
pub async fn upload_genesis_checkpoint(
    config: &Config,
    graph_mem: &BothEyes<GraphRef>,
    s3_client: &S3Client,
    last_indexed_iris_id: IrisSerialId,
    last_indexed_modification_id: i64,
) -> Result<GenesisCheckpointState> {
    tracing::info!(
        "Creating S3 graph checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}",
        last_indexed_iris_id,
        last_indexed_modification_id
    );

    let start = Instant::now();

    let left_graph = graph_mem[LEFT].read().await;
    let right_graph = graph_mem[RIGHT].read().await;

    let data = serialize_both_eyes(&[&*left_graph, &*right_graph])?;
    tracing::info!(
        "Serialized graphs to {} bytes in {:?}",
        data.len(),
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
        config.party_id,
        uuid::Uuid::new_v4()
    );

    let bucket = &config.graph_checkpoint_bucket_name;
    upload_graph(s3_client, bucket, &s3_key, &data).await?;

    let checkpoint = GenesisCheckpointState {
        s3_key: s3_key.clone(),
        last_indexed_iris_id,
        last_indexed_modification_id,
        blake3_hash,
    };

    tracing::info!(
        "S3 graph checkpoint created successfully: s3_key={}, duration={:?}",
        s3_key,
        start.elapsed()
    );

    metrics::histogram!("genesis_checkpoint_duration").record(start.elapsed().as_secs_f64());
    metrics::gauge!("genesis_checkpoint_size_bytes").set(data.len() as f64);
    metrics::gauge!("genesis_checkpoint_last_indexed_id").set(last_indexed_iris_id as f64);

    Ok(checkpoint)
}

pub async fn download_genesis_checkpoint<T: Ref + Display + FromStr + Ord>(
    s3_client: &S3Client,
    config: &Config,
    state: GenesisCheckpointState,
) -> Result<BothEyes<GraphMem<T>>> {
    let bucket = &config.graph_checkpoint_bucket_name;
    let binary_graph = download_graph(s3_client, bucket, &state.s3_key).await?;

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

    let graphs = deserialize_both_eyes(&binary_graph)?;
    Ok(graphs)
}

/// Gets the latest checkpoint from genesis_graph_checkpoint table.
pub async fn get_latest_checkpoint_state<V: VectorStore>(
    graph_store: &GraphPg<V>,
) -> Result<Option<GenesisCheckpointState>> {
    tracing::info!("Retrieving latest graph checkpoint metadata from genesis_graph_checkpoint");

    let row = graph_store.get_latest_genesis_graph_checkpoint().await?;
    let metadata = row
        .map(|row| -> Result<GenesisCheckpointState> {
            let iris_id: IrisSerialId = row.last_indexed_iris_id.try_into().map_err(|_| {
                eyre!(
                    "Invalid last_indexed_iris_id for checkpoint: {}",
                    row.last_indexed_iris_id
                )
            })?;

            Ok(GenesisCheckpointState {
                s3_key: row.s3_key,
                last_indexed_iris_id: iris_id,
                last_indexed_modification_id: row.last_indexed_modification_id,
                blake3_hash: row.blake3_hash,
            })
        })
        .transpose()?;

    if let Some(ref m) = metadata {
        tracing::info!(
            "Found existing checkpoint: s3_key={}, last_indexed_iris_id={}",
            m.s3_key,
            m.last_indexed_iris_id,
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
        "Persisting graph checkpoint state: s3_key={}, last_indexed_iris_id={}",
        state.s3_key,
        state.last_indexed_iris_id,
    );

    let mut tx = tx.tx;
    GraphPg::<V>::insert_genesis_graph_checkpoint(
        &mut tx,
        &state.s3_key,
        i64::from(state.last_indexed_iris_id),
        state.last_indexed_modification_id,
        &state.blake3_hash,
    )
    .await
    .map_err(|e| eyre!("Failed to persist checkpoint state: {:?}", e))?;
    tx.commit().await?;
    Ok(())
}
