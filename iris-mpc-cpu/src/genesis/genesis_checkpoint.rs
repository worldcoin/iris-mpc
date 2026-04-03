use std::{fmt::Display, str::FromStr, time::Instant};

use aws_sdk_s3::Client as S3Client;
use eyre::{eyre, Result};
use iris_mpc_common::{config::Config, IrisSerialId};
use serde::{Deserialize, Serialize};
use sqlx::{Postgres, Transaction};

use crate::{
    execution::hawk_main::{BothEyes, GraphRef, LEFT, RIGHT},
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::{
        graph::{graph_store::GraphPg, layered_graph::GraphMem},
        vector_store::Ref,
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

    let s3_key = generate_checkpoint_key(
        &config.environment,
        config.party_id,
        last_indexed_iris_id,
        last_indexed_modification_id,
    );

    let bucket = &config.graph_checkpoint_bucket_name;
    upload_graph(
        s3_client,
        bucket,
        &s3_key,
        &data,
        DEFAULT_CHECKPOINT_CHUNK_SIZE,
        DEFAULT_CHECKPOINT_PARALLELISM,
    )
    .await?;

    let checkpoint = GenesisCheckpointState {
        s3_key: s3_key.clone(),
        last_indexed_iris_id,
        last_indexed_modification_id,
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
    let binary_graph = download_graph(
        s3_client,
        bucket,
        &state.s3_key,
        DEFAULT_CHECKPOINT_CHUNK_SIZE,
        DEFAULT_CHECKPOINT_PARALLELISM,
    )
    .await?;
    let graphs = deserialize_both_eyes(&binary_graph)?;
    Ok(graphs)
}

/// Gets the latest checkpoint from genesis_graph_checkpoint table.
pub async fn get_latest_checkpoint_state(
    graph_store: &GraphPg<Aby3Store>,
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
pub async fn save_checkpoint_state(
    tx: &mut Transaction<'_, Postgres>,
    state: &GenesisCheckpointState,
) -> Result<()> {
    tracing::info!(
        "Persisting graph checkpoint state: s3_key={}, last_indexed_iris_id={}",
        state.s3_key,
        state.last_indexed_iris_id,
    );

    GraphPg::<Aby3Store>::insert_genesis_graph_checkpoint(
        tx,
        &state.s3_key,
        i64::from(state.last_indexed_iris_id),
        state.last_indexed_modification_id,
    )
    .await
    .map_err(|e| eyre!("Failed to persist checkpoint state: {:?}", e))?;

    Ok(())
}

/// Generates an S3 key for a graph checkpoint.
fn generate_checkpoint_key(
    environment: &str,
    party_id: usize,
    last_indexed_iris_id: IrisSerialId,
    // 0 means None
    last_indexed_modification_id: i64,
) -> String {
    format!(
        "genesis/{}/checkpoint_{}_{}_{}.bin",
        environment, party_id, last_indexed_iris_id, last_indexed_modification_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_checkpoint_key() {
        let key = generate_checkpoint_key("prod", 0, 1000000, 0);
        assert_eq!(key, "genesis/prod/checkpoint_0_1000000_0.bin");

        let key2 = generate_checkpoint_key("dev", 1, 500, 2);
        assert_eq!(key2, "genesis/dev/checkpoint_1_500_2.bin");
    }
}
