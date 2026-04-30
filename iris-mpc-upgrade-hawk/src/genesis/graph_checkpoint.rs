use aws_sdk_s3::Client as S3Client;
use eyre::{bail, Result};
use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::execution::hawk_main::{BothEyes, GraphRef, HawkOps};
use iris_mpc_cpu::genesis::state_accessor::set_last_indexed_iris_id;
use iris_mpc_cpu::graph_checkpoint::{upload_graph_checkpoint, GraphCheckpointState};
use iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store;
use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
use iris_mpc_store::Store as IrisStore;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

use super::{GenesisHawkHandle, JobResult};

/// Uploads a genesis checkpoint, sends the result, and synchronizes peers.
#[allow(clippy::too_many_arguments)]
pub async fn upload_and_sync_genesis_checkpoint(
    checkpoint_bucket: &str,
    party_id: usize,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    s3_client: &S3Client,
    last_indexed_id: u32,
    max_modification_indexed_id: i64,
    is_archival: bool,
    tx_results: &Sender<JobResult>,
    hawk_handle: &mut GenesisHawkHandle,
) -> Result<()> {
    let checkpoint_state = match upload_graph_checkpoint(
        checkpoint_bucket,
        party_id,
        imem_graph_stores,
        s3_client,
        last_indexed_id,
        max_modification_indexed_id,
        None,
        is_archival,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(
                "failed to upload genesis checkpoint for last_indexed_id: {}: {}",
                last_indexed_id,
                e
            );
            bail!(e);
        }
    };

    let (tx, done_rx) = oneshot::channel();
    let result = JobResult::new_s3_checkpoint(checkpoint_state, tx);
    tx_results.send(result).await?;
    hawk_handle.sync_peers(false, Some(done_rx)).await?;
    Ok(())
}

pub async fn maybe_rollback_iris_db(
    graph_checkpoint: &GraphCheckpointState,
    graph_store: &GraphPg<Aby3Store<HawkOps>>,
    iris_store: &IrisStore,
    last_indexed_id: IrisSerialId,
    last_indexed_modification_id: i64,
) -> Result<()> {
    if last_indexed_modification_id != graph_checkpoint.last_indexed_modification_id {
        bail!("mismatch between db and s3 checkpoint for last_indexed_modification_id: db={}, checkpoint={}",
            last_indexed_modification_id, graph_checkpoint.last_indexed_modification_id);
    }

    if last_indexed_id < graph_checkpoint.last_indexed_iris_id {
        bail!("s3 checkpoint is ahead of iris db: db_last_indexed_iris_id={}, checkpoint_last_indexed_iris_id={}",
            last_indexed_id,
            graph_checkpoint.last_indexed_iris_id);
    }

    if last_indexed_id > graph_checkpoint.last_indexed_iris_id {
        tracing::info!("S3 checkpoint is behind the iris db. rolling back the iris db");
        let graph_tx = graph_store.tx().await?;
        let mut tx = graph_tx.tx;
        set_last_indexed_iris_id(&mut tx, graph_checkpoint.last_indexed_iris_id).await?;
        iris_store
            .delete_irises_after_id_tx(&mut tx, graph_checkpoint.last_indexed_iris_id as usize)
            .await?;
        tx.commit().await?;
    }
    Ok(())
}
