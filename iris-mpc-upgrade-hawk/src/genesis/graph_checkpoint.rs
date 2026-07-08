use aws_sdk_s3::Client as S3Client;
use eyre::{bail, Result};
use iris_mpc_common::SerialId;
use iris_mpc_cpu::execution::hawk_main::{BothEyes, GraphRef, HawkOps};
use iris_mpc_cpu::genesis::state_accessor::{
    set_last_indexed_iris_id, set_last_indexed_modification_id,
};
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
        // genesis doesn't need graph_mutation_id, it can just replay irises from the GPU database.
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
    hawk_handle.sync_state(false, Some(done_rx)).await?;
    Ok(())
}

pub async fn maybe_rollback_iris_db(
    graph_checkpoint: &GraphCheckpointState,
    graph_store: &GraphPg<Aby3Store<HawkOps>>,
    iris_store: &IrisStore,
    last_indexed_id: SerialId,
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

/// Reset all HNSW-schema state to `graph_checkpoint` for version-join mode:
/// trim the iris tail beyond the checkpoint, restore the indexed cursors, and
/// clear the WAL and modifications table. When a base checkpoint was pinned,
/// also drop checkpoint rows that post-date it so an abandoned lineage cannot
/// win the next run's latest-common selection. Single transaction: a crash
/// before commit leaves the prior state; a crash after leaves a re-runnable one.
///
/// Takes the **HNSW** iris store (not the source store): all mutated tables live
/// in the HNSW schema.
///
/// # Errors
/// Bails if the checkpoint is ahead of the DB
/// (`last_indexed_id < graph_checkpoint.last_indexed_iris_id`) — corrupt state.
pub async fn reset_to_checkpoint(
    graph_checkpoint: &GraphCheckpointState,
    graph_store: &GraphPg<Aby3Store<HawkOps>>,
    hnsw_iris_store: &IrisStore,
    last_indexed_id: SerialId,
    pinned: bool,
) -> Result<()> {
    if last_indexed_id < graph_checkpoint.last_indexed_iris_id {
        bail!(
            "s3 checkpoint is ahead of iris db: db_last_indexed_iris_id={}, checkpoint_last_indexed_iris_id={}",
            last_indexed_id,
            graph_checkpoint.last_indexed_iris_id
        );
    }

    let s_cp = graph_checkpoint.last_indexed_iris_id;
    let m_cp = graph_checkpoint.last_indexed_modification_id;

    let mut graph_tx = graph_store.tx().await?;
    set_last_indexed_iris_id(&mut graph_tx.tx, s_cp).await?;
    set_last_indexed_modification_id(&mut graph_tx.tx, m_cp).await?;
    hnsw_iris_store
        .delete_irises_after_id_tx(&mut graph_tx.tx, s_cp as usize)
        .await?;
    graph_tx.clear_hawk_graph_mutations().await?;
    hnsw_iris_store
        .clear_modifications_table(&mut graph_tx.tx)
        .await?;
    if pinned {
        graph_tx.delete_checkpoints_after(s_cp).await?;
    }
    graph_tx.tx.commit().await?;

    tracing::info!(
        "reset HNSW state to checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}",
        s_cp,
        m_cp
    );
    Ok(())
}
