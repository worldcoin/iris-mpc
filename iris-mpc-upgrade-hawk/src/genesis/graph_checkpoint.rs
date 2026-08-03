use aws_sdk_s3::Client as S3Client;
use eyre::{bail, Result};
use iris_mpc_common::SerialId;
use iris_mpc_cpu::execution::hawk_main::{BothEyes, GraphRef, HawkOps};
use iris_mpc_cpu::genesis::state_accessor::{
    set_last_indexed_iris_id, set_last_indexed_modification_id,
};
use iris_mpc_cpu::genesis::{Handle as GenesisHawkHandle, JobResult};
use iris_mpc_cpu::graph_checkpoint::{upload_graph_checkpoint, GraphCheckpointState};
use iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store;
use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
use iris_mpc_store::Store as IrisStore;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

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
    let checkpoint_state = upload_graph_checkpoint(
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
    .map_err(|e| {
        tracing::error!(
            "failed to upload genesis checkpoint for last_indexed_id {last_indexed_id}: {e}"
        );
        e
    })?;

    let (tx, done_rx) = oneshot::channel();
    let result = JobResult::new_s3_checkpoint(checkpoint_state, tx);
    tx_results.send(result).await?;
    hawk_handle.sync_state(false, Some(done_rx)).await?;
    Ok(())
}

/// Reset all HNSW-schema state to `graph_checkpoint`:
/// trim the iris tail beyond the checkpoint, restore the indexed cursors, and
/// clear the WAL and modifications table. When a base checkpoint was pinned,
/// `pinned_row_id` carries its `genesis_graph_checkpoint` row id and every row
/// created after it is dropped (by id, so same-height abandoned rows go too);
/// an abandoned lineage must not win the next run's latest-common selection.
/// Single transaction: a crash before commit leaves the prior state; a crash
/// after leaves a re-runnable one.
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
    pinned_row_id: Option<i64>,
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
    if let Some(row_id) = pinned_row_id {
        graph_tx.delete_checkpoints_after_id(row_id).await?;
    }
    graph_tx.tx.commit().await?;

    tracing::info!(
        "reset HNSW state to checkpoint: last_indexed_iris_id={}, last_indexed_modification_id={}",
        s_cp,
        m_cp
    );
    Ok(())
}
