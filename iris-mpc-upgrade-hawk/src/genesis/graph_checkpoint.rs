use ampc_server_utils::{try_get_endpoint_other_nodes, ServerCoordinationConfig};
use eyre::{bail, eyre, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use aws_sdk_s3::Client as S3Client;
use iris_mpc_common::config::Config;
use iris_mpc_common::IrisSerialId;
use iris_mpc_cpu::execution::hawk_main::{BothEyes, GraphRef, HawkOps};
use iris_mpc_cpu::genesis::genesis_checkpoint::{
    upload_genesis_checkpoint, GenesisCheckpointState,
};
use iris_mpc_cpu::genesis::state_accessor::set_last_indexed_iris_id;
use iris_mpc_cpu::hawkers::aby3::aby3_store::Aby3Store;
use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
use iris_mpc_store::Store as IrisStore;
use tokio::sync::mpsc::Sender;
use tokio::sync::oneshot;

use super::{
    log_error, log_info, log_warn, Blake3Hash, GenesisHawkHandle, GraphCheckpointHashes, JobResult,
    GRAPH_CHECKPOINT_ENDPOINT,
};

pub async fn get_common_checkpoint(
    config: &Config,
    my_checkpoint_hashes: GraphCheckpointHashes,
    my_checkpoints: Vec<GenesisCheckpointState>,
) -> Result<Option<GenesisCheckpointState>> {
    let server_coord_config = config
        .server_coordination
        .as_ref()
        .ok_or(eyre!("Missing server coordination config"))?;
    let others_hashes = get_others_graph_hashes(server_coord_config).await?;
    if others_hashes.len() != 2 {
        bail!("invalid number of parties");
    }
    find_common_checkpoint(my_checkpoint_hashes, my_checkpoints, others_hashes)
}

/// subset logic for [`get_common_checkpoint`].
/// Finds the first checkpoint whose hash appears in all parties'
/// hash lists (mine + the X `others_hashes`).  Zero hashes are ignored
/// because they represent empty slots in a [`GraphCheckpointHashes`] array.
pub fn find_common_checkpoint(
    my_checkpoint_hashes: GraphCheckpointHashes,
    my_checkpoints: Vec<GenesisCheckpointState>,
    others_hashes: Vec<GraphCheckpointHashes>,
) -> Result<Option<GenesisCheckpointState>> {
    // using a naive O(n^2) algorithm because the hash lists are length  10 and
    // there are 3 parties, and this code is called infrequently.

    let default_hash: Blake3Hash = [0; 32];
    for (idx, hash) in my_checkpoint_hashes.iter().enumerate() {
        if *hash == default_hash {
            continue;
        }
        // Check if this hash is in all of the others_hashes lists
        let mut found_in_all = true;
        for other_hashes in &others_hashes {
            if !other_hashes.contains(hash) {
                found_in_all = false;
                break;
            }
        }
        if found_in_all {
            let r = my_checkpoints.get(idx).cloned();
            if r.is_none() {
                bail!("unreachable condition in get_common_checkpoint()");
            }
            return Ok(r);
        }
    }
    Ok(None)
}

/// Uploads a genesis checkpoint, sends the result, and synchronizes peers.
#[allow(clippy::too_many_arguments)]
pub async fn upload_and_sync_genesis_checkpoint(
    checkpoint_bucket: &str,
    party_id: usize,
    imem_graph_stores: &Arc<BothEyes<GraphRef>>,
    s3_client: &S3Client,
    last_indexed_id: u32,
    max_modification_indexed_id: i64,
    tx_results: &Sender<JobResult>,
    hawk_handle: &mut GenesisHawkHandle,
) -> Result<()> {
    let checkpoint_state = match upload_genesis_checkpoint(
        checkpoint_bucket,
        party_id,
        imem_graph_stores,
        s3_client,
        last_indexed_id,
        max_modification_indexed_id,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            log_error(format!(
                "failed to upload genesis checkpoint for last_indexed_id: {}: {}",
                last_indexed_id, e
            ));
            bail!(e);
        }
    };

    let (tx, done_rx) = oneshot::channel();
    let result = JobResult::new_s3_checkpoint(checkpoint_state, tx);
    tx_results.send(result).await?;
    hawk_handle.sync_peers(false, Some(done_rx)).await?;
    Ok(())
}

pub async fn get_most_recent_checkpoints(
    graph_store: &GraphPg<Aby3Store<HawkOps>>,
) -> Result<(Vec<GenesisCheckpointState>, GraphCheckpointHashes)> {
    let mut output_hashes: GraphCheckpointHashes = [[0; 32]; 10];
    let db_checkpoints = graph_store.get_genesis_graph_checkpoints().await?;
    let mut valid_tuples = vec![];
    for db_cp in db_checkpoints {
        let genesis_cp_state: Result<GenesisCheckpointState> = db_cp.try_into();
        if let Ok(genesis_cp_state) = genesis_cp_state {
            if let Ok(hash) = blake3::Hash::from_hex(genesis_cp_state.blake3_hash.as_bytes()) {
                valid_tuples.push((genesis_cp_state, hash));
            } else {
                log_warn("checkpoint hashe failed to parse".into());
            }
        } else {
            log_warn("failed to convert GenesisCheckpointRow to GenesisCheckpointState".into());
        }
    }
    let (checkpoints, hashes): (Vec<GenesisCheckpointState>, Vec<blake3::Hash>) =
        valid_tuples.into_iter().unzip();
    for (src, dest) in hashes
        .into_iter()
        .take(output_hashes.len())
        .zip(output_hashes.iter_mut())
    {
        dest.copy_from_slice(src.as_bytes());
    }
    Ok((checkpoints, output_hashes))
}

pub async fn maybe_rollback_iris_db(
    graph_checkpoint: &GenesisCheckpointState,
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
        log_info("S3 checkpoint is behind the iris db. rolling back the iris db".into());
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

pub async fn get_others_graph_hashes(
    config: &ServerCoordinationConfig,
) -> Result<Vec<GraphCheckpointHashes>> {
    tracing::info!("⚓️ ANCHOR: Syncing latest graph checkpoints");

    let connected_and_ready =
        try_get_endpoint_other_nodes(config, GRAPH_CHECKPOINT_ENDPOINT).await?;

    let response_texts_futs: Vec<_> = connected_and_ready
        .into_iter()
        .map(|resp| resp.json())
        .collect();
    let graph_checkpoints: Vec<GraphCheckpointHashes> =
        futures::future::try_join_all(response_texts_futs).await?;

    Ok(graph_checkpoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a `Blake3Hash` where every byte is `b`.
    fn h(b: u8) -> Blake3Hash {
        [b; 32]
    }

    /// Build a `GraphCheckpointHashes` ([Blake3Hash; 10]) from up to 3 hashes.
    /// Remaining slots are filled with the zero hash (treated as "empty").
    fn hashes(slots: &[Blake3Hash]) -> GraphCheckpointHashes {
        assert!(slots.len() <= 3, "test helper: use at most 3 slots");
        let mut arr: GraphCheckpointHashes = [[0u8; 32]; 10];
        for (i, &slot) in slots.iter().enumerate() {
            arr[i] = slot;
        }
        arr
    }

    fn checkpoint(label: &str) -> GenesisCheckpointState {
        GenesisCheckpointState {
            s3_key: label.to_string(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 0,
            blake3_hash: label.to_string(),
        }
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// All three parties share hash A → returns the corresponding checkpoint.
    #[test]
    fn all_three_agree_returns_checkpoint() {
        let a = h(0x01);
        let my_hashes = hashes(&[a]);
        let my_cps = vec![checkpoint("cp_a")];
        let others = vec![hashes(&[a]), hashes(&[a])];

        let result = find_common_checkpoint(my_hashes, my_cps, others).unwrap();
        assert_eq!(result.unwrap().s3_key, "cp_a");
    }

    /// Only my party and one other share hash A; the third has a different
    /// hash → no common checkpoint among all three.
    #[test]
    fn only_two_parties_agree_returns_none() {
        let a = h(0x01);
        let b = h(0x02);
        let my_hashes = hashes(&[a]);
        let my_cps = vec![checkpoint("cp_a")];
        let others = vec![hashes(&[a]), hashes(&[b])];

        let result = find_common_checkpoint(my_hashes, my_cps, others).unwrap();
        assert!(result.is_none());
    }

    /// All three parties have completely disjoint hashes → None.
    #[test]
    fn no_overlap_returns_none() {
        let my_hashes = hashes(&[h(0x01)]);
        let my_cps = vec![checkpoint("cp_a")];
        let others = vec![hashes(&[h(0x02)]), hashes(&[h(0x03)])];

        let result = find_common_checkpoint(my_hashes, my_cps, others).unwrap();
        assert!(result.is_none());
    }

    /// Zero hashes are treated as empty slots and must not be counted as a
    /// common checkpoint even when all three parties "share" them.
    #[test]
    fn zero_hashes_are_skipped() {
        let zero = h(0x00);
        let my_hashes = hashes(&[zero]);
        let my_cps = vec![checkpoint("cp_zero")];
        let others = vec![hashes(&[zero]), hashes(&[zero])];

        let result = find_common_checkpoint(my_hashes, my_cps, others).unwrap();
        assert!(result.is_none());
    }

    /// When multiple hashes are common, the one that appears *first* in my
    /// list is returned (order is determined by my_checkpoint_hashes, not the
    /// HashMap iteration order).
    #[test]
    fn first_matching_hash_in_my_list_wins() {
        let a = h(0x01);
        let b = h(0x02);
        let my_hashes = hashes(&[a, b]);
        let my_cps = vec![checkpoint("cp_a"), checkpoint("cp_b")];
        let others = vec![hashes(&[a, b]), hashes(&[a, b])];

        let result = find_common_checkpoint(my_hashes, my_cps, others).unwrap();
        assert_eq!(result.unwrap().s3_key, "cp_a");
    }

    /// When the first hash in my list is not common but a later one is,
    /// the later checkpoint is returned.
    #[test]
    fn second_hash_matches_when_first_does_not() {
        let a = h(0x01);
        let b = h(0x02);
        // Only `b` is shared by all three parties.
        let my_hashes = hashes(&[a, b]);
        let my_cps = vec![checkpoint("cp_a"), checkpoint("cp_b")];
        let others = vec![hashes(&[b]), hashes(&[b])];

        let result = find_common_checkpoint(my_hashes, my_cps, others).unwrap();
        assert_eq!(result.unwrap().s3_key, "cp_b");
    }

    /// If a hash is common but `my_checkpoints` does not contain a checkpoint
    /// at that index, the function must return an error (the "unreachable"
    /// guard).
    #[test]
    fn missing_checkpoint_for_matching_hash_returns_error() {
        let a = h(0x01);
        let my_hashes = hashes(&[a]);
        let my_cps = vec![]; // intentionally empty — no checkpoint at index 0
        let others = vec![hashes(&[a]), hashes(&[a])];

        let result = find_common_checkpoint(my_hashes, my_cps, others);
        assert!(result.is_err());
    }
}
