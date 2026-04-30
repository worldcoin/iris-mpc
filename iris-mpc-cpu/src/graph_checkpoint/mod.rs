mod data;
mod s3_client;

use ampc_server_utils::{try_get_endpoint_other_nodes, ServerCoordinationConfig};
pub use data::*;
use iris_mpc_common::config::Config;
pub use s3_client::*;

use eyre::{bail, eyre, Result};

use crate::{
    execution::hawk_main::HawkOps, hawkers::aby3::aby3_store::Aby3Store,
    hnsw::graph::graph_store::GraphPg,
};

pub async fn get_common_checkpoint(
    config: &Config,
    my_checkpoint_hashes: GraphCheckpointHashes,
    my_checkpoints: Vec<GraphCheckpointState>,
) -> Result<Option<GraphCheckpointState>> {
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
    my_checkpoints: Vec<GraphCheckpointState>,
    others_hashes: Vec<GraphCheckpointHashes>,
) -> Result<Option<GraphCheckpointState>> {
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

pub async fn get_most_recent_checkpoints(
    graph_store: &GraphPg<Aby3Store<HawkOps>>,
) -> Result<(Vec<GraphCheckpointState>, GraphCheckpointHashes)> {
    let mut output_hashes: GraphCheckpointHashes = [[0; 32]; 10];
    let db_checkpoints = graph_store.get_genesis_graph_checkpoints().await?;
    let mut valid_tuples = vec![];
    for db_cp in db_checkpoints {
        let genesis_cp_state: Result<GraphCheckpointState> = db_cp.try_into();
        if let Ok(genesis_cp_state) = genesis_cp_state {
            if let Ok(hash) = blake3::Hash::from_hex(genesis_cp_state.blake3_hash.as_bytes()) {
                valid_tuples.push((genesis_cp_state, hash));
            } else {
                tracing::warn!("checkpoint hash failed to parse");
            }
        } else {
            tracing::warn!("failed to convert GraphCheckpointRow to GraphCheckpointState");
        }
    }
    let (checkpoints, hashes): (Vec<GraphCheckpointState>, Vec<blake3::Hash>) =
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
    use crate::utils::serialization::graph::GraphFormat;

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

    fn checkpoint(label: &str) -> GraphCheckpointState {
        GraphCheckpointState {
            s3_key: label.to_string(),
            last_indexed_iris_id: 0,
            last_indexed_modification_id: 0,
            graph_mutation_id: None,
            blake3_hash: label.to_string(),
            graph_version: GraphFormat::Current.version(),
            is_archival: false,
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
