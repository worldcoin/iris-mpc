use std::collections::HashMap;

use aws_sdk_s3::Client;
use eyre::{bail, eyre, Result};
use iris_mpc_common::{
    helpers::sync::{SyncResult, SyncState},
    IrisVectorId,
};
use itertools::izip;

use super::{download_graph_checkpoint, GraphCheckpointState};
use crate::{
    execution::hawk_main::{BothEyes, HawkOps},
    hawkers::aby3::aby3_store::Aby3Store,
    hnsw::{
        graph::{
            graph_store::{GraphMutationRow, GraphPg},
            GraphMutation,
        },
        GraphMem,
    },
};

/// Ensures that every modification that was rolled forward during
/// `sync_modifications()` has its graph mutations present in the local
/// `hawk_graph_mutations` table.
///
/// It is assumed that one party could have None for a graph mutation and
/// another party could have Some for a graph mutation. for every modification
/// in to_update, the peer will search SyncResult.all_states for a Some(graph_mutation)
/// to roll forward.
///
/// The graph mutations are applied all together in a transaction.
pub async fn sync_graph_mutations(
    sync_result: &SyncResult,
    graph_pg: &GraphPg<Aby3Store<HawkOps>>,
) -> Result<()> {
    // assuming that compare_modifications() does not return duplicate ids in to_update
    let (to_update, _) = sync_result.compare_modifications();
    if to_update.is_empty() {
        return Ok(());
    }

    let mutation_bytes = build_mutation_bytes(&sync_result.all_states)?;

    let mut tx = graph_pg.begin_tx().await?;
    for modification in &to_update {
        let Some(mutation) = mutation_bytes.get(&modification.id) else {
            bail!(
                "mutation for modification id {} not found in lookup table",
                modification.id
            );
        };

        // note that it is valid for a rolled forward modification to have no graph mutation associated with it. in which case
        // mutation would be None
        if let Some(mutation) = mutation {
            graph_pg
                .upsert_hawk_graph_mutations(&mut tx, modification.id, mutation)
                .await
                .map_err(|e| {
                    eyre!(
                        "Failed to insert graph mutations for modification {}: {e}",
                        modification.id
                    )
                })?;
        }
    }
    tx.commit().await?;
    tracing::info!("synced graph mutations per party");

    Ok(())
}

/// Merges per-party `graph_mutation_bytes` slices into a single
/// `modification_id → &Option<Vec<u8>>` lookup, applying the following
/// peer-merge semantics for each id:
///
/// | existing entry | incoming | result |
/// |---|---|---|
/// | *(absent)* | any | insert |
/// | `None` | `Some(_)` | upgrade to `Some` |
/// | `Some(a)` / `None` | equal value | keep (noop) |
/// | `Some(a)` | `Some(b)` where `a ≠ b` | **error** |
fn build_mutation_bytes(all_states: &[SyncState]) -> Result<HashMap<i64, &Option<Vec<u8>>>> {
    let mut mutation_bytes: HashMap<i64, &Option<Vec<u8>>> = HashMap::new();
    for state in all_states.iter() {
        if state.modifications.len() != state.graph_mutation_bytes.len() {
            bail!(
                "length mismatch in sync_graph_mutations(). modifications len: {}; graph \
                 mutations len: {}",
                state.modifications.len(),
                state.graph_mutation_bytes.len()
            );
        }
        for (modification, graph_mutation) in
            izip!(&state.modifications, &state.graph_mutation_bytes)
        {
            match (mutation_bytes.get(&modification.id), graph_mutation) {
                // first time seeing this id: insert whatever the party reported
                (None, _) => {
                    mutation_bytes.insert(modification.id, graph_mutation);
                }
                // existing entry has no bytes but incoming does: upgrade
                (Some(None), Some(_)) => {
                    mutation_bytes.insert(modification.id, graph_mutation);
                }
                // both sides agree (including both-None): nothing to do
                (Some(existing), incoming) if *existing == incoming => {}
                // conflict: two parties reported different non-None bytes
                _ => {
                    bail!(
                        "graph mutation mismatch between parties. modification id: {}",
                        modification.id
                    );
                }
            }
        }
    }
    Ok(mutation_bytes)
}

/// Loads the in-memory graph from an optional S3 checkpoint and replays a
/// pre-fetched list of WAL mutation rows on top of it.
///
/// - If `checkpoint` is `None` the graph starts empty and `wal_rows` are
///   replayed from the beginning.
/// - If `checkpoint` is `Some` the checkpoint is downloaded from S3 first, then
///   `wal_rows` (already filtered to those that follow the checkpoint) are
///   applied on top.
///
/// `wal_rows` must be ordered by `modification_id` ascending.
pub async fn load_graph_and_roll_forward(
    s3_client: &Client,
    checkpoint_bucket: &str,
    checkpoint: Option<GraphCheckpointState>,
    wal_rows: Vec<GraphMutationRow>,
) -> Result<BothEyes<GraphMem<IrisVectorId>>> {
    let mut both_eyes = if let Some(state) = checkpoint {
        tracing::info!(
            "Loading graph from common S3 checkpoint, hash: {}",
            state.blake3_hash
        );
        download_graph_checkpoint(s3_client, checkpoint_bucket, &state).await?
    } else {
        tracing::info!("No S3 checkpoint found, defaulting to empty graph");
        [GraphMem::new(), GraphMem::new()]
    };

    let n = wal_rows.len();
    apply_graph_mutations(&mut both_eyes, wal_rows)?;
    if n > 0 {
        tracing::info!(n, "applied WAL graph mutations to checkpoint");
    }

    Ok(both_eyes)
}

/// Deserializes and applies a sequence of WAL mutation rows to an in-memory
/// graph, mutating `both_eyes` in place.
///
/// Rows must be ordered by `modification_id` ascending (which is the ordering
/// guaranteed by all `get_hawk_graph_mutations_*` queries).
pub fn apply_graph_mutations(
    both_eyes: &mut BothEyes<GraphMem<IrisVectorId>>,
    mutation_rows: Vec<GraphMutationRow>,
) -> Result<()> {
    debug_assert!(mutation_rows
        .windows(2)
        .all(|w| w[0].modification_id < w[1].modification_id));
    for row in mutation_rows {
        let [left_mutations, right_mutations]: [Vec<GraphMutation<IrisVectorId>>; 2] =
            row.deserialize_mutations()?;
        both_eyes[0].insert_apply(left_mutations);
        both_eyes[1].insert_apply(right_mutations);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::graph::mutation::UpdateEntryPoint;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Serialize a pair of mutation lists into the bincode format expected by
    /// `GraphMutationRow`.
    fn serialize_mutations(
        left: Vec<GraphMutation<IrisVectorId>>,
        right: Vec<GraphMutation<IrisVectorId>>,
    ) -> Vec<u8> {
        let both_eyes: [Vec<GraphMutation<IrisVectorId>>; 2] = [left, right];
        bincode::serialize(&both_eyes).expect("bincode serialization failed")
    }

    fn make_row(
        modification_id: i64,
        left: Vec<GraphMutation<IrisVectorId>>,
        right: Vec<GraphMutation<IrisVectorId>>,
    ) -> GraphMutationRow {
        GraphMutationRow {
            modification_id,
            serialized_mutations: serialize_mutations(left, right),
        }
    }

    /// Construct a `VectorId` from a plain serial id for use in tests.
    fn vid(id: u32) -> IrisVectorId {
        IrisVectorId::from_serial_id(id)
    }

    /// A minimal `AddNode` mutation that adds the node to layer 0 without
    /// updating entry points.
    fn add_node(id: IrisVectorId) -> GraphMutation<IrisVectorId> {
        GraphMutation::AddNode {
            id,
            height: 1,
            update_ep: UpdateEntryPoint::False,
        }
    }

    // ── apply_graph_mutations ────────────────────────────────────────────

    #[test]
    fn empty_wal_leaves_graph_unchanged() {
        let mut both_eyes = [GraphMem::new(), GraphMem::new()];
        apply_graph_mutations(&mut both_eyes, vec![]).unwrap();
        assert!(both_eyes[0].layers.is_empty());
        assert!(both_eyes[1].layers.is_empty());
        assert!(both_eyes[0].entry_points.is_empty());
        assert!(both_eyes[1].entry_points.is_empty());
    }

    #[test]
    fn add_node_appears_in_correct_eye() {
        let mut both_eyes = [GraphMem::new(), GraphMem::new()];
        // Left eye gets node 1; right eye gets nothing.
        let row = make_row(1, vec![add_node(vid(1))], vec![]);
        apply_graph_mutations(&mut both_eyes, vec![row]).unwrap();

        assert_eq!(
            both_eyes[0].layers.len(),
            1,
            "left eye should have one layer"
        );
        assert_eq!(
            both_eyes[0].layers[0].get_links(&vid(1)),
            Some([].as_slice()),
            "node 1 should exist in left eye layer 0"
        );
        assert!(
            both_eyes[1].layers.is_empty(),
            "right eye should be unmodified"
        );
    }

    #[test]
    fn mutations_applied_to_both_eyes_independently() {
        let mut both_eyes = [GraphMem::new(), GraphMem::new()];
        let row = make_row(1, vec![add_node(vid(1))], vec![add_node(vid(10))]);
        apply_graph_mutations(&mut both_eyes, vec![row]).unwrap();

        assert_eq!(
            both_eyes[0].layers[0].get_links(&vid(1)),
            Some([].as_slice())
        );
        assert_eq!(
            both_eyes[1].layers[0].get_links(&vid(10)),
            Some([].as_slice())
        );
        // Cross-check: node from one eye must not appear in the other.
        assert!(both_eyes[0].layers[0].get_links(&vid(10)).is_none());
        assert!(both_eyes[1].layers[0].get_links(&vid(1)).is_none());
    }

    #[test]
    fn multiple_rows_applied_in_order() {
        let mut both_eyes = [GraphMem::new(), GraphMem::new()];
        let rows = vec![
            make_row(1, vec![add_node(vid(1))], vec![add_node(vid(10))]),
            make_row(2, vec![add_node(vid(2))], vec![add_node(vid(20))]),
            make_row(3, vec![add_node(vid(3))], vec![add_node(vid(30))]),
        ];
        apply_graph_mutations(&mut both_eyes, rows).unwrap();

        for node in [1u32, 2, 3] {
            assert_eq!(
                both_eyes[0].layers[0].get_links(&vid(node)),
                Some([].as_slice()),
                "node {node} missing from left eye"
            );
        }
        for node in [10u32, 20, 30] {
            assert_eq!(
                both_eyes[1].layers[0].get_links(&vid(node)),
                Some([].as_slice()),
                "node {node} missing from right eye"
            );
        }
    }

    #[test]
    fn malformed_serialized_bytes_return_error() {
        let mut both_eyes = [GraphMem::new(), GraphMem::new()];
        let bad_row = GraphMutationRow {
            modification_id: 99,
            serialized_mutations: vec![0xFF, 0xFE, 0xFD], // not valid bincode
        };
        assert!(
            apply_graph_mutations(&mut both_eyes, vec![bad_row]).is_err(),
            "expected an error for garbage serialized_mutations"
        );
    }
}

#[cfg(test)]
mod sync_graph_mutations_tests {
    use iris_mpc_common::{
        config::CommonConfig,
        helpers::sync::{Modification, SyncState},
    };

    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a minimal `SyncState` from `(modification_id, graph_mutation_bytes)` pairs.
    fn make_state(pairs: Vec<(i64, Option<Vec<u8>>)>) -> SyncState {
        let (modifications, graph_mutation_bytes): (Vec<_>, Vec<_>) = pairs
            .into_iter()
            .map(|(id, bytes)| {
                (
                    Modification {
                        id,
                        ..Default::default()
                    },
                    bytes,
                )
            })
            .unzip();
        SyncState {
            db_len: 0,
            modifications,
            next_sns_sequence_num: None,
            common_config: CommonConfig::default(),
            graph_mutation_bytes,
        }
    }

    // ── build_mutation_bytes ──────────────────────────────────────────────────

    /// No parties → empty map (mirrors the `to_update.is_empty()` early-return path).
    #[test]
    fn empty_states_returns_empty_map() {
        let result = build_mutation_bytes(&[]).unwrap();
        assert!(result.is_empty());
    }

    /// Single party with a `Some` entry inserts it as-is.
    #[test]
    fn single_party_inserts_some_entry() {
        let bytes = vec![1u8, 2, 3];
        let state = make_state(vec![(1, Some(bytes.clone()))]);
        let result = build_mutation_bytes(&[state]).unwrap();
        assert_eq!(result[&1], &Some(bytes));
    }

    /// All parties report `None` for the same id → the map entry stays `None`.
    #[test]
    fn all_none_across_parties() {
        let a = make_state(vec![(1, None)]);
        let b = make_state(vec![(1, None)]);
        let c = make_state(vec![(1, None)]);
        let result = build_mutation_bytes(&[a, b, c]).unwrap();
        assert_eq!(result[&1], &None);
    }

    /// `None` then `Some`: the `Some` upgrades the existing `None` entry.
    #[test]
    fn none_then_some_upgrades() {
        let bytes = vec![4u8, 5, 6];
        let a = make_state(vec![(1, None)]);
        let b = make_state(vec![(1, Some(bytes.clone()))]);
        let result = build_mutation_bytes(&[a, b]).unwrap();
        assert_eq!(result[&1], &Some(bytes));
    }

    /// `Some` then `None`: the existing `Some` is preserved (no downgrade).
    #[test]
    fn some_then_none_preserves_existing() {
        let bytes = vec![4u8, 5, 6];
        let a = make_state(vec![(1, Some(bytes.clone()))]);
        let b = make_state(vec![(1, None)]);
        let result = build_mutation_bytes(&[a, b]).unwrap();
        assert_eq!(result[&1], &Some(bytes));
    }

    /// Both parties report identical `Some` bytes → noop, result unchanged.
    #[test]
    fn matching_some_bytes_is_noop() {
        let bytes = vec![7u8, 8, 9];
        let a = make_state(vec![(1, Some(bytes.clone()))]);
        let b = make_state(vec![(1, Some(bytes.clone()))]);
        let result = build_mutation_bytes(&[a, b]).unwrap();
        assert_eq!(result[&1], &Some(bytes));
    }

    /// Two parties report different `Some` bytes for the same id → bail.
    #[test]
    fn mismatched_some_bytes_bail() {
        let a = make_state(vec![(1, Some(vec![1, 2, 3]))]);
        let b = make_state(vec![(1, Some(vec![9, 9, 9]))]);
        let err = build_mutation_bytes(&[a, b]).unwrap_err();
        assert!(
            err.to_string().contains("mismatch"),
            "expected a mismatch error, got: {err}"
        );
    }

    /// Multiple modification ids are tracked independently in the same merge.
    #[test]
    fn multiple_ids_tracked_independently() {
        let bytes_a = vec![0xAu8];
        let bytes_b = vec![0xBu8];
        // party 0: mod 1 = Some(A), mod 2 = None
        let p0 = make_state(vec![(1, Some(bytes_a.clone())), (2, None)]);
        // party 1: mod 1 = None,    mod 2 = Some(B)
        let p1 = make_state(vec![(1, None), (2, Some(bytes_b.clone()))]);
        let result = build_mutation_bytes(&[p0, p1]).unwrap();
        assert_eq!(result[&1], &Some(bytes_a), "mod 1 should be upgraded");
        assert_eq!(result[&2], &Some(bytes_b), "mod 2 should be upgraded");
    }

    /// `modifications` and `graph_mutation_bytes` length mismatch → bail.
    #[test]
    fn length_mismatch_bails() {
        let state = SyncState {
            db_len: 0,
            modifications: vec![Modification {
                id: 1,
                ..Default::default()
            }],
            next_sns_sequence_num: None,
            common_config: CommonConfig::default(),
            graph_mutation_bytes: vec![], // one modification, zero byte entries
        };
        let err = build_mutation_bytes(&[state]).unwrap_err();
        assert!(
            err.to_string().contains("length mismatch"),
            "expected a length mismatch error, got: {err}"
        );
    }
}
