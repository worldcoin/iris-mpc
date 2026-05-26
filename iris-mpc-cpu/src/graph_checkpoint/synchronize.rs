use std::collections::HashMap;

use aws_sdk_s3::Client;
use eyre::{bail, eyre, Result};
use iris_mpc_common::{helpers::sync::SyncResult, IrisVectorId};
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

    // lookup: modification_id -> bytes
    let mut mutation_bytes: HashMap<i64, &Option<Vec<u8>>> = HashMap::new();
    for state in sync_result.all_states.iter() {
        if state.modifications.len() != state.graph_mutation_bytes.len() {
            bail!("length mismatch in sync_graph_mutations(). modifications len: {}; graph mutations len: {}", state.modifications.len(), state.graph_mutation_bytes.len());
        }
        for (modification, graph_mutation) in
            izip!(&state.modifications, &state.graph_mutation_bytes)
        {
            let entry = mutation_bytes
                .entry(modification.id)
                .and_modify(|e| {
                    // if one party has not written the graph mutation yet (e would be None) but the other had,
                    // then update the graph mutation
                    if e.is_none() && graph_mutation.is_some() {
                        *e = graph_mutation;
                    }
                })
                .or_insert(graph_mutation);
            if entry.is_some() && graph_mutation.is_some() && *entry != graph_mutation {
                bail!(
                    "graph mutation mismatch between parties. modification id: {}",
                    modification.id
                );
            }
        }
    }

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
