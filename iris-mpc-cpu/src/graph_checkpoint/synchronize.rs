use std::collections::HashMap;

use eyre::{bail, eyre, Result};
use iris_mpc_common::{helpers::sync::SyncResult, IrisVectorId};
use itertools::izip;

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
/// [`sync_modifications`] has its graph mutations present in the local
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
                bail!("graph mutation mismatch between parties. modification id: {}; party A: {:?}; party B: {:?}", modification.id, entry, graph_mutation);
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
                .insert_hawk_graph_mutations(&mut tx, modification.id, mutation)
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
