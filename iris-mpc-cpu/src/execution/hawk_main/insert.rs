use crate::hnsw::{
    graph::{
        mutation::{EdgeType, UnstampedMutation},
        MutationOp, UpdateEntryPoint,
    },
    searcher::{ConnectPlanV, LayerMode},
    vector_store::VectorStoreMut,
    GraphMem, HnswSearcher, VectorStore,
};

use super::VecRequests;

use eyre::{bail, Result};
use iris_mpc_common::vector_id::VectorId;
use itertools::izip;
use std::collections::BTreeSet;

/// A low-level plan for inserting a query into the HNSW graph.
///
/// An `InsertPlanV` represents the *desired* state of a new node's connections after
/// it is added to the graph. It is created during the initial search phase that precedes
/// the actual insertion.
///
/// The `links` field contains a list of neighbors for each layer of the HNSW graph that
/// the new node will be a part of. These are the "ideal" neighbors found during the
/// search. The list of links should already be trimmed to the desired length (e.g., the
/// HNSW parameter `M`).
///
/// This struct is considered a "low-level" plan because it only specifies the outgoing
/// connections from the new node. It does not include the reciprocal (bilateral) connections
/// from the existing neighbors back to the new node. The final, complete set of graph
/// modifications is represented by a `ConnectPlanV`, which is generated from this `InsertPlanV`.
#[derive(Debug)]
pub struct InsertPlanV<V: VectorStore> {
    pub query: V::QueryRef,
    pub links: Vec<Vec<VectorId>>,
    pub update_ep: UpdateEntryPoint,
}

// Manual implementation of Clone for InsertPlanV, since derive(Clone) does not
// propagate the nested Clone bounds on V::QueryRef via TransientRef.
impl<V: VectorStore> Clone for InsertPlanV<V> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            links: self.links.clone(),
            update_ep: self.update_ep.clone(),
        }
    }
}

/// Insert a collection `plans` of `InsertPlanV` structs into the graph and vector store,
/// adjusting the insertion plans as needed to repair any conflict from parallel searches.
///
/// The `insert_ids` argument consists of `Option<VectorId>`s which are `Some(id)` if the
/// associated plan is to be inserted with a specific identifier (e.g. for updates or for
/// insertions which need to parallel an existing iris code database), and `None` if the
/// associated plan is to be inserted at the next available serial ID, with version 0.
///
/// The `replace_ids` argument consists of `Option<VectorId>`s which are `Some(id)` if the
/// associated slot should additionally emit a `RemoveNode(id)` mutation (e.g. for reauth or
/// identity-update replacements, or for pure deletions). Within a slot the `RemoveNode`
/// mutation has a lower seq_no than the new node's `AddNode`/`AddEdges` mutation
/// ("delete-then-add"), each in its own `GraphMutation`.
///
/// Returns a parallel pair of `VecRequests`:
/// - the per-slot `Vec<ConnectPlanV>` carrying the graph mutations to persist (a slot
///   may produce 0 to 3 mutations from the per-slot steps, and the last non-empty slot may
///   additionally carry the batch's single global compaction mutation);
/// - the per-slot `Option<VectorId>` identifying the newly inserted vector, or `None`
///   for pure deletions and no-op slots.
pub async fn insert<V: VectorStoreMut>(
    store: &mut V,
    graph: &mut GraphMem<VectorId>,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<V>>>,
    insert_ids: &VecRequests<Option<VectorId>>,
    replace_ids: &VecRequests<Option<VectorId>>,
) -> Result<(
    VecRequests<Vec<ConnectPlanV>>,
    VecRequests<Option<VectorId>>,
)> {
    tracing::debug!("Inserting {} InsertPlans into store", plans.len());

    assert_eq!(
        plans.len(),
        replace_ids.len(),
        "plans and replace_ids must be the same length"
    );
    assert_eq!(
        plans.len(),
        insert_ids.len(),
        "plans and insert_ids must be the same length"
    );

    let insert_plans = join_plans(plans, &searcher.layer_mode);
    validate_ep_updates(&insert_plans, &searcher.layer_mode)?;

    let mut intra_batch_inserted = vec![];
    let m = searcher.params.get_M(0);

    let mut slot_outputs: Vec<Vec<ConnectPlanV>> = vec![vec![]; insert_plans.len()];
    let mut slot_inserted_ids: Vec<Option<VectorId>> = vec![None; insert_plans.len()];
    let mut batch_expanded: BTreeSet<(VectorId, usize)> = BTreeSet::new();

    for (idx, (plan, insert_id, replace_id)) in
        izip!(insert_plans, insert_ids, replace_ids).enumerate()
    {
        let mut slot_updated: BTreeSet<(VectorId, usize)> = BTreeSet::new();

        // (a) Delete first: own GraphMutation with the lower seq_no.
        if let Some(rid) = replace_id {
            let mutation = graph.apply_new(UnstampedMutation {
                ops: vec![MutationOp::RemoveNode { id: rid.clone() }],
            })?;
            slot_outputs[idx].push(mutation);
        }

        // (b) Insert: own GraphMutation. Collect this slot's updated and
        // expanded neighborhoods for the per-slot prune and the batch
        // compaction respectively.
        if let Some(InsertPlanV {
            query,
            mut links,
            update_ep,
        }) = plan
        {
            if let Some(bottom_layer) = links.first_mut() {
                if bottom_layer.len() < m {
                    bottom_layer.extend_from_slice(&intra_batch_inserted);
                }
            }

            // Vector id actually inserted by this update
            let inserted_id = match insert_id {
                None => store.insert(&query).await,
                Some(id) => store.insert_at(id, &query).await?,
            };
            intra_batch_inserted.push(inserted_id.clone());
            slot_inserted_ids[idx] = Some(inserted_id.clone());

            let mut ops: Vec<MutationOp<VectorId>> = vec![MutationOp::AddNode {
                id: inserted_id.clone(),
                height: links.len(),
                update_ep,
            }];
            for (layer_idx, layer_links) in links.into_iter().enumerate() {
                ops.push(MutationOp::AddEdges {
                    base: inserted_id.clone(),
                    layer: layer_idx,
                    neighbors: layer_links,
                    edge_type: EdgeType::All,
                });
            }
            let unstamped = UnstampedMutation { ops };
            for pair in unstamped.updated_neighborhoods() {
                slot_updated.insert(pair);
            }
            for pair in unstamped.expanded_neighborhoods() {
                batch_expanded.insert(pair);
            }
            let mutation = graph.apply_new(unstamped)?;
            slot_outputs[idx].push(mutation);
        }

        // (c) Per-slot invalid-link prune over any neighborhood this slot's
        // mutations touched. Skipped for pure-delete slots (they don't touch
        // any neighborhood at all here).
        //
        // TODO: remove once neighborhood-versioning lands. With per-edge
        // sequence numbers, stale-edge filtering becomes implicit in
        // `insert_apply` and this step is unnecessary.
        if !slot_updated.is_empty() {
            let ops = searcher
                .prune_invalid_links(store, graph, &slot_updated)
                .await?;
            if !ops.is_empty() {
                let mutation = graph.apply_new(UnstampedMutation { ops })?;
                slot_outputs[idx].push(mutation);
            }
        }
    }

    // (d) Global compaction across the batch, attributed to the last
    // non-empty slot.
    if !batch_expanded.is_empty() {
        let ops = searcher
            .compact_batch(store, graph, &batch_expanded)
            .await?;
        if !ops.is_empty() {
            let mutation = graph.apply_new(UnstampedMutation { ops })?;
            let last_idx = slot_outputs
                .iter()
                .rposition(|v| !v.is_empty())
                .unwrap_or(0);
            slot_outputs[last_idx].push(mutation);
        }
    }

    Ok((slot_outputs, slot_inserted_ids))
}

/// Combine insert plans from parallel searches, repairing any conflict.
///
/// Currently just processes entry point update operations.
fn join_plans<V: VectorStore>(
    mut plans: Vec<Option<InsertPlanV<V>>>,
    layer_mode: &LayerMode,
) -> Vec<Option<InsertPlanV<V>>> {
    match layer_mode {
        LayerMode::Standard { .. } => {
            // Requests to set unique entry point must have strictly increasing layer
            let mut current_max_ep_layer: Option<usize> = None;
            for plan in plans.iter_mut() {
                let Some(plan) = plan else { continue };

                if let UpdateEntryPoint::SetUnique { layer } = plan.update_ep {
                    let update_valid = current_max_ep_layer
                        .map(|max_ep_layer| max_ep_layer < layer)
                        .unwrap_or(true);
                    if update_valid {
                        current_max_ep_layer = Some(layer);
                    } else {
                        plan.update_ep = UpdateEntryPoint::False;
                    }
                }
            }
        }
        LayerMode::LinearScan { .. } => {}
    }

    plans
}

/// Verify that entry point updates are sane for different searcher `LayerMode` variants.
fn validate_ep_updates<V: VectorStore>(
    plans: &Vec<Option<InsertPlanV<V>>>,
    layer_mode: &LayerMode,
) -> Result<()> {
    // For standard mode, check that entry point updates have strictly
    // increasing layers and no "append" updates
    if let LayerMode::Standard { .. } = layer_mode {
        let mut current_max_ep_layer: Option<usize> = None;
        for plan in plans {
            let Some(plan) = plan else { continue };

            match plan.update_ep {
                UpdateEntryPoint::SetUnique { layer } => {
                    if current_max_ep_layer
                        .map(|max_ep_layer| max_ep_layer < layer)
                        .unwrap_or(true)
                    {
                        current_max_ep_layer = Some(layer)
                    } else {
                        bail!("InsertPlan sets entry point at or lower than the current maximum layer");
                    }
                }
                UpdateEntryPoint::Append { .. } => {
                    bail!("Append entry point update encountered during Standard or Bounded layer mode");
                }
                UpdateEntryPoint::False => {}
            }
        }
    }

    // For standard mode with a layer bound specified, check that all updates
    // are at or below the layer bound
    if let LayerMode::Standard {
        max_graph_layer: Some(max_layer),
    } = layer_mode
    {
        for plan in plans {
            let Some(plan) = plan else { continue };

            if let UpdateEntryPoint::SetUnique { layer } = plan.update_ep {
                if layer > *max_layer {
                    bail!(
                        "InsertPlan sets entry point higher than layer bound in Bounded layer mode"
                    );
                }
            }
        }
    }

    // For linear scan mode, check that all updates are "append" updates at the layer bound
    if let LayerMode::LinearScan { max_graph_layer } = layer_mode {
        for plan in plans {
            let Some(plan) = plan else { continue };

            match plan.update_ep {
                UpdateEntryPoint::SetUnique { .. } => {
                    bail!("SetUnique entry point update encountered during LinearScan layer mode");
                }
                UpdateEntryPoint::Append { layer } => {
                    if layer != *max_graph_layer {
                        bail!("InsertPlan adds entry point at different layer than max graph layer during LinearScan layer mode")
                    }
                }
                UpdateEntryPoint::False => {}
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::hawkers::plaintext_store::PlaintextStore;
    use crate::hnsw::graph::GraphMutation;
    use iris_mpc_common::iris_db::iris::IrisCode;
    use itertools::Itertools;
    use std::sync::Arc;

    use super::*;

    fn dummy_insert_plan(ep_update: UpdateEntryPoint) -> InsertPlanV<PlaintextStore> {
        let ins_layer = if let UpdateEntryPoint::SetUnique { layer }
        | UpdateEntryPoint::Append { layer } = ep_update
        {
            layer
        } else {
            0
        };

        InsertPlanV {
            query: Arc::new(IrisCode::default()),
            links: vec![Vec::new(); ins_layer],
            update_ep: ep_update,
        }
    }

    /// Like `dummy_insert_plan` but with caller-provided per-layer links.
    /// Lets tests construct insertions that produce real `AddEdges` ops.
    fn dummy_insert_plan_with_links(
        ep_update: UpdateEntryPoint,
        links: Vec<Vec<VectorId>>,
    ) -> InsertPlanV<PlaintextStore> {
        InsertPlanV {
            query: Arc::new(IrisCode::default()),
            links,
            update_ep: ep_update,
        }
    }

    /// Helper function to test join_plans with multiple scenarios
    fn test_join_plans_helper(
        test_cases: &[UpdateEntryPoint],
        expected_results: &[UpdateEntryPoint],
        layer_mode: &LayerMode,
    ) {
        let mut plans = test_cases
            .iter()
            .cloned()
            .map(dummy_insert_plan)
            .map(Some)
            .collect_vec();
        plans.push(None); // Add a None plan as in the original tests
        let result = join_plans(plans, layer_mode);
        assert_eq!(result.len(), expected_results.len() + 1);
        for (idx, expected_update_ep) in expected_results.iter().enumerate() {
            assert_eq!(result[idx].as_ref().unwrap().update_ep, *expected_update_ep);
        }
    }

    // Test standard operation mode
    #[test]
    fn test_join_plans_standard() {
        // entry points in same layer
        test_join_plans_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::SetUnique { layer: 2 },
            ],
            &[
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::False,
            ],
            &LayerMode::Standard {
                max_graph_layer: None,
            },
        );

        // increasing layer order
        test_join_plans_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 2 },
            ],
            &[
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 2 },
            ],
            &LayerMode::Standard {
                max_graph_layer: None,
            },
        );

        // decreasing layer order
        test_join_plans_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::SetUnique { layer: 1 },
            ],
            &[
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::False,
            ],
            &LayerMode::Standard {
                max_graph_layer: None,
            },
        );

        // exercise more complex case
        test_join_plans_helper(
            &[
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::SetUnique { layer: 2 },
            ],
            &[
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
            ],
            &LayerMode::Standard {
                max_graph_layer: None,
            },
        );
    }

    /// Test bounded operation mode
    #[test]
    fn test_join_plans_bounded() {
        // `join_plans` does not modify entry point updates based on bounded `max_graph_layer`
        test_join_plans_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::SetUnique { layer: 2 },
            ],
            &[
                UpdateEntryPoint::SetUnique { layer: 2 },
                UpdateEntryPoint::False,
            ],
            &LayerMode::Standard {
                max_graph_layer: Some(1),
            },
        );
    }

    /// Test linear scan operation mode
    #[test]
    fn test_join_plans_linear_scan() {
        // `join_plans` does not modify append operations based on bounded `max_graph_layer`
        test_join_plans_helper(
            &[
                UpdateEntryPoint::Append { layer: 1 },
                UpdateEntryPoint::Append { layer: 1 },
            ],
            &[
                UpdateEntryPoint::Append { layer: 1 },
                UpdateEntryPoint::Append { layer: 1 },
            ],
            &LayerMode::LinearScan { max_graph_layer: 1 },
        );

        // `join_plans` does not modify append operations based on bounded `max_graph_layer`
        test_join_plans_helper(
            &[
                UpdateEntryPoint::Append { layer: 2 },
                UpdateEntryPoint::Append { layer: 2 },
            ],
            &[
                UpdateEntryPoint::Append { layer: 2 },
                UpdateEntryPoint::Append { layer: 2 },
            ],
            &LayerMode::LinearScan { max_graph_layer: 1 },
        );

        // `join_plans` does not modify append operations in case of different update layers
        test_join_plans_helper(
            &[
                UpdateEntryPoint::Append { layer: 0 },
                UpdateEntryPoint::Append { layer: 1 },
            ],
            &[
                UpdateEntryPoint::Append { layer: 0 },
                UpdateEntryPoint::Append { layer: 1 },
            ],
            &LayerMode::LinearScan { max_graph_layer: 1 },
        );
    }

    fn test_validate_ep_updates_helper(
        test_cases: &[UpdateEntryPoint],
        expect_ok: bool,
        layer_mode: &LayerMode,
    ) {
        let mut plans = test_cases
            .iter()
            .cloned()
            .map(dummy_insert_plan)
            .map(Some)
            .collect_vec();
        plans.push(None);
        let res = validate_ep_updates(&plans, layer_mode);
        match res {
            Ok(_) => {
                if !expect_ok {
                    panic!("Expected entry point validation to fail, but succeeded instead");
                }
            }
            Err(e) => {
                if expect_ok {
                    panic!(
                        "{}",
                        format!(
                            "Expected entry point validation to succeeed, but failed instead: {}",
                            e
                        )
                    );
                }
            }
        }
    }

    /// Test ep validator Standard layer mode validity checks
    #[test]
    fn test_ep_updates_validator_standard() {
        let standard_layer_mode = LayerMode::Standard {
            max_graph_layer: None,
        };

        // Standard mode layers are strictly increasing
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 1 },
            ],
            false,
            &standard_layer_mode,
        );
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 0 },
            ],
            false,
            &standard_layer_mode,
        );

        // Standard mode doesn't allow Append updates
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::Append { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 1 },
            ],
            false,
            &standard_layer_mode,
        );

        // The following is valid for Standard mode
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 3 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 4 },
                UpdateEntryPoint::SetUnique { layer: 5 },
                UpdateEntryPoint::False,
            ],
            true,
            &standard_layer_mode,
        );
    }

    /// Test ep validator Bounded layer mode validity checks
    #[test]
    fn test_ep_updates_validator_bounded() {
        let bounded_layer_mode = LayerMode::Standard {
            max_graph_layer: Some(3),
        };

        // Bounded mode layers are strictly increasing
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 1 },
            ],
            false,
            &bounded_layer_mode,
        );
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 0 },
            ],
            false,
            &bounded_layer_mode,
        );

        // Bounded mode doesn't allow Append updates
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::Append { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 1 },
            ],
            false,
            &bounded_layer_mode,
        );

        // Bounded mode must set entry points at or below layer bound
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::SetUnique { layer: 3 },
                UpdateEntryPoint::SetUnique { layer: 4 },
            ],
            false,
            &bounded_layer_mode,
        );

        // The following is valid for Bounded mode
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 0 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 1 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
                UpdateEntryPoint::SetUnique { layer: 3 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
            ],
            true,
            &bounded_layer_mode,
        );
    }

    /// Test ep validator LinearScan layer mode validity checks
    #[test]
    fn test_ep_updates_validator_linear_scan() {
        let linear_scan_layer_mode = LayerMode::LinearScan { max_graph_layer: 3 };

        // LinearScan mode cannot have SetUnique updates
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::SetUnique { layer: 3 },
            ],
            false,
            &linear_scan_layer_mode,
        );

        // LinearScan mode cannot append entry points at layers besides the max graph layer
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 4 },
            ],
            false,
            &linear_scan_layer_mode,
        );
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 2 },
            ],
            false,
            &linear_scan_layer_mode,
        );

        // The following is valid for LinearScan mode
        test_validate_ep_updates_helper(
            &[
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::Append { layer: 3 },
                UpdateEntryPoint::False,
                UpdateEntryPoint::False,
            ],
            true,
            &linear_scan_layer_mode,
        );
    }

    /// Pure deletion is encoded as plans[i] = None, replace_ids[i] = Some(id).
    /// The returned grouped_mutations should preserve slot order, with the deletion
    /// slot emitting exactly one RemoveNode for the requested id.
    #[tokio::test]
    async fn test_insert_with_pure_deletion_preserves_slot_order() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<VectorId> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        // Seed the store/graph with two existing vectors A and B so we have something
        // to delete.
        let a = store.insert(&Arc::new(IrisCode::default())).await;
        let b = store.insert(&Arc::new(IrisCode::default())).await;
        // Note: nodes A and B are deliberately not connected by edges — this test
        // only exercises that the pipeline emits the right mutations in the right
        // slots, not the bilateral-edge logic which is tested elsewhere.

        // Batch: [insert C, delete A, delete B]
        let plans = vec![
            Some(dummy_insert_plan(UpdateEntryPoint::SetUnique { layer: 0 })),
            None,
            None,
        ];
        let insert_ids: VecRequests<Option<VectorId>> = vec![None, None, None];
        let replace_ids: VecRequests<Option<VectorId>> = vec![None, Some(a), Some(b)];

        let (grouped, inserted_ids) = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        assert_eq!(grouped.len(), 3, "one output per slot");
        assert_eq!(inserted_ids.len(), 3, "inserted_ids aligned with slots");
        assert!(inserted_ids[0].is_some(), "slot 0 inserted a new vector C");
        assert!(inserted_ids[1].is_none(), "slot 1 is a pure deletion");
        assert!(inserted_ids[2].is_none(), "slot 2 is a pure deletion");

        // Slot 0 contains an AddNode (the insert of C) somewhere in its Vec.
        let slot0 = &grouped[0];
        assert!(
            !slot0.is_empty(),
            "slot 0 should have at least one mutation"
        );
        assert!(
            slot0.iter().any(|g| g
                .ops
                .iter()
                .any(|m| matches!(m, MutationOp::AddNode { .. }))),
            "slot 0 should contain AddNode"
        );

        // Slot 1 is a pure delete: one GraphMutation with one RemoveNode(a).
        let slot1 = &grouped[1];
        assert_eq!(slot1.len(), 1, "pure-delete slot has one GraphMutation");
        assert_eq!(slot1[0].ops.len(), 1, "and one op");
        match &slot1[0].ops[0] {
            MutationOp::RemoveNode { id } => assert_eq!(*id, a),
            other => panic!("expected RemoveNode(a) in slot 1, got {:?}", other),
        }

        // Slot 2 is a pure delete: one GraphMutation with one RemoveNode(b).
        let slot2 = &grouped[2];
        assert_eq!(slot2.len(), 1, "pure-delete slot has one GraphMutation");
        assert_eq!(slot2[0].ops.len(), 1, "and one op");
        match &slot2[0].ops[0] {
            MutationOp::RemoveNode { id } => assert_eq!(*id, b),
            other => panic!("expected RemoveNode(b) in slot 2, got {:?}", other),
        }
    }

    /// Reauth-style replacement is encoded as both plans[i] = Some(plan) AND
    /// replace_ids[i] = Some(old_id). The slot should contain a RemoveNode
    /// GraphMutation with a lower seq_no than the subsequent AddNode
    /// GraphMutation (delete-then-add ordering), each in its own group.
    #[tokio::test]
    async fn test_insert_with_combined_replace_emits_removenode_then_addnode() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<VectorId> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        let old = store.insert(&Arc::new(IrisCode::default())).await;

        let plans = vec![Some(dummy_insert_plan(UpdateEntryPoint::SetUnique {
            layer: 0,
        }))];
        let insert_ids: VecRequests<Option<VectorId>> = vec![None];
        let replace_ids: VecRequests<Option<VectorId>> = vec![Some(old)];

        let (grouped, inserted_ids) = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        assert_eq!(inserted_ids.len(), 1);
        let inserted = inserted_ids[0].expect("combined-replace slot inserts a new vector");
        assert_ne!(
            inserted, old,
            "inserted id is the new vector, not the replaced one"
        );

        let slot0 = &grouped[0];
        // Find the GraphMutation holding RemoveNode(old) and the one holding the AddNode.
        let remove_pos = slot0
            .iter()
            .position(|g| {
                g.ops
                    .iter()
                    .any(|m| matches!(m, MutationOp::RemoveNode { id } if *id == old))
            })
            .expect("slot should contain a GraphMutation with RemoveNode(old)");
        let add_pos = slot0
            .iter()
            .position(|g| {
                g.ops
                    .iter()
                    .any(|m| matches!(m, MutationOp::AddNode { .. }))
            })
            .expect("slot should contain a GraphMutation with AddNode");
        assert!(
            remove_pos < add_pos,
            "delete-then-add ordering: RemoveNode group precedes AddNode group within the slot"
        );

        let remove_seq = slot0[remove_pos].seq_no;
        let add_seq = slot0[add_pos].seq_no;
        assert!(
            remove_seq < add_seq,
            "delete group's seq_no ({remove_seq}) must be strictly less than add group's ({add_seq})"
        );
    }

    /// A None slot in both plans and replace_ids yields an empty Vec
    /// (no mutations) for that slot.
    #[tokio::test]
    async fn test_insert_with_none_slot_yields_empty_vec() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<VectorId> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        let plans: VecRequests<Option<InsertPlanV<PlaintextStore>>> = vec![None];
        let insert_ids: VecRequests<Option<VectorId>> = vec![None];
        let replace_ids: VecRequests<Option<VectorId>> = vec![None];

        let (grouped, inserted_ids) = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        assert!(
            grouped[0].is_empty(),
            "fully-empty slot should yield empty Vec"
        );
        assert!(
            inserted_ids[0].is_none(),
            "fully-empty slot does not insert anything"
        );
    }

    #[tokio::test]
    async fn test_insert_stamps_strictly_increasing_seq_nos_per_slot() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<VectorId> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        let expected_start = graph.next_sequence_number();

        let plans = vec![
            Some(dummy_insert_plan(UpdateEntryPoint::SetUnique { layer: 0 })),
            None,
            Some(dummy_insert_plan(UpdateEntryPoint::False)),
        ];
        let insert_ids: VecRequests<Option<VectorId>> = vec![None, None, None];
        let replace_ids: VecRequests<Option<VectorId>> = vec![None, None, None];

        let (grouped, inserted_ids) = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        // Flatten all GraphMutation seq_nos across all slots in order.
        let seq_nos: Vec<u64> = grouped
            .iter()
            .flat_map(|v| v.iter().map(|g| g.seq_no))
            .collect();
        assert!(
            seq_nos.windows(2).all(|w| w[0] < w[1]),
            "seq_nos across the batch must be strictly increasing: {:?}",
            seq_nos
        );
        assert_eq!(
            *seq_nos.first().expect("at least one mutation expected"),
            expected_start,
            "first seq_no should match graph.next_sequence_number() at entry"
        );
        assert_eq!(
            graph.last_update_seq_no,
            *seq_nos.last().expect("at least one mutation expected"),
            "graph last_update_seq_no should match the last applied seq_no"
        );

        assert_eq!(inserted_ids.len(), 3, "inserted_ids aligned with slots");
        assert!(inserted_ids[0].is_some());
        assert!(inserted_ids[1].is_none());
        assert!(inserted_ids[2].is_some());
        // The inserted ids must match the AddNode ids in their slot's
        // GraphMutation(s).
        for (slot_plans, slot_id) in grouped.iter().zip(inserted_ids.iter()) {
            match slot_id {
                Some(id) => {
                    let add_id = slot_plans.iter().find_map(|g| {
                        g.ops.iter().find_map(|op| match op {
                            MutationOp::AddNode { id, .. } => Some(*id),
                            _ => None,
                        })
                    });
                    assert_eq!(add_id, Some(*id));
                }
                None => {
                    assert!(
                        slot_plans.iter().all(|g| !g
                            .ops
                            .iter()
                            .any(|op| matches!(op, MutationOp::AddNode { .. }))),
                        "no-insert slot should contain no AddNode op"
                    );
                }
            }
        }
    }

    /// When the batch grows neighborhoods past M_limit, the global compaction
    /// mutation is appended to the LAST non-empty slot's Vec, AFTER all
    /// per-slot mutations.
    #[tokio::test]
    async fn test_insert_appends_global_compaction_to_last_nonempty_slot() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<VectorId> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();
        let m_limit = searcher.params.get_M_limit(0);

        // Seed the graph with (m_limit + 1) nodes, each at layer 0, with
        // base-edge neighborhoods exactly at M_limit (so a single new edge
        // tips each over the threshold).
        let seed_count = m_limit + 1;
        let mut seed_ids = Vec::with_capacity(seed_count);
        for _ in 0..seed_count {
            seed_ids.push(store.insert(&Arc::new(IrisCode::default())).await);
        }

        // Apply a setup mutation that:
        //   - Registers each seed node at layer 0.
        //   - Gives each seed node a base-only neighborhood of the OTHER seed
        //     nodes (size m_limit, since there are m_limit + 1 seeds total).
        // EdgeType::Base ensures the back-edges aren't auto-created — keeps
        // each neighborhood exactly at m_limit.
        let mut setup_ops: Vec<MutationOp<VectorId>> = Vec::with_capacity(seed_count * 2);
        for (i, &id) in seed_ids.iter().enumerate() {
            setup_ops.push(MutationOp::AddNode {
                id,
                height: 1,
                update_ep: if i == 0 {
                    UpdateEntryPoint::SetUnique { layer: 0 }
                } else {
                    UpdateEntryPoint::False
                },
            });
        }
        for (i, &id) in seed_ids.iter().enumerate() {
            let neighbors: Vec<_> = seed_ids
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .collect();
            assert_eq!(neighbors.len(), m_limit);
            setup_ops.push(MutationOp::AddEdges {
                base: id,
                layer: 0,
                neighbors,
                edge_type: EdgeType::Base,
            });
        }
        let setup = GraphMutation {
            seq_no: graph.next_sequence_number(),
            ops: setup_ops,
        };
        graph.insert_apply(&setup).expect("setup insert_apply");

        // Now insert one new node whose layer-0 links target every seed node.
        // With EdgeType::All, each seed's neighborhood gains the new node,
        // pushing each from m_limit → m_limit + 1, which exceeds M_limit and
        // triggers compaction.
        let new_plan =
            dummy_insert_plan_with_links(UpdateEntryPoint::False, vec![seed_ids.clone()]);
        let plans = vec![Some(new_plan), None];
        let insert_ids: VecRequests<Option<VectorId>> = vec![None, None];
        let replace_ids: VecRequests<Option<VectorId>> = vec![None, None];

        let (grouped, _) = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        // Slot 1 (the trailing None slot) must be untouched.
        assert!(
            grouped[1].is_empty(),
            "trailing None slot must remain empty"
        );

        // Slot 0 must contain at least the insert mutation AND a compaction
        // mutation (the latter appended by the global compaction step).
        let slot0 = &grouped[0];
        assert!(
            slot0.len() >= 2,
            "slot 0 should contain the insert mutation plus the global compaction; got {} mutations",
            slot0.len()
        );

        // At least one mutation must contain RemoveEdges — i.e. compaction
        // actually fired (not just attribution to an empty op list).
        let has_remove_edges = slot0.iter().any(|g| {
            g.ops
                .iter()
                .any(|op| matches!(op, MutationOp::RemoveEdges { .. }))
        });
        assert!(
            has_remove_edges,
            "global compaction should have emitted at least one RemoveEdges op"
        );

        // The compaction mutation has the maximum seq_no in the batch (it
        // was minted last). Verify by checking that the LAST mutation in
        // slot 0 has the max seq_no across the whole grouped output.
        let max_seq_overall = grouped
            .iter()
            .flat_map(|v| v.iter().map(|g| g.seq_no))
            .max()
            .expect("expected at least one mutation");
        let last_seq_slot0 = slot0.last().expect("slot 0 should be non-empty").seq_no;
        assert_eq!(
            max_seq_overall, last_seq_slot0,
            "global compaction mutation should be the LAST entry in the last non-empty slot"
        );
    }
}
