use crate::hnsw::{
    graph::{mutation::EdgeType, GraphMutation, MutationOp, UpdateEntryPoint},
    searcher::{ConnectPlanV, LayerMode},
    vector_store::VectorStoreMut,
    GraphMem, HnswSearcher, VectorStore,
};

use super::VecRequests;

use eyre::{bail, Result};
use itertools::izip;

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
    pub links: Vec<Vec<V::VectorRef>>,
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
/// identity-update replacements, or for pure deletions). A pure-deletion slot has
/// `plans[i] = None` and `replace_ids[i] = Some(id)`. A slot with both `plans[i] = None` and
/// `replace_ids[i] = None` produces `None` in the output (no mutations for that slot).
pub async fn insert<V: VectorStoreMut>(
    store: &mut V,
    graph: &mut GraphMem<<V as VectorStore>::VectorRef>,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<V>>>,
    insert_ids: &VecRequests<Option<V::VectorRef>>,
    replace_ids: &VecRequests<Option<V::VectorRef>>,
) -> Result<VecRequests<Option<ConnectPlanV<V>>>> {
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

    let mut inserted_ids = vec![];
    let m = searcher.params.get_M(0);

    // Build one Option<GroupedMutations> per batch slot. None slots pass through
    // insert_prepare_batch unchanged; Some slots carry per-request mutations
    // (optional AddNode + AddEdges + optional RemoveNode, OR a pure RemoveNode
    // for deletion-only slots).
    let mut mutations: Vec<Option<GraphMutation<V::VectorRef>>> = vec![None; insert_plans.len()];

    for (idx, (plan, update_id, replace_id)) in
        izip!(insert_plans, insert_ids, replace_ids).enumerate()
    {
        let mut request_mutations: Vec<MutationOp<V::VectorRef>> = vec![];

        if let Some(InsertPlanV {
            query,
            mut links,
            update_ep,
        }) = plan
        {
            // Extend links in bottom layer with items from batch, only when the
            // bottom layer is not large enough to build full neighborhoods,
            // i.e. when the graph does not yet have M elements.
            if let Some(bottom_layer) = links.first_mut() {
                if bottom_layer.len() < m {
                    bottom_layer.extend_from_slice(&inserted_ids);
                }
            }

            // Insert vector in store, getting new persistent vector id if none specified.
            let inserted = match update_id {
                None => store.insert(&query).await,
                Some(id) => store.insert_at(id, &query).await?,
            };

            request_mutations.push(MutationOp::AddNode {
                id: inserted.clone(),
                height: links.len(),
                update_ep,
            });
            for (layer_idx, layer_links) in links.into_iter().enumerate() {
                request_mutations.push(MutationOp::AddEdges {
                    base: inserted.clone(),
                    layer: layer_idx,
                    neighbors: layer_links,
                    edge_type: EdgeType::All,
                });
            }

            inserted_ids.push(inserted);
        }

        if let Some(rid) = replace_id {
            request_mutations.push(MutationOp::RemoveNode { id: rid.clone() });
        }

        if !request_mutations.is_empty() {
            mutations[idx] = Some(GraphMutation {
                id: 0,
                ops: request_mutations,
            });
        }
    }

    let mut grouped_mutations = searcher
        .insert_prepare_batch(store, graph, mutations)
        .await?;

    // Temporary in-place id stamping; Task 4 will move this to mutation
    // construction time via MutationIdAllocator.
    for (next_id, slot) in
        (graph.next_modification_id()..).zip(grouped_mutations.iter_mut().flatten())
    {
        slot.id = next_id;
    }
    // Apply each finalized group; strict-increase ordering is enforced.
    for group in grouped_mutations.iter().flatten() {
        graph.insert_apply(group)?;
    }

    // grouped_mutations is shaped as Vec<Option<ConnectPlanV<V>>>, one entry per
    // batch slot — return it directly as the connect plans.
    Ok(grouped_mutations)
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
        let mut graph: GraphMem<<PlaintextStore as VectorStore>::VectorRef> = GraphMem::new();
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
        let insert_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None, None, None];
        let replace_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None, Some(a), Some(b)];

        let grouped = insert(
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

        // Slot 0 contains an AddNode (the insert of C).
        let slot0 = grouped[0].as_ref().expect("slot 0 should be Some");
        assert!(
            slot0
                .ops
                .iter()
                .any(|m| matches!(m, MutationOp::AddNode { .. })),
            "slot 0 should contain AddNode"
        );

        // Slot 1 contains exactly one mutation, RemoveNode(a).
        let slot1 = grouped[1].as_ref().expect("slot 1 should be Some");
        assert_eq!(slot1.ops.len(), 1, "deletion slot has one mutation");
        match &slot1.ops[0] {
            MutationOp::RemoveNode { id } => assert_eq!(*id, a),
            other => panic!("expected RemoveNode(a) in slot 1, got {:?}", other),
        }

        // Slot 2 contains exactly one mutation, RemoveNode(b).
        let slot2 = grouped[2].as_ref().expect("slot 2 should be Some");
        assert_eq!(slot2.ops.len(), 1, "deletion slot has one mutation");
        match &slot2.ops[0] {
            MutationOp::RemoveNode { id } => assert_eq!(*id, b),
            other => panic!("expected RemoveNode(b) in slot 2, got {:?}", other),
        }
    }

    /// Reauth-style replacement is encoded as both plans[i] = Some(plan) AND
    /// replace_ids[i] = Some(old_id). The slot's group should contain an AddNode
    /// for the new vector followed by a RemoveNode for the old one.
    #[tokio::test]
    async fn test_insert_with_combined_replace_emits_addnode_then_removenode() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<<PlaintextStore as VectorStore>::VectorRef> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        let old = store.insert(&Arc::new(IrisCode::default())).await;

        let plans = vec![Some(dummy_insert_plan(UpdateEntryPoint::SetUnique {
            layer: 0,
        }))];
        let insert_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None];
        let replace_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![Some(old)];

        let grouped = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        let slot0 = grouped[0].as_ref().expect("slot 0 should be Some");
        let mutations: &Vec<_> = &slot0.ops;

        let add_count = mutations
            .iter()
            .filter(|m| matches!(m, MutationOp::AddNode { .. }))
            .count();
        assert_eq!(add_count, 1, "slot should contain exactly one AddNode");

        let remove_old_count = mutations
            .iter()
            .filter(|m| matches!(m, MutationOp::RemoveNode { id } if *id == old))
            .count();
        assert_eq!(
            remove_old_count, 1,
            "slot should contain exactly one RemoveNode(old)"
        );

        let add_pos = mutations
            .iter()
            .position(|m| matches!(m, MutationOp::AddNode { .. }))
            .expect("must contain AddNode");
        let remove_pos = mutations
            .iter()
            .position(|m| matches!(m, MutationOp::RemoveNode { id } if *id == old))
            .expect("must contain RemoveNode(old)");
        assert!(
            add_pos < remove_pos,
            "AddNode should precede the matching RemoveNode in the slot's group"
        );
    }

    /// A None slot in both plans and replace_ids passes through as None.
    #[tokio::test]
    async fn test_insert_with_none_slot_yields_none() {
        let mut store = PlaintextStore::default();
        let mut graph: GraphMem<<PlaintextStore as VectorStore>::VectorRef> = GraphMem::new();
        let searcher = HnswSearcher::new_with_test_parameters();

        let plans: VecRequests<Option<InsertPlanV<PlaintextStore>>> = vec![None];
        let insert_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None];
        let replace_ids: VecRequests<Option<<PlaintextStore as VectorStore>::VectorRef>> =
            vec![None];

        let grouped = insert(
            &mut store,
            &mut graph,
            &searcher,
            plans,
            &insert_ids,
            &replace_ids,
        )
        .await
        .expect("insert should succeed");

        assert!(grouped[0].is_none(), "fully-empty slot should yield None");
    }
}
