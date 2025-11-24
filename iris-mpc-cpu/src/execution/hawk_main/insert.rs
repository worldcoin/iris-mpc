use crate::hnsw::{
    searcher::{ConnectPlanV, LayerMode, UpdateEntryPoint},
    vector_store::VectorStoreMut,
    GraphMem, HnswSearcher, VectorStore,
};

use super::VecRequests;

use eyre::{bail, Result};
use itertools::izip;

/// InsertPlan specifies where a query may be inserted into the HNSW graph.
///
/// The `links` field specifies the final desired links in each layer for the
/// newly inserted query, and should already be trimmed to the desired length,
/// e.g. typically the HNSW parameter M.
#[derive(Debug)]
pub struct InsertPlanV<V: VectorStore> {
    pub query: V::QueryRef,
    pub links: Vec<Vec<V::VectorRef>>,
    pub set_ep: UpdateEntryPoint,
}

// Manual implementation of Clone for InsertPlanV, since derive(Clone) does not
// propagate the nested Clone bounds on V::QueryRef via TransientRef.
impl<V: VectorStore> Clone for InsertPlanV<V> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            links: self.links.clone(),
            set_ep: self.set_ep.clone(),
        }
    }
}

/// Insert a collection `plans` of `InsertPlanV` structs into the graph and vector store,
/// adjusting the insertion plans as needed to repair any conflict from parallel searches.
///
/// The `ids` argument consists of `Option<VectorId>`s which are `Some(id)` if the associated
/// plan is to be inserted with a specific identifier (e.g. for updates or for insertions
/// which need to parallel an existing iris code database), and `None` if the associated plan
/// is to be inserted at the next available serial ID, with version 0.
pub async fn insert<V: VectorStoreMut>(
    store: &mut V,
    graph: &mut GraphMem<<V as VectorStore>::VectorRef>,
    searcher: &HnswSearcher,
    plans: VecRequests<Option<InsertPlanV<V>>>,
    ids: &VecRequests<Option<V::VectorRef>>,
) -> Result<VecRequests<Option<ConnectPlanV<V>>>> {
    tracing::debug!("Inserting {} InsertPlans into store", plans.len());

    let insert_plans = join_plans(plans, &searcher.layer_mode);
    validate_ep_updates(&insert_plans, &searcher.layer_mode)?;

    let mut connect_plans = vec![None; insert_plans.len()];
    let mut inserted_ids = vec![];
    let m = searcher.params.get_M(0);

    let mut update_idxs = vec![];
    let mut updates: Vec<(_, _, _)> = vec![];
    for (idx, (plan, update_id)) in izip!(insert_plans, ids).enumerate() {
        if let Some(InsertPlanV {
            query,
            mut links,
            set_ep,
        }) = plan
        {
            update_idxs.push(idx);

            // Extend links in bottom layer with items from batch, only when the
            // bottom layer is not large enough to build full neighborhoods,
            // i.e. when the graph does not yet have M elements.
            if let Some(bottom_layer) = links.first_mut() {
                if bottom_layer.len() < m {
                    bottom_layer.extend_from_slice(&inserted_ids);
                }
            }

            // Insert vector in store, getting new persistent vector id if none specified.
            let inserted = {
                match update_id {
                    None => store.insert(&query).await,
                    Some(id) => store.insert_at(id, &query).await?,
                }
            };

            updates.push((inserted.clone(), links, set_ep));
            inserted_ids.push(inserted);
        }
    }

    let plans = searcher.insert_prepare_batch(store, graph, updates).await?;
    for (cp_idx, plan) in izip!(update_idxs, plans) {
        graph.insert_apply(plan.clone()).await;
        connect_plans[cp_idx].replace(plan);
    }

    Ok(connect_plans)
}

/// Combine insert plans from parallel searches, repairing any conflict.
///
/// Currently just processes entry point update operations.
fn join_plans<V: VectorStore>(
    mut plans: Vec<Option<InsertPlanV<V>>>,
    layer_mode: &LayerMode,
) -> Vec<Option<InsertPlanV<V>>> {
    match layer_mode {
        LayerMode::Standard | LayerMode::Bounded { .. } => {
            // Requests to set unique entry point must have strictly increasing layer
            let mut current_max_ep_layer: Option<usize> = None;
            for plan in plans.iter_mut() {
                let Some(plan) = plan else { continue };

                if let UpdateEntryPoint::SetUnique { layer } = plan.set_ep {
                    let update_valid = current_max_ep_layer
                        .map(|max_ep_layer| max_ep_layer < layer)
                        .unwrap_or(true);
                    if update_valid {
                        current_max_ep_layer = Some(layer);
                    } else {
                        plan.set_ep = UpdateEntryPoint::False;
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
    // For standard and bounded modes, check that entry point updates have
    // strictly increasing layers and no "append" updates
    if let LayerMode::Standard | LayerMode::Bounded { .. } = layer_mode {
        let mut current_max_ep_layer: Option<usize> = None;
        for plan in plans {
            let Some(plan) = plan else { continue };

            match plan.set_ep {
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

    // For bounded mode, check that all updates are at or below the layer bound
    if let LayerMode::Bounded { max_graph_layer } = layer_mode {
        for plan in plans {
            let Some(plan) = plan else { continue };

            if let UpdateEntryPoint::SetUnique { layer } = plan.set_ep {
                if layer > *max_graph_layer {
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

            match plan.set_ep {
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
            set_ep: ep_update,
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
        for (idx, expected_set_ep) in expected_results.iter().enumerate() {
            assert_eq!(result[idx].as_ref().unwrap().set_ep, *expected_set_ep);
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
            &LayerMode::Standard,
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
            &LayerMode::Standard,
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
            &LayerMode::Standard,
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
            &LayerMode::Standard,
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
            &LayerMode::Bounded { max_graph_layer: 1 },
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
        let standard_layer_mode = LayerMode::Standard;

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
        let bounded_layer_mode = LayerMode::Bounded { max_graph_layer: 3 };

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
}
