use std::{collections::HashMap, sync::Arc};

use crate::{
    hawkers::{aby3::aby3_store::FhdOps, plaintext_store::PlaintextStore},
    hnsw::{
        graph::layered_graph::migrate, vector_store::VectorStoreMut, GraphMem, HnswSearcher,
        VectorStore,
    },
};
use aes_prng::AesRng;
use eyre::Result;
use iris_mpc_common::{iris_db::db::IrisDB, VectorId};

use rand::{RngCore, SeedableRng};

#[allow(dead_code)]
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct TestStore {
    points: HashMap<usize, Point>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Point {
    /// Whatever encoding of a vector.
    data: u64,
    /// Distinguish between queries that are pending, and those that were
    /// ultimately accepted into the vector store.
    is_persistent: bool,
}

fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

impl VectorStore for TestStore {
    type QueryRef = usize; // Vector ID, pending insertion.
    type DistanceRef = u32; // Eager distance representation as fraction.

    async fn vectors_as_queries(&mut self, vectors: Vec<VectorId>) -> Result<Vec<Self::QueryRef>> {
        Ok(vectors
            .into_iter()
            .map(|v| v.serial_id() as usize)
            .collect())
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &VectorId,
    ) -> Result<Self::DistanceRef> {
        // Hamming distance
        let vector_1_idx = vector.serial_id() as usize;
        let vector_0 = self.points[query].data;
        let vector_1 = self.points[&vector_1_idx].data;
        Ok(hamming_distance(vector_0, vector_1))
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(*distance == 0)
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        Ok(*distance1 < *distance2)
    }

    async fn only_valid_entry_points(
        &mut self,
        entry_points: Vec<(VectorId, usize)>,
    ) -> Vec<(VectorId, usize)> {
        entry_points
    }
}

impl VectorStoreMut for TestStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> VectorId {
        // The query is now accepted in the store. It keeps the same ID.
        self.points.get_mut(query).unwrap().is_persistent = true;
        VectorId::from_serial_id(*query as u32)
    }

    async fn insert_at(
        &mut self,
        _vector_ref: &VectorId,
        _query: &Self::QueryRef,
    ) -> Result<VectorId> {
        unimplemented!()
    }
}

#[tokio::test]
async fn test_from_another_naive() -> Result<()> {
    let mut vector_store = PlaintextStore::<FhdOps>::new();
    let mut graph_store = GraphMem::new();
    let searcher = HnswSearcher::new_with_test_parameters();
    let mut rng = AesRng::seed_from_u64(0_u64);

    let raw_queries = IrisDB::new_random_rng(10, &mut rng);

    for raw_query in raw_queries.db {
        let query = Arc::new(raw_query);
        let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
        let (neighbors, update_ep) = searcher
            .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
            .await?;
        let inserted = vector_store.insert(&query).await;
        searcher
            .insert_from_search_results(
                &mut vector_store,
                &mut graph_store,
                inserted,
                neighbors,
                update_ep,
            )
            .await?;
    }

    let different_graph_store: GraphMem = migrate(graph_store.clone(), |v| {
        VectorId::from_0_index(v.index() * 2)
    });
    assert_ne!(graph_store, different_graph_store);

    Ok(())
}

#[test]
fn test_layer_deterministic_serialize_order() {
    let mut layer_a = super::Layer::new();
    let mut layer_b = super::Layer::new();

    let v1 = VectorId::from_serial_id(1);
    let v2 = VectorId::from_serial_id(2);
    let v3 = VectorId::from_serial_id(3);
    let v4 = VectorId::from_serial_id(4);
    let v5 = VectorId::from_serial_id(5);

    layer_a.set_links(v1, vec![v2, v3]);
    layer_a.set_links(v4, vec![v5]);

    layer_b.set_links(v4, vec![v5]);
    layer_b.set_links(v1, vec![v2, v3]);

    let bytes_a = bincode::serialize(&layer_a).expect("layer_a serialize");
    let bytes_b = bincode::serialize(&layer_b).expect("layer_b serialize");

    assert_eq!(bytes_a, bytes_b);
}

#[tokio::test]
async fn test_from_another() -> Result<()> {
    let mut vector_store = PlaintextStore::<FhdOps>::new();
    let mut graph_store = GraphMem::new();
    let mut searcher = HnswSearcher::new_with_test_parameters();
    // Bump layer density so enough nodes roll onto the entry-point layer
    // (max_graph_layer + 1) for the entry-point migration checks below.
    searcher.layer_distribution = crate::hnsw::searcher::LayerDistribution::new_geometric_from_M(2);
    let mut rng = AesRng::seed_from_u64(0_u64);

    let mut point_ids_map: HashMap<VectorId, VectorId> = HashMap::new();

    for raw_query in IrisDB::new_random_rng(20, &mut rng).db {
        let query = Arc::new(raw_query);
        let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
        let (neighbors, update_ep) = searcher
            .search_to_insert(&mut vector_store, &graph_store, &query, insertion_layer)
            .await?;
        let inserted = vector_store.insert(&query).await;
        searcher
            .insert_from_search_results(
                &mut vector_store,
                &mut graph_store,
                inserted,
                neighbors,
                update_ep,
            )
            .await?;

        point_ids_map.insert(inserted, VectorId::from_serial_id(rng.next_u32()));
    }

    let new_graph_store: GraphMem = migrate(graph_store.clone(), |v| point_ids_map[&v]);

    let (entry_point, layer) = graph_store.get_first_entry_point().await.unwrap();
    let (new_entry_point, new_layer) = new_graph_store.get_first_entry_point().await.unwrap();

    // Check that entry points are correct
    assert_eq!(layer, new_layer);
    assert_eq!(point_ids_map[&entry_point], new_entry_point);

    let layers = graph_store.get_layers();
    let new_layers = new_graph_store.get_layers();

    for (layer, new_layer) in layers.iter().zip(new_layers.iter()) {
        let links = layer.get_links_map();
        let new_links = new_layer.get_links_map();

        #[allow(
            clippy::iter_over_hash_type,
            reason = "Iteration is for assertions against a parallel data structure, compared entry by entry."
        )]
        for (point_id, queue) in links.iter() {
            let new_point_id = point_ids_map[point_id];
            let new_queue_vec = new_links[&new_point_id].to_vec();
            for (neighbor_id, new_neighbor_id) in queue.iter().zip(new_queue_vec) {
                assert_eq!(point_ids_map[neighbor_id], new_neighbor_id);
            }
        }
    }

    Ok(())
}

use crate::hnsw::graph::mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint};

#[test]
fn add_edges_outgoing_writes_only_to_id_list() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let c = VectorId::from_serial_id(3);
    // Seed: a, b, c all exist at layer 0 with no edges.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            ops: vec![
                MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddNode {
                    id: c,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
            ],
        })
        .unwrap();
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            ops: vec![MutationOp::AddEdges {
                base: a,
                layer: 0,
                neighbors: vec![b, c],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[b, c]);
    assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[] as &[VectorId]);
    assert_eq!(graph.layers[0].get_links(&c).unwrap(), &[] as &[VectorId]);
}

#[test]
fn add_edges_incoming_writes_only_to_target_lists() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let c = VectorId::from_serial_id(3);
    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            ops: vec![
                MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddNode {
                    id: c,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
            ],
        })
        .unwrap();
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            ops: vec![MutationOp::AddEdges {
                base: a,
                layer: 0,
                neighbors: vec![b, c],
                edge_type: EdgeType::Neighbors,
            }],
        })
        .unwrap();
    assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[] as &[VectorId]);
    assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[a]);
    assert_eq!(graph.layers[0].get_links(&c).unwrap(), &[a]);
}

#[test]
fn add_edges_bidirectional_writes_both_sides() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let c = VectorId::from_serial_id(3);
    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            ops: vec![
                MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddNode {
                    id: c,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
            ],
        })
        .unwrap();
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            ops: vec![MutationOp::AddEdges {
                base: a,
                layer: 0,
                neighbors: vec![b, c],
                edge_type: EdgeType::All,
            }],
        })
        .unwrap();
    assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[b, c]);
    assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[a]);
    assert_eq!(graph.layers[0].get_links(&c).unwrap(), &[a]);
}

#[test]
fn remove_edges_outgoing_only_modifies_id_list() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let c = VectorId::from_serial_id(3);
    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            ops: vec![
                MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b, c],
                    edge_type: EdgeType::Base,
                },
                MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddEdges {
                    base: b,
                    layer: 0,
                    neighbors: vec![a],
                    edge_type: EdgeType::Base,
                },
                MutationOp::AddNode {
                    id: c,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddEdges {
                    base: c,
                    layer: 0,
                    neighbors: vec![a],
                    edge_type: EdgeType::Base,
                },
            ],
        })
        .unwrap();
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            ops: vec![MutationOp::RemoveEdges {
                base: a,
                layer: 0,
                neighbors: vec![b],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[c]);
    // Bidirectional cleanup is not implied — b's list still contains a.
    assert_eq!(graph.layers[0].get_links(&b).unwrap(), &[a]);
}

#[test]
fn two_phase_apply_edges_before_node_in_vec_still_works() {
    // Pass 1 should apply AddNode before pass 2 applies AddEdges, regardless
    // of their order in the input Vec.
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            ops: vec![
                // Listed first: an edge op that references a node not yet created.
                MutationOp::AddEdges {
                    base: a,
                    layer: 0,
                    neighbors: vec![b],
                    edge_type: EdgeType::Base,
                },
                // Listed second: the node creation.
                MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 0 },
                },
                MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
            ],
        })
        .unwrap();
    // Pass-1 created the nodes, then pass-2 applied the edge — so a should
    // now have b in its outgoing list.
    assert_eq!(graph.layers[0].get_links(&a).unwrap(), &[b]);
}

#[test]
fn next_seq_no_is_one_past_last_and_does_not_mutate() {
    use crate::hnsw::GraphMem;
    let mut graph = GraphMem::new();
    assert_eq!(graph.last_update_seq_no, 0);
    assert_eq!(graph.next_sequence_number(), 1);
    assert_eq!(graph.next_sequence_number(), 1, "peek must not mutate");
    graph.last_update_seq_no = 42;
    assert_eq!(graph.next_sequence_number(), 43);
    assert_eq!(graph.last_update_seq_no, 42, "peek must not mutate");
}

#[test]
fn insert_apply_advances_last_update_seq_no_on_success() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let mutation = GraphMutation {
        seq_no: 1,
        ops: vec![MutationOp::AddNode {
            id: a,
            height: 1,
            update_ep: UpdateEntryPoint::Append { layer: 0 },
        }],
    };
    graph
        .insert_apply(&mutation)
        .expect("strict-increase should hold");
    assert_eq!(graph.last_update_seq_no, 1);
}

#[test]
fn insert_apply_rejects_seq_no_equal_to_last_update_seq_no() {
    let mut graph = GraphMem::new();
    graph.last_update_seq_no = 5;
    let mutation = GraphMutation {
        seq_no: 5,
        ops: vec![MutationOp::AddNode {
            id: VectorId::from_serial_id(1),
            height: 1,
            update_ep: UpdateEntryPoint::Append { layer: 0 },
        }],
    };
    let res = graph.insert_apply(&mutation);
    assert!(res.is_err(), "equal seq_no must be rejected");
    assert_eq!(
        graph.last_update_seq_no, 5,
        "state must be unchanged on Err"
    );
    assert_eq!(graph.layers.len(), 0, "no ops should have been applied");
}

#[test]
fn insert_apply_rejects_seq_no_below_last_update_seq_no() {
    let mut graph = GraphMem::new();
    graph.last_update_seq_no = 10;
    let mutation = GraphMutation {
        seq_no: 9,
        ops: vec![MutationOp::AddNode {
            id: VectorId::from_serial_id(1),
            height: 1,
            update_ep: UpdateEntryPoint::Append { layer: 0 },
        }],
    };
    let res = graph.insert_apply(&mutation);
    assert!(res.is_err());
    assert_eq!(graph.last_update_seq_no, 10);
}

#[test]
fn insert_apply_all_short_circuits_on_first_violation() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let mutations = vec![
        GraphMutation {
            seq_no: 1,
            ops: vec![MutationOp::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        },
        // Equal seq_no — should fail.
        GraphMutation {
            seq_no: 1,
            ops: vec![MutationOp::AddNode {
                id: b,
                height: 1,
                update_ep: UpdateEntryPoint::False,
            }],
        },
    ];
    let res = graph.insert_apply_all(&mutations);
    assert!(res.is_err(), "second mutation must be rejected");
    assert_eq!(
        graph.last_update_seq_no, 1,
        "first applied; last_update_seq_no at 1"
    );
    // First mutation's AddNode took effect, second did not.
    assert!(graph.layers[0].get_links(&a).is_some());
    assert!(graph.layers[0].get_links(&b).is_none());
}
