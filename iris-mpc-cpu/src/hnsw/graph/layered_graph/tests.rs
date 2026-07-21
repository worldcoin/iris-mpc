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
use iris_mpc_common::{iris_db::db::IrisDB, SerialId, VectorId};

use rand::{RngCore, SeedableRng};

/// The consensus checksum must change when either clock changes, even with
/// identical neighbor sets and entry points — otherwise parties that skip
/// different edges via `is_active` could agree on the checksum.
#[test]
fn checksum_folds_content_and_neighborhood_clocks() {
    use super::{EntryPoint, Layer, NodeInit};
    // node1 neighborhood seq = `nbhd_seq`; node1 content clock = `content`.
    let build = |nbhd_seq: u64, content: u64, version: i16| -> GraphMem {
        let mut layer = Layer::new();
        layer.set_links_trusted(1u32, vec![2u32], nbhd_seq);
        layer.set_links_trusted(2u32, vec![], 0);
        let node_init = HashMap::from([
            (
                1u32,
                NodeInit {
                    seq_no: content,
                    version,
                },
            ),
            (
                2u32,
                NodeInit {
                    seq_no: 0,
                    version: 0,
                },
            ),
        ]);
        GraphMem::from_parts(
            vec![EntryPoint { point: 1, layer: 0 }],
            vec![layer],
            0,
            node_init,
        )
    };

    let base = build(0, 0, 0);
    assert_eq!(base.checksum(), build(0, 0, 0).checksum(), "deterministic");
    assert_ne!(
        base.checksum(),
        build(5, 0, 0).checksum(),
        "per-neighborhood seq_no must be folded into the checksum"
    );
    assert_ne!(
        base.checksum(),
        build(0, 7, 0).checksum(),
        "node_init (content clock) must be folded into the checksum"
    );
    assert_ne!(
        base.checksum(),
        build(0, 0, 3).checksum(),
        "node version must be folded into the checksum"
    );
}

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

    async fn vectors_as_queries(
        &mut self,
        vectors: Vec<VectorId>,
    ) -> Result<Vec<Self::QueryRef>> {
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

#[test]
fn test_layer_deterministic_serialize_order() {
    let mut layer_a = super::Layer::new();
    let mut layer_b = super::Layer::new();

    layer_a.set_links_trusted(1, vec![2, 3], 0);
    layer_a.set_links_trusted(4, vec![5], 0);

    layer_b.set_links_trusted(4, vec![5], 0);
    layer_b.set_links_trusted(1, vec![2, 3], 0);

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
    searcher.layer_distribution =
        crate::hnsw::searcher::LayerDistribution::new_geometric_from_M(2);
    let mut rng = AesRng::seed_from_u64(0_u64);

    let mut point_ids_map: HashMap<SerialId, SerialId> = HashMap::new();

    for raw_query in IrisDB::new_random_rng(20, &mut rng).db {
        let query = Arc::new(raw_query);
        let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
        let (neighbors, update_ep, as_of) = searcher
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
                as_of,
            )
            .await?;

        point_ids_map.insert(inserted.serial_id(), rng.next_u32());
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
        for (point_id, nbhd) in links.iter() {
            let new_point_id = point_ids_map[point_id];
            let new_nbhd = &new_links[&new_point_id];
            // Neighbor lists are stored serial-sorted, and the id remap does
            // not preserve that order, so compare the remapped sets.
            let expected: std::collections::HashSet<SerialId> =
                nbhd.neighbors.iter().map(|n| point_ids_map[n]).collect();
            let got: std::collections::HashSet<SerialId> =
                new_nbhd.neighbors.iter().copied().collect();
            assert_eq!(expected, got, "neighbors of {point_id} -> {new_point_id}");
        }
    }

    Ok(())
}

use crate::hnsw::graph::mutation::{
    EdgeType, GraphMutation, MutationOp, UnstampedMutation, UpdateEntryPoint,
};

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
            as_of: 0,
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
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2, 3],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors,
        vec![2u32, 3u32]
    );
    assert_eq!(
        graph.layers[0].get_links(&2).unwrap().neighbors,
        vec![] as Vec<SerialId>
    );
    assert_eq!(
        graph.layers[0].get_links(&3).unwrap().neighbors,
        vec![] as Vec<SerialId>
    );
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
            as_of: 0,
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
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2, 3],
                edge_type: EdgeType::Neighbors,
            }],
        })
        .unwrap();
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors,
        vec![] as Vec<SerialId>
    );
    assert_eq!(graph.layers[0].get_links(&2).unwrap().neighbors, vec![1u32]);
    assert_eq!(graph.layers[0].get_links(&3).unwrap().neighbors, vec![1u32]);
}

/// Filter-on-bump: an asymmetric stale edge `a -> b` (no `b -> a`
/// back-edge, as arises after compaction) is dropped the next time `a`'s
/// neighborhood is touched once `b`'s content clock has advanced via
/// reauth.
#[test]
fn stale_edge_dropped_on_neighborhood_touch_after_reauth() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let d = VectorId::from_serial_id(3);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    // seq 1: insert a and b.
    graph
        .apply_new(UnstampedMutation {
            as_of: graph.last_update_seq_no,
            ops: vec![node(a), node(b)],
        })
        .unwrap();
    // seq 2: asymmetric forward edge a -> b only; b's list stays empty.
    graph
        .apply_new(UnstampedMutation {
            as_of: graph.last_update_seq_no,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
        vec![2u32]
    );

    // seq 3: reauth b. RemoveNode(b) does no backlink cleanup, so a -> b
    // survives — then AddNode(b) advances content[b] to 3.
    graph
        .apply_new(UnstampedMutation {
            as_of: graph.last_update_seq_no,
            ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
        })
        .unwrap();
    // a's neighborhood (last touched at seq 2) hasn't been revisited, so the
    // stale edge is still present — detection happens on next touch.
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
        vec![2u32]
    );

    // seq 4/5: touch a's neighborhood with a fresh edge a -> d. The filter
    // runs against a's prior seq (2) and drops a -> b (content[b] = 3 > 2)
    // before appending d.
    graph
        .apply_new(UnstampedMutation {
            as_of: graph.last_update_seq_no,
            ops: vec![node(d)],
        })
        .unwrap();
    let touch = MutationOp::AddEdges {
        base: 1,
        layer: 0,
        neighbors: vec![3],
        edge_type: EdgeType::Base,
    };
    let minted = graph
        .apply_new(UnstampedMutation {
            as_of: graph.last_update_seq_no,
            ops: vec![touch.clone()],
        })
        .unwrap();
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
        vec![3u32]
    );

    // The drop is not reflected in the op list; it re-derives on replay.
    assert_eq!(minted.ops, vec![touch]);
}

/// Resolution: an `AddEdges` record identified before its target's
/// reauth must not re-point at — and certify — the target's new content.
/// The record carries the intent verbatim; the reference is void at every
/// apply, so replay reproduces the same graph.
#[test]
fn resolution_drops_edge_ref_to_target_reauthed_after_identification() {
    let mut graph = GraphMem::new();
    let mut wal: Vec<GraphMutation> = Vec::new();
    let x0 = VectorId::new(1, 0);
    let x1 = VectorId::new(1, 1);
    let y = VectorId::from_serial_id(2);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    // seq 1: X enters the graph.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![node(x0)],
            })
            .unwrap(),
    );
    // Y's insert search identified X here.
    let as_of = graph.last_update_seq_no;
    // seq 2: X reauths — its content is no longer what the search saw.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![MutationOp::RemoveNode { id: x1 }, node(x1)],
            })
            .unwrap(),
    );
    // seq 3: Y's insert lands with the pre-reauth identification.
    let ops = vec![
        node(y),
        MutationOp::AddEdges {
            base: 2,
            neighbors: vec![1],
            layer: 0,
            edge_type: EdgeType::All,
        },
    ];
    let minted = graph
        .apply_new(UnstampedMutation {
            as_of,
            ops: ops.clone(),
        })
        .unwrap();

    // The record carries the intent verbatim; the void reference is
    // dropped at apply: no edge in either direction.
    assert_eq!(minted.ops, ops);
    assert_eq!(minted.as_of, as_of);
    assert_eq!(graph.get_active_links(&2, 0), Vec::<VectorId>::new());
    assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());

    // Replay re-resolves to the identical graph.
    wal.push(minted);
    let mut replayed = GraphMem::new();
    replayed.insert_apply_all(&wal).unwrap();
    assert_eq!(graph.checksum(), replayed.checksum());
    assert_eq!(replayed.get_active_links(&2, 0), Vec::<VectorId>::new());
}

/// Resolution: a `RemoveEdges` record ranked before its target's reauth
/// must not evict the fresh edge the reauth re-added — the removal names
/// the old edge, which is already gone. Replay skips it identically.
#[test]
fn resolution_skips_removal_of_edge_readded_after_identification() {
    let mut graph = GraphMem::new();
    let mut wal: Vec<GraphMutation> = Vec::new();
    let x0 = VectorId::new(1, 0);
    let x1 = VectorId::new(1, 1);
    let y = VectorId::from_serial_id(2);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    // seq 1: Y -> X.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![
                    node(x0),
                    node(y),
                    MutationOp::AddEdges {
                        base: 2,
                        neighbors: vec![1],
                        layer: 0,
                        edge_type: EdgeType::Base,
                    },
                ],
            })
            .unwrap(),
    );
    // Compaction ranked Y's neighborhood and chose to evict X here.
    let as_of = graph.last_update_seq_no;
    // seq 2: X reauths and re-links to Y — Y -> X now points at vetted
    // fresh content.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![
                    MutationOp::RemoveNode { id: x1 },
                    node(x1),
                    MutationOp::AddEdges {
                        base: 1,
                        neighbors: vec![2],
                        layer: 0,
                        edge_type: EdgeType::All,
                    },
                ],
            })
            .unwrap(),
    );
    assert_eq!(graph.get_active_links(&2, 0), vec![x1]);

    // seq 3: the stale-ranked eviction lands; the fresh edge survives.
    let ops = vec![MutationOp::RemoveEdges {
        base: 2,
        neighbors: vec![1],
        layer: 0,
        edge_type: EdgeType::Base,
    }];
    let minted = graph
        .apply_new(UnstampedMutation {
            as_of,
            ops: ops.clone(),
        })
        .unwrap();
    assert_eq!(minted.ops, ops);
    assert_eq!(graph.get_active_links(&2, 0), vec![x1]);

    // Replay re-resolves to the identical graph — the fresh edge
    // survives there too.
    wal.push(minted);
    let mut replayed = GraphMem::new();
    replayed.insert_apply_all(&wal).unwrap();
    assert_eq!(graph.checksum(), replayed.checksum());
    assert_eq!(replayed.get_active_links(&2, 0), vec![x1]);
}

/// Resolution: a `base` names the current node with that serial, so an
/// edge op whose base reauthed after identification applies to the
/// reauthed node.
#[test]
fn resolution_applies_op_whose_base_reauthed_after_identification() {
    let mut graph = GraphMem::new();
    let mut wal: Vec<GraphMutation> = Vec::new();
    let x0 = VectorId::new(1, 0);
    let x1 = VectorId::new(1, 1);
    let z = VectorId::from_serial_id(3);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    // seq 1: X and Z.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![node(x0), node(z)],
            })
            .unwrap(),
    );
    // An edge touch X <-> Z was identified here.
    let as_of = graph.last_update_seq_no;
    // seq 2: X reauths.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![MutationOp::RemoveNode { id: x1 }, node(x1)],
            })
            .unwrap(),
    );
    // seq 3: the touch lands on the reauthed base.
    let ops = vec![MutationOp::AddEdges {
        base: 1,
        neighbors: vec![3],
        layer: 0,
        edge_type: EdgeType::All,
    }];
    let minted = graph
        .apply_new(UnstampedMutation {
            as_of,
            ops: ops.clone(),
        })
        .unwrap();

    assert_eq!(minted.ops, ops);
    assert_eq!(graph.get_active_links(&1, 0), vec![z]);
    assert_eq!(graph.get_active_links(&3, 0), vec![x1]);

    // Replay re-resolves to the identical graph.
    wal.push(minted);
    let mut replayed = GraphMem::new();
    replayed.insert_apply_all(&wal).unwrap();
    assert_eq!(graph.checksum(), replayed.checksum());
    assert_eq!(replayed.get_active_links(&1, 0), vec![z]);
}

/// Resolution: an edge op whose base has no current node is dropped
/// whole — edges must not wire to an absent endpoint.
#[test]
fn resolution_drops_op_whose_base_is_absent() {
    let mut graph = GraphMem::new();
    let mut wal: Vec<GraphMutation> = Vec::new();
    let x0 = VectorId::new(1, 0);
    let z = VectorId::from_serial_id(3);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    // seq 1: X and Z.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![node(x0), node(z)],
            })
            .unwrap(),
    );
    // An edge touch X <-> Z was identified here.
    let as_of = graph.last_update_seq_no;
    // seq 2: X is removed.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![MutationOp::RemoveNode { id: x0 }],
            })
            .unwrap(),
    );
    // seq 3: the touch lands with its base gone and is void.
    let ops = vec![MutationOp::AddEdges {
        base: 1,
        neighbors: vec![3],
        layer: 0,
        edge_type: EdgeType::All,
    }];
    let minted = graph
        .apply_new(UnstampedMutation {
            as_of,
            ops: ops.clone(),
        })
        .unwrap();

    assert_eq!(minted.ops, ops);
    assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());
    assert_eq!(graph.get_active_links(&3, 0), Vec::<VectorId>::new());

    // Replay re-resolves to the identical graph.
    wal.push(minted);
    let mut replayed = GraphMem::new();
    replayed.insert_apply_all(&wal).unwrap();
    assert_eq!(graph.checksum(), replayed.checksum());
    assert_eq!(replayed.get_active_links(&3, 0), Vec::<VectorId>::new());
}

/// Resolution: a reauth plan's search almost always finds the node's own
/// old version, and the serial collapse turns it into a self-reference.
/// It was identified against the content the reauth replaces, so it must
/// not survive as a self-edge — while legitimate neighbors do.
#[test]
fn resolution_drops_reauth_self_reference() {
    let mut graph = GraphMem::new();
    let mut wal: Vec<GraphMutation> = Vec::new();
    let s0 = VectorId::new(1, 0);
    let s1 = VectorId::new(1, 1);
    let a = VectorId::from_serial_id(2);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    // seq 1: S and anchor A, linked.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![
                    node(s0),
                    node(a),
                    MutationOp::AddEdges {
                        base: 1,
                        neighbors: vec![2],
                        layer: 0,
                        edge_type: EdgeType::All,
                    },
                ],
            })
            .unwrap(),
    );
    // The reauth's search identified [S(old), A] here.
    let as_of = graph.last_update_seq_no;
    // seq 2: teardown, its own record (the production shape).
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of: graph.last_update_seq_no,
                ops: vec![MutationOp::RemoveNode { id: s1 }],
            })
            .unwrap(),
    );
    // seq 3: re-insert wiring the searched links, self-echo included.
    wal.push(
        graph
            .apply_new(UnstampedMutation {
                as_of,
                ops: vec![
                    node(s1),
                    MutationOp::AddEdges {
                        base: 1,
                        neighbors: vec![1, 2],
                        layer: 0,
                        edge_type: EdgeType::All,
                    },
                ],
            })
            .unwrap(),
    );

    // No self-edge, physically or actively; the real neighbor survives
    // in both directions.
    assert_eq!(graph.get_raw_links(&1, 0), &[2u32]);
    assert_eq!(graph.get_active_links(&1, 0), vec![a]);
    assert_eq!(graph.get_active_links(&2, 0), vec![s1]);

    // Replay re-resolves to the identical graph.
    let mut replayed = GraphMem::new();
    replayed.insert_apply_all(&wal).unwrap();
    assert_eq!(graph.checksum(), replayed.checksum());
    assert_eq!(replayed.get_raw_links(&1, 0), &[2u32]);
}

/// Read-path skip: `get_active_links` omits a content-stale neighbor even
/// when the neighborhood was never touched after the reauth, so the
/// physical edge is still present.
#[test]
fn get_active_links_skips_content_stale_neighbor() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![node(a), node(b)],
        })
        .unwrap();
    // Asymmetric a -> b; a's neighborhood certified at seq 2.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(graph.get_active_links(&1, 0), vec![b]);

    // Reauth b (content[b] -> 3) without touching a.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
        })
        .unwrap();

    // Physical edge survives (a untouched), but read-path skips it.
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
        vec![2u32]
    );
    assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());
}

/// Read-path liveness: `get_active_links` drops a dangling edge to a
/// removed node. The physical edge survives `RemoveNode` (no backlink
/// cleanup); the removed node has no content-clock entry, so `is_active`
/// treats it as dead.
#[test]
fn get_active_links_drops_edge_to_removed_node() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![node(a), node(b)],
        })
        .unwrap();
    // Asymmetric a -> b so b's removal can't clean a's back-edge.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    // Remove b (no re-insert): content[b] is dropped.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveNode { id: b }],
        })
        .unwrap();

    // Physical edge a -> b survives, but read-path skips the dead node.
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors().to_vec(),
        vec![2u32]
    );
    assert_eq!(graph.get_active_links(&1, 0), Vec::<VectorId>::new());
}

/// Read-path selectivity: a live neighbor survives while a content-stale
/// neighbor is dropped. Guards against `get_active_links` blanket-emptying
/// (e.g. an inverted `is_active`), which the single-neighbor tests above
/// would not catch. `c` carries a non-zero version, pinning that the
/// surviving link resolves to the version its `AddNode` carried.
#[test]
fn get_active_links_keeps_live_drops_stale() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let c = VectorId::new(3, 5);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![node(a), node(b), node(c)],
        })
        .unwrap();
    // a -> [b, c], certified at seq 2.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2, 3],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(graph.get_active_links(&1, 0), vec![b, c]);

    // Reauth only b (content[b] -> 3); a and c untouched.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
        })
        .unwrap();

    // c (content[c]=1 <= 2) survives at its inserted version; b
    // (content[b]=3 > 2) is skipped.
    assert_eq!(graph.get_active_links(&1, 0), vec![c]);
    assert_eq!(graph.vector_id_of(3), Some(c));
    assert_eq!(graph.vector_id_of(4), None, "never-inserted serial");
}

/// The incremental `node_init_hash` (mutation apply) must match the bulk
/// recompute (`from_parts`, via deserialize) after a mutation history that
/// includes a reauth.
#[test]
fn incremental_checksum_matches_bulk_after_reauth() {
    let mut graph = GraphMem::new();
    let a = VectorId::from_serial_id(1);
    let b = VectorId::from_serial_id(2);
    let node = |id| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };

    graph
        .insert_apply(&GraphMutation {
            seq_no: 1,
            as_of: 0,
            ops: vec![node(a), node(b)],
        })
        .unwrap();
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2],
                edge_type: EdgeType::All,
            }],
        })
        .unwrap();
    // Reauth b: RemoveNode + AddNode, incrementally updating node_init_hash.
    graph
        .insert_apply(&GraphMutation {
            seq_no: 3,
            as_of: 2,
            ops: vec![MutationOp::RemoveNode { id: b }, node(b)],
        })
        .unwrap();

    // bincode round trip: Deserialize routes through from_parts, which
    // recomputes node_init_hash in bulk.
    let buf = bincode::serialize(&graph).unwrap();
    let bulk: GraphMem = bincode::deserialize(&buf).unwrap();

    assert_eq!(
        graph.checksum(),
        bulk.checksum(),
        "incremental node_init_hash drifted from bulk recompute"
    );
}

/// A graph aged through insert, mixed edge types, reauth, and delete
/// round-trips through the current pair format, and its incremental
/// checksum matches a bulk recompute.
#[tokio::test]
async fn graph_v5_evolving_roundtrip() {
    use crate::utils::serialization::graph::{
        read_graph_pair, write_graph_pair_current, GraphFormat,
    };
    let node = |id: VectorId| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };
    let v = VectorId::from_serial_id;
    let mut g = GraphMem::new();

    // Build: node 1 spans layers 0,1 (entry point at L1); 2..=4 in layer 0.
    g.insert_apply(&GraphMutation {
        seq_no: 1,
        as_of: 0,
        ops: vec![
            MutationOp::AddNode {
                id: v(1),
                height: 2,
                update_ep: UpdateEntryPoint::Append { layer: 1 },
            },
            node(v(2)),
            node(v(3)),
            node(v(4)),
        ],
    })
    .unwrap();
    assert_eq!(g.get_num_layers(), 2);
    assert_eq!(g.get_entry_points(), Some(vec![1]));

    // Mixed edge types at layer 0: All(1<->{2,3}) + Base(3->4).
    g.insert_apply(&GraphMutation {
        seq_no: 2,
        as_of: 1,
        ops: vec![
            MutationOp::AddEdges {
                base: 1,
                neighbors: vec![2, 3],
                layer: 0,
                edge_type: EdgeType::All,
            },
            MutationOp::AddEdges {
                base: 3,
                neighbors: vec![4],
                layer: 0,
                edge_type: EdgeType::Base,
            },
        ],
    })
    .unwrap();
    assert_eq!(g.get_active_links(&1, 0), vec![v(2), v(3)]);
    assert_eq!(g.get_active_links(&2, 0), vec![v(1)]);

    // Reauth node 3 (bumps its content clock) and delete node 4, so the aged
    // graph carries a non-uniform content clock and a removed node.
    g.insert_apply(&GraphMutation {
        seq_no: 3,
        as_of: 2,
        ops: vec![MutationOp::RemoveNode { id: v(3) }, node(v(3))],
    })
    .unwrap();
    g.insert_apply(&GraphMutation {
        seq_no: 4,
        as_of: 3,
        ops: vec![MutationOp::RemoveNode { id: v(4) }],
    })
    .unwrap();
    assert_eq!(g.last_update_seq_no, 4);

    // Incremental checksum matches a bulk (from_parts) recompute over the
    // aged graph, and it round-trips through the production pair format.
    let bulk: GraphMem = bincode::deserialize(&bincode::serialize(&g).unwrap()).unwrap();
    assert_eq!(
        g.checksum(),
        bulk.checksum(),
        "incremental checksum drifted"
    );

    let mut pair_buf = Vec::new();
    write_graph_pair_current(&mut pair_buf, [g.clone(), g.clone()]).unwrap();
    let pair =
        read_graph_pair(&mut std::io::Cursor::new(&pair_buf), GraphFormat::Current).unwrap();
    for restored in &pair {
        assert_eq!(*restored, g, "pair round-trip changed the aged graph");
    }
}

/// `RemoveEdges` fused retain: one teardown drops the explicitly-named
/// edge AND sweeps a content-stale sibling in a single re-stamp, on both
/// the base-list half (EdgeType::Base) and the target-loop back half
/// (EdgeType::All). The sweep is not reflected in the returned op list.
#[tokio::test]
async fn remove_edges_op_fused_retain_drops_explicit_and_sweeps_stale() {
    let node = |id: VectorId| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };
    let v = VectorId::from_serial_id;
    let mut g = GraphMem::new();
    g.apply_new(UnstampedMutation {
        as_of: g.last_update_seq_no,
        ops: (1..=6).map(|i| node(v(i))).collect(),
    })
    .unwrap();

    // ── EdgeType::Base, forward-half retain. 1 -> {2,3} forward-only, then
    // reauth 3 so 1->3 is content-stale while 1's list is untouched.
    g.apply_new(UnstampedMutation {
        as_of: g.last_update_seq_no,
        ops: vec![MutationOp::AddEdges {
            base: 1,
            neighbors: vec![2, 3],
            layer: 0,
            edge_type: EdgeType::Base,
        }],
    })
    .unwrap();
    g.apply_new(UnstampedMutation {
        as_of: g.last_update_seq_no,
        ops: vec![MutationOp::RemoveNode { id: v(3) }, node(v(3))],
    })
    .unwrap();
    assert_eq!(g.get_raw_links(&1, 0), &[2u32, 3u32], "stale 3 still raw");
    assert_eq!(g.get_active_links(&1, 0), vec![v(2)], "3 masked as stale");

    // RemoveEdges Base(1 -/-> 2): retain drops explicit 2 AND stale 3; the
    // returned op list carries the intent unchanged.
    let teardown = MutationOp::RemoveEdges {
        base: 1,
        neighbors: vec![2],
        layer: 0,
        edge_type: EdgeType::Base,
    };
    let minted = g
        .apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: vec![teardown.clone()],
        })
        .unwrap();
    assert_eq!(
        g.get_raw_links(&1, 0),
        &[] as &[u32],
        "explicit 2 removed AND content-stale 3 swept in one retain"
    );
    assert_eq!(minted.ops, vec![teardown], "ops returned unchanged");

    // ── EdgeType::All, back-half (target-loop) retain. 4<->5 symmetric plus
    // asymmetric 5->6; reauth 6 so 5->6 is stale.
    g.apply_new(UnstampedMutation {
        as_of: g.last_update_seq_no,
        ops: vec![
            MutationOp::AddEdges {
                base: 4,
                neighbors: vec![5],
                layer: 0,
                edge_type: EdgeType::All,
            },
            MutationOp::AddEdges {
                base: 5,
                neighbors: vec![6],
                layer: 0,
                edge_type: EdgeType::Base,
            },
        ],
    })
    .unwrap();
    g.apply_new(UnstampedMutation {
        as_of: g.last_update_seq_no,
        ops: vec![MutationOp::RemoveNode { id: v(6) }, node(v(6))],
    })
    .unwrap();
    assert_eq!(g.get_raw_links(&5, 0), &[4u32, 6u32], "stale 6 still raw");

    // RemoveEdges All(4 -/- 5): forward half empties 4; back-half retain on
    // target 5 drops explicit 4 AND stale 6.
    let teardown = MutationOp::RemoveEdges {
        base: 4,
        neighbors: vec![5],
        layer: 0,
        edge_type: EdgeType::All,
    };
    let minted = g
        .apply_new(UnstampedMutation {
            as_of: g.last_update_seq_no,
            ops: vec![teardown.clone()],
        })
        .unwrap();
    assert_eq!(
        g.get_raw_links(&4, 0),
        &[] as &[u32],
        "forward half drops 5"
    );
    assert_eq!(
        g.get_raw_links(&5, 0),
        &[] as &[u32],
        "back-half retain drops explicit 4 AND content-stale 6"
    );
    assert_eq!(minted.ops, vec![teardown], "ops returned unchanged");
}

/// A stream minted by `apply_new` replays via `insert_apply_all` onto a
/// fresh graph to the identical state — checksum included. Exercises:
/// reauth whose re-wiring touches a neighborhood holding its own stale
/// back-edge, deletion followed by a touch sweeping the dangling edge,
/// both `RemoveEdges` fused-retain halves (Base forward, All back), a
/// compaction-shaped `RemoveEdges`, multi-layer edges, and entry-point
/// churn.
#[test]
fn minted_stream_replays_to_identical_graph() {
    fn mint(g: &mut GraphMem, wal: &mut Vec<GraphMutation>, ops: Vec<MutationOp>) {
        wal.push(
            g.apply_new(UnstampedMutation {
                as_of: g.last_update_seq_no,
                ops,
            })
            .unwrap(),
        );
    }
    let node = |s: u32| MutationOp::AddNode {
        id: VectorId::from_serial_id(s),
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };
    // Node 1 spans layers 0,1 and is the entry point; re-minted on reauth.
    let node1 = || MutationOp::AddNode {
        id: VectorId::from_serial_id(1),
        height: 2,
        update_ep: UpdateEntryPoint::Append { layer: 1 },
    };
    let all = |base: u32, neighbors: Vec<u32>, layer: usize| MutationOp::AddEdges {
        base,
        neighbors,
        layer,
        edge_type: EdgeType::All,
    };

    let mut l = GraphMem::new();
    let mut wal = Vec::new();

    // seq 1: nodes 1..=7 (1 and 5 span layer 1).
    let mut ops = vec![node1()];
    ops.extend((2..=7).map(node));
    ops[4] = MutationOp::AddNode {
        id: VectorId::from_serial_id(5),
        height: 2,
        update_ep: UpdateEntryPoint::False,
    };
    mint(&mut l, &mut wal, ops);
    // seq 2: 1<->2 at layer 0, 1<->5 at layer 1.
    mint(
        &mut l,
        &mut wal,
        vec![all(1, vec![2], 0), all(1, vec![5], 1)],
    );
    // seq 3: reauth 1, phase one — node teardown (2 and 5 keep stale
    // back-edges to 1; the entry point is dropped).
    mint(
        &mut l,
        &mut wal,
        vec![MutationOp::RemoveNode {
            id: VectorId::from_serial_id(1),
        }],
    );
    // seq 4: reauth 1, phase two — KEY fixture: the re-wiring back-halves
    // touch 2 (layer 0) and 5 (layer 1), each holding the stale edge to 1
    // that the same op re-adds (drop-then-re-add within one apply).
    mint(
        &mut l,
        &mut wal,
        vec![node1(), all(1, vec![2], 0), all(1, vec![5], 1)],
    );
    assert_eq!(
        wal[3].ops,
        vec![node1(), all(1, vec![2], 0), all(1, vec![5], 1)],
        "minted ops are the intent, unmodified"
    );

    // seq 5: pure deletion of 2 — node 1 keeps a dangling 1->2 edge.
    mint(
        &mut l,
        &mut wal,
        vec![MutationOp::RemoveNode {
            id: VectorId::from_serial_id(2),
        }],
    );
    // seq 6: 3<->1 — the back-half touch of 1 sweeps the dangling 2.
    mint(&mut l, &mut wal, vec![all(3, vec![1], 0)]);
    // seq 7: forward-only fan-out 5 -> {6,7} at layer 0.
    mint(
        &mut l,
        &mut wal,
        vec![MutationOp::AddEdges {
            base: 5,
            neighbors: vec![6, 7],
            layer: 0,
            edge_type: EdgeType::Base,
        }],
    );
    // seq 8: reauth 6 in one mutation (no edges reference 6 here).
    mint(
        &mut l,
        &mut wal,
        vec![
            MutationOp::RemoveNode {
                id: VectorId::from_serial_id(6),
            },
            node(6),
        ],
    );
    // seq 9: compaction-shaped RemoveEdges (Base): drops named 7 and sweeps
    // stale 6 from 5's list in the fused forward retain.
    mint(
        &mut l,
        &mut wal,
        vec![MutationOp::RemoveEdges {
            base: 5,
            neighbors: vec![7],
            layer: 0,
            edge_type: EdgeType::Base,
        }],
    );
    // seq 10: 4<->3.
    mint(&mut l, &mut wal, vec![all(4, vec![3], 0)]);
    // seq 11: reauth 1 again — 3's list (certified at seq 10) holds a now
    // stale edge to 1.
    mint(
        &mut l,
        &mut wal,
        vec![
            MutationOp::RemoveNode {
                id: VectorId::from_serial_id(1),
            },
            node1(),
        ],
    );
    // seq 12: RemoveEdges All 4 -/- 3: the back-half target retain on 3
    // drops named 4 and sweeps stale 1.
    mint(
        &mut l,
        &mut wal,
        vec![MutationOp::RemoveEdges {
            base: 4,
            neighbors: vec![3],
            layer: 0,
            edge_type: EdgeType::All,
        }],
    );

    // Only the two explicit RemoveEdges (seq 9, 12) appear in the stream;
    // filter drops are never recorded.
    let remove_edges = wal
        .iter()
        .flat_map(|m| m.ops.iter())
        .filter(|op| matches!(op, MutationOp::RemoveEdges { .. }))
        .count();
    assert_eq!(remove_edges, 2, "ops must record intent only");

    let mut r = GraphMem::new();
    r.insert_apply_all(&wal).unwrap();
    assert_eq!(l.checksum(), r.checksum(), "replay diverged from mint");
    assert_eq!(l, r, "replay diverged from mint");
}

/// Edge ops never perturb the content clock: a node's `NodeInit` moves
/// only on its own `AddNode`/`RemoveNode`.
#[tokio::test]
async fn edge_ops_do_not_perturb_content_clock() {
    let node = |id: VectorId| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };
    let v = VectorId::from_serial_id;
    let mut g = GraphMem::new();
    // Node 2 carries a non-zero version so the clock has real content.
    g.insert_apply(&GraphMutation {
        seq_no: 1,
        as_of: 0,
        ops: vec![node(v(1)), node(VectorId::new(2, 7)), node(v(3))],
    })
    .unwrap();
    let clock_before = g.node_init.clone();

    g.insert_apply(&GraphMutation {
        seq_no: 2,
        as_of: 1,
        ops: vec![MutationOp::AddEdges {
            base: 1,
            neighbors: vec![2, 3],
            layer: 0,
            edge_type: EdgeType::All,
        }],
    })
    .unwrap();
    assert_eq!(g.get_raw_links(&1, 0), &[2u32, 3u32]);
    assert_eq!(
        g.get_raw_links(&2, 0),
        &[1u32],
        "back-edge keyed on serial 2"
    );
    assert_eq!(
        g.node_init, clock_before,
        "AddEdges must not perturb the content clock"
    );

    g.insert_apply(&GraphMutation {
        seq_no: 3,
        as_of: 2,
        ops: vec![MutationOp::RemoveEdges {
            base: 1,
            neighbors: vec![2],
            layer: 0,
            edge_type: EdgeType::Base,
        }],
    })
    .unwrap();
    assert_eq!(g.get_raw_links(&1, 0), &[3u32]);
    assert_eq!(
        g.node_init, clock_before,
        "RemoveEdges must not perturb the content clock"
    );
}

/// Deleting the sole entry point: RemoveNode drops it from every layer,
/// the entry-point list, and the content clock. Backlinks to it linger in
/// raw but are masked by is_active; get_temporary_entry_point falls back
/// to the min-serial node of the top non-empty layer.
#[tokio::test]
async fn remove_node_of_entry_point_falls_back_to_min_serial() {
    let node = |id: VectorId| MutationOp::AddNode {
        id,
        height: 1,
        update_ep: UpdateEntryPoint::False,
    };
    let v = VectorId::from_serial_id;
    let mut g = GraphMem::new();
    // Node 1 spans layers 0,1 and is the sole entry point (at L1); 2,3 in L0.
    g.insert_apply(&GraphMutation {
        seq_no: 1,
        as_of: 0,
        ops: vec![
            MutationOp::AddNode {
                id: v(1),
                height: 2,
                update_ep: UpdateEntryPoint::Append { layer: 1 },
            },
            node(v(2)),
            node(v(3)),
        ],
    })
    .unwrap();
    // Symmetric backlinks 1<->2, 1<->3 at layer 0.
    g.insert_apply(&GraphMutation {
        seq_no: 2,
        as_of: 1,
        ops: vec![MutationOp::AddEdges {
            base: 1,
            neighbors: vec![2, 3],
            layer: 0,
            edge_type: EdgeType::All,
        }],
    })
    .unwrap();
    assert_eq!(g.get_entry_points(), Some(vec![1]));

    g.insert_apply(&GraphMutation {
        seq_no: 3,
        as_of: 2,
        ops: vec![MutationOp::RemoveNode { id: v(1) }],
    })
    .unwrap();

    assert!(g.layers[0].get_links(&1).is_none(), "gone at layer 0");
    assert!(g.layers[1].get_links(&1).is_none(), "gone at layer 1");
    // Backlinks 2->1 / 3->1 linger physically but are masked at read: 1 is
    // gone from the content clock, so is_active hides the dangling edge.
    assert_eq!(g.get_raw_links(&2, 0), &[1u32], "2->1 lingers in raw");
    assert!(
        g.get_active_links(&2, 0).is_empty(),
        "2->1 masked by is_active"
    );
    assert_eq!(g.get_raw_links(&3, 0), &[1u32], "3->1 lingers in raw");
    assert!(
        g.get_active_links(&3, 0).is_empty(),
        "3->1 masked by is_active"
    );
    assert_eq!(g.get_entry_points(), None, "entry-point list emptied");
    // L1 now empty, so fall back to min serial of L0.
    assert_eq!(g.get_temporary_entry_point(), Some((2u32, 0)));
    assert!(!g.node_init.contains_key(&1), "content[1] dropped");
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
            as_of: 0,
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
            as_of: 1,
            ops: vec![MutationOp::AddEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2, 3],
                edge_type: EdgeType::All,
            }],
        })
        .unwrap();
    assert_eq!(
        graph.layers[0].get_links(&1).unwrap().neighbors,
        vec![2u32, 3u32]
    );
    assert_eq!(graph.layers[0].get_links(&2).unwrap().neighbors, vec![1u32]);
    assert_eq!(graph.layers[0].get_links(&3).unwrap().neighbors, vec![1u32]);
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
            as_of: 0,
            ops: vec![
                MutationOp::AddNode {
                    id: a,
                    height: 1,
                    update_ep: UpdateEntryPoint::Append { layer: 1 },
                },
                MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2, 3],
                    edge_type: EdgeType::Base,
                },
                MutationOp::AddNode {
                    id: b,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddEdges {
                    base: 2,
                    layer: 0,
                    neighbors: vec![1],
                    edge_type: EdgeType::Base,
                },
                MutationOp::AddNode {
                    id: c,
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                },
                MutationOp::AddEdges {
                    base: 3,
                    layer: 0,
                    neighbors: vec![1],
                    edge_type: EdgeType::Base,
                },
            ],
        })
        .unwrap();
    graph
        .insert_apply(&GraphMutation {
            seq_no: 2,
            as_of: 1,
            ops: vec![MutationOp::RemoveEdges {
                base: 1,
                layer: 0,
                neighbors: vec![2],
                edge_type: EdgeType::Base,
            }],
        })
        .unwrap();
    assert_eq!(graph.layers[0].get_links(&1).unwrap().neighbors, vec![3u32]);
    // Bidirectional cleanup is not implied — b's list still contains a.
    assert_eq!(graph.layers[0].get_links(&2).unwrap().neighbors, vec![1u32]);
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
            as_of: 0,
            ops: vec![
                // Listed first: an edge op that references a node not yet created.
                MutationOp::AddEdges {
                    base: 1,
                    layer: 0,
                    neighbors: vec![2],
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
    assert_eq!(graph.layers[0].get_links(&1).unwrap().neighbors, vec![2u32]);
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
        as_of: 0,
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
        as_of: 4,
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
        as_of: 8,
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
            as_of: 0,
            ops: vec![MutationOp::AddNode {
                id: a,
                height: 1,
                update_ep: UpdateEntryPoint::Append { layer: 0 },
            }],
        },
        // Equal seq_no — should fail.
        GraphMutation {
            seq_no: 1,
            as_of: 0,
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
    assert!(graph.layers[0].get_links(&1).is_some());
    assert!(graph.layers[0].get_links(&2).is_none());
}
