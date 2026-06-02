use super::cpu_node::CpuNodes;

// TODO (open question #8): confirm BothEyes import path.
// use iris_mpc_cpu::hnsw::graph::BothEyes;   OR   iris_mpc_common::...

// TODO (open question #7/8): confirm GraphMutation / MutationOp / IrisVectorId paths.
// use iris_mpc_cpu::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
// use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
// use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;

/// Builds and inserts synthetic `hawk_graph_mutations` rows without requiring a live
/// MPC request pipeline or real iris data.
///
/// Each inserted row corresponds to one `modification_id` and contains a
/// bincode-serialized `BothEyes<Vec<GraphMutation<IrisVectorId>>>`.
///
/// ## Constraints
///
/// - `AddNode` and `AddEdges` are both safe.  Reset-update, recovery-update, and
///   other modification types assume a node already exists and are not supported here.
/// - Use sequential `node_id` values (0, 1, 2, …).
/// - Keep neighbor lists under 100 entries per `add_edges` call to stay within
///   realistic HNSW degree bounds.
///
/// ## Usage
///
/// ```rust
/// WalMutationBuilder::new()
///     .add_node(1, 0, 3)                     // mod_id=1, node_id=0, height=3
///     .add_node(2, 1, 2)                     // mod_id=2, node_id=1, height=2
///     .add_edges(3, 0, vec![1], 0)           // mod_id=3, base=0, neighbors=[1], layer=0
///     .seed_all(&nodes)
///     .await?;
/// ```
pub struct WalMutationBuilder {
    entries: Vec<WalEntry>,
}

/// Internal representation of one WAL row to be seeded.
enum WalEntry {
    AddNode {
        modification_id: i64,
        /// Sequence number within the mutation; equals modification_id for synthetic rows.
        seq_no: u64,
        node_id: u32,
        height: usize,
    },
    AddEdges {
        modification_id: i64,
        seq_no: u64,
        base: u32,
        /// Neighbor node IDs.  Keep under 100 to stay within realistic HNSW degree bounds.
        neighbors: Vec<u32>,
        layer: usize,
    },
}

impl WalEntry {
    fn modification_id(&self) -> i64 {
        match self {
            Self::AddNode { modification_id, .. } => *modification_id,
            Self::AddEdges { modification_id, .. } => *modification_id,
        }
    }
}

impl WalMutationBuilder {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add an `AddNode` mutation for both eyes at the given `modification_id`.
    ///
    /// `node_id` is the HNSW node identifier (use sequential values starting from 0).
    /// `height` is the HNSW layer height for this node.
    pub fn add_node(mut self, modification_id: i64, node_id: u32, height: usize) -> Self {
        self.entries.push(WalEntry::AddNode {
            modification_id,
            seq_no: modification_id as u64,
            node_id,
            height,
        });
        self
    }

    /// Add an `AddEdges` mutation for both eyes at the given `modification_id`.
    ///
    /// `base` is the source node; `neighbors` are the nodes it connects to at `layer`.
    /// Keep `neighbors.len()` under 100 to stay within realistic HNSW degree bounds.
    pub fn add_edges(
        mut self,
        modification_id: i64,
        base: u32,
        neighbors: Vec<u32>,
        layer: usize,
    ) -> Self {
        assert!(
            neighbors.len() <= 100,
            "neighbors.len() = {} exceeds the 100-edge limit per add_edges call",
            neighbors.len()
        );
        self.entries.push(WalEntry::AddEdges {
            modification_id,
            seq_no: modification_id as u64,
            base,
            neighbors,
            layer,
        });
        self
    }

    /// Persist all mutations to one party's graph store.
    ///
    /// For each entry constructs a `BothEyes<Vec<GraphMutation<IrisVectorId>>>`,
    /// serializes it with bincode, and calls
    /// `graph.upsert_hawk_graph_mutations(tx, mod_id, bytes)`.
    pub async fn seed(&self, _graph: &()) /* TODO: &GraphPg<PlaintextStore> */ -> eyre::Result<()> {
        for entry in &self.entries {
            // TODO (open questions #7, #8, #9): replace placeholders with real types.
            //
            // let op = match entry {
            //     WalEntry::AddNode { node_id, height, seq_no, .. } => GraphMutation {
            //         seq_no: *seq_no,
            //         ops: vec![MutationOp::AddNode {
            //             id: IrisVectorId::from(*node_id),
            //             height: *height,
            //             update_ep: UpdateEntryPoint::False,
            //         }],
            //     },
            //     WalEntry::AddEdges { base, neighbors, layer, seq_no, .. } => GraphMutation {
            //         seq_no: *seq_no,
            //         ops: vec![MutationOp::AddEdges {
            //             base: IrisVectorId::from(*base),
            //             neighbors: neighbors.iter().map(|&n| IrisVectorId::from(n)).collect(),
            //             layer: *layer,
            //             edge_type: EdgeType::All,
            //         }],
            //     },
            // };
            // let both_eyes = BothEyes { left: vec![op.clone()], right: vec![op] };
            // let bytes = bincode::serialize(&both_eyes)?;
            // let mut tx = graph.pool.begin().await?;
            // graph.upsert_hawk_graph_mutations(&mut tx, entry.modification_id(), bytes).await?;
            // tx.commit().await?;

            let _ = entry;
            todo!("serialize GraphMutation and upsert into hawk_graph_mutations")
        }
        Ok(())
    }

    /// Convenience: seed the same mutations into all 3 parties' stores.
    pub async fn seed_all(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        for node in &nodes.0 {
            // TODO: self.seed(&node.stores.graph).await?;
            let _ = node;
            todo!("seed WAL mutations into all 3 parties")
        }
        Ok(())
    }

    /// Number of WAL entries that will be seeded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for WalMutationBuilder {
    fn default() -> Self {
        Self::new()
    }
}
