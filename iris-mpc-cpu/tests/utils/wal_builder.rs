use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes,
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::{
        graph_store::GraphPg,
        mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint},
    },
};

use super::cpu_node::CpuNodes;

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
    entries: Vec<(i64, GraphMutation<IrisVectorId>)>,
}

impl WalMutationBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an `AddNode` mutation for both eyes at the given `modification_id`.
    ///
    /// `node_id` is the HNSW node identifier (use sequential values starting from 0).
    /// `height` is the HNSW layer height for this node.
    pub fn add_node(mut self, modification_id: i64, node_id: u32, height: usize) -> Self {
        let mutation = GraphMutation {
            seq_no: modification_id as u64,
            ops: vec![MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(node_id),
                height,
                update_ep: UpdateEntryPoint::False,
            }],
        };
        self.entries.push((modification_id, mutation));
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
        let mutation = GraphMutation {
            seq_no: modification_id as u64,
            ops: vec![MutationOp::AddEdges {
                base: IrisVectorId::from_serial_id(base),
                neighbors: neighbors
                    .iter()
                    .map(|&n| IrisVectorId::from_serial_id(n))
                    .collect(),
                layer,
                edge_type: EdgeType::All,
            }],
        };
        self.entries.push((modification_id, mutation));
        self
    }

    /// Persist all mutations to one party's graph store.
    ///
    /// For each entry serializes `BothEyes<Vec<GraphMutation<IrisVectorId>>>` with
    /// bincode and calls `graph.upsert_hawk_graph_mutations(tx, mod_id, bytes)`.
    pub async fn seed(&self, graph: &GraphPg<PlaintextStore>) -> eyre::Result<()> {
        for (modification_id, mutation) in &self.entries {
            let both_eyes: BothEyes<Vec<GraphMutation<IrisVectorId>>> =
                [vec![mutation.clone()], vec![mutation.clone()]];
            let bytes = bincode::serialize(&both_eyes)?;
            let mut tx = graph.pool().begin().await?;
            graph
                .upsert_hawk_graph_mutations(&mut tx, *modification_id, &bytes)
                .await?;
            tx.commit().await?;
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
