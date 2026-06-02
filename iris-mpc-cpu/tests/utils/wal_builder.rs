use super::cpu_node::CpuNodes;

// TODO: replace with real imports once paths are confirmed.
// use iris_mpc_cpu::hnsw::graph::mutation::{GraphMutation, MutationOp, UpdateEntryPoint};
// use iris_mpc_cpu::hnsw::graph::graph_store::GraphPg;
// use iris_mpc_cpu::hawkers::plaintext_store::PlaintextStore;
// use iris_mpc_common::iris_db::iris::IrisVectorId;

/// Builds and inserts synthetic `hawk_graph_mutations` rows without requiring a live
/// MPC request pipeline or real iris data.
///
/// Each inserted row corresponds to one `modification_id` and contains a
/// bincode-serialized `BothEyes<Vec<GraphMutation<IrisVectorId>>>`.
///
/// # Usage
///
/// ```rust
/// WalMutationBuilder::new()
///     .add_node(1, 0, 3)
///     .add_node(2, 1, 2)
///     .add_edges(3, 0, vec![1], 0)
///     .seed_all(&nodes)
///     .await?;
/// ```
pub struct WalMutationBuilder {
    // Each element: (modification_id, left_mutations, right_mutations)
    // TODO: replace Vec<u8> placeholders with real GraphMutation types
    entries: Vec<WalEntry>,
}

struct WalEntry {
    modification_id: i64,
    // Placeholder: will become BothEyes<Vec<GraphMutation<IrisVectorId>>>
    // TODO: use real types (open question #6 in readme — do mutations need to form
    // a valid HNSW graph or can they be arbitrary for WAL replay tests?)
    description: String,
}

impl WalMutationBuilder {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add an `AddNode` mutation for both eyes at the given `modification_id`.
    pub fn add_node(mut self, modification_id: i64, node_id: u32, height: usize) -> Self {
        self.entries.push(WalEntry {
            modification_id,
            description: format!("AddNode(id={node_id}, height={height})"),
        });
        self
    }

    /// Add an `AddEdges` mutation for both eyes at the given `modification_id`.
    pub fn add_edges(
        mut self,
        modification_id: i64,
        base: u32,
        neighbors: Vec<u32>,
        layer: usize,
    ) -> Self {
        self.entries.push(WalEntry {
            modification_id,
            description: format!("AddEdges(base={base}, neighbors={neighbors:?}, layer={layer})"),
        });
        self
    }

    /// Persist all mutations to one party's graph store.
    pub async fn seed(&self, _graph: &()) -> eyre::Result<()> {
        // TODO:
        //   for each entry:
        //     1. construct BothEyes<Vec<GraphMutation<IrisVectorId>>> from the entry
        //     2. bincode::serialize it
        //     3. call graph.upsert_hawk_graph_mutations(tx, entry.modification_id, bytes)
        todo!("serialize and insert WAL mutations into graph store")
    }

    /// Convenience: seed the same mutations into all 3 parties' stores.
    pub async fn seed_all(&self, _nodes: &CpuNodes) -> eyre::Result<()> {
        // TODO: call self.seed(&node.stores.graph) for each party
        todo!("seed WAL mutations into all 3 parties")
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
