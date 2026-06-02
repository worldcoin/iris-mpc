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
/// ## Constraint: only `AddNode` mutations
///
/// Only `MutationOp::AddNode` (Uniqueness) mutations are safe on an empty starting
/// graph.  Reset-update, recovery-update, and other modification types assume a node
/// already exists and will panic or corrupt state if replayed against an empty graph.
///
/// Use sequential `node_id` values (0, 1, 2, …) to build a coherent graph.
///
/// ## Usage
///
/// ```rust
/// WalMutationBuilder::new()
///     .add_node(1, 0, 3)   // mod_id=1, node_id=0, height=3
///     .add_node(2, 1, 2)   // mod_id=2, node_id=1, height=2
///     .seed_all(&nodes)
///     .await?;
/// ```
pub struct WalMutationBuilder {
    entries: Vec<WalEntry>,
}

/// Internal representation of one WAL row to be seeded.
struct WalEntry {
    modification_id: i64,
    node_id: u32,
    height: usize,
    /// Sequence number within the mutation (mirrors seq_no in GraphMutation).
    /// For synthetic mutations, this equals modification_id.
    seq_no: u64,
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
        self.entries.push(WalEntry {
            modification_id,
            node_id,
            height,
            seq_no: modification_id as u64,
        });
        self
    }

    /// Persist all mutations to one party's graph store.
    ///
    /// For each entry, constructs:
    ///   `BothEyes { left: vec![GraphMutation { seq_no, ops: [AddNode { id, height, update_ep: False }] }],
    ///               right: <same> }`
    /// serializes with bincode, and calls `graph.upsert_hawk_graph_mutations(tx, mod_id, bytes)`.
    pub async fn seed(&self, _graph: &()) /* TODO: &GraphPg<PlaintextStore> */ -> eyre::Result<()> {
        for entry in &self.entries {
            // TODO (open questions #7, #8, #9):
            //
            //   let mutation = GraphMutation {
            //       seq_no: entry.seq_no,
            //       ops: vec![MutationOp::AddNode {
            //           id: IrisVectorId::from(entry.node_id),
            //           height: entry.height,
            //           update_ep: UpdateEntryPoint::False,
            //       }],
            //   };
            //   let both_eyes = BothEyes { left: vec![mutation.clone()], right: vec![mutation] };
            //   let bytes = bincode::serialize(&both_eyes)?;
            //   let mut tx = graph.pool.begin().await?;
            //   graph.upsert_hawk_graph_mutations(&mut tx, entry.modification_id, bytes).await?;
            //   tx.commit().await?;

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
