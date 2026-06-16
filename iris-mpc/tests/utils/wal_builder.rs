use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes,
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::{
        graph_store::GraphPg,
        mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint},
    },
};
use std::collections::HashMap;

use super::cpu_node::CpuNodes;
use crate::utils::cpu_node::CpuNode;

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModificationStatus {
    Completed,
    Pending,
}

impl ModificationStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ModificationStatus::Completed => "COMPLETED",
            ModificationStatus::Pending => "PENDING",
        }
    }
}

/// Stateful test utility for building WAL mutations incrementally.
///
/// Call [`add_nodes`] one or more times to append batches of nodes. Each call
/// continues sequentially from the last assigned modification ID / node ID, so
/// you can interleave calls with [`build`] and checkpoint operations:
///
/// ```ignore
/// let mut builder = WalMutationBuilder::new();
/// builder.add_nodes(50);
/// builder.build(&nodes).await?;
/// nodes.make_checkpoints(49, 49).await?;   // checkpoint the first batch
/// builder.add_nodes(50);                    // 50 more nodes on top
/// builder.build(&nodes).await?;             // upserts are idempotent
/// ```
///
/// For each new node, `add_nodes` creates two [`GraphMutation`]s:
///   1. `AddNode`  – height 1
///   2. `AddEdges` – layer 0, [`EdgeType::All`], connecting to **every
///      previously added node** (full mesh up to that point)
///
/// Default `persisted = true`, default `status = COMPLETED`.
/// Use [`set_persisted`] / [`set_status`] to override individual modifications.
#[derive(Default)]
pub struct WalMutationBuilder {
    entries: HashMap<i64, GraphMutation>,
    persisted: HashMap<i64, bool>,
    status: HashMap<i64, ModificationStatus>,
    processed: usize,
}

impl WalMutationBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `count` new nodes, continuing from the last assigned IDs.
    ///
    /// Each node connects (via `AddEdges`) to every node added in prior calls
    /// as well as every node added earlier within the same call.
    pub fn add_nodes(&mut self, count: usize) -> &mut Self {
        let starting_idx = self.entries.len();
        let new_len = starting_idx + count;
        for idx in starting_idx..new_len {
            let node_id = idx + 1;
            // All nodes added before this one become neighbors.
            let neighbors: Vec<IrisVectorId> = (1..=node_id)
                .filter(|x| x != &node_id)
                .map(|x| x as u32)
                .map(IrisVectorId::from_serial_id)
                .collect();

            let mut mutation = GraphMutation {
                seq_no: node_id as u64,
                ops: vec![],
            };
            mutation.ops.push(MutationOp::AddNode {
                id: IrisVectorId::from_serial_id(node_id as _),
                height: 1,
                update_ep: UpdateEntryPoint::False,
            });

            if !neighbors.is_empty() {
                mutation.ops.push(MutationOp::AddEdges {
                    base: IrisVectorId::from_serial_id(node_id as _),
                    neighbors,
                    layer: 0,
                    edge_type: EdgeType::All,
                });
            }
            self.entries.insert(node_id as _, mutation);
            self.persisted.insert(node_id as _, true);
            self.status
                .insert(node_id as _, ModificationStatus::Completed);
        }
        self
    }

    /// Sets the `persisted` flag for a single modification.
    /// must precede `build()`
    pub fn set_persisted(&mut self, modification_id: i64, value: bool) -> &mut Self {
        self.persisted.insert(modification_id, value);
        self
    }

    /// Sets the `status` for a single modification to `COMPLETED` or `PENDING`.
    #[allow(dead_code)]
    pub fn set_status(&mut self, modification_id: i64, value: ModificationStatus) -> &mut Self {
        self.status.insert(modification_id, value);
        self
    }

    /// Add a single node with explicit neighbor IDs, bypassing the auto-neighbor logic.
    ///
    /// Unlike [`add_nodes`], this always emits an `AddEdges` op regardless of how many
    /// nodes have been added previously — useful for crafting WAL entries that are
    /// intentionally incompatible with those produced by [`add_nodes`] at the same
    /// modification ID.
    pub fn add_node_with_neighbors(&mut self, neighbor_ids: &[u32]) -> &mut Self {
        let node_id = (self.entries.len() + 1) as u32;
        let modification_id = node_id as i64;
        let neighbors: Vec<IrisVectorId> = neighbor_ids
            .iter()
            .map(|&id| IrisVectorId::from_serial_id(id))
            .collect();
        let mut ops = vec![MutationOp::AddNode {
            id: IrisVectorId::from_serial_id(node_id),
            height: 1,
            update_ep: UpdateEntryPoint::False,
        }];
        if !neighbors.is_empty() {
            ops.push(MutationOp::AddEdges {
                base: IrisVectorId::from_serial_id(node_id),
                neighbors,
                layer: 0,
                edge_type: EdgeType::All,
            });
        }
        let mutation = GraphMutation {
            seq_no: node_id as u64,
            ops,
        };
        self.entries.insert(modification_id, mutation);
        self.persisted.insert(modification_id, true);
        self.status
            .insert(modification_id, ModificationStatus::Completed);
        self
    }

    pub async fn insert_mutations(&self, graph: &GraphPg<PlaintextStore>) -> eyre::Result<()> {
        let mut mutations: Vec<_> = self
            .entries
            .values()
            .filter(|x| x.seq_no > self.processed as u64)
            .cloned()
            .collect();
        mutations.sort_by_key(|a| a.seq_no);
        let mut tx = graph.pool().begin().await?;
        for m in mutations {
            let modification_id = m.seq_no as _;
            let m = vec![m];
            let both_eyes: BothEyes<Vec<GraphMutation>> = [m.clone(), m];
            let bytes = bincode::serialize(&both_eyes)?;
            graph
                .upsert_hawk_graph_mutations(&mut tx, modification_id, &bytes)
                .await?;
        }
        tx.commit().await?;
        Ok(())
    }

    pub async fn insert_mutations_all(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        for node in &nodes.0 {
            self.insert_mutations(&node.store.graph).await?;
        }
        Ok(())
    }

    pub async fn seed_modifications(
        &self,
        graph: &GraphPg<PlaintextStore>,
        party_id: usize,
    ) -> eyre::Result<()> {
        let mut mutations: Vec<_> = self
            .entries
            .values()
            .filter(|x| x.seq_no > self.processed as u64)
            .cloned()
            .collect();
        mutations.sort_by_key(|a| a.seq_no);
        let result_message_body = format!(r#"{{"node_id":{party_id}}}"#);
        for m in mutations {
            let modification_id = m.seq_no as i64;
            let serial_id: i64 = match m.ops.first() {
                Some(MutationOp::AddNode { id, .. }) => id.serial_id() as i64,
                Some(MutationOp::AddEdges { base, .. }) => base.serial_id() as i64,
                _ => 0,
            };
            let s3_url = uuid::Uuid::from_u128(modification_id as u128).to_string();
            let persisted = self
                .persisted
                .get(&modification_id)
                .copied()
                .unwrap_or(true);
            let status = self
                .status
                .get(&modification_id)
                .map(|s| s.as_str())
                .unwrap_or("COMPLETED");
            sqlx::query(
                r#"
                INSERT INTO modifications
                    (id, serial_id, request_type, s3_url, status, persisted, result_message_body)
                VALUES ($1, $2, 'uniqueness', $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
                "#,
            )
            .bind(modification_id)
            .bind(serial_id)
            .bind(&s3_url)
            .bind(status)
            .bind(persisted)
            .bind(&result_message_body)
            .execute(graph.pool())
            .await?;
        }
        Ok(())
    }

    pub async fn seed_modifications_all(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        for node in &nodes.0 {
            self.seed_modifications(&node.store.graph, node.config.party_id)
                .await?;
        }
        Ok(())
    }

    pub async fn build(&mut self, nodes: &CpuNodes) -> eyre::Result<()> {
        self.insert_mutations_all(nodes).await?;
        self.seed_modifications_all(nodes).await?;
        self.processed = self.entries.len();
        Ok(())
    }

    /// used to simulate the mutations (WAL) falling behind the modifications.
    /// set to mutations = false to only seed modifications.
    /// note that if any mutations are skipped, there is no way to use the WalBuilder
    /// to seed them later
    pub async fn build_single(&mut self, node: &CpuNode, mutations: bool) -> eyre::Result<()> {
        if mutations {
            self.insert_mutations(&node.store.graph).await?;
        }
        self.seed_modifications(&node.store.graph, node.config.party_id)
            .await?;
        self.processed = self.entries.len();
        Ok(())
    }
}
