use std::collections::HashMap;

use iris_mpc_common::IrisVectorId;
use iris_mpc_cpu::{
    execution::hawk_main::BothEyes,
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::{
        graph_store::GraphPg,
        mutation::{EdgeType, GraphMutation, MutationOp, UpdateEntryPoint},
    },
};
use iris_mpc_utils::{aws::AwsClient, irises::generate_iris_shares_for_upload_both_eyes};
use rand::{rngs::StdRng, SeedableRng};

use super::cpu_node::CpuNodes;

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

/// Test utility that pre-builds one modification per node.
///
/// Each modification (keyed by `modification_id = node_index as i64`) contains
/// two [`GraphMutation`]s:
///   1. `AddNode`  – layer 0, height 1
///   2. `AddEdges` – layer 0, edges to `(base+1) % n` and `(base+2) % n`, [`EdgeType::All`]
///
/// Default `persisted = true`, default `status = COMPLETED`.
/// Use [`set_persisted`] / [`set_status`] to override individual modifications.
pub struct WalMutationBuilder {
    entries: HashMap<i64, Vec<GraphMutation<IrisVectorId>>>,
    persisted: HashMap<i64, bool>,
    status: HashMap<i64, ModificationStatus>,
}

impl WalMutationBuilder {
    pub fn new(num_nodes: usize) -> Self {
        let mut entries = HashMap::new();
        let mut persisted = HashMap::new();
        let mut status = HashMap::new();

        let n = num_nodes as u32;

        for i in 0..num_nodes {
            let modification_id = i as i64;
            let node_id = i as u32;

            let add_node = GraphMutation {
                seq_no: modification_id as u64,
                ops: vec![MutationOp::AddNode {
                    id: IrisVectorId::from_serial_id(node_id),
                    height: 1,
                    update_ep: UpdateEntryPoint::False,
                }],
            };

            let add_edges = GraphMutation {
                seq_no: modification_id as u64,
                ops: vec![MutationOp::AddEdges {
                    base: IrisVectorId::from_serial_id(node_id),
                    neighbors: vec![
                        IrisVectorId::from_serial_id((node_id + 1) % n),
                        IrisVectorId::from_serial_id((node_id + 2) % n),
                    ],
                    layer: 0,
                    edge_type: EdgeType::All,
                }],
            };

            entries.insert(modification_id, vec![add_node, add_edges]);
            persisted.insert(modification_id, true);
            status.insert(modification_id, ModificationStatus::Completed);
        }

        Self {
            entries,
            persisted,
            status,
        }
    }

    /// Sets the `persisted` flag for a single modification.
    pub fn set_persisted(&mut self, modification_id: i64, value: bool) -> &mut Self {
        self.persisted.insert(modification_id, value);
        self
    }

    /// Sets the `status` for a single modification to `COMPLETED` or `PENDING`.
    pub fn set_status(&mut self, modification_id: i64, value: ModificationStatus) -> &mut Self {
        self.status.insert(modification_id, value);
        self
    }

    pub async fn insert_mutations(&self, graph: &GraphPg<PlaintextStore>) -> eyre::Result<()> {
        for (modification_id, mutations) in &self.entries {
            let both_eyes: BothEyes<Vec<GraphMutation<IrisVectorId>>> =
                [mutations.clone(), mutations.clone()];
            let bytes = bincode::serialize(&both_eyes)?;
            let mut tx = graph.pool().begin().await?;
            graph
                .upsert_hawk_graph_mutations(&mut tx, *modification_id, &bytes)
                .await?;
            tx.commit().await?;
        }
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
        let result_message_body = format!(r#"{{"node_id":{party_id}}}"#);
        for (modification_id, mutations) in &self.entries {
            let serial_id: i64 = match mutations.first().and_then(|m| m.ops.first()) {
                Some(MutationOp::AddNode { id, .. }) => id.serial_id() as i64,
                Some(MutationOp::AddEdges { base, .. }) => base.serial_id() as i64,
                _ => 0,
            };
            let s3_url = uuid::Uuid::from_u128(*modification_id as u128).to_string();
            let persisted = self.persisted.get(modification_id).copied().unwrap_or(true);
            let status = self
                .status
                .get(modification_id)
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

    pub async fn build(&self, nodes: &CpuNodes) -> eyre::Result<()> {
        self.insert_mutations_all(nodes).await?;
        self.seed_modifications_all(nodes).await?;
        Ok(())
    }

    pub async fn upload_iris_shares(&self, aws_client: &AwsClient) -> eyre::Result<()> {
        let mut rng = StdRng::seed_from_u64(42);
        for (modification_id, _) in &self.entries {
            let uuid = uuid::Uuid::from_u128(*modification_id as u128);
            let shares = generate_iris_shares_for_upload_both_eyes(&mut rng, None, None);
            aws_client
                .s3_upload_iris_shares(&uuid, &shares)
                .await
                .map_err(|e| {
                    eyre::eyre!("S3 upload failed for modification_id={modification_id}: {e}")
                })?;
        }
        Ok(())
    }
}
