use super::constants::COUNT_OF_PARTIES;
use crate::utils::{modifications::ModificationInput, GaloisRingSharedIrisPair, HawkConfigs};
use eyre::Result;
use iris_mpc_common::{
    config::Config,
    postgres::{AccessMode, PostgresClient},
    IrisSerialId,
};
use iris_mpc_cpu::{
    execution::hawk_main::StoreId,
    genesis::{
        plaintext::GenesisState,
        state_accessor::{unset_last_indexed_iris_id, unset_last_indexed_modification_id},
    },
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};
use itertools::Itertools;

// these constants were copied from genesis because genesis requires using an Aby3Store while the tests use a PlainTextStore
mod constants {
    /// Domain for persistent state store entry for last indexed id
    pub const STATE_DOMAIN: &str = "genesis";
    // Key for persistent state store entry for last indexed iris id
    pub const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";
    // Key for persistent state store entry for last indexed modification id
    pub const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "last_indexed_modification_id";
}
/// simulates a MPC node, complete with configuration (HAWK and Genesis) and database connections.
/// a simulation consists of 3 MPC nodes; see MpcNodes
pub struct MpcNode {
    // databases
    pub gpu_iris_store: IrisStore,
    pub cpu_iris_store: IrisStore,
    pub graph_store: GraphStore<PlaintextStore>,
}

/// Simulates a 3 party multi party computation. Is intended to be built from a list of configurations in a way that
/// allows the MpcNode instances to be passed to async move closures, for concurrent tasks.
pub struct MpcNodes {
    nodes: [MpcNode; COUNT_OF_PARTIES],
}

impl MpcNodes {
    pub async fn new(config: &HawkConfigs) -> Self {
        Self {
            nodes: [
                MpcNode::new(config[0].clone()).await,
                MpcNode::new(config[1].clone()).await,
                MpcNode::new(config[2].clone()).await,
            ],
        }
    }
}

impl IntoIterator for MpcNodes {
    type Item = MpcNode;
    type IntoIter = std::array::IntoIter<MpcNode, COUNT_OF_PARTIES>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

// entry points
impl MpcNode {
    pub async fn new(config: Config) -> Self {
        let cpu_client = PostgresClient::new(
            &config.get_cpu_db_url().unwrap(),
            &config.get_cpu_db_schema(),
            AccessMode::ReadWrite,
        )
        .await
        .unwrap();

        let gpu_client = PostgresClient::new(
            &config.get_gpu_db_url().unwrap(),
            &config.get_gpu_db_schema(),
            AccessMode::ReadWrite,
        )
        .await
        .unwrap();

        cpu_client.migrate().await;
        gpu_client.migrate().await;

        Self {
            gpu_iris_store: IrisStore::new(&gpu_client).await.unwrap(),
            cpu_iris_store: IrisStore::new(&cpu_client).await.unwrap(),
            graph_store: GraphStore::new(&cpu_client).await.unwrap(),
        }
    }

    /// clear all tables and insert irises into the GPU database
    pub async fn init_tables(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        self.clear_all_tables().await?;
        self.insert_into_gpu_iris_store(shares).await?;
        Ok(())
    }
}

// utilities for unit testing, such as assertions
impl MpcNode {
    pub async fn assert_graphs_match(&self, expected: &GenesisState) {
        let graph_left = {
            let mut graph_tx = self.graph_store.tx().await.unwrap();
            graph_tx
                .with_graph(StoreId::Left)
                .load_to_mem(self.graph_store.pool(), 2)
                .await
        }
        .expect("Could not load left graph");
        let graph_right = {
            let mut graph_tx = self.graph_store.tx().await.unwrap();
            graph_tx
                .with_graph(StoreId::Right)
                .load_to_mem(self.graph_store.pool(), 2)
                .await
        }
        .expect("Could not load right graph");

        assert!(graph_left == expected.dst_db.graphs[0]);
        assert!(graph_right == expected.dst_db.graphs[1]);
    }

    pub async fn get_last_indexed_iris_id(&self) -> IrisSerialId {
        self.graph_store
            .get_persistent_state(
                constants::STATE_DOMAIN,
                constants::STATE_KEY_LAST_INDEXED_IRIS_ID,
            )
            .await
            .unwrap()
            .unwrap_or_default()
    }

    pub async fn get_last_indexed_modification_id(&self) -> i64 {
        self.graph_store
            .get_persistent_state(
                constants::STATE_DOMAIN,
                constants::STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
            )
            .await
            .unwrap()
            .unwrap_or_default()
    }
}

// misc
impl MpcNode {
    pub async fn insert_modifications(&self, mods: &[ModificationInput]) -> Result<()> {
        let mut updates = vec![];
        let tx = self.gpu_iris_store.tx().await?;
        for m in mods {
            let mut m2 = self
                .gpu_iris_store
                .insert_modification(Some(m.serial_id), m.request_type.to_str(), None)
                .await?;
            m2.status = m.get_status().to_string();
            m2.persisted = m.persisted;
            updates.push(m2);
        }
        tx.commit().await?;

        let mut tx = self.gpu_iris_store.tx().await?;
        self.gpu_iris_store
            .update_modifications(&mut tx, updates.iter().collect::<Vec<_>>().as_slice())
            .await?;
        tx.commit().await?;
        Ok(())
    }

    async fn clear_all_tables(&self) -> Result<()> {
        let mut graph_tx = self.graph_store.tx().await?;
        graph_tx
            .with_graph(StoreId::Left)
            .clear_tables()
            .await
            .expect("Could not clear left graph");
        graph_tx
            .with_graph(StoreId::Right)
            .clear_tables()
            .await
            .expect("Could not clear right graph");

        unset_last_indexed_iris_id(&mut graph_tx.tx).await?;
        unset_last_indexed_modification_id(&mut graph_tx.tx).await?;
        graph_tx.tx.commit().await?;

        // delete irises
        self.gpu_iris_store.rollback(0).await?;
        self.cpu_iris_store.rollback(0).await?;

        // clear modifications tables
        let mut tx = self.cpu_iris_store.tx().await?;
        self.cpu_iris_store
            .clear_modifications_table(&mut tx)
            .await?;
        tx.commit().await?;

        let mut tx = self.gpu_iris_store.tx().await?;
        self.gpu_iris_store
            .clear_modifications_table(&mut tx)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// Adds arbitrary irises to the database. The iris ID will be the new
    /// number of entries after the insertion
    async fn insert_into_gpu_iris_store(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        const SECRET_SHARING_PG_TX_SIZE: usize = 100;

        let mut tx = self.gpu_iris_store.tx().await?;
        let starting_len = self.gpu_iris_store.count_irises().await?;

        let chunks: Vec<Vec<_>> = shares
            .iter()
            .enumerate()
            .chunks(SECRET_SHARING_PG_TX_SIZE)
            .into_iter()
            .map(|chunk| chunk.collect())
            .collect();

        for batch in chunks.into_iter() {
            // use the idx as the id field
            let iris_refs: Vec<_> = batch
                .into_iter()
                .map(|(idx, (iris_l, iris_r))| StoredIrisRef {
                    // warning: id should be >= 1
                    id: (starting_len + idx + 1) as _,
                    left_code: &iris_l.code.coefs,
                    left_mask: &iris_l.mask.coefs,
                    right_code: &iris_r.code.coefs,
                    right_mask: &iris_r.mask.coefs,
                })
                .collect();

            self.gpu_iris_store
                .insert_irises(&mut tx, &iris_refs)
                .await?;
            tx.commit().await?;
            tx = self.gpu_iris_store.tx().await?;
        }

        Ok(())
    }
}
