use super::{
    constants::{COUNT_OF_PARTIES, PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2},
    types::HawkConfigs,
};
use crate::utils::GaloisRingSharedIrisPair;
use eyre::Result;
use iris_mpc_common::{
    config::Config as NodeConfig,
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::{Store as IrisStore, StoredIrisRef};

/// Encapsulates API pointers to a database.
pub struct NodeDbProvider {
    pub gpu_iris_store: IrisStore,
    pub cpu_iris_store: IrisStore,
    // todo: wrap this in a BothEyes
    pub graph_store: GraphStore<PlaintextStore>,
}

impl NodeDbProvider {
    pub async fn new_from_config(config: &NodeConfig) -> Self {
        let cpu_client = PostgresClient::new(
            &config.get_db_url(),
            &config.get_db_schema(config.hnsw_schema_name_suffix()),
            AccessMode::ReadWrite,
        )
        .await
        .unwrap();

        let gpu_client = PostgresClient::new(
            &config.get_db_url(),
            &config.get_db_schema(config.gpu_schema_name_suffix()),
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

    pub async fn clear_all_tables(&self) -> Result<()> {
        // only the cpu database uses the graph store.
        self.graph_store.clear_hawk_graph_tables().await?;

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
    pub async fn insert_gpu_iris_store(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        let starting_len = self.gpu_iris_store.count_irises().await?;

        let mut tx = self.gpu_iris_store.tx().await?;
        // use the idx as the id field
        let mut to_insert = vec![];
        for (idx, shares) in shares.iter().enumerate() {
            to_insert.push(StoredIrisRef {
                // warning: id should be >= 1
                id: (starting_len + idx + 1) as _,
                left_code: &shares.0.code.coefs,
                left_mask: &shares.0.mask.coefs,
                right_code: &shares.1.code.coefs,
                right_mask: &shares.1.mask.coefs,
            })
        }

        self.gpu_iris_store
            .insert_irises(&mut tx, &to_insert)
            .await?;
        tx.commit().await?;
        Ok(())
    }

    /// clears the cpu iris store and grah table, and inserts into the gpu iris store
    pub async fn init_iris_stores(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        self.clear_all_tables().await?;
        self.insert_gpu_iris_store(shares).await?;
        Ok(())
    }
}

/// Encapsulates API pointers to set of network databases.
pub struct NetDbProvider {
    /// Pointer to set of network node db providers.
    nodes: [NodeDbProvider; COUNT_OF_PARTIES],
}

/// Constructor.
impl NetDbProvider {
    pub fn new(node_0: NodeDbProvider, node_1: NodeDbProvider, node_2: NodeDbProvider) -> Self {
        Self {
            nodes: [node_0, node_1, node_2],
        }
    }

    pub async fn new_from_config(config: &HawkConfigs) -> Self {
        Self::new(
            NodeDbProvider::new_from_config(&config[PARTY_IDX_0]).await,
            NodeDbProvider::new_from_config(&config[PARTY_IDX_1]).await,
            NodeDbProvider::new_from_config(&config[PARTY_IDX_2]).await,
        )
    }

    /// Returns an iterator over references to the node database providers.
    pub fn iter(&self) -> impl Iterator<Item = &NodeDbProvider> {
        self.nodes.iter()
    }
}
