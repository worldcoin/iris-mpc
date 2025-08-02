use std::collections::HashMap;

use super::{
    constants::{COUNT_OF_PARTIES, PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2},
    types::HawkConfigs,
};
use crate::utils::{store::construct_initial_genesis_state, GaloisRingSharedIrisPair};
use eyre::Result;
use futures::StreamExt;
use iris_mpc_common::{
    config::Config as NodeConfig,
    iris_db::iris::{IrisCode, IrisCodeArray},
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_cpu::{
    genesis::plaintext::{run_plaintext_genesis, GenesisArgs, GenesisConfig, GenesisState},
    hawkers::plaintext_store::PlaintextStore,
    hnsw::graph::graph_store::GraphPg as GraphStore,
    protocol::shared_iris::GaloisRingSharedIris,
};
use iris_mpc_store::{DbStoredIris, Store as IrisStore, StoredIrisRef};

/// Encapsulates API pointers to a database.
pub struct NodeDbProvider {
    pub gpu_iris_store: IrisStore,
    pub cpu_iris_store: IrisStore,
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

    pub async fn simulate_genesis(
        &self,
        genesis_config: GenesisConfig,
        genesis_args: GenesisArgs,
    ) -> Result<GenesisState> {
        todo!("convert from database iris shares to a IrisCode")
        /*let mut input = HashMap::new();
        let mut stream = self.cpu_iris_store.stream_irises().await;
        while let Some(iris) = stream.next().await {
            let iris: DbStoredIris = iris?;
            let left_iris = GaloisRingSharedIris::try_from_buffers_inner(
                0,
                iris.left_code(),
                iris.left_mask(),
            )?;

            let right_iris = GaloisRingSharedIris::try_from_buffers_inner(
                0,
                iris.right_code(),
                iris.right_mask(),
            )?;

            input.insert(
                iris.id() as u32,
                (
                    iris.version_id(),
                    IrisCode {
                        code: IrisCodeArray(left_iris.code.coefs),
                        mask: IrisCodeArray(left_iris.mask.coefs),
                    },
                    IrisCode {
                        code: IrisCodeArray(right_iris.code.coefs),
                        mask: IrisCodeArray(right_iris.mask.coefs),
                    },
                ),
            );
        }

        let genesis_state = construct_initial_genesis_state(genesis_config, genesis_args, input);
        let r = run_plaintext_genesis(genesis_state).await?;
        Ok(r)
        */
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

    pub fn into_iter(self) -> impl Iterator<Item = NodeDbProvider> {
        self.nodes.into_iter()
    }
}
