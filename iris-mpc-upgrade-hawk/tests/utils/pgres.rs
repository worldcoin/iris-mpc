use super::{
    constants::{COUNT_OF_PARTIES, PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2},
    types::NetConfig,
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

/// Encapsulates information required to connect to a database.
pub struct DbConnectionInfo {
    /// Connection schema.
    access_mode: AccessMode,

    /// Connection schema.
    schema: String,

    /// Connection URL.
    url: String,
}

/// Constructors.
impl DbConnectionInfo {
    fn new(config: &NodeConfig, schema_suffix: &String, access_mode: AccessMode) -> Self {
        Self {
            access_mode,
            schema: config.get_db_schema(schema_suffix),
            url: config.get_db_url(),
        }
    }

    pub fn new_read_only(config: &NodeConfig, schema_suffix: &String) -> Self {
        Self::new(config, schema_suffix, AccessMode::ReadOnly)
    }

    pub fn new_read_write(config: &NodeConfig, schema_suffix: &String) -> Self {
        Self::new(config, schema_suffix, AccessMode::ReadWrite)
    }
}

/// Accessors.
impl DbConnectionInfo {
    pub fn access_mode(&self) -> AccessMode {
        self.access_mode
    }

    pub fn schema(&self) -> &String {
        &self.schema
    }

    pub fn url(&self) -> &String {
        &self.url
    }
}

/// Encapsulates API pointers to a database.
pub struct NodeDbContext {
    /// Pointer to HNSW Graph store API.
    graph_store: GraphStore<PlaintextStore>,

    /// Pointer to Iris store API.
    iris_store: IrisStore,
}

impl NodeDbContext {
    pub async fn new(connection_info: DbConnectionInfo) -> Self {
        let client = PostgresClient::new(
            connection_info.url(),
            connection_info.schema(),
            connection_info.access_mode(),
        )
        .await
        .unwrap();

        Self {
            iris_store: IrisStore::new(&client).await.unwrap(),
            graph_store: GraphStore::new(&client).await.unwrap(),
        }
    }

    pub fn graph_store(&self) -> &GraphStore<PlaintextStore> {
        &self.graph_store
    }

    pub fn iris_store(&self) -> &IrisStore {
        &self.iris_store
    }

    /// Initializes the iris database with the given slice of pairs.
    pub async fn init_iris_db(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        self.clear_all_tables().await?;
        self.insert_irises(shares).await?;
        Ok(())
    }

    /// Adds arbitrary irises to the database. The iris ID will be the new
    /// number of entries after the insertion
    pub async fn insert_irises(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        let starting_len = self.iris_store.count_irises().await?;

        let mut tx = self.iris_store.tx().await?;
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

        self.iris_store.insert_irises(&mut tx, &to_insert).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn clear_all_tables(&self) -> Result<()> {
        self.graph_store.clear_hawk_graph_tables().await?;
        self.iris_store.rollback(0).await?;
        Ok(())
    }
}

/// Encapsulates API pointers to a database.
pub struct NodeDbProvider {
    /// Pointer to HNSW Graph store API.
    pub cpu_store: NodeDbContext,

    /// Pointer to Iris store API.
    /// Note that the gpu_store doesn't use its graph_store
    pub gpu_store: NodeDbContext,
}

/// Constructor.
impl NodeDbProvider {
    pub fn new(cpu_store: NodeDbContext, gpu_store: NodeDbContext) -> Self {
        Self {
            cpu_store,
            gpu_store,
        }
    }

    pub async fn new_from_config(config: &NodeConfig) -> Self {
        Self::new(
            NodeDbContext::new(DbConnectionInfo::new_read_write(
                config,
                config.hnsw_schema_name_suffix(),
            ))
            .await,
            NodeDbContext::new(DbConnectionInfo::new_read_only(
                config,
                config.gpu_schema_name_suffix(),
            ))
            .await,
        )
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

    pub async fn new_from_config(config: &NetConfig) -> Self {
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
