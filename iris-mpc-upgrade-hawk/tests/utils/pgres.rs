use super::{
    constants::{COUNT_OF_PARTIES, PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2},
    types::NetConfig,
};
use iris_mpc_common::{
    config::Config as NodeConfig,
    postgres::{AccessMode, PostgresClient},
};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::Store as IrisStore;

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

/// Constructor.
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
}

/// Accessors.
impl NodeDbContext {
    pub fn graph_store(&self) -> &GraphStore<PlaintextStore> {
        &self.graph_store
    }

    pub fn iris_store(&self) -> &IrisStore {
        &self.iris_store
    }
}

/// Encapsulates API pointers to a database.
pub struct NodeDbProvider {
    /// Pointer to HNSW Graph store API.
    cpu_store: NodeDbContext,

    /// Pointer to Iris store API.
    gpu_store: NodeDbContext,

    /// Ordinal index of MPC party.
    party_idx: usize,
}

/// Constructor.
impl NodeDbProvider {
    pub fn new(party_idx: usize, cpu_store: NodeDbContext, gpu_store: NodeDbContext) -> Self {
        Self {
            cpu_store,
            gpu_store,
            party_idx,
        }
    }

    pub async fn new_from_config(config: &NodeConfig) -> Self {
        Self::new(
            config.party_id,
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

/// Accessors.
impl NodeDbProvider {
    pub fn cpu_store(&self) -> &NodeDbContext {
        &self.cpu_store
    }

    pub fn gpu_store(&self) -> &NodeDbContext {
        &self.gpu_store
    }

    pub fn party_idx(&self) -> usize {
        self.party_idx
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
}

/// Accessors.
impl NetDbProvider {
    pub fn of_node(&self, idx: usize) -> &NodeDbProvider {
        &self.nodes[idx]
    }
}
