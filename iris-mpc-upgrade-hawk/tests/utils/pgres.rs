use super::{
    constants::{PARTY_COUNT, PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2},
    types::{NetConfig, NodeType, PartyIdx},
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
    schema_name: String,

    /// Connection URL.
    url: String,
}

/// Constructors.
impl DbConnectionInfo {
    fn new(config: &NodeConfig, schema_suffix: &String, access_mode: AccessMode) -> Self {
        Self {
            access_mode,
            schema_name: config.get_db_schema(schema_suffix),
            url: config.get_db_url(),
        }
    }

    #[allow(dead_code)]
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

    pub fn schema_name(&self) -> &String {
        &self.schema_name
    }

    pub fn url(&self) -> &String {
        &self.url
    }
}

/// Encapsulates API pointers to a database.
pub struct NodeDbContext {
    /// Pointer to HNSW Graph store API.
    #[allow(dead_code)]
    graph_store: GraphStore<PlaintextStore>,

    /// Pointer to Iris store API.
    iris_store: IrisStore,
}

/// Constructor.
impl NodeDbContext {
    pub async fn new(connection_info: DbConnectionInfo) -> Self {
        let client = PostgresClient::new(
            connection_info.url(),
            connection_info.schema_name(),
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    cpu_store: NodeDbContext,

    /// Pointer to Iris store API.
    gpu_store: NodeDbContext,
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
        // TODO:
        // Ensure gpu store is readonly also as this is the case in PROD.
        // There is an issue when running against a fresh dB as the migration
        // is not being executed if AccessMode=ReadOnly.
        Self::new(
            NodeDbContext::new(DbConnectionInfo::new_read_write(
                config,
                config.hnsw_schema_name_suffix(),
            ))
            .await,
            NodeDbContext::new(DbConnectionInfo::new_read_write(
                config,
                config.gpu_schema_name_suffix(),
            ))
            .await,
        )
    }
}

/// Accessors.
impl NodeDbProvider {
    fn store(&self, node_type: &NodeType) -> &NodeDbContext {
        match node_type {
            NodeType::CPU => &self.cpu_store,
            NodeType::GPU => &self.gpu_store,
        }
    }

    #[allow(dead_code)]
    pub fn graph_store(&self, node_type: &NodeType) -> &GraphStore<PlaintextStore> {
        self.store(node_type).graph_store()
    }

    pub fn iris_store(&self, node_type: &NodeType) -> &IrisStore {
        self.store(node_type).iris_store()
    }
}

/// Encapsulates API pointers to set of network databases.
pub struct NetDbProvider {
    /// Pointer to set of network node db providers.
    nodes: [NodeDbProvider; PARTY_COUNT],
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
    pub fn provider(&self, party_idx: PartyIdx) -> &NodeDbProvider {
        &self.nodes[party_idx]
    }

    pub fn providers(&self) -> &[NodeDbProvider; PARTY_COUNT] {
        &self.nodes
    }

    #[allow(dead_code)]
    pub fn graph_store(
        &self,
        party_idx: PartyIdx,
        node_type: &NodeType,
    ) -> &GraphStore<PlaintextStore> {
        self.provider(party_idx).store(node_type).graph_store()
    }

    #[allow(dead_code)]
    pub fn graph_stores(&self, node_type: &NodeType) -> Vec<&GraphStore<PlaintextStore>> {
        self.providers()
            .iter()
            .map(|provider| provider.graph_store(node_type))
            .collect()
    }

    pub fn iris_store(&self, party_idx: PartyIdx, node_type: &NodeType) -> &IrisStore {
        self.provider(party_idx).store(node_type).iris_store()
    }

    pub fn iris_stores(&self, node_type: &NodeType) -> Vec<&IrisStore> {
        self.providers()
            .iter()
            .map(|provider| provider.iris_store(node_type))
            .collect()
    }
}
