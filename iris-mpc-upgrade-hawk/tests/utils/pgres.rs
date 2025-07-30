use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::Store as IrisStore;

use crate::utils::constants::COUNT_OF_PARTIES;

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
    fn new(schema: String, url: String, access_mode: AccessMode) -> Self {
        Self {
            access_mode,
            schema,
            url,
        }
    }

    pub fn new_read_only(schema: String, url: String) -> Self {
        Self::new(schema, url, AccessMode::ReadOnly)
    }

    pub fn new_read_write(schema: String, url: String) -> Self {
        Self::new(schema, url, AccessMode::ReadWrite)
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
pub struct NodeDbStore {
    /// Pointer to HNSW Graph store API.
    graph: GraphStore<PlaintextStore>,

    /// Pointer to Iris store API.
    iris: IrisStore,
}

/// Constructor.
impl NodeDbStore {
    pub async fn new(connection_info: DbConnectionInfo) -> Self {
        let client = PostgresClient::new(
            connection_info.url(),
            connection_info.schema(),
            connection_info.access_mode(),
        )
        .await
        .unwrap();

        Self {
            iris: IrisStore::new(&client).await.unwrap(),
            graph: GraphStore::new(&client).await.unwrap(),
        }
    }
}

/// Accessors.
impl NodeDbStore {
    pub fn graph(&self) -> &GraphStore<PlaintextStore> {
        &self.graph
    }

    pub fn iris(&self) -> &IrisStore {
        &self.iris
    }
}

/// Encapsulates API pointers to a database.
pub struct NodeDbProvider {
    /// Pointer to HNSW Graph store API.
    cpu_store: NodeDbStore,

    /// Pointer to Iris store API.
    gpu_store: NodeDbStore,

    /// Ordinal index of MPC party.
    party_idx: usize,
}

/// Constructor.
impl NodeDbProvider {
    pub fn new(party_idx: usize, cpu_store: NodeDbStore, gpu_store: NodeDbStore) -> Self {
        Self {
            cpu_store,
            gpu_store,
            party_idx,
        }
    }
}

/// Accessors.
impl NodeDbProvider {
    pub fn cpu_store(&self) -> &NodeDbStore {
        &self.cpu_store
    }

    pub fn gpu_store(&self) -> &NodeDbStore {
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
}

/// Accessors.
impl NetDbProvider {
    pub fn node(&self, idx: usize) -> &NodeDbProvider {
        &self.nodes[idx]
    }
}
