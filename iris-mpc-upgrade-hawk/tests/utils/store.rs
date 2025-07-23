use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::Store as IrisStore;

// pub async fn new(url: &str, schema: &str) -> Self {
//     let client = PostgresClient::new(url, schema, AccessMode::ReadWrite)
//         .await
//         .unwrap();
//     let store = Store::new(&client).await.unwrap();
//     let graph_pg = GraphPg::new(&client).await.unwrap();
//     Self { store, graph_pg }
// }

/// Encapsulates information required to connect to a database.
struct DatabaseConnectionInfo {
    /// Connection schema.
    db_schema: String,

    /// Connection URL.
    db_url: String,
}

/// Accessors.
impl DatabaseConnectionInfo {
    pub fn db_schema(&self) -> &String {
        &self.db_schema
    }

    pub fn db_url(&self) -> &String {
        &self.db_url
    }
}

/// Encapsulates API pointers to a database.
pub struct DatabaseContext {
    /// Pointer to HNSW Graph store API.
    graph_store: GraphStore<PlaintextStore>,

    /// Pointer to Iris store API.
    iris_store: IrisStore,
}

/// Accessors.
impl DatabaseContext {
    pub fn graph_store(&self) -> &GraphStore<PlaintextStore> {
        &self.graph_store
    }

    pub fn iris_store(&self) -> &IrisStore {
        &self.iris_store
    }
}

/// Constructor.
impl DatabaseContext {
    pub async fn new(connection_info: DatabaseConnectionInfo) -> Self {
        let client = PostgresClient::new(
            connection_info.db_url(),
            connection_info.db_schema(),
            AccessMode::ReadWrite,
        )
        .await
        .unwrap();

        Self {
            iris_store: IrisStore::new(&client).await.unwrap(),
            graph_store: GraphStore::new(&client).await.unwrap(),
        }
    }
}
