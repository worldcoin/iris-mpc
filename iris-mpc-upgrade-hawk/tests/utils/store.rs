use super::{
    constants::COUNT_OF_PARTIES, resources::read_iris_shares_batch, types::GaloisRingSharedIrisPair,
};
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::Store as IrisStore;
use itertools::{IntoChunks, Itertools};

/// Encapsulates information required to connect to a database.
pub struct DatabaseConnectionInfo {
    /// Connection schema.
    access_mode: AccessMode,

    /// Connection schema.
    schema: String,

    /// Connection URL.
    url: String,
}

/// Constructors.
impl DatabaseConnectionInfo {
    pub fn new(schema: String, url: String, access_mode: AccessMode) -> Self {
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
impl DatabaseConnectionInfo {
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

/// Persists Iris shares to remote databases.
///
/// # Arguments
///
/// * `batch_size` - Size of chunks to split Iris shares into.
/// * `rng_state` - State of an RNG being used to inject entropy to share creation.
/// * `skip_offset` - Number of Iris code pairs within ndjson file to skip.
/// * `max_items` - Maximum number of Iris code pairs to read.
///
/// # Returns
///
/// A chunked iterator over Iris shares.
///
pub async fn write_iris_shares_to_stores(
    _stores: Option<[&IrisStore; COUNT_OF_PARTIES]>,
    shares_batch_generator: IntoChunks<
        impl Iterator<Item = Box<[GaloisRingSharedIrisPair; COUNT_OF_PARTIES]>>,
    >,
) -> Result<()> {
    for (batch_idx, batch) in shares_batch_generator.into_iter().enumerate() {
        println!("batch-idx {}", batch_idx);
        for (shares_idx, shares) in batch.into_iter().enumerate() {
            println!("shares {} :: {}", shares_idx, shares.len());
            for (share_idx, _share) in shares.to_vec().iter().enumerate() {
                println!("share-idx {}", share_idx);
            }
        }
    }

    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::{read_iris_shares_batch, write_iris_shares_to_stores};

    #[tokio::test]
    async fn test_write_iris_shares_to_store() {
        let batch_size = 10;
        let rng_state = 93;
        let skip_offset = 0;
        let max_items = 100;

        let shares_batch_generator =
            read_iris_shares_batch(batch_size, rng_state, skip_offset, max_items).unwrap();
        write_iris_shares_to_stores(None, shares_batch_generator)
            .await
            .unwrap();
    }
}
