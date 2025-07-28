use super::types::IrisCodePair;
use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use iris_mpc_cpu::{
    hawkers::plaintext_store::PlaintextStore, hnsw::graph::graph_store::GraphPg as GraphStore,
};
use iris_mpc_store::Store as IrisStore;
use itertools::IntoChunks;

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

pub async fn write_iris_shares_to_store(
    batch_size: usize,
    iris_shares_rnd: u64,
    iris_serial_id_max: usize,
    iris_code_pair_batch_stream: IntoChunks<impl Iterator<Item = IrisCodePair>>,
) -> Result<()> {
    unimplemented!()
    // let mut n_iris_code_pair_read: usize = 0;
    // let mut batch: Vec<Vec<(GaloisRingSharedIris, GaloisRingSharedIris)>> = (0..COUNT_OF_PARTIES)
    //     .map(|_| Vec::with_capacity(batch_size))
    //     .collect();

    // for (iris_code_pair_batch_idx, iris_code_pair_batch_iter) in
    //     iris_code_pair_batch_stream.into_iter().enumerate()
    // {
    //     let iris_code_pair_batch: Vec<(_, _)> = iris_code_pair_batch_iter.collect();
    //     n_iris_code_pair_read += iris_code_pair_batch.len();

    //     for (left, right) in iris_code_pair_batch {
    //         // Reset RNG for each pair to match shares_encoding.rs behavior
    //         let mut shares_seed = StdRng::seed_from_u64(iris_shares_rnd);

    //         let left_shares =
    //             GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, left.clone());
    //         let right_shares =
    //             GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, right.clone());
    //         for (party, (shares_l, shares_r)) in izip!(left_shares, right_shares).enumerate() {
    //             batch[party].push((shares_l, shares_r));
    //         }
    //     }

    //     let cur_batch_len = batch[0].len();
    //     let iris_serial_id_last =
    //         (iris_code_pair_batch_idx * batch_size) + cur_batch_len + iris_serial_id_max;

    //     for (db, shares) in izip!(&dbs, batch.iter_mut()) {
    //         #[allow(clippy::drain_collect)]
    //         let (_, iris_serial_id_last_persisted) =
    //             db.persist_vector_shares(shares.drain(..).collect()).await?;
    //         assert_eq!(iris_serial_id_last_persisted, iris_serial_id_last);
    //     }

    //     logger::log_info(
    //         "Store",
    //         format!(
    //             "Persisted {} locally generated shares",
    //             iris_serial_id_last - iris_serial_id_max
    //         )
    //         .as_str(),
    //     );
    // }

    // Ok(())
}
