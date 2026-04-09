use std::sync::Arc;

use super::constants::COUNT_OF_PARTIES;
use crate::utils::{
    modifications::{self, ModificationInput},
    s3_deletions::get_aws_clients,
    GaloisRingSharedIrisPair, HawkConfigs,
};
use eyre::{bail, Result};
use iris_mpc_common::{
    config::Config,
    postgres::{AccessMode, PostgresClient},
    IrisSerialId, IrisVectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{BothEyes, StoreId},
    genesis::{
        genesis_checkpoint::{download_genesis_checkpoint, get_latest_checkpoint_state},
        state_accessor::{unset_last_indexed_iris_id, unset_last_indexed_modification_id},
    },
    hawkers::plaintext_store::{PlaintextStore, PlaintextVectorRef},
    hnsw::{graph::graph_store::GraphPg as GraphStore, GraphMem},
};
use iris_mpc_store::{Store, StoredIrisRef};
use itertools::Itertools;
use tokio::task::JoinSet;

// these constants were copied from genesis because genesis requires using an Aby3Store while the tests use a PlainTextStore
pub mod constants {
    /// Domain for persistent state store entry for last indexed id
    pub const STATE_DOMAIN: &str = "genesis";
    // Key for persistent state store entry for last indexed iris id
    pub const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";
    // Key for persistent state store entry for last indexed modification id
    pub const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "last_indexed_modification_id";
}

/// Struct holds database references for iris and graph store functionality
pub struct DbStores {
    pub iris: Store,
    pub graph: GraphStore<PlaintextStore>,
}

impl DbStores {
    pub async fn new(url: &str, schema_name: &str, access_mode: AccessMode) -> Self {
        let client = PostgresClient::new(url, schema_name, access_mode)
            .await
            .unwrap();
        client.migrate().await;
        let iris = Store::new(&client).await.unwrap();
        let graph = GraphStore::new(&client).await.unwrap();

        Self { iris, graph }
    }
}

/// simulates a MPC node, complete with configuration (HAWK and Genesis) and database connections.
/// a simulation consists of 3 MPC nodes; see MpcNodes
pub struct MpcNode {
    // databases
    pub gpu_stores: DbStores,
    pub cpu_stores: DbStores,
}

/// Simulates a 3 party multi party computation. Is intended to be built from a list of configurations in a way that
/// allows the MpcNode instances to be passed to async move closures, for concurrent tasks.
pub struct MpcNodes {
    nodes: [Arc<MpcNode>; COUNT_OF_PARTIES],
}

impl MpcNodes {
    pub async fn new(configs: &HawkConfigs) -> Self {
        Self {
            nodes: [
                Arc::new(MpcNode::new(configs[0].clone()).await),
                Arc::new(MpcNode::new(configs[1].clone()).await),
                Arc::new(MpcNode::new(configs[2].clone()).await),
            ],
        }
    }

    pub async fn apply_assertions(&self, gpu_asserts: DbAssertions, cpu_asserts: DbAssertions) {
        let gpu_asserts = Arc::new(gpu_asserts);
        let cpu_asserts = Arc::new(cpu_asserts);

        let mut join_set = JoinSet::new();
        for node in self.nodes.iter().cloned() {
            let gpu_asserts = gpu_asserts.clone();
            let cpu_asserts = cpu_asserts.clone();
            join_set.spawn(async move {
                gpu_asserts.assert(&node.gpu_stores).await.unwrap();
                cpu_asserts.assert(&node.cpu_stores).await.unwrap();
            });
        }
        join_set.join_all().await;
    }

    /// Asserts that the S3 checkpoint graphs match the expected graphs for all nodes.
    pub async fn assert_s3_checkpoint_graphs(
        &self,
        configs: &HawkConfigs,
        expected_graphs: &BothEyes<GraphMem<PlaintextVectorRef>>,
    ) -> Result<()> {
        for (i, (node, config)) in self.nodes.iter().zip(configs.iter()).enumerate() {
            let aws_clients = get_aws_clients(config).await?;
            let checkpoint_state = get_latest_checkpoint_state(&node.cpu_stores.graph)
                .await?
                .ok_or_else(|| eyre::eyre!("No checkpoint found for node {}", i))?;
            let s3_graphs: BothEyes<GraphMem<PlaintextVectorRef>> = download_genesis_checkpoint(
                &aws_clients.s3_client,
                &config.graph_checkpoint_bucket_name,
                checkpoint_state,
            )
            .await?;

            assert_eq!(
                s3_graphs[0], expected_graphs[0],
                "Left graph mismatch for node {}",
                i
            );
            assert_eq!(
                s3_graphs[1], expected_graphs[1],
                "Right graph mismatch for node {}",
                i
            );
        }

        Ok(())
    }
}

impl IntoIterator for MpcNodes {
    type Item = Arc<MpcNode>;
    type IntoIter = std::array::IntoIter<Arc<MpcNode>, COUNT_OF_PARTIES>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.into_iter()
    }
}

// entry points
impl MpcNode {
    pub async fn new(config: Config) -> Self {
        let gpu_stores = DbStores::new(
            &config.get_gpu_db_url().unwrap(),
            &config.get_gpu_db_schema(),
            AccessMode::ReadWrite,
        )
        .await;

        let cpu_stores = DbStores::new(
            &config.get_cpu_db_url().unwrap(),
            &config.get_cpu_db_schema(),
            AccessMode::ReadWrite,
        )
        .await;

        Self {
            gpu_stores,
            cpu_stores,
        }
    }

    /// clear all tables and insert irises into the GPU database
    pub async fn init_tables(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        self.clear_all_tables().await?;
        self.insert_into_gpu_iris_store(shares).await?;
        Ok(())
    }
}

// misc
impl MpcNode {
    pub async fn apply_modifications(
        &self,
        last_mods: &[ModificationInput],
        cur_mods: &[ModificationInput],
    ) -> Result<()> {
        if !modifications::modifications_extension_is_valid(last_mods, cur_mods) {
            bail!("Specified modifications are not a valid extension of the last modifications state.")
        }

        let mut tx = self.gpu_stores.iris.tx().await?;

        let mods: Vec<_> = cur_mods.iter().cloned().map(|m| m.into()).collect();
        for m in mods.iter() {
            db_ops::write_modification(&mut tx, m).await?;
        }

        let update_serial_ids = modifications::modifications_extension_updates(last_mods, cur_mods);
        for serial_id in update_serial_ids {
            db_ops::increment_iris_version(&mut tx, serial_id).await?;
        }

        tx.commit().await?;

        Ok(())
    }

    #[allow(dead_code)]
    pub async fn increment_iris_version(&self, serial_id: i64) -> Result<()> {
        let mut tx = self.gpu_stores.iris.tx().await?;
        db_ops::increment_iris_version(&mut tx, serial_id).await?;
        tx.commit().await?;

        Ok(())
    }

    #[allow(dead_code)]
    pub async fn persist_modification(&self, id: i64) -> Result<()> {
        let mut tx = self.gpu_stores.iris.tx().await?;
        db_ops::persist_modification(&mut tx, id).await?;
        tx.commit().await?;

        Ok(())
    }

    /// Insert additional irises into the CPU iris store and update the persistent state
    /// to indicate those irises were indexed. This simulates a scenario where the CPU
    /// database thinks it indexed more irises than what's in the S3 checkpoint.
    pub async fn insert_extra_irises_into_cpu_store(
        &self,
        starting_id: usize,
        count: usize,
    ) -> Result<()> {
        use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};

        // Create dummy iris data
        let dummy_code = vec![0u16; IRIS_CODE_LENGTH];
        let dummy_mask = vec![0u16; MASK_CODE_LENGTH];

        let (irises, vector_ids): (Vec<StoredIrisRef>, Vec<IrisVectorId>) = (200..=count + 200)
            .map(|i| {
                let iris_id = (starting_id + i);
                (
                    StoredIrisRef {
                        id: iris_id as i64,
                        left_code: &dummy_code,
                        left_mask: &dummy_mask,
                        right_code: &dummy_code,
                        right_mask: &dummy_mask,
                    },
                    IrisVectorId::new(iris_id as u32, 500),
                )
            })
            .collect();

        // Get a transaction from the graph store (which gives us access to postgres tx)
        let graph_tx = self.cpu_stores.graph.tx().await?;
        let mut tx = graph_tx.tx;

        // Insert irises into CPU store
        self.cpu_stores
            .iris
            .insert_copy_irises(&mut tx, &vector_ids, &irises)
            .await?;

        // Update the persistent state to indicate we indexed these irises
        let new_last_indexed_id = (starting_id + count) as u32;
        GraphStore::<PlaintextStore>::set_persistent_state(
            &mut tx,
            constants::STATE_DOMAIN,
            constants::STATE_KEY_LAST_INDEXED_IRIS_ID,
            &new_last_indexed_id,
        )
        .await?;

        tx.commit().await?;

        Ok(())
    }

    async fn clear_all_tables(&self) -> Result<()> {
        for stores in [&self.gpu_stores, &self.cpu_stores] {
            // delete irises
            stores.iris.rollback(0).await?;

            let mut graph_tx = stores.graph.tx().await?;

            // clear graphs
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

            let mut tx = graph_tx.tx;

            // clear modifications tables
            stores.iris.clear_modifications_table(&mut tx).await?;

            // clear persistent state
            unset_last_indexed_iris_id(&mut tx).await?;
            unset_last_indexed_modification_id(&mut tx).await?;

            // clear genesis graph checkpoint table
            sqlx::query("DELETE FROM genesis_graph_checkpoint")
                .execute(&mut *tx)
                .await?;

            tx.commit().await?;
        }

        Ok(())
    }

    /// Adds arbitrary irises to the database. The iris ID will be the new
    /// number of entries after the insertion
    async fn insert_into_gpu_iris_store(&self, shares: &[GaloisRingSharedIrisPair]) -> Result<()> {
        const SECRET_SHARING_PG_TX_SIZE: usize = 100;

        let mut tx = self.gpu_stores.iris.tx().await?;
        let starting_len = self.gpu_stores.iris.count_irises().await?;

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

            self.gpu_stores
                .iris
                .insert_irises(&mut tx, &iris_refs)
                .await?;
            tx.commit().await?;
            tx = self.gpu_stores.iris.tx().await?;
        }

        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct DbAssertions {
    pub num_irises: Option<usize>,
    pub vector_ids: Option<Vec<IrisVectorId>>,
    pub num_modifications: Option<usize>,
    pub last_indexed_iris_id: Option<IrisSerialId>,
    pub last_indexed_modification_id: Option<i64>,
}

impl DbAssertions {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn assert_num_irises(mut self, num: usize) -> Self {
        self.num_irises = Some(num);
        self
    }

    pub fn assert_vector_ids(mut self, vector_ids: Vec<IrisVectorId>) -> Self {
        self.vector_ids = Some(vector_ids);
        self
    }

    pub fn assert_num_modifications(mut self, num: usize) -> Self {
        self.num_modifications = Some(num);
        self
    }

    pub fn assert_last_indexed_iris_id(mut self, id: IrisSerialId) -> Self {
        self.last_indexed_iris_id = Some(id);
        self
    }

    pub fn assert_last_indexed_modification_id(mut self, id: i64) -> Self {
        self.last_indexed_modification_id = Some(id);
        self
    }

    pub async fn assert(&self, stores: &DbStores) -> Result<()> {
        if let Some(num_irises) = self.num_irises {
            let store_num_irises = stores.iris.count_irises().await?;
            assert_eq!(num_irises, store_num_irises);
        }

        if let Some(vector_ids) = &self.vector_ids {
            let store_vector_ids = db_ops::get_iris_vector_ids(&stores.iris).await?;
            assert_eq!(store_vector_ids, *vector_ids);
        }

        if let Some(num_modifications) = self.num_modifications {
            let store_num_modifications = stores.iris.last_modifications(1000).await?.len();
            assert_eq!(store_num_modifications, num_modifications);
        }

        if let Some(last_indexed_iris_id) = self.last_indexed_iris_id {
            let store_last_indexed_iris_id: u32 = stores
                .graph
                .get_persistent_state(
                    constants::STATE_DOMAIN,
                    constants::STATE_KEY_LAST_INDEXED_IRIS_ID,
                )
                .await?
                .unwrap_or(0);
            assert_eq!(store_last_indexed_iris_id, last_indexed_iris_id);
        }

        if let Some(last_indexed_modification_id) = self.last_indexed_modification_id {
            let store_last_indexed_modification_id: i64 = stores
                .graph
                .get_persistent_state(
                    constants::STATE_DOMAIN,
                    constants::STATE_KEY_LAST_INDEXED_MODIFICATION_ID,
                )
                .await?
                .unwrap_or_default();
            assert_eq!(
                store_last_indexed_modification_id,
                last_indexed_modification_id
            );
        }

        Ok(())
    }
}

mod db_ops {
    use std::ops::DerefMut;

    use eyre::Result;
    use iris_mpc_common::{helpers::sync::Modification, IrisVectorId};
    use iris_mpc_store::Store;
    use sqlx::{Postgres, Transaction};

    /// Test functionality which updates an iris only by incrementing its version,
    /// without changing the underlying iris code.
    pub async fn increment_iris_version(
        tx: &mut Transaction<'_, Postgres>,
        serial_id: i64,
    ) -> Result<()> {
        let query = sqlx::query(
            r#"
            UPDATE irises SET version_id = version_id + 1
            WHERE id = $1;
            "#,
        )
        .bind(serial_id);
        query.execute(tx.deref_mut()).await?;

        Ok(())
    }

    pub async fn get_iris_vector_ids(store: &Store) -> Result<Vec<IrisVectorId>> {
        let ids: Vec<(i64, i16)> = sqlx::query_as(
            r#"
            SELECT
                id,
                version_id
            FROM irises
            ORDER BY id ASC;
            "#,
        )
        .fetch_all(&store.pool)
        .await?;

        let ids = ids
            .into_iter()
            .map(|(serial_id, version)| IrisVectorId::new(serial_id as u32, version))
            .collect();

        Ok(ids)
    }

    pub async fn persist_modification(
        tx: &mut Transaction<'_, Postgres>,
        modification_id: i64,
    ) -> Result<()> {
        let query = sqlx::query(
            r#"
            UPDATE modifications SET status = 'COMPLETED', persisted = true
            WHERE id = $1;
            "#,
        )
        .bind(modification_id);
        query.execute(tx.deref_mut()).await?;

        Ok(())
    }

    /// Writes a modification to the modifications table, overwriting fields if the specified
    /// modification id already exists.
    pub async fn write_modification(
        tx: &mut Transaction<'_, Postgres>,
        m: &Modification,
    ) -> Result<()> {
        let query = sqlx::query(
            r#"
            INSERT INTO modifications (id, serial_id, request_type, s3_url, status, persisted)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO UPDATE
            SET serial_id = EXCLUDED.serial_id,
                request_type = EXCLUDED.request_type,
                s3_url = EXCLUDED.s3_url,
                status = EXCLUDED.status,
                persisted = EXCLUDED.persisted;
            "#,
        )
        .bind(m.id)
        .bind(m.serial_id)
        .bind(m.request_type.as_str())
        .bind(m.s3_url.as_ref())
        .bind(m.status.as_str())
        .bind(m.persisted);
        query.execute(tx.deref_mut()).await?;

        Ok(())
    }
}
