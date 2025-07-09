use std::sync::Arc;

use crate::hnsw::{
    metrics::ops_counter::Operation::{CompareDistance, EvaluateDistance},
    vector_store::VectorStoreMut,
    GraphMem, HnswSearcher, VectorStore,
};
use aes_prng::AesRng;
use iris_mpc_common::{
    iris_db::{
        db::IrisDB,
        iris::{IrisCode, MATCH_THRESHOLD_RATIO},
    },
    vector_id::SerialId,
};
use rand::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::debug;

use super::aby3::aby3_store::VectorId;
use eyre::{bail, Result};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct IrisCodeWithSerialId {
    pub iris_code: IrisCode,
    pub serial_id: SerialId,
}

/// Vector store which works over plaintext iris codes and distance computations.
///
/// This variant is only suitable for single-threaded operation.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlaintextStore {
    pub points: HashMap<SerialId, IrisCode>,
}

impl PlaintextStore {
    /// Generate a new empty `PlaintextStore`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a new `PlaintextStore` of specified size with random entries.
    pub fn new_random<R: RngCore + Clone + CryptoRng>(rng: &mut R, store_size: usize) -> Self {
        let points_codes = IrisDB::new_random_rng(store_size, rng).db;
        let points = points_codes
            .into_iter()
            .enumerate()
            .map(|(idx, iris)| (idx as u32 + 1, iris))
            .collect::<HashMap<u32, IrisCode>>();
        Self { points }
    }

    /// Generate an HNSW graph over the first `graph_size` entries of this
    /// `PlaintextStore`, using the specified searcher and randomness.
    pub async fn generate_graph<R: RngCore + Clone + CryptoRng>(
        &mut self,
        rng: &mut R,
        graph_size: usize,
        searcher: &HnswSearcher,
    ) -> Result<GraphMem<Self>> {
        let mut graph = GraphMem::new();
        let mut rng = AesRng::from_rng(rng.clone())?;

        if graph_size > self.points.len() {
            bail!("Cannot generate graph larger than underlying vector store");
        }

        // sort in order to ensure deterministic behavior
        let mut keys: Vec<_> = self.points.keys().cloned().collect();
        keys.sort();
        keys.truncate(graph_size);

        for key in keys {
            let iris_code = self.points[&key].clone();
            let query_id = VectorId::from_serial_id(key);
            let insertion_layer = searcher.select_layer_rng(&mut rng)?;
            let query = IrisCodeWithSerialId {
                iris_code,
                serial_id: key,
            };
            let (neighbors, set_ep) = searcher
                .search_to_insert(self, &graph, &query, insertion_layer)
                .await?;
            searcher
                .insert_from_search_results(self, &mut graph, query_id, neighbors, set_ep)
                .await?;
        }

        Ok(graph)
    }
}

fn fraction_is_match(dist: &(u16, u16)) -> bool {
    let (a, b) = *dist;
    (a as f64) < (b as f64) * MATCH_THRESHOLD_RATIO
}

fn fraction_less_than(dist_1: &(u16, u16), dist_2: &(u16, u16)) -> bool {
    let (a, b) = *dist_1; // a/b
    let (c, d) = *dist_2; // c/d
    (a as u32) * (d as u32) < (b as u32) * (c as u32)
}

impl VectorStore for PlaintextStore {
    type QueryRef = IrisCodeWithSerialId;
    type VectorRef = VectorId;
    type DistanceRef = (u16, u16);

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        vectors
            .iter()
            .map(|id| IrisCodeWithSerialId {
                iris_code: self.points.get(&id.serial_id()).unwrap().clone(),
                serial_id: id.serial_id(),
            })
            .collect()
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        debug!(event_type = EvaluateDistance.id());
        let vector_code = &self
            .points
            .get(&vector.serial_id())
            .ok_or_else(|| eyre::eyre!("Vector ID not found in store"))?;
        Ok(query.iris_code.get_distance_fraction(vector_code))
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(fraction_is_match(distance))
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(fraction_less_than(distance1, distance2))
    }
}

impl VectorStoreMut for PlaintextStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.points.insert(query.serial_id, query.iris_code.clone());
        VectorId::from_serial_id(query.serial_id)
    }
}

/// PlaintextStore with synchronization primitives for multithreaded use.
#[derive(Default, Debug, Clone)]
pub struct SharedPlaintextStore {
    pub points: Arc<RwLock<HashMap<u32, IrisCode>>>,
}

impl SharedPlaintextStore {
    pub fn new() -> Self {
        Default::default()
    }
}

impl From<PlaintextStore> for SharedPlaintextStore {
    fn from(value: PlaintextStore) -> Self {
        Self {
            points: Arc::new(RwLock::new(value.points)),
        }
    }
}

impl VectorStore for SharedPlaintextStore {
    type QueryRef = IrisCodeWithSerialId;
    type VectorRef = VectorId;
    type DistanceRef = (u16, u16);

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        let store = self.points.read().await;
        vectors
            .iter()
            .map(|id| IrisCodeWithSerialId {
                iris_code: store.get(&id.serial_id()).unwrap().clone(),
                serial_id: id.serial_id(),
            })
            .collect()
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        debug!(event_type = EvaluateDistance.id());
        let store = self.points.read().await;
        let vector_code = store
            .get(&vector.serial_id())
            .ok_or_else(|| eyre::eyre!("Vector ID not found in store"))?;
        Ok(query.iris_code.get_distance_fraction(vector_code))
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(fraction_is_match(distance))
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(fraction_less_than(distance1, distance2))
    }
}

impl VectorStoreMut for SharedPlaintextStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        let mut store = self.points.write().await;
        store.insert(query.serial_id, query.iris_code.clone());
        VectorId::from_serial_id(query.serial_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{graph::layered_graph::migrate, HnswSearcher};
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tokio::task::JoinSet;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_basic_ops() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut store = PlaintextStore::new();

        let db_code = IrisDB::new_random_rng(10, &mut rng).db;

        let db = db_code
            .iter()
            .enumerate()
            .map(|(idx, iris)| IrisCodeWithSerialId {
                iris_code: iris.clone(),
                serial_id: idx as u32 + 1,
            })
            .collect::<Vec<_>>();

        let mut ids = Vec::new();
        for q in db.iter() {
            ids.push(store.insert(q).await);
        }

        let d01 = store.eval_distance(&db[0], &ids[1]).await?;
        let d02 = store.eval_distance(&db[0], &ids[2]).await?;
        let d03 = store.eval_distance(&db[0], &ids[3]).await?;
        let d10 = store.eval_distance(&db[1], &ids[0]).await?;
        let d12 = store.eval_distance(&db[1], &ids[2]).await?;
        let d13 = store.eval_distance(&db[1], &ids[3]).await?;
        let d23 = store.eval_distance(&db[2], &ids[3]).await?;
        let d30 = store.eval_distance(&db[3], &ids[0]).await?;

        assert_eq!(
            store.less_than(&d01, &d23).await?,
            db[0].iris_code.get_distance(&db[1].iris_code)
                < db[2].iris_code.get_distance(&db[3].iris_code)
        );

        assert_eq!(
            store.less_than(&d23, &d01).await?,
            db[2].iris_code.get_distance(&db[3].iris_code)
                < db[0].iris_code.get_distance(&db[1].iris_code)
        );

        assert_eq!(
            store.less_than(&d02, &d13).await?,
            db[0].iris_code.get_distance(&db[2].iris_code)
                < db[1].iris_code.get_distance(&db[3].iris_code)
        );

        assert_eq!(
            store.less_than(&d03, &d12).await?,
            db[0].iris_code.get_distance(&db[3].iris_code)
                < db[1].iris_code.get_distance(&db[2].iris_code)
        );

        assert_eq!(
            store.less_than(&d10, &d23).await?,
            db[1].iris_code.get_distance(&db[0].iris_code)
                < db[2].iris_code.get_distance(&db[3].iris_code)
        );

        assert_eq!(
            store.less_than(&d12, &d30).await?,
            db[1].iris_code.get_distance(&db[2].iris_code)
                < db[3].iris_code.get_distance(&db[0].iris_code)
        );

        assert_eq!(
            store.less_than(&d02, &d01).await?,
            db[0].iris_code.get_distance(&db[2].iris_code)
                < db[0].iris_code.get_distance(&db[1].iris_code)
        );

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_plaintext_hnsw_matcher() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 1;
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut ptxt_vector = PlaintextStore::new_random(&mut rng, database_size);
        let ptxt_graph = ptxt_vector
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;
        for i in 0..database_size {
            let query = ptxt_vector.points.get(&(i as u32 + 1)).unwrap().clone();
            let query = IrisCodeWithSerialId {
                iris_code: query.clone(),
                serial_id: i as u32 + 1,
            };
            let cleartext_neighbors = searcher
                .search(&mut ptxt_vector, &ptxt_graph, &query, 1)
                .await?;
            assert!(
                searcher
                    .is_match(&mut ptxt_vector, &[cleartext_neighbors])
                    .await?,
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_parallel_plaintext_hnsw_matcher() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 16;

        let searcher = HnswSearcher::new_with_test_parameters();
        let mut ptxt_vector = PlaintextStore::new_random(&mut rng, database_size);
        let ptxt_graph = ptxt_vector
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;

        let mut shared_vector = SharedPlaintextStore::from(ptxt_vector);
        let shared_graph = Arc::new(migrate(ptxt_graph, |id| id));

        for ids in (0..database_size)
            .map(|id| VectorId::from_0_index(id as u32))
            .chunks(4)
            .into_iter()
        {
            let ids: Vec<_> = ids.into_iter().collect();
            let queries = shared_vector.vectors_as_queries(ids.clone()).await;

            let mut jobs = JoinSet::new();
            for query in queries {
                let mut sh_vec = shared_vector.clone();
                let searcher = searcher.clone();
                let graph = Arc::clone(&shared_graph);
                jobs.spawn(async move { searcher.search(&mut sh_vec, &graph, &query, 1).await });
            }

            let results = jobs
                .join_all()
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            for result_neighbors in results {
                assert!(
                    searcher
                        .is_match(&mut shared_vector, &[result_neighbors])
                        .await?
                )
            }
        }

        Ok(())
    }
}
