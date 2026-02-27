use crate::{
    hawkers::{
        aby3::aby3_store::{DistanceFn, DistanceOps, FhdOps},
        shared_irises::{SharedIrises, SharedIrisesRef},
        TEST_DISTANCE_FN,
    },
    hnsw::{
        metrics::ops_counter::Operation::{CompareDistance, EvaluateDistance},
        vector_store::VectorStoreMut,
        GraphMem, HnswSearcher, SortedNeighborhood, VectorStore,
    },
};
use aes_prng::AesRng;
use iris_mpc_common::{
    iris_db::{db::IrisDB, iris::IrisCode},
    vector_id::VectorId,
};
use rand::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, marker::PhantomData, sync::Arc};
use tracing::debug;

use eyre::{bail, Result};

pub type PlaintextVectorRef = <PlaintextStore as VectorStore>::VectorRef;
pub type PlaintextStoredIris = Arc<IrisCode>;

pub type PlaintextSharedIrises = SharedIrises<PlaintextStoredIris>;
pub type PlaintextSharedIrisesRef = SharedIrisesRef<PlaintextStoredIris>;

/// Vector store which works over plaintext iris codes and distance computations.
///
/// Generic over `D: DistanceOps` which determines the comparison protocol
/// (FHD cross-multiply vs NHD polynomial). Defaults to `FhdOps`.
///
/// This variant is only suitable for single-threaded operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct PlaintextStore<D: DistanceOps = FhdOps> {
    pub storage: PlaintextSharedIrises,
    pub distance_fn: DistanceFn,
    #[serde(skip)]
    _phantom: PhantomData<D>,
}

impl<D: DistanceOps> Default for PlaintextStore<D> {
    fn default() -> Self {
        Self::with_storage(PlaintextSharedIrises::default())
    }
}

impl<D: DistanceOps> PlaintextStore<D> {
    /// Generate a new empty `PlaintextStore`.
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_storage(storage: PlaintextSharedIrises) -> Self {
        Self {
            storage,
            distance_fn: TEST_DISTANCE_FN,
            _phantom: PhantomData,
        }
    }

    pub fn from_irises_iter(iter: impl Iterator<Item = IrisCode>) -> Self {
        let mut vector = Self::new();
        for (idx, iris) in iter.enumerate() {
            let id = VectorId::from_0_index(idx as u32);
            vector.insert_with_id(id, Arc::new(iris));
        }
        vector
    }

    /// Return the size of the underlying set of irises.
    pub fn len(&self) -> usize {
        self.storage.db_size()
    }

    /// Return whether the underlying iris set is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.db_size() == 0
    }

    pub fn insert_with_id(
        &mut self,
        id: VectorId,
        query: <Self as VectorStore>::QueryRef,
    ) -> <Self as VectorStore>::VectorRef {
        self.storage.insert(id, query)
    }

    /// Generate a new `PlaintextStore` of specified size with random entries.
    pub fn new_random<R: RngCore + Clone + CryptoRng>(rng: &mut R, store_size: usize) -> Self {
        let points_codes = IrisDB::new_random_rng(store_size, rng).db;
        let points = points_codes
            .into_iter()
            .enumerate()
            .map(|(idx, iris)| (VectorId::from_0_index(idx as u32), Arc::new(iris)))
            .collect::<HashMap<VectorId, PlaintextStoredIris>>();
        Self::with_storage(SharedIrises::new(points, Default::default()))
    }

    /// Generate an HNSW graph over the first `graph_size` entries of this `PlaintextStore`, sorted in increasing
    /// order of serial ids, using the specified searcher and randomness.
    pub async fn generate_graph<R: RngCore + Clone + CryptoRng>(
        &mut self,
        rng: &mut R,
        graph_size: usize,
        searcher: &HnswSearcher,
    ) -> Result<GraphMem<<Self as VectorStore>::VectorRef>> {
        let mut graph = GraphMem::new();
        let mut rng = AesRng::from_rng(rng.clone())?;

        if graph_size > self.len() {
            bail!("Cannot generate graph larger than underlying vector store");
        }

        // sort in order to ensure deterministic behavior
        let mut serial_ids: Vec<_> = self.storage.get_sorted_serial_ids();
        serial_ids.truncate(graph_size);

        for serial_id in serial_ids {
            let query = self
                .storage
                .get_vector_by_serial_id(serial_id)
                .unwrap()
                .clone();
            let query_id = VectorId::from_serial_id(serial_id);
            let insertion_layer = searcher.gen_layer_rng(&mut rng)?;
            let (neighbors, update_ep) = searcher
                .search_to_insert::<_, SortedNeighborhood<_>>(self, &graph, &query, insertion_layer)
                .await?;
            searcher
                .insert_from_search_results(self, &mut graph, query_id, neighbors, update_ep)
                .await?;
        }

        Ok(graph)
    }
}

pub fn fraction_ordering(dist_1: &(u16, u16), dist_2: &(u16, u16)) -> Ordering {
    FhdOps::plaintext_ordering(dist_1, dist_2)
}

impl<D: DistanceOps> VectorStore for PlaintextStore<D> {
    type QueryRef = Arc<IrisCode>;
    type VectorRef = VectorId;
    type DistanceRef = (u16, u16);

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        vectors
            .iter()
            .map(|id| self.storage.get_vector(id).unwrap().clone())
            .collect()
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        debug!(event_type = EvaluateDistance.id());
        let vector_code = self.storage.get_vector(vector).ok_or_else(|| {
            eyre::eyre!(
                "Vector ID not found in store for serial {}",
                vector.serial_id()
            )
        })?;
        let distance = D::plaintext_distance(vector_code, query, self.distance_fn);
        Ok(distance)
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(D::plaintext_is_match(distance))
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(D::plaintext_less_than(distance1, distance2))
    }

    // Note: default implementation + metrics
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        let mut results: Vec<bool> = Vec::with_capacity(distances.len());
        for (d1, d2) in distances {
            results.push(self.less_than(d1, d2).await?);
        }
        metrics::counter!("less_than").increment(distances.len() as u64);
        Ok(results)
    }

    async fn only_valid_vectors(
        &mut self,
        mut vectors: Vec<Self::VectorRef>,
    ) -> Vec<Self::VectorRef> {
        vectors.retain(|v| self.storage.contains(v));
        vectors
    }
}

impl<D: DistanceOps> VectorStoreMut for PlaintextStore<D> {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.append(query.clone())
    }

    async fn insert_at(
        &mut self,
        vector_ref: &Self::VectorRef,
        query: &Self::QueryRef,
    ) -> Result<Self::VectorRef> {
        Ok(self.storage.insert(*vector_ref, query.clone()))
    }
}

/// PlaintextStore with synchronization primitives for multithreaded use.
#[derive(Debug, Clone)]
pub struct SharedPlaintextStore<D: DistanceOps = FhdOps> {
    pub storage: PlaintextSharedIrisesRef,
    pub distance_fn: DistanceFn,
    _phantom: PhantomData<D>,
}

impl<D: DistanceOps> Default for SharedPlaintextStore<D> {
    fn default() -> Self {
        Self {
            storage: SharedIrises::default().to_arc(),
            distance_fn: TEST_DISTANCE_FN,
            _phantom: PhantomData,
        }
    }
}

impl<D: DistanceOps> SharedPlaintextStore<D> {
    pub fn new() -> Self {
        Default::default()
    }

    pub async fn len(&self) -> usize {
        self.storage.read().await.db_size()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

impl<D: DistanceOps> From<PlaintextStore<D>> for SharedPlaintextStore<D> {
    fn from(value: PlaintextStore<D>) -> Self {
        Self {
            storage: value.storage.to_arc(),
            distance_fn: value.distance_fn,
            _phantom: PhantomData,
        }
    }
}

impl<D: DistanceOps> VectorStore for SharedPlaintextStore<D> {
    type QueryRef = Arc<IrisCode>;
    type VectorRef = VectorId;
    type DistanceRef = (u16, u16);

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        let store = self.storage.read().await;
        vectors
            .iter()
            .map(|id| store.get_vector(id).unwrap().clone())
            .collect()
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        let distances = self.eval_distance_batch(query, &[*vector]).await?;
        Ok(distances[0])
    }

    async fn eval_distance_batch(
        &mut self,
        query: &Self::QueryRef,
        vectors: &[Self::VectorRef],
    ) -> Result<Vec<Self::DistanceRef>> {
        debug!(event_type = EvaluateDistance.id());
        let store = self.storage.read().await;
        let vector_codes = vectors
            .iter()
            .map(|v| {
                let serial_id = v.serial_id();
                store.get_vector(v).ok_or_else(|| {
                    eyre::eyre!("Vector ID not found in store for serial {}", serial_id)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(vector_codes
            .into_iter()
            .map(|v| D::plaintext_distance(v, query, self.distance_fn))
            .collect())
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        Ok(D::plaintext_is_match(distance))
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        Ok(D::plaintext_less_than(distance1, distance2))
    }

    // Note: default implementation + metrics
    async fn less_than_batch(
        &mut self,
        distances: &[(Self::DistanceRef, Self::DistanceRef)],
    ) -> Result<Vec<bool>> {
        let mut results: Vec<bool> = Vec::with_capacity(distances.len());
        for (d1, d2) in distances {
            results.push(self.less_than(d1, d2).await?);
        }
        metrics::counter!("less_than").increment(distances.len() as u64);
        Ok(results)
    }

    async fn only_valid_vectors(
        &mut self,
        mut vectors: Vec<Self::VectorRef>,
    ) -> Vec<Self::VectorRef> {
        let storage = self.storage.read().await;
        vectors.retain(|v| storage.contains(v));
        vectors
    }
}

impl<D: DistanceOps> VectorStoreMut for SharedPlaintextStore<D> {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.storage.append(query).await
    }

    async fn insert_at(
        &mut self,
        vector_ref: &Self::VectorRef,
        query: &Self::QueryRef,
    ) -> Result<Self::VectorRef> {
        Ok(self.storage.insert(*vector_ref, query).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkers::aby3::aby3_store::FhdOps;
    use crate::hnsw::HnswSearcher;
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
        let mut store = PlaintextStore::<FhdOps>::new();

        let db = IrisDB::new_random_rng(10, &mut rng)
            .db
            .into_iter()
            .map(Arc::new)
            .collect_vec();

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

        let distance = |a, b| FhdOps::plaintext_distance(a, b, TEST_DISTANCE_FN);

        assert_eq!(
            store.less_than(&d01, &d23).await?,
            distance(&db[0], &db[1]) < distance(&db[2], &db[3])
        );

        assert_eq!(
            store.less_than(&d23, &d01).await?,
            distance(&db[2], &db[3]) < distance(&db[0], &db[1])
        );

        assert_eq!(
            store.less_than(&d02, &d13).await?,
            distance(&db[0], &db[2]) < distance(&db[1], &db[3])
        );

        assert_eq!(
            store.less_than(&d03, &d12).await?,
            distance(&db[0], &db[3]) < distance(&db[1], &db[2])
        );

        assert_eq!(
            store.less_than(&d10, &d23).await?,
            distance(&db[1], &db[0]) < distance(&db[2], &db[3])
        );

        assert_eq!(
            store.less_than(&d12, &d30).await?,
            distance(&db[1], &db[2]) < distance(&db[3], &db[0])
        );

        assert_eq!(
            store.less_than(&d02, &d01).await?,
            distance(&db[0], &db[2]) < distance(&db[0], &db[1])
        );

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_plaintext_hnsw_matcher() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 1;
        let searcher = HnswSearcher::new_with_test_parameters();
        let mut ptxt_vector = PlaintextStore::<FhdOps>::new_random(&mut rng, database_size);
        let ptxt_graph = ptxt_vector
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;
        for i in 0..database_size {
            let serial_id = i as u32 + 1;
            let vector_id = VectorId::from_serial_id(serial_id);
            let query = ptxt_vector.storage.get_vector(&vector_id).unwrap().clone();
            let cleartext_neighbors: SortedNeighborhood<_> = searcher
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
        let mut ptxt_vector = PlaintextStore::<FhdOps>::new_random(&mut rng, database_size);
        let ptxt_graph = ptxt_vector
            .generate_graph(&mut rng, database_size, &searcher)
            .await?;

        let mut shared_vector = SharedPlaintextStore::from(ptxt_vector);
        let shared_graph = Arc::new(ptxt_graph);

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

            let results: Vec<SortedNeighborhood<_>> = jobs
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
