use std::sync::Arc;

use crate::hnsw::{
    metrics::ops_counter::Operation::{CompareDistance, EvaluateDistance},
    vector_store::VectorStoreMut,
    GraphMem, HnswSearcher, VectorStore,
};
use aes_prng::AesRng;
use iris_mpc_common::iris_db::{
    db::IrisDB,
    iris::{IrisCode, MATCH_THRESHOLD_RATIO},
};
use rand::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::aby3::aby3_store::VectorId;
use eyre::Result;

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlaintextStore {
    pub points: Vec<Arc<IrisCode>>,
}

impl PlaintextStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorStore for PlaintextStore {
    type QueryRef = Arc<IrisCode>;
    type VectorRef = VectorId;
    type DistanceRef = (u16, u16);

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        vectors
            .iter()
            .map(|id| self.points.get(id.index() as usize).unwrap().clone())
            .collect()
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        debug!(event_type = EvaluateDistance.id());
        let vector_code = &self.points[vector.index() as usize];
        Ok(query.get_distance_fraction(vector_code))
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> Result<bool> {
        let (a, b) = *distance; // a/b
        Ok((a as f64) < (b as f64) * MATCH_THRESHOLD_RATIO)
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> Result<bool> {
        debug!(event_type = CompareDistance.id());
        let (a, b) = *distance1; // a/b
        let (c, d) = *distance2; // c/d
        Ok((a as i32) * (d as i32) - (b as i32) * (c as i32) < 0)
    }
}

impl VectorStoreMut for PlaintextStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        self.points.push(query.clone());
        VectorId::from_0_index((self.points.len() - 1) as u32)
    }
}

impl PlaintextStore {
    pub async fn create_random<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
        searcher: &HnswSearcher,
    ) -> Result<(Self, GraphMem<Self>)> {
        // makes sure the searcher produces same graph structure by having the same rng
        let mut rng_searcher1 = AesRng::from_rng(rng.clone())?;
        let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;

        let mut plaintext_vector_store = PlaintextStore::default();
        let mut plaintext_graph_store = GraphMem::new();

        for raw_query in cleartext_database {
            let query = Arc::new(raw_query);
            searcher
                .insert(
                    &mut plaintext_vector_store,
                    &mut plaintext_graph_store,
                    &query,
                    &mut rng_searcher1,
                )
                .await?;
        }

        Ok((plaintext_vector_store, plaintext_graph_store))
    }

    pub async fn create_random_store<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
    ) -> Result<Self> {
        let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;

        let mut plaintext_vector_store = PlaintextStore::default();

        for raw_query in cleartext_database {
            let query = Arc::new(raw_query);
            let _ = plaintext_vector_store.insert(&query).await;
        }

        Ok(plaintext_vector_store)
    }

    pub async fn create_random_store_with_db(cleartext_database: Vec<IrisCode>) -> Result<Self> {
        let mut plaintext_vector_store = PlaintextStore::default();

        for raw_query in cleartext_database {
            let query = Arc::new(raw_query);
            let _ = plaintext_vector_store.insert(&query).await;
        }

        Ok(plaintext_vector_store)
    }

    pub async fn create_graph<R: RngCore + Clone + CryptoRng>(
        &mut self,
        rng: &mut R,
        graph_size: usize,
    ) -> Result<GraphMem<Self>> {
        let mut rng_searcher1 = AesRng::from_rng(rng.clone())?;

        let mut graph = GraphMem::new();
        let searcher = HnswSearcher::default();

        for idx in 0..graph_size {
            let query = self.points.get(idx).unwrap().clone();
            let query_id = VectorId::from_0_index(idx as u32);
            let insertion_layer = searcher.select_layer(&mut rng_searcher1)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswSearcher;
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use itertools::Itertools;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_basic_ops() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let mut plaintext_store = PlaintextStore::default();

        let cleartext_database = IrisDB::new_random_rng(10, &mut rng).db;

        let queries = cleartext_database
            .iter()
            .cloned()
            .map(Arc::new)
            .collect_vec();

        let mut ids = Vec::new();
        for q in queries.iter() {
            ids.push(plaintext_store.insert(q).await);
        }

        let d01 = plaintext_store.eval_distance(&queries[0], &ids[1]).await?;
        let d02 = plaintext_store.eval_distance(&queries[0], &ids[2]).await?;
        let d03 = plaintext_store.eval_distance(&queries[0], &ids[3]).await?;
        let d10 = plaintext_store.eval_distance(&queries[1], &ids[0]).await?;
        let d12 = plaintext_store.eval_distance(&queries[1], &ids[2]).await?;
        let d13 = plaintext_store.eval_distance(&queries[1], &ids[3]).await?;
        let d23 = plaintext_store.eval_distance(&queries[2], &ids[3]).await?;
        let d30 = plaintext_store.eval_distance(&queries[3], &ids[0]).await?;

        let db0 = &cleartext_database[0];
        let db1 = &cleartext_database[1];
        let db2 = &cleartext_database[2];
        let db3 = &cleartext_database[3];

        assert_eq!(
            plaintext_store.less_than(&d01, &d23).await?,
            db0.get_distance(db1) < db2.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&d23, &d01).await?,
            db2.get_distance(db3) < db0.get_distance(db1)
        );

        assert_eq!(
            plaintext_store.less_than(&d02, &d13).await?,
            db0.get_distance(db2) < db1.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&d03, &d12).await?,
            db0.get_distance(db3) < db1.get_distance(db2)
        );

        assert_eq!(
            plaintext_store.less_than(&d10, &d23).await?,
            db1.get_distance(db0) < db2.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&d12, &d30).await?,
            db1.get_distance(db2) < db3.get_distance(db0)
        );

        assert_eq!(
            plaintext_store.less_than(&d02, &d01).await?,
            db0.get_distance(db2) < db0.get_distance(db1)
        );

        Ok(())
    }

    #[tokio::test]
    #[traced_test]
    async fn test_plaintext_hnsw_matcher() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 1;
        let searcher = HnswSearcher::default();
        let (mut ptxt_vector, mut ptxt_graph) =
            PlaintextStore::create_random(&mut rng, database_size, &searcher).await?;
        for i in 0..database_size {
            let query = ptxt_vector.points.get(i).unwrap().clone();
            let cleartext_neighbors = searcher
                .search(&mut ptxt_vector, &mut ptxt_graph, &query, 1)
                .await?;
            assert!(
                searcher
                    .is_match(&mut ptxt_vector, &[cleartext_neighbors])
                    .await?,
            );
        }

        Ok(())
    }
}
