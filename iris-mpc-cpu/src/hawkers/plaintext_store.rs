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
use std::{
    fmt::Display,
    num::ParseIntError,
    ops::{Index, IndexMut},
    str::FromStr,
};
use tracing::debug;

use super::aby3::aby3_store::VectorId;
use eyre::Result;

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlaintextIris(pub IrisCode);

impl PlaintextIris {
    /// Return the fractional Hamming distance with another PlaintextIris,
    /// represented as u16 numerator and denominator.
    pub fn distance_fraction(&self, other: &Self) -> (u16, u16) {
        let combined_mask = self.0.mask & other.0.mask;
        let combined_mask_len = combined_mask.count_ones();

        let combined_code = (self.0.code ^ other.0.code) & combined_mask;
        let code_distance = combined_code.count_ones();

        (code_distance as u16, combined_mask_len as u16)
    }

    /// Return the fractional Hamming distance with another PlaintextIris,
    /// represented as the i16 dot product of associated masked-bit vectors
    /// and the u16 size of the common unmasked region
    pub fn dot_distance_fraction(&self, other: &Self) -> (i16, u16) {
        let (code_distance, combined_mask_len) = self.distance_fraction(other);

        // `code_distance` gives the number of common unmasked bits which are
        // different between two iris codes, and `combined_mask_len` gives the
        // total number of common unmasked bits. The dot product of masked-bit
        // vectors adds 1 for each unmasked bit which is equal, and subtracts 1
        // for each unmasked bit which is unequal; so this can be computed by
        // starting with 1 for every unmasked bit, and subtracting 2 for every
        // unequal unmasked bit, as follows.
        let dot_product = combined_mask_len.wrapping_sub(2 * code_distance) as i16;

        (dot_product, combined_mask_len)
    }
}

// TODO refactor away is_persistent flag; should probably be stored in a
// separate buffer instead whenever working with non-persistent iris codes
#[derive(Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlaintextPoint {
    /// Whatever encoding of a vector.
    pub data: PlaintextIris,
    /// Distinguish between queries that are pending, and those that were
    /// ultimately accepted into the vector store.
    pub is_persistent: bool,
}

#[derive(Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PointId(pub u32);

impl Display for PointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FromStr for PointId {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(PointId(FromStr::from_str(s)?))
    }
}

impl<T> Index<PointId> for Vec<T> {
    type Output = T;

    fn index(&self, index: PointId) -> &Self::Output {
        self.index(index.0 as usize)
    }
}

impl<T> IndexMut<PointId> for Vec<T> {
    fn index_mut(&mut self, index: PointId) -> &mut Self::Output {
        self.index_mut(index.0 as usize)
    }
}

impl From<usize> for PointId {
    fn from(value: usize) -> Self {
        PointId(value as u32)
    }
}

impl From<u32> for PointId {
    fn from(value: u32) -> Self {
        PointId(value)
    }
}

impl From<PointId> for VectorId {
    fn from(id: PointId) -> Self {
        VectorId::from_0_index(id.0)
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlaintextStore {
    pub points: Vec<PlaintextPoint>,
}

impl PlaintextStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prepare_query(&mut self, raw_query: IrisCode) -> <Self as VectorStore>::QueryRef {
        self.points.push(PlaintextPoint {
            data: PlaintextIris(raw_query),
            is_persistent: false,
        });

        let idx = (self.points.len() - 1) as u32;
        VectorId::from_0_index(idx)
    }
}

impl VectorStore for PlaintextStore {
    type QueryRef = VectorId; // Vector ID, pending insertion.
    type VectorRef = VectorId; // Vector ID, inserted.
    type DistanceRef = (u16, u16);

    async fn vectors_as_queries(&mut self, vectors: Vec<Self::VectorRef>) -> Vec<Self::QueryRef> {
        vectors
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Result<Self::DistanceRef> {
        debug!(event_type = EvaluateDistance.id());
        let query_code = &self.points[query.index() as usize];
        let vector_code = &self.points[vector.index() as usize];
        Ok(query_code.data.distance_fraction(&vector_code.data))
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

        for raw_query in cleartext_database.iter() {
            let query = plaintext_vector_store.prepare_query(raw_query.clone());
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

        for raw_query in cleartext_database.iter() {
            let query = plaintext_vector_store.prepare_query(raw_query.clone());
            let _ = plaintext_vector_store.insert(&query).await;
        }

        Ok(plaintext_vector_store)
    }

    pub async fn create_random_store_with_db(cleartext_database: Vec<IrisCode>) -> Result<Self> {
        let mut plaintext_vector_store = PlaintextStore::default();

        for raw_query in cleartext_database.iter() {
            let query = plaintext_vector_store.prepare_query(raw_query.clone());
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

        let mut plaintext_graph_store = GraphMem::new();
        let searcher = HnswSearcher::default();

        for i in 0..graph_size {
            let id = VectorId::from_0_index(i as u32);
            searcher
                .insert(self, &mut plaintext_graph_store, &id, &mut rng_searcher1)
                .await?;
        }

        Ok(plaintext_graph_store)
    }
}

impl VectorStoreMut for PlaintextStore {
    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[query.index() as usize].is_persistent = true;
        *query
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswSearcher;
    use aes_prng::AesRng;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_basic_ops() -> Result<()> {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let cleartext_database = IrisDB::new_random_rng(10, &mut rng).db;
        let formatted_database: Vec<_> = cleartext_database
            .iter()
            .map(|code| PlaintextIris(code.clone()))
            .collect();
        let mut plaintext_store = PlaintextStore::default();

        let pid0 = plaintext_store.prepare_query(cleartext_database[0].clone());
        let pid1 = plaintext_store.prepare_query(cleartext_database[1].clone());
        let pid2 = plaintext_store.prepare_query(cleartext_database[2].clone());
        let pid3 = plaintext_store.prepare_query(cleartext_database[3].clone());

        let q0 = plaintext_store.insert(&pid0).await;
        let q1 = plaintext_store.insert(&pid1).await;
        let q2 = plaintext_store.insert(&pid2).await;
        let q3 = plaintext_store.insert(&pid3).await;

        let d01 = plaintext_store.eval_distance(&q0, &q1).await?;
        let d02 = plaintext_store.eval_distance(&q0, &q2).await?;
        let d03 = plaintext_store.eval_distance(&q0, &q3).await?;
        let d12 = plaintext_store.eval_distance(&q1, &q2).await?;
        let d13 = plaintext_store.eval_distance(&q1, &q3).await?;
        let d23 = plaintext_store.eval_distance(&q2, &q3).await?;

        let d10 = plaintext_store.eval_distance(&q1, &q0).await?;
        let d30 = plaintext_store.eval_distance(&q3, &q0).await?;

        let db0 = &formatted_database[0].0;
        let db1 = &formatted_database[1].0;
        let db2 = &formatted_database[2].0;
        let db3 = &formatted_database[3].0;

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
            let id = VectorId::from_0_index(i as u32);
            let cleartext_neighbors = searcher
                .search(&mut ptxt_vector, &mut ptxt_graph, &id, 1)
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
