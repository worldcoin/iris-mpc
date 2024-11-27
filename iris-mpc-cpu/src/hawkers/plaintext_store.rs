use aes_prng::AesRng;
use hawk_pack::{graph_store::GraphMem, hnsw_db::HawkSearcher, VectorStore};
use iris_mpc_common::iris_db::{
    db::IrisDB,
    iris::{IrisCode, MATCH_THRESHOLD_RATIO},
};
use rand::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
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
        // total number of common unmasked bits.  The dot product of masked-bit
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
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct PlaintextPoint {
    /// Whatever encoding of a vector.
    pub data:          PlaintextIris,
    /// Distinguish between queries that are pending, and those that were
    /// ultimately accepted into the vector store.
    pub is_persistent: bool,
}

#[derive(Copy, Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PointId(pub u32);

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

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct PlaintextStore {
    pub points: Vec<PlaintextPoint>,
}

impl PlaintextStore {
    pub fn prepare_query(&mut self, raw_query: IrisCode) -> <Self as VectorStore>::QueryRef {
        self.points.push(PlaintextPoint {
            data:          PlaintextIris(raw_query),
            is_persistent: false,
        });

        let point_id = self.points.len() - 1;
        point_id.into()
    }
}

impl VectorStore for PlaintextStore {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (u16, u16);

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[*query].is_persistent = true;
        *query
    }

    async fn eval_distance(
        &mut self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        let query_code = &self.points[*query];
        let vector_code = &self.points[*vector];
        query_code.data.distance_fraction(&vector_code.data)
    }

    async fn is_match(&mut self, distance: &Self::DistanceRef) -> bool {
        let (a, b) = *distance; // a/b
        (a as f64) < (b as f64) * MATCH_THRESHOLD_RATIO
    }

    async fn less_than(
        &mut self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let (a, b) = *distance1; // a/b
        let (c, d) = *distance2; // c/d
        (a as i32) * (d as i32) - (b as i32) * (c as i32) < 0
    }
}

impl PlaintextStore {
    pub async fn create_random<R: RngCore + Clone + CryptoRng>(
        rng: &mut R,
        database_size: usize,
    ) -> eyre::Result<(Vec<IrisCode>, Self, GraphMem<Self>)> {
        // makes sure the searcher produces same graph structure by having the same rng
        let mut rng_searcher1 = AesRng::from_rng(rng.clone())?;
        let cleartext_database = IrisDB::new_random_rng(database_size, rng).db;

        let mut plaintext_vector_store = PlaintextStore::default();
        let mut plaintext_graph_store = GraphMem::new();
        let searcher = HawkSearcher::default();

        for raw_query in cleartext_database.iter() {
            let query = plaintext_vector_store.prepare_query(raw_query.clone());
            let neighbors = searcher
                .search_to_insert(
                    &mut plaintext_vector_store,
                    &mut plaintext_graph_store,
                    &query,
                )
                .await;
            let inserted = plaintext_vector_store.insert(&query).await;
            searcher
                .insert_from_search_results(
                    &mut plaintext_vector_store,
                    &mut plaintext_graph_store,
                    &mut rng_searcher1,
                    inserted,
                    neighbors,
                )
                .await;
        }

        Ok((
            cleartext_database,
            plaintext_vector_store,
            plaintext_graph_store,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_prng::AesRng;
    use hawk_pack::hnsw_db::HawkSearcher;
    use iris_mpc_common::iris_db::db::IrisDB;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_basic_ops() {
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

        let d01 = plaintext_store.eval_distance(&q0, &q1).await;
        let d02 = plaintext_store.eval_distance(&q0, &q2).await;
        let d03 = plaintext_store.eval_distance(&q0, &q3).await;
        let d12 = plaintext_store.eval_distance(&q1, &q2).await;
        let d13 = plaintext_store.eval_distance(&q1, &q3).await;
        let d23 = plaintext_store.eval_distance(&q2, &q3).await;

        let d10 = plaintext_store.eval_distance(&q1, &q0).await;
        let d30 = plaintext_store.eval_distance(&q3, &q0).await;

        let db0 = &formatted_database[0].0;
        let db1 = &formatted_database[1].0;
        let db2 = &formatted_database[2].0;
        let db3 = &formatted_database[3].0;

        assert_eq!(
            plaintext_store.less_than(&d01, &d23).await,
            db0.get_distance(db1) < db2.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&d23, &d01).await,
            db2.get_distance(db3) < db0.get_distance(db1)
        );

        assert_eq!(
            plaintext_store.less_than(&d02, &d13).await,
            db0.get_distance(db2) < db1.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&d03, &d12).await,
            db0.get_distance(db3) < db1.get_distance(db2)
        );

        assert_eq!(
            plaintext_store.less_than(&d10, &d23).await,
            db1.get_distance(db0) < db2.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&d12, &d30).await,
            db1.get_distance(db2) < db3.get_distance(db0)
        );

        assert_eq!(
            plaintext_store.less_than(&d02, &d01).await,
            db0.get_distance(db2) < db0.get_distance(db1)
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_plaintext_hnsw_matcher() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 1;
        let searcher = HawkSearcher::default();
        let (_, mut ptxt_vector, mut ptxt_graph) =
            PlaintextStore::create_random(&mut rng, database_size)
                .await
                .unwrap();
        for i in 0..database_size {
            let cleartext_neighbors = searcher
                .search_to_insert(&mut ptxt_vector, &mut ptxt_graph, &i.into())
                .await;
            assert!(
                searcher
                    .is_match(&mut ptxt_vector, &cleartext_neighbors)
                    .await,
            );
        }
    }
}
