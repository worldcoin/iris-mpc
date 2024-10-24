use hawk_pack::VectorStore;
use iris_mpc_common::iris_db::iris::IrisCode;

#[derive(Default, Debug, Clone)]
pub struct PlaintextStore {
    pub points: Vec<PlaintextPoint>,
}

#[derive(Default, Debug, Clone)]
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
        let dot_product = combined_mask_len.wrapping_sub(2 * code_distance) as i16;

        (dot_product, combined_mask_len)
    }
}

#[derive(Clone, Default, Debug)]
pub struct PlaintextPoint {
    /// Whatever encoding of a vector.
    data:          PlaintextIris,
    /// Distinguish between queries that are pending, and those that were
    /// ultimately accepted into the vector store.
    is_persistent: bool,
}

pub type PointId = u32;

impl PlaintextStore {
    pub fn prepare_query(&mut self, raw_query: IrisCode) -> <Self as VectorStore>::QueryRef {
        self.points.push(PlaintextPoint {
            data:          PlaintextIris(raw_query),
            is_persistent: false,
        });

        let point_id = self.points.len() - 1;
        point_id as PointId
    }

    /// Compare two distances between pairs of iris codes
    ///
    /// Returns terms obtained by cross-multiplying numerators with opposite
    /// denominators.
    pub fn distance_computation(
        &self,
        distance1: &(PointId, PointId),
        distance2: &(PointId, PointId),
    ) -> (i32, i32) {
        let (x1, y1) = (
            &self.points[distance1.0 as usize],
            &self.points[distance1.1 as usize],
        );
        let (x2, y2) = (
            &self.points[distance2.0 as usize],
            &self.points[distance2.1 as usize],
        );
        let (a, b) = x1.data.distance_fraction(&y1.data);
        let (c, d) = x2.data.distance_fraction(&y2.data);
        let cross_1 = a as i32 * d as i32;
        let cross_2 = c as i32 * b as i32;

        // for Hamming distances a/b and c/d, return (a*d, b*c)
        (cross_1, cross_2)
    }
}

impl VectorStore for PlaintextStore {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[*query as usize].is_persistent = true;
        *query
    }

    async fn eval_distance(
        &self,
        query: &Self::QueryRef,
        vector: &Self::VectorRef,
    ) -> Self::DistanceRef {
        // Do not compute the distance yet, just forward the IDs.
        (*query, *vector)
    }

    async fn is_match(&self, distance: &Self::DistanceRef) -> bool {
        let x = &self.points[distance.0 as usize];
        let y = &self.points[distance.1 as usize];
        x.data.0.is_close(&y.data.0)
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let (cross_1, cross_2) = self.distance_computation(distance1, distance2);
        (cross_1 - cross_2) < 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hawkers::galois_store::gr_create_ready_made_hawk_searcher;
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

        let db0 = &formatted_database[0].0;
        let db1 = &formatted_database[1].0;
        let db2 = &formatted_database[2].0;
        let db3 = &formatted_database[3].0;

        assert_eq!(
            plaintext_store.less_than(&(q0, q1), &(q2, q3)).await,
            db0.get_distance(db1) < db2.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&(q2, q3), &(q0, q1)).await,
            db2.get_distance(db3) < db0.get_distance(db1)
        );

        assert_eq!(
            plaintext_store.less_than(&(q0, q2), &(q1, q3)).await,
            db0.get_distance(db2) < db1.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&(q0, q3), &(q1, q2)).await,
            db0.get_distance(db3) < db1.get_distance(db2)
        );

        assert_eq!(
            plaintext_store.less_than(&(q1, q0), &(q2, q3)).await,
            db1.get_distance(db0) < db2.get_distance(db3)
        );

        assert_eq!(
            plaintext_store.less_than(&(q1, q2), &(q3, q0)).await,
            db1.get_distance(db2) < db3.get_distance(db0)
        );

        assert_eq!(
            plaintext_store.less_than(&(q0, q2), &(q0, q1)).await,
            db0.get_distance(db2) < db0.get_distance(db1)
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_plaintext_hnsw_matcher() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 1;
        let searcher = HawkSearcher::default();
        let ((mut ptxt_vector, mut ptxt_graph), _) =
            gr_create_ready_made_hawk_searcher(&mut rng, database_size)
                .await
                .unwrap();
        for i in 0..database_size {
            let cleartext_neighbors = searcher
                .search_to_insert(&mut ptxt_vector, &mut ptxt_graph, &(i as PointId))
                .await;
            assert!(searcher.is_match(&ptxt_vector, &cleartext_neighbors).await,);
        }
    }
}
