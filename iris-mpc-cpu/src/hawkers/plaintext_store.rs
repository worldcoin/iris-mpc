use crate::shares::ring_impl::RingElement;
use hawk_pack::VectorStore;
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray, MATCH_THRESHOLD_RATIO};
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone)]
pub struct PlaintextStore {
    pub points: Vec<PlaintextPoint>,
}

#[derive(Default, Debug, Clone)]
pub struct FormattedIris {
    data: Vec<i8>,
    mask: IrisCodeArray,
}

impl From<IrisCode> for FormattedIris {
    fn from(value: IrisCode) -> Self {
        let data: Vec<i8> = (0..IrisCode::IRIS_CODE_SIZE)
            .map(|i| {
                let bi = value.code.get_bit(i);
                let mi = value.mask.get_bit(i);
                let bi_mi = (bi & mi) as i8;
                (mi as i8) - 2 * bi_mi
            })
            .collect();
        FormattedIris {
            data,
            mask: value.mask,
        }
    }
}

impl From<Vec<RingElement<u16>>> for FormattedIris {
    fn from(value: Vec<RingElement<u16>>) -> Self {
        let data: Vec<i8> = (0..IrisCode::IRIS_CODE_SIZE)
            .map(|i| {
                let v = value[i].convert();
                if v > 32768 {
                    (v as i32 - 65536) as i8
                } else {
                    v as i8
                }
            })
            .collect();
        FormattedIris {
            data,
            mask: IrisCodeArray::ONES,
        }
    }
}
impl FormattedIris {
    #[cfg(test)]
    pub fn compute_real_distance(&self, other: &FormattedIris) -> f64 {
        let hd = self.dot_on_code(other);
        let mask_ones = (self.mask & other.mask).count_ones();
        ((mask_ones as f64) - (hd as f64)) / (2. * mask_ones as f64)
    }
}

#[derive(Clone, Default, Debug)]
pub struct PlaintextPoint {
    /// Whatever encoding of a vector.
    data:          FormattedIris,
    /// Distinguish between queries that are pending, and those that were
    /// ultimately accepted into the vector store.
    is_persistent: bool,
}

impl FormattedIris {
    pub fn dot_on_code(&self, other: &Self) -> i32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(0_i32, |sum, (i, j)| sum + (*i as i32) * (*j as i32))
    }
}

impl PlaintextPoint {
    fn compute_distance(&self, other: &PlaintextPoint) -> (i16, usize) {
        let combined_mask = self.data.mask & other.data.mask;
        let dot = self.data.dot_on_code(&other.data) as i16;
        (dot, combined_mask.count_ones())
    }

    fn is_close(&self, other: &PlaintextPoint) -> bool {
        let hd = self.data.dot_on_code(&other.data);
        let mask_ones = (self.data.mask & other.data.mask).count_ones();
        let threshold = (mask_ones as f64) * (1. - 2. * MATCH_THRESHOLD_RATIO);
        (threshold as i32 - hd) < 0
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointId(pub usize);

impl PointId {
    pub fn val(&self) -> usize {
        self.0
    }
}

impl PlaintextStore {
    pub fn prepare_query(&mut self, raw_query: IrisCode) -> <Self as VectorStore>::QueryRef {
        self.points.push(PlaintextPoint {
            data:          FormattedIris::from(raw_query),
            is_persistent: false,
        });

        let point_id = self.points.len() - 1;
        PointId(point_id)
    }

    pub fn distance_computation(
        &self,
        distance1: &(PointId, PointId),
        distance2: &(PointId, PointId),
    ) -> (i32, i32) {
        let (x1, y1) = (
            &self.points[distance1.0.val()],
            &self.points[distance1.1.val()],
        );
        let (x2, y2) = (
            &self.points[distance2.0.val()],
            &self.points[distance2.1.val()],
        );
        let (d1, t1) = x1.compute_distance(y1);
        let (d2, t2) = x2.compute_distance(y2);

        let cross_1 = d2 as i32 * t1 as i32;
        let cross_2 = d1 as i32 * t2 as i32;
        (cross_1, cross_2)
    }
}

impl VectorStore for PlaintextStore {
    type QueryRef = PointId; // Vector ID, pending insertion.
    type VectorRef = PointId; // Vector ID, inserted.
    type DistanceRef = (PointId, PointId); // Lazy distance representation.

    async fn insert(&mut self, query: &Self::QueryRef) -> Self::VectorRef {
        // The query is now accepted in the store. It keeps the same ID.
        self.points[query.0].is_persistent = true;
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
        let x = &self.points[distance.0 .0];
        let y = &self.points[distance.1 .0];
        x.is_close(y)
    }

    async fn less_than(
        &self,
        distance1: &Self::DistanceRef,
        distance2: &Self::DistanceRef,
    ) -> bool {
        let (d2t1, d1t2) = self.distance_computation(distance1, distance2);
        (d2t1 - d1t2) < 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        database_generators::create_ground_truth_database,
        hawkers::aby3_store::create_ready_made_hawk_searcher,
    };
    use aes_prng::AesRng;
    use rand::SeedableRng;
    use tracing_test::traced_test;

    #[tokio::test]
    #[traced_test]
    async fn test_basic_ops() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let cleartext_database = create_ground_truth_database(&mut rng, 10).unwrap();
        let formatted_database: Vec<_> = cleartext_database
            .iter()
            .map(|code| FormattedIris::from(code.clone()))
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

        let c1 = plaintext_store.less_than(&(q0, q1), &(q2, q3)).await;

        let d01 = formatted_database[0].compute_real_distance(&formatted_database[1]);
        let d23 = formatted_database[2].compute_real_distance(&formatted_database[3]);
        assert_eq!(c1, d01 < d23);

        let c2 = plaintext_store.less_than(&(q2, q3), &(q0, q1)).await;
        assert_eq!(c2, d23 < d01);

        assert_eq!(
            plaintext_store.less_than(&(q0, q2), &(q1, q3)).await,
            formatted_database[0].compute_real_distance(&formatted_database[2])
                < formatted_database[1].compute_real_distance(&formatted_database[3])
        );

        assert_eq!(
            plaintext_store.less_than(&(q0, q3), &(q1, q2)).await,
            formatted_database[0].compute_real_distance(&formatted_database[3])
                < formatted_database[1].compute_real_distance(&formatted_database[2])
        );

        assert_eq!(
            plaintext_store.less_than(&(q1, q0), &(q2, q3)).await,
            formatted_database[1].compute_real_distance(&formatted_database[0])
                < formatted_database[2].compute_real_distance(&formatted_database[3])
        );

        assert_eq!(
            plaintext_store.less_than(&(q1, q2), &(q3, q0)).await,
            formatted_database[1].compute_real_distance(&formatted_database[2])
                < formatted_database[3].compute_real_distance(&formatted_database[0])
        );

        assert_eq!(
            plaintext_store.less_than(&(q0, q2), &(q0, q1)).await,
            formatted_database[0].compute_real_distance(&formatted_database[2])
                < formatted_database[0].compute_real_distance(&formatted_database[1])
        );
    }

    #[tokio::test]
    #[traced_test]
    async fn test_plaintext_hnsw_matcher() {
        let mut rng = AesRng::seed_from_u64(0_u64);
        let database_size = 1;
        let (cleartext_searcher, _) = create_ready_made_hawk_searcher(&mut rng, database_size)
            .await
            .unwrap();
        for i in 0..database_size {
            let cleartext_neighbors = cleartext_searcher.search_to_insert(&PointId(i)).await;
            assert!(cleartext_searcher.is_match(&cleartext_neighbors).await,);
        }
    }
}
