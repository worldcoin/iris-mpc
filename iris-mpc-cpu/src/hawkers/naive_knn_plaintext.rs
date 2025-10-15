use iris_mpc_common::{iris_db::iris::IrisCode, IrisSerialId};
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPool,
};
use serde::{Deserialize, Serialize};

use crate::hawkers::plaintext_store::fraction_ordering;

#[derive(Serialize, Deserialize)]
pub struct KNNResult {
    pub node: IrisSerialId,
    neighbors: Vec<IrisSerialId>,
}

pub fn naive_knn(
    irises: &[IrisCode],
    k: usize,
    start: usize,
    end: usize,
    pool: &ThreadPool,
) -> Vec<KNNResult> {
    pool.install(|| {
        (start..end)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| {
                let current_iris = &irises[i - 1];
                let mut neighbors = irises
                    .iter()
                    .enumerate()
                    .flat_map(|(j, other_iris)| {
                        (i != j + 1).then_some((
                            (j + 1) as u32,
                            current_iris.get_distance_fraction(other_iris),
                        ))
                    })
                    .collect::<Vec<_>>();
                neighbors
                    .select_nth_unstable_by(k - 1, |lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
                neighbors.shrink_to_fit(); // just to make sure
                neighbors.sort_by(|lhs, rhs| fraction_ordering(&lhs.1, &rhs.1));
                let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                KNNResult {
                    node: i as u32,
                    neighbors,
                }
            })
            .collect::<Vec<_>>()
    })
}
