use iris_mpc_common::iris_db::iris::IrisCode;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPool,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct KNNResult {
    pub node: usize,
    neighbors: Vec<usize>,
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
                        (i != j + 1).then_some((j + 1, current_iris.get_distance(other_iris)))
                    })
                    .collect::<Vec<_>>();
                neighbors.select_nth_unstable_by(k - 1, |(_, d1), (_, d2)| d1.total_cmp(d2));
                let mut neighbors = neighbors.drain(0..k).collect::<Vec<_>>();
                neighbors.shrink_to_fit(); // just to make sure
                neighbors.sort_by(|(_, d1), (_, d2)| d1.total_cmp(d2));
                let neighbors = neighbors.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
                KNNResult { node: i, neighbors }
            })
            .collect::<Vec<_>>()
    })
}
