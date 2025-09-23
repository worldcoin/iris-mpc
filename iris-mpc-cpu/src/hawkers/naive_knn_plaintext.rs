use iris_mpc_common::iris_db::iris::IrisCode;
use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};

pub fn naive_knn(irises: Vec<IrisCode>, num_threads: usize) {
    let k = 320;
    let n = irises.len();

    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();

    let _results = pool.install(|| {
        (0..n)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|i| {
                let current_iris = &irises[i];
                let mut distances = irises
                    .iter()
                    .enumerate()
                    .map(|(j, other_iris)| (j, current_iris.get_distance(other_iris)))
                    .collect::<Vec<_>>();
                distances.select_nth_unstable_by(k - 1, |(_, d1), (_, d2)| d1.total_cmp(d2));
                distances.truncate(k);
                distances.sort_by(|(_, d1), (_, d2)| d1.total_cmp(d2));
                distances
            })
            .collect::<Vec<_>>()
    });
}
