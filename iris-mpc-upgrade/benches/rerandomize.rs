use criterion::{criterion_group, criterion_main, BatchSize};
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn iris_rerandomize(c: &mut criterion::Criterion) {
    use iris_mpc_store::DbStoredIris;
    use iris_mpc_upgrade::rerandomization::randomize_iris;

    let master_seed = [123u8; 32];

    c.bench_function("iris_rerandomize", |b| {
        b.iter_batched(
            || {
                DbStoredIris::new(
                    42,
                    1,
                    vec![1u8; IRIS_CODE_LENGTH * std::mem::size_of::<u16>()],
                    vec![2u8; MASK_CODE_LENGTH * std::mem::size_of::<u16>()],
                    vec![3u8; IRIS_CODE_LENGTH * std::mem::size_of::<u16>()],
                    vec![4u8; MASK_CODE_LENGTH * std::mem::size_of::<u16>()],
                )
            },
            |iris| randomize_iris(iris, &master_seed, 0),
            BatchSize::SmallInput,
        );
    });
    c.bench_function("iris_rerandomize 1024", |b| {
        b.iter_batched(
            || {
                (0..1024)
                    .map(|_| {
                        DbStoredIris::new(
                            42,
                            1,
                            vec![1u8; IRIS_CODE_LENGTH * std::mem::size_of::<u16>()],
                            vec![2u8; MASK_CODE_LENGTH * std::mem::size_of::<u16>()],
                            vec![3u8; IRIS_CODE_LENGTH * std::mem::size_of::<u16>()],
                            vec![4u8; MASK_CODE_LENGTH * std::mem::size_of::<u16>()],
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |irises| {
                irises
                    .into_iter()
                    .map(|iris| randomize_iris(iris, &master_seed, 0))
                    .collect::<Vec<_>>()
            },
            BatchSize::LargeInput,
        );
    });
    c.bench_function("iris_rerandomize par_iter 1024", |b| {
        b.iter_batched(
            || {
                (0..1024)
                    .map(|_| {
                        DbStoredIris::new(
                            42,
                            1,
                            vec![1u8; IRIS_CODE_LENGTH * std::mem::size_of::<u16>()],
                            vec![2u8; MASK_CODE_LENGTH * std::mem::size_of::<u16>()],
                            vec![3u8; IRIS_CODE_LENGTH * std::mem::size_of::<u16>()],
                            vec![4u8; MASK_CODE_LENGTH * std::mem::size_of::<u16>()],
                        )
                    })
                    .collect::<Vec<_>>()
            },
            |irises| {
                irises
                    .into_par_iter()
                    .map(|iris| randomize_iris(iris, &master_seed, 0))
                    .collect::<Vec<_>>()
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, iris_rerandomize);

criterion_main!(benches);
