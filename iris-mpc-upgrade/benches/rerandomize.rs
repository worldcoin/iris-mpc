use criterion::{criterion_group, criterion_main, BatchSize};
use iris_mpc_common::{IRIS_CODE_LENGTH, MASK_CODE_LENGTH};

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
}

criterion_group!(benches, iris_rerandomize);

criterion_main!(benches);
