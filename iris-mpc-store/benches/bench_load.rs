use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use eyre::{eyre, Result};
use futures::{StreamExt, TryStreamExt};
use iris_mpc_common::config::Config;
use iris_mpc_store::{Store, StoredIris, StoredIrisRef};
use tokio::runtime::Runtime;

const IRIS_CODE_LENGTH: usize = 12800;

async fn setup(count: usize) -> Result<(Store, String)> {
    let schema_name = temporary_name();
    let store = Store::new(&test_db_url()?, &schema_name).await?;

    let chunk_size = count.min(1_000);
    let chunk_count = count.div_ceil(chunk_size);
    assert_eq!(chunk_size * chunk_count, count);

    let iris = StoredIrisRef {
        left_code:  &[123_u16; IRIS_CODE_LENGTH],
        left_mask:  &[456_u16; IRIS_CODE_LENGTH],
        right_code: &[789_u16; IRIS_CODE_LENGTH],
        right_mask: &[101_u16; IRIS_CODE_LENGTH],
    };
    let codes_and_masks = vec![iris; chunk_size];

    for _ in 0..chunk_count {
        let mut tx = store.tx().await?;
        store.insert_irises(&mut tx, &codes_and_masks).await?;
        tx.commit().await?;
    }

    Ok((store, schema_name))
}

async fn load_collect(store: &Store, count_irises: usize, parallelism: usize) -> Result<()> {
    let data_len = count_irises * IRIS_CODE_LENGTH;
    let mut left_codes_db = vec![0; data_len];
    let mut left_masks_db = vec![0; data_len];
    let mut right_codes_db = vec![0; data_len];
    let mut right_masks_db = vec![0; data_len];

    let mut stream = store.stream_irises_par(parallelism).await;
    while let Some(iris) = stream.try_next().await? {
        if iris.index() >= count_irises {
            return Err(eyre!("Inconsistent iris index {}", iris.index()));
        }

        let start = iris.index() * IRIS_CODE_LENGTH;
        left_codes_db[start..start + IRIS_CODE_LENGTH].copy_from_slice(iris.left_code());
        left_masks_db[start..start + IRIS_CODE_LENGTH].copy_from_slice(iris.left_mask());
        right_codes_db[start..start + IRIS_CODE_LENGTH].copy_from_slice(iris.right_code());
        right_masks_db[start..start + IRIS_CODE_LENGTH].copy_from_slice(iris.right_mask());
    }

    black_box(left_codes_db);
    black_box(left_masks_db);
    black_box(right_codes_db);
    black_box(right_masks_db);
    Ok(())
}

async fn load_drop(store: &Store, _count_irises: usize, parallelism: usize) -> Result<()> {
    store
        .stream_irises_par(parallelism)
        .await
        .try_for_each_concurrent(None, |_iris| async { Ok(()) })
        .await?;
    Ok(())
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    println!("Runtime: {}", rt.metrics().num_workers());
    let count = 16_000;
    let (store, schema_name) = rt.block_on(setup(count)).unwrap();

    let mut c = c.benchmark_group(format!("load count={}", count));
    c.sample_size(10);

    for parallelism in [16, 4, 1] {
        c.throughput(Throughput::Elements(count as u64));

        c.bench_function(format!("drop parallelism={}", parallelism), |b| {
            b.to_async(&rt)
                .iter(|| async { load_drop(&store, count, parallelism).await.unwrap() })
        });

        c.bench_function(format!("collect parallelism={}", parallelism), |b| {
            b.to_async(&rt)
                .iter(|| async { load_collect(&store, count, parallelism).await.unwrap() })
        });
    }

    c.finish();
    rt.block_on(cleanup(&store, &schema_name)).unwrap();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// TODO: reuse from src/lib.rs
const APP_NAME: &str = "SMPC";
const DOTENV_TEST: &str = ".env.test";

fn test_db_url() -> Result<String> {
    dotenvy::from_filename(DOTENV_TEST)?;
    Ok(Config::load_config(APP_NAME)?
        .database
        .ok_or(eyre!("Missing database config"))?
        .url)
}

fn temporary_name() -> String {
    format!("smpc_test{}_0", rand::random::<u32>())
}

async fn cleanup(store: &Store, schema_name: &str) -> Result<()> {
    assert!(schema_name.starts_with("smpc_test"));
    sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", schema_name))
        .execute(&store.pool)
        .await?;
    Ok(())
}
