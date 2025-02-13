use criterion::{criterion_group, criterion_main, Criterion};
use iris_mpc_common::{
    galois_engine::degree4::{
        FullGaloisRingIrisCodeShare, GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
    },
    helpers::smpc_request::UNIQUENESS_MESSAGE_TYPE,
    iris_db::iris::IrisCode,
    job::BatchQuery,
};
use iris_mpc_gpu::server::PreprocessedBatchQuery;
use rand::thread_rng;
use uuid::Uuid;

pub fn criterion_benchmark_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Preprocessing");
    for batch_size in [1, 4, 16, 64] {
        let mut batch_query = BatchQuery::default();
        for _ in 0..batch_size {
            let iris = IrisCode::random_rng(&mut thread_rng());
            let [shares, _, _] =
                FullGaloisRingIrisCodeShare::encode_iris_code(&iris, &mut thread_rng());
            let code_shares_store = shares.code;
            let mask_shares_store = shares.mask;
            let mut code_shares_query = code_shares_store.clone();
            let mut mask_shares_query = mask_shares_store.clone();
            GaloisRingIrisCodeShare::preprocess_iris_code_query_share(&mut code_shares_query);
            GaloisRingTrimmedMaskCodeShare::preprocess_mask_code_query_share(
                &mut mask_shares_query,
            );
            let code_shares_query = code_shares_query.all_rotations();
            let mask_shares_query = mask_shares_query.all_rotations();
            let code_shares_db = code_shares_store.all_rotations();
            let mask_shares_db = mask_shares_store.all_rotations();
            batch_query.request_ids.push(Uuid::new_v4().to_string());
            batch_query
                .request_types
                .push(UNIQUENESS_MESSAGE_TYPE.to_owned());
            batch_query
                .query_left
                .code
                .extend(code_shares_query.clone());
            batch_query.query_right.code.extend(code_shares_query);
            batch_query
                .query_left
                .mask
                .extend(mask_shares_query.clone());
            batch_query.query_right.mask.extend(mask_shares_query);

            batch_query.store_left.code.push(code_shares_store.clone());
            batch_query.store_right.code.push(code_shares_store);
            batch_query.store_left.mask.push(mask_shares_store.clone());
            batch_query.store_right.mask.push(mask_shares_store);

            batch_query.db_left.code.extend(code_shares_db.clone());
            batch_query.db_right.code.extend(code_shares_db);
            batch_query.db_left.mask.extend(mask_shares_db.clone());
            batch_query.db_right.mask.extend(mask_shares_db);
        }

        group.bench_function(format!("batch_size={}", batch_size), |b| {
            b.iter_batched(
                || batch_query.clone(),
                |batch| {
                    let _ = PreprocessedBatchQuery::from(batch);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

criterion_group!(
    name = preprocessing;
    config = Criterion::default();
    targets = criterion_benchmark_preprocessing
);
criterion_main!(preprocessing);
