#![cfg(feature = "db_dependent")]

use eyre::Result;
use iris_mpc_common::{
    postgres::{run_migrations, AccessMode, PostgresClient},
    VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{
        iris_worker::{cache_iris, init_workers, IrisWorkerPool, LocalIrisWorkerPool},
        RIGHT,
    },
    hawkers::{
        aby3::aby3_store::{Aby3Store, DistanceMode, FhdOps},
        shared_irises::SharedIrises,
    },
    protocol::shared_iris::GaloisRingSharedIris,
};
use iris_mpc_store::{
    test_utils::{cleanup, temporary_name, test_db_url},
    Store, StoredIrisRef,
};
use std::{collections::HashMap, sync::Arc};

#[tokio::test]
async fn cold_eye_dot_product_matches_resident_without_retaining_target() -> Result<()> {
    let schema_name = temporary_name();
    let postgres =
        PostgresClient::new(&test_db_url()?, &schema_name, AccessMode::ReadWrite).await?;
    run_migrations(&postgres.pool, false).await?;
    let db = Store::new(&postgres).await?;

    let target = Arc::new(GaloisRingSharedIris::default_for_party(0));
    let vector_id = VectorId::new(1, 0);
    let mut tx = db.tx().await?;
    db.insert_irises(
        &mut tx,
        &[StoredIrisRef {
            id: 1,
            left_code: &target.code.coefs,
            left_mask: &target.mask.coefs,
            right_code: &target.code.coefs,
            right_mask: &target.mask.coefs,
        }],
    )
    .await?;
    tx.commit().await?;

    let resident_store =
        Aby3Store::<FhdOps>::new_storage(Some(HashMap::from([(vector_id, target.clone())])))
            .to_arc();
    let resident: Arc<dyn IrisWorkerPool> = Arc::new(LocalIrisWorkerPool::new_local(
        resident_store,
        DistanceMode::MinRotation,
        0,
    ));

    let cold_store = SharedIrises::to_arc(Aby3Store::<FhdOps>::new_storage(None));
    let cold: Arc<dyn IrisWorkerPool> = Arc::new(LocalIrisWorkerPool::new_cold(
        init_workers(RIGHT, cold_store.clone(), false),
        cold_store.clone(),
        DistanceMode::MinRotation,
        0,
        db,
        RIGHT,
    ));

    let query = Arc::new(GaloisRingSharedIris::default_for_party(0));
    let resident_query = cache_iris(resident.as_ref(), query.clone()).await?;
    let cold_query = cache_iris(cold.as_ref(), query).await?;
    let resident_result = resident
        .compute_dot_products_full_rotations(resident_query, vec![vector_id])
        .await?;
    let cold_result = cold
        .compute_dot_products_full_rotations(cold_query, vec![vector_id])
        .await?;

    assert_eq!(cold_result, resident_result);
    assert!(cold_store.get_vector(&vector_id).await.is_none());

    cleanup(&postgres, &schema_name).await?;
    Ok(())
}
