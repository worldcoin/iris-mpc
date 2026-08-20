#![cfg(feature = "db_dependent")]

use eyre::Result;
use iris_mpc_common::{
    postgres::{run_migrations, AccessMode, PostgresClient},
    VectorId,
};
use iris_mpc_cpu::{
    execution::hawk_main::{
        iris_worker::{
            cache_iris, init_workers, ColdStorageInit, IrisWorkerPool, LocalIrisWorkerPool, QueryId,
        },
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
async fn cold_eye_prefetched_dot_product_matches_resident_and_populates_lfu() -> Result<()> {
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
    let cold: Arc<dyn IrisWorkerPool> = Arc::new(
        LocalIrisWorkerPool::new_cold(
            init_workers(RIGHT, cold_store.clone(), false),
            cold_store.clone(),
            DistanceMode::MinRotation,
            0,
            ColdStorageInit {
                store: db.clone(),
                side: RIGHT,
                luc_window_ids: Vec::new(),
                luc_window_capacity: 0,
                lfu_cache_capacity: 8,
            },
        )
        .await?,
    );

    // Prefetch the old version, then advance Postgres. The cold computation
    // must consume the prefetched record rather than issue a late DB read.
    cold.prefetch_irises(vec![vector_id]).await?;
    cold.wait_for_prefetch().await?;
    let replacement = GaloisRingSharedIris::dummy_for_party(0);
    db.update_iris(
        None,
        1,
        &replacement.code,
        &replacement.mask,
        &replacement.code,
        &replacement.mask,
    )
    .await?;

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

    // Consuming the prefetch promotes this immutable version into TinyLFU.
    // Postgres now contains v1, so a second v0 read can only succeed from RAM.
    assert_eq!(cold.fetch_irises(vec![vector_id]).await?[0], target);

    // A cache hit for v0 must not mask v1. A foreground v1 read comes from the
    // database and is promoted independently under its exact VectorId.
    let next_id = vector_id.next_version();
    let next = cold.fetch_irises(vec![next_id]).await?;
    assert_eq!(*next[0], replacement);
    db.update_iris(
        None,
        1,
        &target.code,
        &target.mask,
        &target.code,
        &target.mask,
    )
    .await?;
    assert_eq!(*cold.fetch_irises(vec![next_id]).await?[0], replacement);
    assert_eq!(
        cold.fetch_irises(vec![next_id.next_version()]).await?[0],
        target
    );

    cleanup(&postgres, &schema_name).await?;
    Ok(())
}

#[tokio::test]
async fn cold_eye_luc_window_rolls_forward_and_survives_persistence_ack() -> Result<()> {
    let schema_name = temporary_name();
    let postgres =
        PostgresClient::new(&test_db_url()?, &schema_name, AccessMode::ReadWrite).await?;
    run_migrations(&postgres.pool, false).await?;
    let db = Store::new(&postgres).await?;

    let original = Arc::new(GaloisRingSharedIris::default_for_party(0));
    let vector_id = VectorId::new(1, 0);
    let mut tx = db.tx().await?;
    db.insert_irises(
        &mut tx,
        &[StoredIrisRef {
            id: 1,
            left_code: &original.code.coefs,
            left_mask: &original.mask.coefs,
            right_code: &original.code.coefs,
            right_mask: &original.mask.coefs,
        }],
    )
    .await?;
    tx.commit().await?;

    let cold_store = SharedIrises::to_arc(Aby3Store::<FhdOps>::new_storage(None));
    let cold: Arc<dyn IrisWorkerPool> = Arc::new(
        LocalIrisWorkerPool::new_cold(
            init_workers(RIGHT, cold_store.clone(), false),
            cold_store,
            DistanceMode::MinRotation,
            0,
            ColdStorageInit {
                store: db.clone(),
                side: RIGHT,
                luc_window_ids: vec![vector_id],
                luc_window_capacity: 1,
                lfu_cache_capacity: 8,
            },
        )
        .await?,
    );

    // Change Postgres after startup. The old current LUC version must still be
    // served from RAM rather than becoming a miss against the newer DB row.
    let replacement = Arc::new(GaloisRingSharedIris::dummy_for_party(0));
    db.update_iris(
        None,
        1,
        &replacement.code,
        &replacement.mask,
        &replacement.code,
        &replacement.mask,
    )
    .await?;
    assert_eq!(cold.fetch_irises(vec![vector_id]).await?[0], original);

    // The actor-side update advances the rolling cache immediately. Releasing
    // its pending-write entry after commit must not evict the LUC copy.
    let query_id = QueryId::new();
    cold.cache_queries(vec![(query_id, replacement.clone())])
        .await?;
    let next_id = vector_id.next_version();
    cold.insert_irises(vec![(query_id, next_id)]).await?;
    assert_eq!(cold.fetch_irises(vec![next_id]).await?[0], replacement);
    assert_eq!(cold.acknowledge_persisted_irises(vec![next_id]).await, 1);
    assert_eq!(cold.fetch_irises(vec![next_id]).await?[0], replacement);

    cleanup(&postgres, &schema_name).await?;
    Ok(())
}

#[tokio::test]
async fn cold_eye_version_miss_fails_prefetch_and_foreground_fetch() -> Result<()> {
    let schema_name = temporary_name();
    let postgres =
        PostgresClient::new(&test_db_url()?, &schema_name, AccessMode::ReadWrite).await?;
    run_migrations(&postgres.pool, false).await?;
    let db = Store::new(&postgres).await?;

    let stored = GaloisRingSharedIris::dummy_for_party(0);
    let current_id = VectorId::new(1, 0);
    let mut tx = db.tx().await?;
    db.insert_irises(
        &mut tx,
        &[StoredIrisRef {
            id: 1,
            left_code: &stored.code.coefs,
            left_mask: &stored.mask.coefs,
            right_code: &stored.code.coefs,
            right_mask: &stored.mask.coefs,
        }],
    )
    .await?;
    tx.commit().await?;

    let cold_store = SharedIrises::to_arc(Aby3Store::<FhdOps>::new_storage(None));
    let cold: Arc<dyn IrisWorkerPool> = Arc::new(
        LocalIrisWorkerPool::new_cold(
            init_workers(RIGHT, cold_store.clone(), false),
            cold_store,
            DistanceMode::MinRotation,
            0,
            ColdStorageInit {
                store: db,
                side: RIGHT,
                luc_window_ids: Vec::new(),
                luc_window_capacity: 0,
                lfu_cache_capacity: 8,
            },
        )
        .await?,
    );

    // Model the registry being one version ahead of Postgres. Neither the
    // asynchronous prefetch nor its foreground fallback may fabricate an
    // all-zero iris for this exact-version miss.
    let registry_id = current_id.next_version();
    cold.prefetch_irises(vec![registry_id]).await?;
    let prefetch_error = cold.wait_for_prefetch().await.unwrap_err();
    assert!(
        format!("{prefetch_error:#}").contains("cold-eye database missing"),
        "unexpected prefetch error: {prefetch_error:#}"
    );

    let foreground_error = cold.fetch_irises(vec![registry_id]).await.unwrap_err();
    assert!(
        format!("{foreground_error:#}").contains("cold-eye database missing"),
        "unexpected foreground error: {foreground_error:#}"
    );

    cleanup(&postgres, &schema_name).await?;
    Ok(())
}
