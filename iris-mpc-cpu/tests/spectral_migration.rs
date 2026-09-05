#![cfg(feature = "db_dependent")]

use ampc_actor_utils::execution::local::LocalRuntime;
use eyre::Result;
use futures::future::try_join_all;
use iris_mpc_common::{
    iris_db::iris::IrisCode,
    postgres::{run_migrations, AccessMode, PostgresClient},
    VectorId,
};
use iris_mpc_cpu::protocol::{
    ntt::persistence::load_or_convert_batch, shared_iris::GaloisRingSharedIris,
};
use iris_mpc_store::{
    test_utils::{cleanup, temporary_name, test_db_url},
    Store, StoredIrisRef,
};
use rand::{rngs::StdRng, SeedableRng};

#[tokio::test]
async fn migration_reuses_complete_generations_and_repairs_partial_writes() -> Result<()> {
    let mut rng = StdRng::seed_from_u64(25_601);
    let plaintext = IrisCode::random_rng(&mut rng);
    let shares = GaloisRingSharedIris::generate_shares_locally(&mut rng, plaintext);
    let mut clients = Vec::new();
    let mut stores = Vec::new();
    for share in &shares {
        let schema = temporary_name();
        let client = PostgresClient::new(&test_db_url()?, &schema, AccessMode::ReadWrite).await?;
        run_migrations(&client.pool, false).await?;
        let store = Store::new(&client).await?;
        let mut tx = store.tx().await?;
        store
            .insert_irises(
                &mut tx,
                &[StoredIrisRef {
                    id: 1,
                    left_code: &share.code.coefs,
                    left_mask: &share.mask.coefs,
                    right_code: &share.code.coefs,
                    right_mask: &share.mask.coefs,
                }],
            )
            .await?;
        tx.commit().await?;
        clients.push((client, schema));
        stores.push(store);
    }
    let ids = [VectorId::new(1, 0)];
    let migrate = async || {
        let runtime = LocalRuntime::mock_setup_with_channel().await?;
        try_join_all(runtime.sessions.into_iter().zip(&stores).map(
            |(mut session, store)| async move {
                load_or_convert_batch(&mut session, store, 0, &ids).await
            },
        ))
        .await
    };
    let initial = migrate().await?;
    let reused = migrate().await?;
    for party in 0..3 {
        assert_eq!(
            initial[party][0].1.packed_bytes(),
            reused[party][0].1.packed_bytes()
        );
    }
    let before: Vec<Vec<u8>> = try_join_all(stores.iter().map(|store| async {
        sqlx::query_scalar(
            "SELECT generation FROM cpu_spectral_irises WHERE serial_id=1 AND side=0",
        )
        .fetch_one(&store.pool)
        .await
    }))
    .await?;
    assert!(before.windows(2).all(|w| w[0] == w[1]));
    // Simulate a crash after precisely one party committed a different sharing.
    sqlx::query("UPDATE cpu_spectral_irises SET generation=$1 WHERE serial_id=1 AND side=0")
        .bind([0x55u8; 16].as_slice())
        .execute(&stores[1].pool)
        .await?;
    migrate().await?;
    let after: Vec<Vec<u8>> = try_join_all(stores.iter().map(|store| async {
        sqlx::query_scalar(
            "SELECT generation FROM cpu_spectral_irises WHERE serial_id=1 AND side=0",
        )
        .fetch_one(&store.pool)
        .await
    }))
    .await?;
    assert!(after.windows(2).all(|w| w[0] == w[1]));
    assert_ne!(before[0], after[0]);
    // A cache from the old prime must force joint reconstruction, even when
    // the payload checksum and generation still agree with the other parties.
    sqlx::query("UPDATE cpu_spectral_irises SET format_version=1 WHERE serial_id=1 AND side=0")
        .execute(&stores[2].pool)
        .await?;
    migrate().await?;
    let refreshed: Vec<(i16, Vec<u8>)> = try_join_all(stores.iter().map(|store| async {
        sqlx::query_as("SELECT format_version,generation FROM cpu_spectral_irises WHERE serial_id=1 AND side=0")
            .fetch_one(&store.pool).await
    })).await?;
    assert!(refreshed.iter().all(|(version, _)| *version == 2));
    assert!(refreshed.windows(2).all(|w| w[0] == w[1]));
    assert_ne!(after[0], refreshed[0].1);
    // Mismatched authoritative versions must fail on every party before resharing.
    let runtime = LocalRuntime::mock_setup_with_channel().await?;
    let failures =
        futures::future::join_all(runtime.sessions.into_iter().zip(&stores).enumerate().map(
            |(party, (mut session, store))| async move {
                load_or_convert_batch(
                    &mut session,
                    store,
                    0,
                    &[VectorId::new(1, i16::from(party == 1))],
                )
                .await
            },
        ))
        .await;
    assert!(failures.iter().all(Result::is_err));
    for (client, schema) in clients {
        cleanup(&client, &schema).await?;
    }
    Ok(())
}
