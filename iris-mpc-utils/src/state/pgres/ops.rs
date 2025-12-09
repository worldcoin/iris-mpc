use eyre::Result;
use iris_mpc_common::{helpers::sync::Modification, IrisVectorId};
use iris_mpc_store::Store;
use sqlx::{Postgres, Transaction};
use std::ops::DerefMut;

/// Increments an Iris's version.
pub async fn increment_iris_version(
    tx: &mut Transaction<'_, Postgres>,
    serial_id: i64,
) -> Result<()> {
    let query = sqlx::query(
        r#"
            UPDATE irises SET version_id = version_id + 1
            WHERE id = $1;
            "#,
    )
    .bind(serial_id);
    query.execute(tx.deref_mut()).await?;

    Ok(())
}

/// Returns set of an Iris's versions - historical plus present.
pub async fn get_iris_vector_ids(store: &Store) -> Result<Vec<IrisVectorId>> {
    let ids: Vec<(i64, i16)> = sqlx::query_as(
        r#"
            SELECT
                id,
                version_id
            FROM irises
            ORDER BY id ASC;
            "#,
    )
    .fetch_all(&store.pool)
    .await?;

    let ids = ids
        .into_iter()
        .map(|(serial_id, version)| IrisVectorId::new(serial_id as u32, version))
        .collect();

    Ok(ids)
}

/// Updates a modification's status to COMPLETED and sets persisted flag to true.
pub async fn persist_modification(
    tx: &mut Transaction<'_, Postgres>,
    modification_id: i64,
) -> Result<()> {
    let query = sqlx::query(
        r#"
            UPDATE modifications SET status = 'COMPLETED', persisted = true
            WHERE id = $1;
            "#,
    )
    .bind(modification_id);
    query.execute(tx.deref_mut()).await?;

    Ok(())
}

/// Upserts an Iris modification.
pub async fn write_modification(
    tx: &mut Transaction<'_, Postgres>,
    m: &Modification,
) -> Result<()> {
    let query = sqlx::query(
        r#"
            INSERT INTO modifications (id, serial_id, request_type, s3_url, status, persisted)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO UPDATE
            SET serial_id = EXCLUDED.serial_id,
                request_type = EXCLUDED.request_type,
                s3_url = EXCLUDED.s3_url,
                status = EXCLUDED.status,
                persisted = EXCLUDED.persisted;
            "#,
    )
    .bind(m.id)
    .bind(m.serial_id)
    .bind(m.request_type.as_str())
    .bind(m.s3_url.as_ref())
    .bind(m.status.as_str())
    .bind(m.persisted);
    query.execute(tx.deref_mut()).await?;

    Ok(())
}
