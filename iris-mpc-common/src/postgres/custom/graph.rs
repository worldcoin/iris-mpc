use crate::{IrisVectorId, SortedEdgeIds};
use eyre::Result;
use futures::TryStreamExt;
use sqlx::{types::Json, PgPool, Row};

const BATCH_SIZE: usize = 500;

pub async fn entry_points_to_bytea(pool: &PgPool) -> Result<()> {
    // todo: verify that this sql query is correct
    let mut stream = sqlx::query(
        r#"
        SELECT graph_id, entry_point
        FROM hawk_graph_entry
        WHERE entry_point NOT NULL
        "#,
    )
    .fetch(pool);

    let mut buffer = Vec::with_capacity(BATCH_SIZE);
    while let Some(row) = stream.try_next().await.expect("failed to get next") {
        let graph_id: i32 = row.get("graph_id");
        let entry_point: Json<IrisVectorId> = row.get("entry_point");

        let entry_point = entry_point.0.to_packed()?;
        buffer.push((graph_id, entry_point));

        if buffer.len() >= BATCH_SIZE {
            batch_entry_point_to_bytea(pool, &buffer).await?;
            buffer.clear();
        }
    }

    if !buffer.is_empty() {
        batch_entry_point_to_bytea(pool, &buffer).await?;
    }
    Ok(())
}

pub async fn links_to_bytea(pool: &PgPool) -> Result<()> {
    let mut stream = sqlx::query(
        r#"
        SELECT graph_id, source_ref, layer, links
        FROM hawk_graph_links
        "#,
    )
    .fetch(pool);

    let mut buffer = Vec::with_capacity(BATCH_SIZE);
    while let Some(row) = stream.try_next().await.expect("failed to get next") {
        let graph_id: i32 = row.get("graph_id");
        let source_ref: String = row.get("source_ref");
        let layer: i32 = row.get("layer");
        let links: Json<SortedEdgeIds<IrisVectorId>> = row.get("links");

        let links = links.0.to_packed()?;
        buffer.push((graph_id, source_ref, layer, links));

        if buffer.len() >= BATCH_SIZE {
            batch_links_to_bytea(pool, &buffer).await?;
            buffer.clear();
        }
    }

    if !buffer.is_empty() {
        batch_links_to_bytea(pool, &buffer).await?;
    }
    Ok(())
}

async fn batch_links_to_bytea(
    pool: &PgPool,
    batch: &[(i32, String, i32, Vec<u8>)],
) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;

    for (graph_id, source_ref, layer, links) in batch {
        sqlx::query(
            r#"
            UPDATE hawk_graph_links
            SET links_b = $1
            WHERE graph_id = $2 AND source_ref = $3 AND layer = $4
            "#,
        )
        .bind(links)
        .bind(graph_id)
        .bind(source_ref)
        .bind(layer)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await
}

async fn batch_entry_point_to_bytea(
    pool: &PgPool,
    batch: &[(i32, Vec<u8>)],
) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;

    for (graph_id, entry_point) in batch {
        sqlx::query(
            r#"
            UPDATE hawk_graph_entry
            SET entry_point_b = $1
            WHERE graph_id = $2
            "#,
        )
        .bind(entry_point)
        .bind(graph_id)
        .execute(&mut *tx)
        .await?;
    }
    tx.commit().await
}
