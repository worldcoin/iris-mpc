use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use sqlx::{PgPool, Postgres, Transaction};

use crate::anon_stats::types::DistanceBundle1D;

#[derive(Clone, Debug)]
pub struct AnonStatsStore {
    pub pool: PgPool,
    pub schema_name: String,
}

impl AnonStatsStore {
    pub async fn new(postgres_client: &PostgresClient) -> Result<Self> {
        tracing::info!(
            "Created anon-stats-mpc-store with schema: {}",
            postgres_client.schema_name
        );

        if postgres_client.access_mode == AccessMode::ReadOnly {
            tracing::info!("Not migrating anon-stats-mpc-store DB in read-only mode");
        } else {
            sqlx::migrate!("./anon_stats_migrations/")
                .run(&postgres_client.pool)
                .await?;
        }

        Ok(AnonStatsStore {
            pool: postgres_client.pool.clone(),
            schema_name: postgres_client.schema_name.to_string(),
        })
    }

    pub async fn tx(&self) -> Result<Transaction<'_, Postgres>> {
        Ok(self.pool.begin().await?)
    }

    pub async fn num_available_anon_stats(&self) -> Result<i64> {
        let row: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM anon_stats_1d WHERE processed = FALSE
            "#,
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(row.0)
    }
    pub async fn get_available_anon_stats(
        &self,
        limit: usize,
    ) -> Result<Vec<(i64, DistanceBundle1D)>> {
        let res: Vec<(i64, i64, Vec<u8>)> = sqlx::query_as(
            r#"
            SELECT (id, match_id,bundle) FROM anon_stats_1d WHERE processed = FALSE
            ORDER BY id ASC
            LIMIT $1
            "#,
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let distance_bundles = res
            .into_iter()
            .map(|(id, match_id, bundle_bytes)| {
                let bundle: DistanceBundle1D =
                    bincode::deserialize(&bundle_bytes).map_err(|e| {
                        eyre::eyre!(
                            "Failed to deserialize DistanceBundle1D for anon_stats id {}: {:?}",
                            id,
                            e
                        )
                    })?;
                Result::<_, eyre::Report>::Ok((match_id, bundle))
            })
            .collect::<Result<Vec<_>, eyre::Report>>()?;

        Ok(distance_bundles)
    }

    pub async fn mark_anon_stats_processed(&self, ids: &[i64]) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE anon_stats_1d SET processed = TRUE WHERE id = ANY($1)
            "#,
        )
        .bind(ids)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    const ANON_STATS_1D_INSERT_BATCH_SIZE: usize = 10000;

    pub async fn insert_anon_stats_batch(
        &self,
        anon_stats: &[(i64, DistanceBundle1D)],
    ) -> Result<()> {
        if anon_stats.is_empty() {
            return Ok(());
        }
        let mut tx = self.pool.begin().await?;
        for chunk in anon_stats.chunks(Self::ANON_STATS_1D_INSERT_BATCH_SIZE) {
            let mapped_chunk = chunk.iter().map(|(id, bundle)| {
                let bundle_bytes =
                    bincode::serialize(bundle).expect("Failed to serialize DistanceBundle1D");
                (id, bundle_bytes)
            });
            let mut query = sqlx::QueryBuilder::new(
                r#"INSERT INTO anon_stats_1d (match_id, bundle, processed)"#,
            );
            query.push_values(mapped_chunk, |mut query, (id, bytes)| {
                query.push_bind(id);
                query.push_bind(bytes);
            });

            let res = query.build().execute(&mut *tx).await?;
            if res.rows_affected() != chunk.len() as u64 {
                return Err(eyre::eyre!(
                    "Expected to insert {} rows, but only inserted {} rows",
                    chunk.len(),
                    res.rows_affected()
                ));
            }
        }
        tx.commit().await?;

        Ok(())
    }
}
