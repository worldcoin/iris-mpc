use eyre::Result;
use iris_mpc_common::postgres::{AccessMode, PostgresClient};
use sqlx::{PgPool, Postgres, Transaction};

use crate::anon_stats::types::{AnonStatsOrigin, DistanceBundle1D, LiftedDistanceBundle1D};

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

    /// Get number of available anon stats entries from the DB for the given origin.
    pub async fn num_available_anon_stats(&self, origin: AnonStatsOrigin) -> Result<i64> {
        let row: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM anon_stats_1d WHERE processed = FALSE and origin = $1
            "#,
        )
        .bind(i16::from(origin))
        .fetch_one(&self.pool)
        .await?;

        Ok(row.0)
    }

    /// Get number of available lifted anon stats entries from the DB for the given origin.
    pub async fn num_available_anon_stats_lifted(&self, origin: AnonStatsOrigin) -> Result<i64> {
        let row: (i64,) = sqlx::query_as(
            r#"
            SELECT COUNT(*) FROM anon_stats_1d_lifted WHERE processed = FALSE and origin = $1
            "#,
        )
        .bind(i16::from(origin))
        .fetch_one(&self.pool)
        .await?;

        Ok(row.0)
    }
    /// Get available anon stats entries from the DB for the given origin, up to the given limit.
    /// Returns a tuple of (ids, Vec<(match_id, DistanceBundle1D)>)
    pub async fn get_available_anon_stats(
        &self,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, DistanceBundle1D)>)> {
        let res: Vec<(i64, i64, Vec<u8>)> = sqlx::query_as(
            r#"
            SELECT id, match_id, bundle FROM anon_stats_1d WHERE processed = FALSE and origin = $1
            ORDER BY id ASC
            LIMIT $2
            "#,
        )
        .bind(i16::from(origin))
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let (ids, distance_bundles) = res
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
                Result::<_, eyre::Report>::Ok((id, (match_id, bundle)))
            })
            .collect::<Result<(Vec<_>, Vec<_>), eyre::Report>>()?;

        Ok((ids, distance_bundles))
    }

    /// Get available lifted anon stats entries from the DB for the given origin, up to the given limit.
    /// Returns a tuple of (ids, Vec<(match_id, LiftedDistanceBundle1D)>)
    pub async fn get_available_anon_stats_lifted(
        &self,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, LiftedDistanceBundle1D)>)> {
        let res: Vec<(i64, i64, Vec<u8>)> = sqlx::query_as(
            r#"
            SELECT id, match_id, bundle FROM anon_stats_1d_lifted WHERE processed = FALSE and origin = $1
            ORDER BY id ASC
            LIMIT $2
            "#,
        )
        .bind(i16::from(origin))
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let (ids, distance_bundles) = res
            .into_iter()
            .map(|(id, match_id, bundle_bytes)| {
                let bundle: LiftedDistanceBundle1D =
                    bincode::deserialize(&bundle_bytes).map_err(|e| {
                        eyre::eyre!(
                            "Failed to deserialize DistanceBundle1D for anon_stats id {}: {:?}",
                            id,
                            e
                        )
                    })?;
                Result::<_, eyre::Report>::Ok((id, (match_id, bundle)))
            })
            .collect::<Result<(Vec<_>, Vec<_>), eyre::Report>>()?;

        Ok((ids, distance_bundles))
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
    pub async fn mark_lifted_anon_stats_processed(&self, ids: &[i64]) -> Result<()> {
        sqlx::query(
            r#"
            UPDATE anon_stats_1d_lifted SET processed = TRUE WHERE id = ANY($1)
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
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        if anon_stats.is_empty() {
            return Ok(());
        }
        let origin = i16::from(origin);
        let mut tx = self.pool.begin().await?;
        for chunk in anon_stats.chunks(Self::ANON_STATS_1D_INSERT_BATCH_SIZE) {
            let mapped_chunk = chunk.iter().map(|(id, bundle)| {
                let bundle_bytes =
                    bincode::serialize(bundle).expect("Failed to serialize DistanceBundle1D");
                (id, bundle_bytes)
            });
            let mut query =
                sqlx::QueryBuilder::new(r#"INSERT INTO anon_stats_1d (match_id, bundle, origin)"#);
            query.push_values(mapped_chunk, |mut query, (id, bytes)| {
                query.push_bind(id);
                query.push_bind(bytes);
                query.push_bind(origin);
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

    pub async fn insert_anon_stats_batch_lifted(
        &self,
        anon_stats: &[(i64, LiftedDistanceBundle1D)],
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        if anon_stats.is_empty() {
            return Ok(());
        }
        let origin = i16::from(origin);
        let mut tx = self.pool.begin().await?;
        for chunk in anon_stats.chunks(Self::ANON_STATS_1D_INSERT_BATCH_SIZE) {
            let mapped_chunk = chunk.iter().map(|(id, bundle)| {
                let bundle_bytes =
                    bincode::serialize(bundle).expect("Failed to serialize DistanceBundle1D");
                (id, bundle_bytes)
            });
            let mut query = sqlx::QueryBuilder::new(
                r#"INSERT INTO anon_stats_1d_lifted (match_id, bundle, origin)"#,
            );
            query.push_values(mapped_chunk, |mut query, (id, bytes)| {
                query.push_bind(id);
                query.push_bind(bytes);
                query.push_bind(origin);
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
