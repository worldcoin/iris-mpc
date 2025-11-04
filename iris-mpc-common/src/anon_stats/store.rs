use crate::postgres::{AccessMode, PostgresClient};
use eyre::Result;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Postgres, Transaction};

use super::types::AnonStatsOrigin;

#[derive(Clone, Debug)]
pub struct AnonStatsStore {
    pub pool: PgPool,
    pub schema_name: String,
}

const ANON_STATS_1D_TABLE: &str = "anon_stats_1d";
const ANON_STATS_1D_LIFTED_TABLE: &str = "anon_stats_1d_lifted";
const ANON_STATS_2D_TABLE: &str = "anon_stats_2d";
const ANON_STATS_2D_LIFTED_TABLE: &str = "anon_stats_2d_lifted";

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

    async fn num_available_anon_stats(
        &self,
        table_name: &'static str,
        origin: AnonStatsOrigin,
    ) -> Result<i64> {
        let row: (i64,) = sqlx::query_as(
            &[
                r#"
            SELECT COUNT(*) FROM "#,
                table_name,
                r#" WHERE processed = FALSE and origin = $1
            "#,
            ]
            .concat(),
        )
        .bind(i16::from(origin))
        .fetch_one(&self.pool)
        .await?;

        Ok(row.0)
    }
    /// Get number of available lifted anon stats entries from the DB for the given origin.
    pub async fn num_available_anon_stats_1d(&self, origin: AnonStatsOrigin) -> Result<i64> {
        self.num_available_anon_stats(ANON_STATS_1D_TABLE, origin)
            .await
    }
    /// Get number of available lifted anon stats entries from the DB for the given origin.
    pub async fn num_available_anon_stats_1d_lifted(&self, origin: AnonStatsOrigin) -> Result<i64> {
        self.num_available_anon_stats(ANON_STATS_1D_LIFTED_TABLE, origin)
            .await
    }
    /// Get number of available lifted anon stats entries from the DB for the given origin.
    pub async fn num_available_anon_stats_2d(&self, origin: AnonStatsOrigin) -> Result<i64> {
        self.num_available_anon_stats(ANON_STATS_2D_TABLE, origin)
            .await
    }
    /// Get number of available lifted anon stats entries from the DB for the given origin.
    pub async fn num_available_anon_stats_2d_lifted(&self, origin: AnonStatsOrigin) -> Result<i64> {
        self.num_available_anon_stats(ANON_STATS_2D_LIFTED_TABLE, origin)
            .await
    }

    async fn get_available_anon_stats<T: for<'a> Deserialize<'a>>(
        &self,
        table_name: &'static str,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, T)>)> {
        let res: Vec<(i64, i64, Vec<u8>)> = sqlx::query_as(
            &[
                r#"
            SELECT id, match_id, bundle FROM "#,
                table_name,
                r#" WHERE processed = FALSE and origin = $1
            ORDER BY id ASC
            LIMIT $2
            "#,
            ]
            .concat(),
        )
        .bind(i16::from(origin))
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;

        let (ids, distance_bundles) = res
            .into_iter()
            .map(|(id, match_id, bundle_bytes)| {
                let bundle: T = bincode::deserialize(&bundle_bytes).map_err(|e| {
                    eyre::eyre!(
                        "Failed to deserialize distance bundle from table {} for anon_stats id {}: {:?}",
                        table_name,
                        id,
                        e
                    )
                })?;
                Result::<_, eyre::Report>::Ok((id, (match_id, bundle)))
            })
            .collect::<Result<(Vec<_>, Vec<_>), eyre::Report>>()?;

        Ok((ids, distance_bundles))
    }

    /// Get available anon stats entries from the DB for the given origin, up to the given limit.
    /// Returns a tuple of (ids, Vec<(match_id, T)>)
    pub async fn get_available_anon_stats_1d<T: for<'a> Deserialize<'a>>(
        &self,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, T)>)> {
        self.get_available_anon_stats(ANON_STATS_1D_TABLE, origin, limit)
            .await
    }
    /// Get available lifted anon stats entries from the DB for the given origin, up to the given limit.
    /// Returns a tuple of (ids, Vec<(match_id, T)>)
    pub async fn get_available_anon_stats_1d_lifted<T: for<'a> Deserialize<'a>>(
        &self,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, T)>)> {
        self.get_available_anon_stats(ANON_STATS_1D_LIFTED_TABLE, origin, limit)
            .await
    }
    /// Get available anon stats entries from the DB for the given origin, up to the given limit.
    /// Returns a tuple of (ids, Vec<(match_id, T)>)
    pub async fn get_available_anon_stats_2d<T: for<'a> Deserialize<'a>>(
        &self,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, T)>)> {
        self.get_available_anon_stats(ANON_STATS_2D_TABLE, origin, limit)
            .await
    }
    /// Get available lifted anon stats entries from the DB for the given origin, up to the given limit.
    /// Returns a tuple of (ids, Vec<(match_id, T)>)
    pub async fn get_available_anon_stats_2d_lifted<T: for<'a> Deserialize<'a>>(
        &self,
        origin: AnonStatsOrigin,
        limit: usize,
    ) -> Result<(Vec<i64>, Vec<(i64, T)>)> {
        self.get_available_anon_stats(ANON_STATS_2D_LIFTED_TABLE, origin, limit)
            .await
    }

    async fn mark_anon_stats_processed(&self, table_name: &'static str, ids: &[i64]) -> Result<()> {
        sqlx::query(
            &[
                "UPDATE ",
                table_name,
                r#" SET processed = TRUE WHERE id = ANY($1)
            "#,
            ]
            .concat(),
        )
        .bind(ids)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
    pub async fn mark_anon_stats_processed_1d(&self, ids: &[i64]) -> Result<()> {
        self.mark_anon_stats_processed(ANON_STATS_1D_TABLE, ids)
            .await
    }
    pub async fn mark_anon_stats_processed_1d_lifted(&self, ids: &[i64]) -> Result<()> {
        self.mark_anon_stats_processed(ANON_STATS_1D_LIFTED_TABLE, ids)
            .await
    }
    pub async fn mark_anon_stats_processed_2d(&self, ids: &[i64]) -> Result<()> {
        self.mark_anon_stats_processed(ANON_STATS_2D_TABLE, ids)
            .await
    }
    pub async fn mark_anon_stats_processed_2d_lifted(&self, ids: &[i64]) -> Result<()> {
        self.mark_anon_stats_processed(ANON_STATS_2D_LIFTED_TABLE, ids)
            .await
    }

    async fn clear_unprocessed_anon_stats(
        &self,
        table_name: &'static str,
        origin: AnonStatsOrigin,
    ) -> Result<u64> {
        let result = sqlx::query(
            &[
                "DELETE FROM ",
                table_name,
                r#" WHERE processed = FALSE AND origin = $1"#,
            ]
            .concat(),
        )
        .bind(i16::from(origin))
        .execute(&self.pool)
        .await?;
        Ok(result.rows_affected())
    }

    pub async fn clear_unprocessed_anon_stats_1d(&self, origin: AnonStatsOrigin) -> Result<u64> {
        self.clear_unprocessed_anon_stats(ANON_STATS_1D_TABLE, origin)
            .await
    }

    pub async fn clear_unprocessed_anon_stats_1d_lifted(
        &self,
        origin: AnonStatsOrigin,
    ) -> Result<u64> {
        self.clear_unprocessed_anon_stats(ANON_STATS_1D_LIFTED_TABLE, origin)
            .await
    }

    pub async fn clear_unprocessed_anon_stats_2d(&self, origin: AnonStatsOrigin) -> Result<u64> {
        self.clear_unprocessed_anon_stats(ANON_STATS_2D_TABLE, origin)
            .await
    }

    pub async fn clear_unprocessed_anon_stats_2d_lifted(
        &self,
        origin: AnonStatsOrigin,
    ) -> Result<u64> {
        self.clear_unprocessed_anon_stats(ANON_STATS_2D_LIFTED_TABLE, origin)
            .await
    }

    const ANON_STATS_INSERT_BATCH_SIZE: usize = 10000;

    async fn insert_anon_stats_batch<T: Serialize>(
        &self,
        table_name: &'static str,
        anon_stats: &[(i64, T)],
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        if anon_stats.is_empty() {
            return Ok(());
        }
        let origin = i16::from(origin);
        let mut tx = self.pool.begin().await?;
        for chunk in anon_stats.chunks(Self::ANON_STATS_INSERT_BATCH_SIZE) {
            let mapped_chunk = chunk.iter().map(|(id, bundle)| {
                let bundle_bytes =
                    bincode::serialize(bundle).expect("Failed to serialize DistanceBundle");
                (id, bundle_bytes)
            });
            let mut query = sqlx::QueryBuilder::new(
                ["INSERT INTO ", table_name, r#" (match_id, bundle, origin)"#].concat(),
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

    pub async fn insert_anon_stats_batch_1d<T: Serialize>(
        &self,
        anon_stats: &[(i64, T)],
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        self.insert_anon_stats_batch(ANON_STATS_1D_TABLE, anon_stats, origin)
            .await
    }
    pub async fn insert_anon_stats_batch_1d_lifted<T: Serialize>(
        &self,
        anon_stats: &[(i64, T)],
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        self.insert_anon_stats_batch(ANON_STATS_1D_LIFTED_TABLE, anon_stats, origin)
            .await
    }
    pub async fn insert_anon_stats_batch_2d<T: Serialize>(
        &self,
        anon_stats: &[(i64, T)],
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        self.insert_anon_stats_batch(ANON_STATS_2D_TABLE, anon_stats, origin)
            .await
    }
    pub async fn insert_anon_stats_batch_2d_lifted<T: Serialize>(
        &self,
        anon_stats: &[(i64, T)],
        origin: AnonStatsOrigin,
    ) -> Result<()> {
        self.insert_anon_stats_batch(ANON_STATS_2D_LIFTED_TABLE, anon_stats, origin)
            .await
    }
}
