use eyre::Result;
use iris_mpc_common::postgres::PostgresClient;
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

        postgres_client.migrate().await;

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
}
