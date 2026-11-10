//! Durable local handoff progress. Peer agreement stays in the server's existing
//! drain/reconciliation protocol; these operations do not establish consensus.
use eyre::{ensure, Result, WrapErr};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Postgres, Transaction};
use std::time::Duration;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum HandoffPhase {
    Requested,
    Draining,
    Drained,
    Activated,
}

#[derive(Clone, Debug, PartialEq, Eq, sqlx::FromRow)]
pub struct Handoff {
    pub handoff_id: Uuid,
    pub source: Uuid,
    pub destination: Uuid,
    pub phase: HandoffPhase,
    pub source_process: Option<Uuid>,
    pub candidate_processes: Option<Vec<Uuid>>,
    pub drain_report: Option<serde_json::Value>,
}

#[derive(Clone)]
pub struct CpuCellHandoffs {
    pool: PgPool,
    table: String,
}

impl CpuCellHandoffs {
    pub fn new(pool: PgPool, schema: &str) -> Result<Self> {
        ensure!(
            !schema.is_empty()
                && schema.len() <= 63
                && schema
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
            "invalid handoff schema identifier"
        );
        Ok(Self {
            pool,
            table: format!("\"{schema}\".cpu_cell_handoffs"),
        })
    }

    async fn transaction(&self) -> Result<Transaction<'_, Postgres>> {
        let mut tx = self.pool.begin().await?;
        sqlx::raw_sql("SET LOCAL lock_timeout = '2s'; SET LOCAL statement_timeout = '5s'")
            .execute(&mut *tx)
            .await?;
        Ok(tx)
    }

    pub async fn get(&self, id: Uuid) -> Result<Handoff> {
        tokio::time::timeout(
            Duration::from_secs(5),
            sqlx::query_as(&format!("SELECT * FROM {} WHERE handoff_id=$1", self.table))
                .bind(id)
                .fetch_one(&self.pool),
        )
        .await
        .wrap_err("reading handoff timed out")?
        .map_err(Into::into)
    }

    /// Called by deployment wiring. A completed deployment is never overwritten
    /// or reused as the destination of a later handoff.
    pub async fn request(&self, id: Uuid, source: Uuid, destination: Uuid) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(10), async {
            ensure!(
                !id.is_nil() && !source.is_nil() && !destination.is_nil() && source != destination,
                "handoff requires distinct non-nil deployment identities"
            );
            let mut tx = self.transaction().await?;
            sqlx::query(&format!(
                "LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE",
                self.table
            ))
            .execute(&mut *tx)
            .await?;
            let previous: Option<Handoff> =
                sqlx::query_as(&format!("SELECT * FROM {} WHERE handoff_id=$1", self.table))
                    .bind(id)
                    .fetch_optional(&mut *tx)
                    .await?;
            if let Some(previous) = previous {
                ensure!(
                    previous.source == source && previous.destination == destination,
                    "handoff ID already names another deployment"
                );
                tx.commit().await?;
                return Ok(());
            }
            let latest: Option<Handoff> = sqlx::query_as(&format!(
                "SELECT * FROM {} ORDER BY sequence DESC LIMIT 1",
                self.table
            ))
            .fetch_optional(&mut *tx)
            .await?;
            if let Some(latest) = latest {
                ensure!(
                    latest.phase == HandoffPhase::Activated && latest.destination == source,
                    "previous handoff must activate this source before another deployment"
                );
            }
            let reused: bool = sqlx::query_scalar(&format!(
                "SELECT EXISTS (SELECT 1 FROM {} WHERE source=$1 OR destination=$1)",
                self.table
            ))
            .bind(destination)
            .fetch_one(&mut *tx)
            .await?;
            ensure!(!reused, "destination deployment identity was already used");
            sqlx::query(&format!(
                "INSERT INTO {} (handoff_id,source,destination) VALUES($1,$2,$3)",
                self.table
            ))
            .bind(id)
            .bind(source)
            .bind(destination)
            .execute(&mut *tx)
            .await?;
            tx.commit().await?;
            Ok(())
        })
        .await
        .wrap_err("CPU handoff request timed out; inspect durable state before retry")?
    }

    pub async fn start_drain(
        &self,
        id: Uuid,
        source: Uuid,
        process: Uuid,
        candidates: &[Uuid; 3],
    ) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(10), async {
        ensure!(
            !process.is_nil()
                && candidates.iter().all(|id| !id.is_nil() && *id != process)
                && candidates[0] != candidates[1]
                && candidates[0] != candidates[2]
                && candidates[1] != candidates[2],
            "drain requires distinct current process identities"
        );
        let mut tx = self.transaction().await?;
        let result = sqlx::query(&format!("UPDATE {}
                 SET phase='draining', source_process=$3, candidate_processes=$4
                 WHERE handoff_id=$1 AND source=$2
                   AND (phase='requested' OR (phase='draining' AND source_process=$3 AND candidate_processes=$4))", self.table))
            .bind(id).bind(source).bind(process).bind(candidates.as_slice()).execute(&mut *tx).await?;
        ensure!(
            result.rows_affected() == 1,
            "handoff is not awaiting this source's drain"
        );
        tx.commit().await?;
        Ok(())
        }).await.wrap_err("CPU handoff start_drain timed out; inspect durable state before retry")?
    }

    /// The caller supplies the existing agreed drain report only after all
    /// admitted effects have settled. Exact retries preserve the original proof.
    pub async fn complete_drain(
        &self,
        id: Uuid,
        source: Uuid,
        process: Uuid,
        report: &serde_json::Value,
    ) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(10), async {
        ensure!(report.is_object(), "drain report must be an object");
        let mut tx = self.transaction().await?;
        let result = sqlx::query(&format!("UPDATE {}
                 SET phase=CASE WHEN phase='activated' THEN phase ELSE 'drained' END, drain_report=$3
                 WHERE handoff_id=$1 AND source=$2 AND source_process=$4
                   AND (phase='draining' OR (phase IN ('drained','activated') AND drain_report=$3))", self.table))
            .bind(id).bind(source).bind(report).bind(process).execute(&mut *tx).await?;
        ensure!(
            result.rows_affected() == 1,
            "handoff is not draining or completed with another report"
        );
        tx.commit().await?;
        Ok(())
        }).await.wrap_err("CPU handoff complete_drain timed out; inspect durable state before retry")?
    }

    /// Local activation bookkeeping. The caller must first validate all peers'
    /// reports and must pass the existing cohort barrier before serving.
    pub async fn activate(
        &self,
        id: Uuid,
        destination: Uuid,
        candidates: &[Uuid; 3],
        report: &serde_json::Value,
    ) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(10), async {
        let mut tx = self.transaction().await?;
        sqlx::query(&format!(
            "LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE",
            self.table
        ))
        .execute(&mut *tx)
        .await?;
        let latest: Option<Uuid> = sqlx::query_scalar(&format!(
            "SELECT handoff_id FROM {} ORDER BY sequence DESC LIMIT 1",
            self.table
        ))
        .fetch_optional(&mut *tx)
        .await?;
        ensure!(latest == Some(id), "cannot activate a superseded handoff");
        let result = sqlx::query(&format!("UPDATE {}
                 SET phase='activated'
                 WHERE handoff_id=$1 AND destination=$2
                   AND phase IN ('drained','activated') AND drain_report=$3 AND candidate_processes=$4", self.table))
            .bind(id).bind(destination).bind(report).bind(candidates.as_slice()).execute(&mut *tx).await?;
        ensure!(
            result.rows_affected() == 1,
            "handoff lacks the matching completed drain"
        );
        tx.commit().await?;
        Ok(())
        }).await.wrap_err("CPU handoff activate timed out; inspect durable state before retry")?
    }
}

#[cfg(all(test, feature = "db_dependent"))]
#[path = "cpu_cell_tests.rs"]
mod tests;
