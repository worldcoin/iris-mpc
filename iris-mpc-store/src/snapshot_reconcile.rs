use crate::{DbStoredIris, Store};
use eyre::{ensure, Result, WrapErr};
use sqlx::{pool::PoolConnection, Executor, FromRow, Postgres};

const SEMANTIC_ID_SIZE: usize = 16;

#[derive(Debug)]
pub(crate) struct IrisMetadata {
    pub id: i64,
    pub version_id: i16,
    pub rerand_epoch: i32,
    pub semantic_id: Option<[u8; SEMANTIC_ID_SIZE]>,
}

#[derive(FromRow)]
struct RawIrisMetadata {
    id: i64,
    version_id: i16,
    rerand_epoch: i32,
    semantic_id: Option<Vec<u8>>,
}

#[derive(FromRow)]
struct RawInventory {
    row_count: i64,
    min_id: i64,
    max_id: i64,
}

fn validate_inventory(inventory: RawInventory) -> Result<usize> {
    ensure!(inventory.row_count >= 0, "negative Aurora iris count");
    if inventory.row_count == 0 {
        ensure!(
            inventory.min_id == 0 && inventory.max_id == 0,
            "empty Aurora inventory has a nonempty id range"
        );
        return Ok(0);
    }
    ensure!(
        inventory.min_id == 1 && inventory.max_id == inventory.row_count,
        "Aurora inventory is not exactly contiguous: count {}, min id {}, max id {}",
        inventory.row_count,
        inventory.min_id,
        inventory.max_id
    );
    usize::try_from(inventory.row_count).wrap_err("Aurora iris count does not fit usize")
}

/// One authoritative read view for both metadata reconciliation and cache
/// misses. This prevents a version check from racing the corresponding blob
/// fetch without holding back ordinary writers.
pub(crate) struct AuroraSnapshot {
    connection: PoolConnection<Postgres>,
}

impl Store {
    pub(crate) async fn begin_aurora_snapshot(&self) -> Result<AuroraSnapshot> {
        let mut connection = self.pool.acquire().await?;
        // A failed load may drop this object while the raw transaction is
        // active. Never return such a backend to the pool.
        connection.close_on_drop();
        connection
            .execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            .await?;
        Ok(AuroraSnapshot { connection })
    }
}

impl AuroraSnapshot {
    /// Read and validate the complete authoritative inventory in this same
    /// repeatable-read view. A rerandomized server must never interpret a
    /// caller's earlier count as permission to serve only a prefix.
    pub(crate) async fn authoritative_row_count(&mut self) -> Result<usize> {
        let inventory = sqlx::query_as::<_, RawInventory>(
            "SELECT COUNT(*)::bigint AS row_count, \
                    COALESCE(MIN(id), 0)::bigint AS min_id, \
                    COALESCE(MAX(id), 0)::bigint AS max_id \
               FROM irises",
        )
        .fetch_one(&mut *self.connection)
        .await?;
        validate_inventory(inventory)
    }

    pub(crate) async fn metadata_page(
        &mut self,
        after_id: i64,
        max_id: i64,
        limit: i64,
    ) -> Result<Vec<IrisMetadata>> {
        ensure!(limit > 0, "metadata page size must be positive");
        let rows = sqlx::query_as::<_, RawIrisMetadata>(
            "SELECT id, version_id, rerand_epoch, \
                    uuid_send(semantic_id) AS semantic_id FROM irises \
             WHERE id > $1 AND id <= $2 ORDER BY id LIMIT $3",
        )
        .bind(after_id)
        .bind(max_id)
        .bind(limit)
        .fetch_all(&mut *self.connection)
        .await?;
        rows.into_iter()
            .map(|row| {
                let semantic_id = row
                    .semantic_id
                    .map(|value| {
                        value.try_into().map_err(|value: Vec<u8>| {
                            eyre::eyre!(
                                "Aurora iris {} has semantic id length {}, expected {SEMANTIC_ID_SIZE}",
                                row.id,
                                value.len()
                            )
                        })
                    })
                    .transpose()?;
                Ok(IrisMetadata {
                    id: row.id,
                    version_id: row.version_id,
                    rerand_epoch: row.rerand_epoch,
                    semantic_id,
                })
            })
            .collect()
    }

    pub(crate) async fn irises(&mut self, ids: &[i64]) -> Result<Vec<DbStoredIris>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        Ok(sqlx::query_as::<_, DbStoredIris>(
            "SELECT id, version_id, left_code, left_mask, right_code, right_mask, rerand_epoch \
             FROM irises WHERE id = ANY($1) ORDER BY id",
        )
        .bind(ids)
        .fetch_all(&mut *self.connection)
        .await?)
    }

    pub(crate) async fn finish(mut self) -> Result<()> {
        self.connection.execute("COMMIT").await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authoritative_inventory_rejects_prefixes_and_gaps() {
        assert_eq!(
            validate_inventory(RawInventory {
                row_count: 0,
                min_id: 0,
                max_id: 0,
            })
            .unwrap(),
            0
        );
        assert_eq!(
            validate_inventory(RawInventory {
                row_count: 3,
                min_id: 1,
                max_id: 3,
            })
            .unwrap(),
            3
        );
        assert!(validate_inventory(RawInventory {
            row_count: 2,
            min_id: 1,
            max_id: 3,
        })
        .is_err());
        assert!(validate_inventory(RawInventory {
            row_count: 2,
            min_id: 2,
            max_id: 3,
        })
        .is_err());
    }
}
