use bytemuck::cast_slice;
use eyre::Result;
use futures::StreamExt;
use sqlx::{postgres::PgPoolOptions, PgPool};
use std::env;

// TODO: pending config mechanism.
const DB_URL: &str = "postgres://postgres:postgres@localhost/postgres";
const POOL_SIZE: u32 = 5;

const CREATE_TABLE_IRISES: &str = "
    CREATE TABLE IF NOT EXISTS irises (
        id SERIAL PRIMARY KEY,
        code BYTEA,
        mask BYTEA
    );
";

#[derive(sqlx::FromRow, Debug)]
pub struct StoredIris {
    pub id: i32,
    pub code: Vec<u8>,
    pub mask: Vec<u8>,
}

impl StoredIris {
    pub fn code(&self) -> &[u16] {
        cast_slice(&self.code)
    }

    pub fn mask(&self) -> &[u16] {
        cast_slice(&self.mask)
    }
}

pub struct Store {
    pool: PgPool,
}

impl Store {
    pub async fn new_from_env() -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(POOL_SIZE)
            .connect(env::var("DB_URL").unwrap_or(DB_URL.to_string()).as_str())
            .await?;

        sqlx::query(CREATE_TABLE_IRISES).execute(&pool).await?;

        Ok(Store { pool })
    }

    pub async fn iter_irises(&self) -> Result<impl Iterator<Item = StoredIris>> {
        let mut rows = sqlx::query_as::<_, StoredIris>("SELECT * FROM irises").fetch(&self.pool);

        let mut items = Vec::new();

        while let Some(row) = rows.next().await {
            if let Ok(row) = row {
                items.push(row);
            }
        }

        Ok(items.into_iter())
    }

    pub async fn insert_irises(&self, codes_and_masks: &[(&[u16], &[u16])]) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for (code, mask) in codes_and_masks {
            sqlx::query("INSERT INTO irises (code, mask) VALUES ($1, $2)")
                .bind(cast_slice::<u16, u8>(code))
                .bind(cast_slice::<u16, u8>(mask))
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;

        Ok(())
    }

    #[cfg(test)]
    async fn clear(&self) -> Result<()> {
        sqlx::query("DELETE FROM irises")
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_store() -> Result<()> {
        let store = Store::new_from_env().await?;
        store.clear().await.unwrap();

        let codes_and_masks: &[(&[u16], &[u16]); 2] = &[
            (&[1, 2, 3, 4], &[5, 6, 7, 8]),
            (&[9, 10, 11, 12], &[13, 14, 15, 16]),
        ];

        store.insert_irises(codes_and_masks).await?;

        let got = store.iter_irises().await?.collect::<Vec<_>>();

        assert_eq!(got.len(), 2);

        for i in 0..2 {
            assert_eq!(got[i].code(), codes_and_masks[i].0);
            assert_eq!(got[i].mask(), codes_and_masks[i].1);
        }
        Ok(())
    }
}
