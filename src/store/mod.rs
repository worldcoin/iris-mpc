use crate::config::Config;
use bytemuck::cast_slice;
use eyre::{eyre, Result};
use futures::Stream;
use sqlx::{migrate::Migrator, postgres::PgPoolOptions, Executor, PgPool};

const APP_NAME: &str = "SMPC";
const POOL_SIZE: u32 = 5;

static MIGRATOR: Migrator = sqlx::migrate!("./migrations");

fn sql_switch_schema(schema_name: &str) -> Result<String> {
    sanitize_identifier(schema_name)?;
    Ok(format!(
        "
        CREATE SCHEMA IF NOT EXISTS \"{}\";
        SET search_path TO \"{}\";
        ",
        schema_name, schema_name
    ))
}

#[derive(sqlx::FromRow, Debug, Default)]
pub struct StoredIris {
    id:   i64,     // BIGSERIAL, non-negative.
    code: Vec<u8>, // BYTEA
    mask: Vec<u8>, // BYTEA
}

impl StoredIris {
    pub fn id(&self) -> u64 {
        self.id as u64
    }

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
    /// Connect to a database based on Config URL, environment, and party_id.
    pub async fn new_from_config(config: &Config) -> Result<Self> {
        let db_config = config
            .database
            .as_ref()
            .ok_or(eyre!("Missing database config"))?;
        let schema_name = format!("{}_{}_{}", APP_NAME, config.environment, config.party_id);
        Self::new(&db_config.url, &schema_name).await
    }

    pub async fn new(url: &str, schema_name: &str) -> Result<Self> {
        let connect_sql = sql_switch_schema(schema_name)?;

        let pool = PgPoolOptions::new()
            .max_connections(POOL_SIZE)
            .after_connect(move |conn, _meta| {
                // Switch to the given schema in every connection.
                let connect_sql = connect_sql.clone();
                Box::pin(async move {
                    conn.execute(connect_sql.as_ref()).await.inspect_err(|e| {
                        eprintln!("error in after_connect: {:?}", e);
                    })?;
                    Ok(())
                })
            })
            .connect(url)
            .await?;

        // Create the schema on the first startup.
        MIGRATOR.run(&pool).await?;

        Ok(Store { pool })
    }

    pub async fn stream_irises(&self) -> impl Stream<Item = Result<StoredIris, sqlx::Error>> + '_ {
        sqlx::query_as::<_, StoredIris>("SELECT * FROM irises").fetch(&self.pool)
    }

    /// Insert irises into the database.
    /// `next_id` is the next available ID greater than any existing one.
    pub async fn insert_irises(
        &self,
        next_id: u64,
        codes_and_masks: &[(&[u16], &[u16])],
    ) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let mut query = sqlx::QueryBuilder::new("INSERT INTO irises (id, code, mask) ");
        query.push_values(
            codes_and_masks.iter().enumerate(),
            |mut query, (i, (code, mask))| {
                query.push_bind(next_id as i64 + i as i64);
                query.push_bind(cast_slice::<u16, u8>(code));
                query.push_bind(cast_slice::<u16, u8>(mask));
            },
        );

        query.build().execute(&mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }
}

fn sanitize_identifier(input: &str) -> Result<()> {
    if input.chars().all(|c| c.is_alphanumeric() || c == '_') {
        Ok(())
    } else {
        Err(eyre!("Invalid SQL identifier"))
    }
}

#[cfg(test)]
mod tests {
    const DOTENV_TEST: &str = ".env.test";

    use super::*;
    use futures::TryStreamExt;
    use tokio;

    #[tokio::test]
    async fn test_store() -> Result<()> {
        // Create a unique schema for this test.
        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 0);

        let next_id = 123;
        let codes_and_masks: &[(&[u16], &[u16]); 2] = &[
            (&[1, 2, 3, 4], &[5, 6, 7, 8]),
            (&[9, 10, 11, 12], &[13, 14, 15, 16]),
        ];
        store.insert_irises(next_id, codes_and_masks).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;

        assert_eq!(got.len(), 2);
        for i in 0..2 {
            assert_eq!(got[i].id(), next_id + i as u64);
            assert_eq!(got[i].code(), codes_and_masks[i].0);
            assert_eq!(got[i].mask(), codes_and_masks[i].1);
        }

        // Clean up on success.
        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_many() -> Result<()> {
        let count = 1 << 14;

        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let codes_and_masks = vec![(&[123_u16; 12800][..], &[456_u16; 12800][..]); count];
        store.insert_irises(0, &codes_and_masks).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), count);

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    fn test_db_url() -> Result<String> {
        dotenvy::from_filename(DOTENV_TEST)?;
        Ok(Config::load_config(APP_NAME)?
            .database
            .ok_or(eyre!("Missing database config"))?
            .url)
    }

    fn temporary_name() -> String {
        format!("smpc_test{}_0", rand::random::<u32>())
    }

    async fn cleanup(store: &Store, schema_name: &str) -> Result<()> {
        assert!(schema_name.starts_with("smpc_test"));
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", schema_name))
            .execute(&store.pool)
            .await?;
        Ok(())
    }
}
