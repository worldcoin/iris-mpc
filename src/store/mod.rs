use bytemuck::cast_slice;
use eyre::{eyre, Result};
use futures::Stream;
use sqlx::{migrate::Migrator, postgres::PgPoolOptions, Executor, PgPool};
use std::env;

// Environment variables.
const DATABASE_URL: &str = "DATABASE_URL";
const ENVIRONMENT: &str = "SMPC__ENVIRONMENT";
const PARTY_ID: &str = "SMPC__PARTY_ID";

const SCHEMA_PREFIX: &str = "smpc";
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
    #[allow(dead_code)]
    id: i64, // BIGSERIAL
    code: Vec<u8>, // BYTEA
    mask: Vec<u8>, // BYTEA
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
    /// Connect to a database based on env-vars DATABASE_URL, SMPC__ENVIRONMENT, and SMPC__PARTY_ID.
    pub async fn new_from_env() -> Result<Self> {
        Ok(Self::new(&Self::env_url()?, &Self::env_schema_name()?).await?)
    }

    fn env_url() -> Result<String> {
        env::var(DATABASE_URL).map_err(|_| eyre!("Missing env-var {}", DATABASE_URL))
    }

    fn env_schema_name() -> Result<String> {
        let environment =
            env::var(ENVIRONMENT).map_err(|_| eyre!("Missing env-var {}", ENVIRONMENT))?;
        let party_id = env::var(PARTY_ID).map_err(|_| eyre!("Missing env-var {}", PARTY_ID))?;
        Ok(format!("{}_{}_{}", SCHEMA_PREFIX, environment, party_id))
    }

    pub async fn new(url: &str, schema_name: &str) -> Result<Self> {
        let connect_sql = sql_switch_schema(&schema_name)?;

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

    pub async fn insert_irises(&self, codes_and_masks: &[(&[u16], &[u16])]) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let mut query = sqlx::QueryBuilder::new("INSERT INTO irises (code, mask) ");
        query.push_values(codes_and_masks, |mut query, (code, mask)| {
            query.push_bind(cast_slice::<u16, u8>(code));
            query.push_bind(cast_slice::<u16, u8>(mask));
        });

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
        dotenvy::from_filename(DOTENV_TEST)?;

        // Create a unique schema for this test.
        let schema_name = temporary_name();

        let store = Store::new(&Store::env_url()?, &schema_name).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), 0);

        let codes_and_masks: &[(&[u16], &[u16]); 2] = &[
            (&[1, 2, 3, 4], &[5, 6, 7, 8]),
            (&[9, 10, 11, 12], &[13, 14, 15, 16]),
        ];
        store.insert_irises(codes_and_masks).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;

        assert_eq!(got.len(), 2);
        for i in 0..2 {
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

        dotenvy::from_filename(DOTENV_TEST)?;
        let schema_name = temporary_name();
        let store = Store::new(&Store::env_url()?, &schema_name).await?;

        let codes_and_masks = vec![(&[123 as u16; 12800][..], &[456 as u16; 12800][..]); count];
        store.insert_irises(&codes_and_masks).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), count);

        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[test]
    fn test_env_vars() -> Result<()> {
        dotenvy::from_filename(DOTENV_TEST)?;
        assert!(Store::env_url()?.starts_with("postgres://"));
        assert!(Store::env_schema_name()?.starts_with(&format!("{}_test", SCHEMA_PREFIX)));
        Ok(())
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
