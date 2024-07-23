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
    #[allow(dead_code)]
    id:         i64, // BIGSERIAL
    left_code:  Vec<u8>, // BYTEA
    left_mask:  Vec<u8>, // BYTEA
    right_code: Vec<u8>, // BYTEA
    right_mask: Vec<u8>, // BYTEA
}

impl StoredIris {
    pub fn left_code(&self) -> &[u16] {
        cast_u8_to_u16(&self.left_code)
    }

    pub fn left_mask(&self) -> &[u16] {
        cast_u8_to_u16(&self.left_mask)
    }

    pub fn right_code(&self) -> &[u16] {
        cast_u8_to_u16(&self.right_code)
    }

    pub fn right_mask(&self) -> &[u16] {
        cast_u8_to_u16(&self.right_mask)
    }
}

#[derive(Clone)]
pub struct StoredIrisRef<'a> {
    pub left_code:  &'a [u16],
    pub left_mask:  &'a [u16],
    pub right_code: &'a [u16],
    pub right_mask: &'a [u16],
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
        sqlx::query_as::<_, StoredIris>("SELECT * FROM irises ORDER BY id").fetch(&self.pool)
    }

    pub async fn insert_irises(&self, codes_and_masks: &[StoredIrisRef<'_>]) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        let mut query = sqlx::QueryBuilder::new(
            "INSERT INTO irises (left_code, left_mask, right_code, right_mask)",
        );
        query.push_values(codes_and_masks, |mut query, iris| {
            query.push_bind(cast_slice::<u16, u8>(iris.left_code));
            query.push_bind(cast_slice::<u16, u8>(iris.left_mask));
            query.push_bind(cast_slice::<u16, u8>(iris.right_code));
            query.push_bind(cast_slice::<u16, u8>(iris.right_mask));
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

fn cast_u8_to_u16(s: &[u8]) -> &[u16] {
    if s.is_empty() {
        &[] // A literal empty &[u8] may be unaligned.
    } else {
        cast_slice(s)
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

        let codes_and_masks = &[
            StoredIrisRef {
                left_code:  &[1, 2, 3, 4],
                left_mask:  &[5, 6, 7, 8],
                right_code: &[9, 10, 11, 12],
                right_mask: &[13, 14, 15, 16],
            },
            StoredIrisRef {
                left_code:  &[1117, 18, 19, 20],
                left_mask:  &[21, 1122, 23, 24],
                right_code: &[25, 26, 1127, 28],
                right_mask: &[29, 30, 31, 1132],
            },
            StoredIrisRef {
                left_code:  &[17, 18, 19, 20],
                left_mask:  &[21, 22, 23, 24],
                // Empty is allowed until stereo is implemented.
                right_code: &[],
                right_mask: &[],
            },
        ];
        store.insert_irises(&codes_and_masks[0..2]).await?;
        store.insert_irises(&codes_and_masks[2..3]).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;

        assert_eq!(got.len(), 3);
        for i in 0..3 {
            assert_eq!(got[i].left_code(), codes_and_masks[i].left_code);
            assert_eq!(got[i].left_mask(), codes_and_masks[i].left_mask);
            assert_eq!(got[i].right_code(), codes_and_masks[i].right_code);
            assert_eq!(got[i].right_mask(), codes_and_masks[i].right_mask);
        }

        // Clean up on success.
        cleanup(&store, &schema_name).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_insert_many() -> Result<()> {
        let count = 1 << 13;

        let schema_name = temporary_name();
        let store = Store::new(&test_db_url()?, &schema_name).await?;

        let iris = StoredIrisRef {
            left_code:  &[123_u16; 12800],
            left_mask:  &[456_u16; 12800],
            right_code: &[789_u16; 12800],
            right_mask: &[101_u16; 12800],
        };
        let codes_and_masks = vec![iris; count];
        store.insert_irises(&codes_and_masks).await?;

        let got: Vec<StoredIris> = store.stream_irises().await.try_collect().await?;
        assert_eq!(got.len(), count);
        assert_sorted_by_id(&got);

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

    fn assert_sorted_by_id(vec: &[StoredIris]) {
        assert!(
            vec.windows(2).all(|w| w[0].id <= w[1].id),
            "Vector is not sorted by id"
        );
    }

    async fn cleanup(store: &Store, schema_name: &str) -> Result<()> {
        assert!(schema_name.starts_with("smpc_test"));
        sqlx::query(&format!("DROP SCHEMA \"{}\" CASCADE", schema_name))
            .execute(&store.pool)
            .await?;
        Ok(())
    }
}
