use futures::Stream;
use mpc_uniqueness_check::{bits::Bits, encoded_bits::EncodedBits};
use sqlx::Postgres;

pub struct Db {
    pool: sqlx::Pool<Postgres>,
}

impl Db {
    pub async fn new(url: &str) -> eyre::Result<Self> {
        tracing::info!("Connecting to database");

        let pool = sqlx::Pool::connect(url).await?;

        tracing::info!("Connected to database");

        Ok(Self { pool })
    }

    #[tracing::instrument(skip(self))]
    pub async fn fetch_share(&self, id: u64) -> eyre::Result<(i64, EncodedBits)> {
        Ok(sqlx::query_as(
            r#"
            SELECT id, share
            FROM shares
            WHERE id = $1
            "#,
        )
        .bind(i64::try_from(id).expect("id fits into i64"))
        .fetch_one(&self.pool)
        .await?)
    }

    #[tracing::instrument(skip(self))]
    pub async fn fetch_mask(&self, id: u64) -> eyre::Result<(i64, Bits)> {
        Ok(sqlx::query_as(
            r#"
            SELECT id, masks
            FROM masks
            WHERE id = $1
            "#,
        )
        .bind(i64::try_from(id).expect("id fits into i64"))
        .fetch_one(&self.pool)
        .await?)
    }

    #[tracing::instrument(skip(self))]
    pub fn stream_shares(
        &self,
        id_range: std::ops::Range<u64>,
    ) -> impl Stream<Item = sqlx::Result<(i64, EncodedBits)>> + '_ {
        sqlx::query_as(
            r#"
            SELECT id, share
            FROM shares
            WHERE id >= $1 AND id < $2
            ORDER BY id ASC
            "#,
        )
        .bind(i64::try_from(id_range.start).expect("id fits into i64"))
        .bind(i64::try_from(id_range.end).expect("id fits into i64"))
        .fetch(&self.pool)
    }
    #[tracing::instrument(skip(self))]
    pub fn stream_masks(
        &self,
        id_range: std::ops::Range<u64>,
    ) -> impl Stream<Item = sqlx::Result<(i64, Bits)>> + '_ {
        sqlx::query_as(
            r#"
            SELECT id, mask
            FROM masks
            WHERE id >= $1 AND id < $2
            ORDER BY id ASC
            "#,
        )
        .bind(i64::try_from(id_range.start).expect("id fits into i64"))
        .bind(i64::try_from(id_range.end).expect("id fits into i64"))
        .fetch(&self.pool)
    }
}
