use super::utils::{errors::IndexationError, fetcher, logger};
use aws_sdk_s3::Client as S3Client;
use eyre::Result;
use iris_mpc_common::{config::Config, IrisSerialId};
use iris_mpc_store::{DbStoredIris, Store as IrisStore};
use std::future::Future;
use std::{iter::Peekable, ops::Range};

// Generates batches of Iris identifiers for processing.
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // Size of generated batches.
    batch_size: usize,

    // Set of Iris serial identifiers to exclude from indexing.
    exclusions: Vec<IrisSerialId>,

    // Range of Iris serial identifiers to be indexed.
    range: Range<IrisSerialId>,

    // Iterator over range of Iris serial identifiers to be indexed.
    range_iter: Peekable<Range<IrisSerialId>>,
}

// Constructor.
impl BatchGenerator {
    pub fn new(
        batch_size: usize,
        range: Range<IrisSerialId>,
        exclusions: Vec<IrisSerialId>,
    ) -> Self {
        Self {
            batch_size,
            exclusions,
            batch_count: 0,
            range: range.clone(),
            range_iter: range.peekable(),
        }
    }
}

// Constructor.
impl BatchGenerator {
    pub async fn new_from_services(
        config: &Config,
        last_indexation_height: IrisSerialId,
        max_indexation_height: IrisSerialId,
        s3_client: &S3Client,
    ) -> Result<Self, IndexationError> {
        // Set exclusions, i.e. identifiers marked as deleted.
        let exclusions = fetcher::fetch_iris_deletions(config, s3_client)
            .await
            .unwrap();

        let range_end = max_indexation_height + 1;
        let range_start = last_indexation_height;
        let range = range_start..range_end + 1;

        tracing::info!(
            "HNSW GENESIS :: Batch Generator :: Deletions for exclusion count = {}",
            exclusions.len(),
        );
        tracing::info!(
            "HNSW GENESIS :: Batch Generator :: Range of serial-id's to index = {}..{}",
            range.start,
            range.end
        );

        Ok(Self::new(config.max_batch_size, range, exclusions))
    }
}

// Methods.
impl BatchGenerator {
    // Returns next batch of Iris serial identifiers to be indexed.
    fn get_identifiers(&mut self) -> Option<Vec<IrisSerialId>> {
        // Escape if exhausted.
        self.range_iter.peek()?;

        // Construct next batch.
        let mut batch = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && batch.len() < self.batch_size {
            let next_id = self.range_iter.by_ref().next().unwrap();
            if next_id > self.range.end {
                break;
            }
            if !self.exclusions.contains(&next_id) {
                batch.push(next_id);
            } else {
                Self::log_info(format!("Excluding deletion :: serial-id={}", next_id));
            }
        }

        // Escape if empty otherwise increment count.
        if batch.is_empty() {
            return None;
        } else {
            self.batch_count += 1;
        }

        Self::log_info(format!(
            "Constructed new batch for indexation: idx={} :: irises={}",
            self.batch_count,
            batch.len()
        ));

        Some(batch)
    }

    // Helper: component logging.
    fn log_info(msg: String) {
        logger::log_info("Batch Generator", msg);
    }
}

// Batch iterator interface.
pub trait BatchIterator {
    // Count of generated batches.
    fn batch_count(&self) -> usize;

    // Iterator over batches of Iris data to be indexed.
    fn next_batch(
        &mut self,
        iris_store: &IrisStore,
    ) -> impl Future<Output = Result<Option<Vec<DbStoredIris>>, IndexationError>> + Send;
}

// Batch iterator implementation.
impl BatchIterator for BatchGenerator {
    // Count of generated batches.
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    // Returns next batch of Iris data to be indexed or None if exhausted.
    async fn next_batch(
        &mut self,
        iris_store: &IrisStore,
    ) -> Result<Option<Vec<DbStoredIris>>, IndexationError> {
        if let Some(identifiers) = self.get_identifiers() {
            let batch = fetcher::fetch_iris_batch(iris_store, identifiers).await?;
            Self::log_info(format!("Iris batch fetched: idx={}", self.batch_count,));

            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
}

// ------------------------------------------------------------------------
// Tests.
// ------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;
    use iris_mpc_common::postgres::{AccessMode, PostgresClient};
    use iris_mpc_store::test_utils::{cleanup, temporary_name, test_db_url};

    // Defaults.
    const DEFAULT_RNG_SEED: u64 = 0;
    const DEFAULT_PARTY_ID: usize = 0;
    const DEFAULT_SIZE_OF_IRIS_DB: usize = 100;

    // Returns a set of test resources.
    async fn get_resources() -> Result<(IrisStore, PostgresClient, String)> {
        // Set PostgreSQL client + store.
        let pg_schema = temporary_name();
        let pg_client =
            PostgresClient::new(&test_db_url()?, &pg_schema, AccessMode::ReadWrite).await?;

        // Set store.
        let iris_store = IrisStore::new(&pg_client).await?;

        // Set dB with 100 Iris's.
        iris_store
            .init_db_with_random_shares(
                DEFAULT_RNG_SEED,
                DEFAULT_PARTY_ID,
                DEFAULT_SIZE_OF_IRIS_DB,
                true,
            )
            .await?;

        Ok((iris_store, pg_client, pg_schema))
    }

    // Test new from range pulled from dB.
    #[tokio::test]
    async fn test_new() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        let instance = BatchGenerator::new(
            10,
            1..(iris_store.count_irises().await.unwrap() as u32),
            Vec::new(),
        );
        assert_eq!(instance.range.end, 100);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }

    // Test iteration.
    #[tokio::test]
    async fn test_iterator() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        let mut instance = BatchGenerator::new(
            10,
            1..(iris_store.count_irises().await.unwrap() as u32 + 1),
            Vec::new(),
        );

        // Expecting 10 batches of 10 Iris's per batch.
        while let Some(batch) = instance.next_batch(&iris_store).await? {
            assert_eq!(batch.len(), 10);
        }
        assert_eq!(instance.batch_count, 10);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }
}
