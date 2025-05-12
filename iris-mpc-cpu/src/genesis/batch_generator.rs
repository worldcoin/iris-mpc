use super::utils::{errors::IndexationError, fetcher, logger, types::IrisSerialId};
use crate::{hawkers::aby3::aby3_store::Aby3Store, hnsw::graph::graph_store::GraphPg};
use aws_sdk_s3::Client as S3Client;
use eyre::Result;
use iris_mpc_common::config::Config;
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

// Batch generation iterator interface.
pub trait BatchIterator {
    // Count of generated batches.
    fn batch_count(&self) -> usize;

    // Iterator over batches of Iris data to be indexed.
    fn next_batch(
        &mut self,
        iris_store: &IrisStore,
    ) -> impl Future<Output = Result<Option<Vec<DbStoredIris>>, IndexationError>> + Send;
}

// Constructors.
impl BatchGenerator {
    pub fn new_with_range(
        batch_size: usize,
        indexation_range: Range<IrisSerialId>,
        exclusions: Vec<IrisSerialId>,
    ) -> Self {
        Self {
            batch_size,
            exclusions,
            batch_count: 0,
            range: indexation_range.clone(),
            range_iter: indexation_range.peekable(),
        }
    }
}

// Defaults.
impl Default for BatchGenerator {
    fn default() -> Self {
        let indexation_range = 0..0;

        Self {
            batch_size: 64,
            exclusions: Vec::<IrisSerialId>::new(),
            batch_count: 0,
            range: indexation_range.clone(),
            range_iter: indexation_range.peekable(),
        }
    }
}

// Initializer.
impl BatchGenerator {
    pub async fn init(
        &mut self,
        config: &Config,
        iris_store: &IrisStore,
        _graph_store: &GraphPg<Aby3Store>,
        s3_client: &S3Client,
    ) -> Result<(), IndexationError> {
        // Set serial identifiers to be excluded form indexation.
        self.exclusions = fetcher::fetch_iris_deletions(config, s3_client)
            .await
            .unwrap();
        tracing::info!(
            "HNSW GENESIS :: Batch Generator :: Deletions for exclusion count = {}",
            self.exclusions.len(),
        );

        // Set heights.
        let height_of_protocol = fetcher::fetch_height_of_protocol(iris_store).await?;
        let height_of_indexed = fetcher::fetch_height_of_indexed(iris_store).await?;

        // Set range of indexation.
        self.range_iter = (height_of_indexed..height_of_protocol + 1).peekable();
        tracing::info!(
            "HNSW GENESIS :: Batch Generator :: Range of serial-id's to index = {}..{}",
            height_of_indexed,
            height_of_protocol
        );

        Ok(())
    }
}

// Methods.
impl BatchGenerator {
    // Returns next batch of Iris serial identifiers to be indexed.
    fn get_identifiers(&mut self) -> Option<Vec<IrisSerialId>> {
        self.range_iter.peek()?;
        // if self.range_iter.peek().is_none() {
        //     return None;
        // }

        let mut batch = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && batch.len() < self.batch_size {
            let next_id = self.range_iter.by_ref().next().unwrap();
            if next_id > self.range.end {
                break;
            }
            if !self.exclusions.contains(&next_id) {
                batch.push(next_id);
            } else {
                tracing::info!(
                    "HNSW GENESIS :: Batch Generator :: Excluding deletion :: serial-id={}",
                    next_id
                );
            }
        }

        if batch.is_empty() {
            return None;
        } else {
            self.batch_count += 1;
        }

        #[cfg(test)]
        println!(
            "HNSW GENESIS :: Batch Generator :: Constructed new batch for indexation: idx={} :: irises={}",
            self.batch_count,
            batch.len(),
        );
        tracing::info!(
            "HNSW GENESIS :: Batch Generator :: Constructed new batch for indexation: idx={} :: irises={}",
            self.batch_count,
            batch.len(),
        );

        Some(batch)
    }

    // Helper: component logging.
    pub fn log_info(&self, msg: String) {
        logger::log_info("Batch Generator", msg.as_str());
    }
}

// Implement our BatchIterator trait
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
            self.log_info(format!(
                "Fetching Iris batch for indexation: idx={} :: irises={}",
                self.batch_count,
                identifiers.len()
            ));

            let batch = fetcher::fetch_iris_batch(iris_store, identifiers).await?;
            logger::log_info(
                "Batch Generator",
                format!("Iris batch fetched: idx={}", self.batch_count,).as_str(),
            );

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
            .init_db_with_random_shares(0, 0, 100, true)
            .await?;

        Ok((iris_store, pg_client, pg_schema))
    }

    // Test new from default.
    #[tokio::test]
    async fn test_new_01() -> Result<()> {
        let instance = BatchGenerator::default();
        assert_eq!(instance.batch_count, 0);
        assert_eq!(instance.batch_size, 64);
        assert_eq!(instance.range.start, 0);
        assert_eq!(instance.range.end, 0);
        assert_eq!(instance.exclusions.len(), 0);

        Ok(())
    }

    // Test new from specific range.
    #[tokio::test]
    async fn test_new_02() -> Result<()> {
        let instance = BatchGenerator::new_with_range(10, 1..100, Vec::new());
        assert_eq!(instance.batch_count, 0);
        assert_eq!(instance.batch_size, 10);
        assert_eq!(instance.range.start, 1);
        assert_eq!(instance.range.end, 100);
        assert_eq!(instance.exclusions.len(), 0);

        Ok(())
    }

    // Test new from range pulled from dB.
    #[tokio::test]
    async fn test_new_03() -> Result<()> {
        // Set resources.
        let (iris_store, pg_client, pg_schema) = get_resources().await.unwrap();

        let instance = BatchGenerator::new_with_range(
            10,
            1..(iris_store.count_irises().await.unwrap() as u64),
            Vec::new(),
        );
        assert_eq!(instance.range.end, 100);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }
}
