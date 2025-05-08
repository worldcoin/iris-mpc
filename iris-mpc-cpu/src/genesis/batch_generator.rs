use super::utils::{errors::IndexationError, fetcher, types::IrisSerialId};
use crate::{hawkers::aby3::aby3_store::Aby3Store, hnsw::graph::graph_store::GraphPg};
use aws_sdk_s3::Client as S3Client;
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

    // Maximum height to which to index.
    max_indexation_height: IrisSerialId,

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

// Constructor.
impl BatchGenerator {
    pub fn new(batch_size: usize, max_indexation_height: IrisSerialId) -> Self {
        Self {
            batch_size,
            max_indexation_height,
            batch_count: 0,
            exclusions: vec![],
            range_iter: (0..0).peekable(),
        }
    }
}

// Initializer.
impl BatchGenerator {
    pub async fn init(
        &mut self,
        iris_store: &IrisStore,
        _graph_store: &GraphPg<Aby3Store>,
        s3_client: &S3Client,
        env: String,
    ) -> Result<(), IndexationError> {
        // Set indexation exclusions.
        self.exclusions = fetcher::fetch_iris_deletions(s3_client, env).await.unwrap();
        tracing::info!(
            "HNSW GENESIS: Deletions for exclusion count = {}",
            self.exclusions.len(),
        );

        // Set indexation range.
        let height_of_protocol = fetcher::fetch_height_of_protocol(iris_store).await?;
        let height_of_indexed = fetcher::fetch_height_of_indexed(iris_store).await?;
        self.range_iter = (height_of_indexed..height_of_protocol + 1).peekable();

        tracing::info!(
            "HNSW GENESIS: Range of serial-id's to index = {}..{}",
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
            if next_id > self.max_indexation_height {
                break;
            }
            if !self.exclusions.contains(&next_id) {
                batch.push(next_id);
            } else {
                tracing::info!("HNSW GENESIS: Excluding deletion :: serial-id={}", next_id);
            }
        }

        if batch.is_empty() {
            return None;
        }

        self.batch_count += 1;
        tracing::info!(
            "HNSW GENESIS: Constructed new batch for indexation: idx={} :: irises={}",
            self.batch_count,
            batch.len(),
        );

        Some(batch)
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
            let batch = fetcher::fetch_iris_batch(iris_store, identifiers).await?;

            tracing::info!(
                "HNSW GENESIS: Fetched new batch for indexation: idx={} :: irises={}",
                self.batch_count,
                batch.len(),
            );

            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
}
