use super::utils::{errors::IndexationError, fetcher, types::IrisSerialId};
use crate::{hawkers::aby3::aby3_store::Aby3Store, hnsw::graph::graph_store::GraphPg};
use aws_sdk_s3::Client as S3Client;
use iris_mpc_store::{DbStoredIris as IrisData, Store as IrisStore};
use std::{iter::Peekable, ops::Range};

// Generates batches of Iris identifiers for processing.
#[allow(dead_code)]
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // Size of generated batches.
    batch_size: usize,

    // Set of Iris serial identifiers to exclude from indexing.
    exclusions: Vec<IrisSerialId>,

    // Iterator over range of Iris serial identifiers to be indexed.
    range_iter: Peekable<Range<IrisSerialId>>,
}

// Constructor.
#[allow(dead_code)]
impl BatchGenerator {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
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
    ) -> Result<(), IndexationError> {
        // Set indexation exclusions.
        self.exclusions = fetcher::fetch_iris_deletions(s3_client).await.unwrap();
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

#[allow(dead_code)]
impl BatchGenerator {
    // Returns next batch of Iris data to be indexed.
    pub async fn next_batch(
        &mut self,
        iris_store: &IrisStore,
    ) -> Result<Vec<IrisData>, IndexationError> {
        let identifiers = self.get_identifiers();
        let batch = fetcher::fetch_iris_batch(iris_store, identifiers).await?;
        tracing::info!(
            "HNSW GENESIS: Fetched new batch for indexation: idx={} :: irises={}",
            self.batch_count,
            batch.len(),
        );

        Ok(batch)
    }

    // Returns next batch of Iris serial identifiers to be indexed.
    fn get_identifiers(&mut self) -> Vec<IrisSerialId> {
        let mut batch = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && batch.len() < self.batch_size {
            let next_id = self.range_iter.by_ref().next().unwrap();
            if !self.exclusions.contains(&next_id) {
                batch.push(next_id);
            } else {
                tracing::info!("HNSW GENESIS: Excluding deletion :: serial-id={}", next_id);
            }
        }
        self.batch_count += 1;
        tracing::info!(
            "HNSW GENESIS: Constructed new batch for indexation: idx={} :: irises={}",
            self.batch_count,
            batch.len(),
        );

        batch
    }
}
