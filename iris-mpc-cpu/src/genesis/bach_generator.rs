use super::utils::{errors::IndexationError, fetcher, types::IrisSerialId};
use iris_mpc_common::config::Config;
use iris_mpc_store::Store as IrisStore;
use std::{iter::Peekable, ops::Range};

// Generates batches of Iris identifiers for processing.
#[allow(dead_code)]
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // System configuration information.
    config: Config,

    // System configuration information.
    store: IrisStore,

    // Iterator over range of Iris serial identifiers to be indexed.
    indexation_range_iter: Peekable<Range<IrisSerialId>>,

    // Set of Iris serial identifiers to exclude from indexing.
    indexation_exclusions: Vec<IrisSerialId>,
}

// Constructor.
#[allow(dead_code)]
impl BatchGenerator {
    pub fn new(config: Config, store: IrisStore) -> Self {
        Self {
            config,
            store,
            batch_count: 0,
            indexation_exclusions: vec![],
            indexation_range_iter: (0..0).peekable(),
        }
    }
}

// Initializer.
#[allow(dead_code)]
impl BatchGenerator {
    async fn init(&mut self) -> Result<(), IndexationError> {
        // Set indexation exclusions.
        self.indexation_exclusions = fetcher::fetch_iris_deletions(&self.config).await.unwrap();

        // Set indexation range.
        let height_of_protocol = fetcher::fetch_height_of_protocol(&self.store).await?;
        let height_of_indexed = fetcher::fetch_height_of_indexed(&self.store).await?;
        self.indexation_range_iter = (height_of_indexed..height_of_protocol + 1).peekable();

        // Emit log entries.
        tracing::info!(
            "HNSW GENESIS: Range of serial-id's to index = {}..{}",
            height_of_indexed,
            height_of_protocol
        );
        tracing::info!(
            "HNSW GENESIS: Deletions for exclusion = {}",
            self.indexation_exclusions.len(),
        );

        Ok(())
    }
}
