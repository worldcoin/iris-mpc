use super::utils::{self, errors::IndexationError, fetcher, types::IrisSerialId};
use iris_mpc_common::config::Config;
use std::{iter::Peekable, ops::Range};

// Generates batches of Iris identifiers for processing.
#[allow(dead_code)]
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // System configuration information.
    config: Config,

    // Iterator over range of Iris serial identifiers to be indexed.
    indexation_range_iter: Peekable<Range<IrisSerialId>>,

    // Set of Iris serial identifiers to exclude from indexing.
    indexation_exclusions: Vec<IrisSerialId>,
}

// Constructor.
#[allow(dead_code)]
impl BatchGenerator {
    pub fn new(config: Config) -> Self {
        Self {
            batch_count: 0,
            config,
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
        let store = utils::pgres::get_store_instance(&self.config).await;
        let height_of_protocol = fetcher::fetch_height_of_protocol(&store).await?;
        let height_of_indexed = fetcher::fetch_height_of_indexed(&store).await?;
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
