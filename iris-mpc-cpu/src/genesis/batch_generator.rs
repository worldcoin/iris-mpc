use super::{
    state_accessor as fetcher,
    utils::{errors::IndexationError, logger},
};
use eyre::Result;
use iris_mpc_common::IrisSerialId;
use iris_mpc_store::{DbStoredIris, Store as IrisStore};
use std::{future::Future, iter::Peekable, ops::RangeInclusive};

// Component name for logging purposes.
const COMPONENT: &str = "Batch-Generator";

// A batch for upstream indexation.
pub struct Batch {
    // Array of stored Iris's to be indexed.
    pub data: Vec<DbStoredIris>,

    // Ordinal batch identifier scoped by processing context.
    pub id: usize,
}

// Constructor.
impl Batch {
    pub fn new(batch_id: usize, data: Vec<DbStoredIris>) -> Self {
        Self { data, id: batch_id }
    }
}

// Methods.
impl Batch {
    /// Return the Iris serial id of batch's first element.
    pub fn id_start(&self) -> IrisSerialId {
        self.data
            .first()
            .map(|value| value.id() as IrisSerialId)
            .unwrap()
    }

    /// Return the Iris serial id of batch's last element.
    pub fn id_end(&self) -> IrisSerialId {
        self.data
            .last()
            .map(|value| value.id() as IrisSerialId)
            .unwrap()
    }

    /// Return the size of the batch.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// Generates batches of Iris identifiers for processing.
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // Size of generated batches.
    batch_size: usize,

    // Set of Iris serial identifiers to exclude from indexing.
    exclusions: Vec<IrisSerialId>,

    // Range of Iris serial identifiers to be indexed.
    range: RangeInclusive<IrisSerialId>,

    // Iterator over range of Iris serial identifiers to be indexed.
    range_iter: Peekable<RangeInclusive<IrisSerialId>>,
}

// Constructor.
impl BatchGenerator {
    /// Create a new `BatchGenerator` with the following properties:
    ///
    /// - Range of iris serial ids to be produced is those between `start_id` and `end_id`,
    ///   inclusive.
    /// - Batches produced are of size `batch_size` until the last batch, which may be smaller.
    /// - Any serial ids contained in `exclusions` are skipped, and not included in batches.
    pub fn new(
        start_id: IrisSerialId,
        end_id: IrisSerialId,
        batch_size: usize,
        exclusions: Vec<IrisSerialId>,
    ) -> Self {
        let range = start_id..=end_id;

        Self::log_info(format!(
            "Deletions for exclusion count = {}",
            exclusions.len(),
        ));
        Self::log_info(format!(
            "Range of serial-id's to index = {}..{}",
            range.start(),
            range.end()
        ));

        Self {
            batch_size,
            exclusions,
            batch_count: 0,
            range: range.clone(),
            range_iter: range.peekable(),
        }
    }
}

// Accessors.
impl BatchGenerator {
    pub fn get_identifiers_range(&self) -> RangeInclusive<IrisSerialId> {
        self.range.clone()
    }

    pub fn get_exclusions(&self) -> Vec<IrisSerialId> {
        self.exclusions.clone()
    }
}

// Methods.
impl BatchGenerator {
    /// Returns next batch of Iris serial identifiers to be indexed.
    pub fn next_identifiers(&mut self) -> Option<Vec<IrisSerialId>> {
        // Escape if exhausted.
        self.range_iter.peek()?;

        // Construct next batch.
        let mut batch = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && batch.len() < self.batch_size {
            let next_id = self.range_iter.by_ref().next().unwrap();
            if !self.exclusions.contains(&next_id) {
                batch.push(next_id);
            } else {
                Self::log_info(format!("Excluding deletion :: iris-serial-id={}", next_id));
            }
        }

        // Escape if empty otherwise increment count.
        if batch.is_empty() {
            return None;
        } else {
            self.batch_count += 1;
        }

        Self::log_info(format!(
            "Constructed new batch for indexation: batch-id={} :: batch-size={}",
            self.batch_count,
            batch.len()
        ));

        Some(batch)
    }

    // Helper: component logging.
    fn log_info(msg: String) {
        logger::log_info(COMPONENT, msg);
    }
}

/// Batch iterator interface.
pub trait BatchIterator {
    // Count of generated batches.
    fn batch_count(&self) -> usize;

    // Iterator over batches of Iris data to be indexed.
    fn next_batch(
        &mut self,
        iris_store: &IrisStore,
    ) -> impl Future<Output = Result<Option<Batch>, IndexationError>> + Send;
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
    ) -> Result<Option<Batch>, IndexationError> {
        if let Some(identifiers) = self.next_identifiers() {
            let data = fetcher::fetch_iris_batch(iris_store, identifiers).await?;
            let batch = Batch::new(self.batch_count, data);
            Self::log_info(format!("Iris batch fetched: batch-id={}", batch.id,));
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
#[cfg(feature = "db_dependent")]
mod tests {
    use super::*;
    use eyre::Result;
    use iris_mpc_common::postgres::{AccessMode, PostgresClient};
    use iris_mpc_store::test_utils::{cleanup, temporary_name, test_db_url};

    // Defaults.
    const DEFAULT_RNG_SEED: u64 = 0;
    const DEFAULT_PARTY_ID: usize = 0;
    const DEFAULT_SIZE_OF_IRIS_DB: usize = 100;
    const DEFAULT_SIZE_OF_BATCH: usize = 10;
    const DEFAULT_COUNT_OF_BATCHES: usize = 10;

    // Returns a set of test resources.
    async fn get_resources() -> Result<(IrisStore, PostgresClient, String)> {
        // Set PostgreSQL client + store.
        let pg_schema = temporary_name();
        let pg_client =
            PostgresClient::new(&test_db_url()?, &pg_schema, AccessMode::ReadWrite).await?;

        // Set store.
        let iris_store = IrisStore::new(&pg_client).await?;

        // Set dB with 100 irises.
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
            1,
            iris_store.count_irises().await.unwrap() as IrisSerialId,
            DEFAULT_SIZE_OF_BATCH,
            Vec::new(),
        );
        assert_eq!(*instance.range.end() as usize, DEFAULT_SIZE_OF_IRIS_DB);

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
            1,
            iris_store.count_irises().await.unwrap() as IrisSerialId,
            DEFAULT_SIZE_OF_BATCH,
            Vec::new(),
        );

        // Expecting M batches of N Iris's per batch.
        while let Some(batch) = instance.next_batch(&iris_store).await? {
            assert_eq!(batch.size(), DEFAULT_SIZE_OF_BATCH);
        }
        assert_eq!(instance.batch_count, DEFAULT_COUNT_OF_BATCHES);

        // Unset resources.
        cleanup(&pg_client, &pg_schema).await?;

        Ok(())
    }

    #[test]
    fn test_exclusions() -> Result<()> {
        let mut instance = BatchGenerator::new(1, 30, 10, vec![3, 7, 12, 15, 22, 30, 70]);

        let mut batches = Vec::new();
        while let Some(batch) = instance.next_identifiers() {
            batches.push(batch);
        }
        assert_eq!(
            batches,
            vec![
                vec![1, 2, 4, 5, 6, 8, 9, 10, 11, 13],
                vec![14, 16, 17, 18, 19, 20, 21, 23, 24, 25],
                vec![26, 27, 28, 29],
            ]
        );

        Ok(())
    }
}
