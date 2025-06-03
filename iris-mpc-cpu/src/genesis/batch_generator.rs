use crate::{
    execution::hawk_main::BothEyes,
    hawkers::aby3::aby3_store::{QueryRef, SharedIrisesRef},
};

use super::utils::{errors::IndexationError, logger};
use eyre::Result;
use iris_mpc_common::{vector_id::VectorId, IrisSerialId};
use std::{fmt, future::Future, iter::Peekable, ops::RangeInclusive};

/// Component name for logging purposes.
const COMPONENT: &str = "Batch-Generator";

/// A batch for upstream indexation.
#[derive(Debug)]
pub struct Batch {
    /// Ordinal batch identifier scoped by processing context.
    pub batch_id: usize,

    /// Array of vector ids of iris enrollments
    pub vector_ids: Vec<VectorId>,

    /// Array of left iris codes, in query format
    pub left_queries: Vec<QueryRef>,

    /// Array of right iris codes, in query format
    pub right_queries: Vec<QueryRef>,
}

/// Trait: fmt::Display.
impl fmt::Display for Batch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "id={}, size={}, range=({}..{})",
            self.batch_id,
            self.size(),
            self.id_start(),
            self.id_end()
        )
    }
}

/// Methods.
impl Batch {
    // Returns Iris serial id of batch's last element.
    pub fn id_end(&self) -> IrisSerialId {
        self.vector_ids.last().map(|id| id.serial_id()).unwrap()
    }

    // Returns Iris serial id of batch's first element.
    pub fn id_start(&self) -> IrisSerialId {
        self.vector_ids.first().map(|id| id.serial_id()).unwrap()
    }

    // Returns size of the batch.
    pub fn size(&self) -> usize {
        self.vector_ids.len()
    }
}

/// Generates batches of Iris identifiers for processing.
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

/// Constructor.
impl BatchGenerator {
    /// Create a new `BatchGenerator` with the following properties:
    ///
    /// # Arguments
    ///
    /// * `start_id` - Identifier of first Iris to be indexed.
    /// * `end_id` - Identifier of last Iris to be indexed.
    /// * `batch_size` - Maximum size of a batch.
    /// * `exclusions` - Identifier Iris's not to be indexed.
    ///
    pub fn new(
        start_id: IrisSerialId,
        end_id: IrisSerialId,
        batch_size: usize,
        exclusions: Vec<IrisSerialId>,
    ) -> Self {
        assert!(
            end_id > start_id,
            "Invalid indexation range: {}..{}.",
            start_id,
            end_id
        );

        let range = start_id..=end_id;

        Self {
            batch_size,
            exclusions,
            batch_count: 0,
            range: range.clone(),
            range_iter: range.peekable(),
        }
    }
}

/// Trait: fmt::Display.
impl fmt::Display for BatchGenerator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "batch-size={}, count-of-exclusions={}, range-of-iris-ids=({}..{}), current-batch-id={}",
            self.batch_size,
            self.range.start(),
            self.range.end(),
            self.exclusions.len(),
            self.batch_count
        )
    }
}

/// Methods.
impl BatchGenerator {
    /// Returns next batch of Iris serial identifiers to be indexed.
    fn next_identifiers(&mut self) -> Option<Vec<IrisSerialId>> {
        // Escape if exhausted.
        self.range_iter.peek()?;

        // Construct next batch.
        let mut identifiers = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && identifiers.len() < self.batch_size {
            let next_id = self.range_iter.by_ref().next().unwrap();
            if !self.exclusions.contains(&next_id) {
                identifiers.push(next_id);
            } else {
                Self::log_info(format!("Excluding deletion :: iris-serial-id={}", next_id));
            }
        }

        if identifiers.is_empty() {
            None
        } else {
            Some(identifiers)
        }
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
        iris_stores: &BothEyes<SharedIrisesRef>,
    ) -> impl Future<Output = Result<Option<Batch>, IndexationError>> + Send;
}

/// Batch iterator implementation.
impl BatchIterator for BatchGenerator {
    // Count of generated batches.
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    // Returns next batch of Iris data to be indexed or None if exhausted.
    async fn next_batch(
        &mut self,
        iris_stores: &BothEyes<SharedIrisesRef>,
    ) -> Result<Option<Batch>, IndexationError> {
        if let Some(identifiers) = self.next_identifiers() {
            // Assumption: ids are the same in both left and right stores, esp. versions
            let vector_ids = iris_stores[0]
                .get_vector_ids(&identifiers)
                .await
                .into_iter()
                .zip(identifiers)
                .map(|(id_opt, serial_id)| {
                    id_opt.ok_or(IndexationError::MissingSerialId(serial_id))
                })
                .collect::<Result<Vec<_>, IndexationError>>()?;

            let left_queries = iris_stores[0].get_queries(vector_ids.iter()).await;
            let right_queries = iris_stores[1].get_queries(vector_ids.iter()).await;

            self.batch_count += 1;
            let batch = Batch {
                batch_id: self.batch_count,
                vector_ids,
                left_queries,
                right_queries,
            };
            Self::log_info(format!("Generated batch: {}", batch));
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        execution::hawk_main::StoreId,
        hawkers::{
            aby3::test_utils::setup_aby3_shared_iris_stores_with_preloaded_db,
            plaintext_store::PlaintextStore,
        },
    };

    use super::*;
    use aes_prng::AesRng;
    use eyre::Result;
    use rand::SeedableRng;

    // Defaults.
    const DEFAULT_RNG_SEED: u64 = 0;
    const DEFAULT_PARTY_ID: usize = 0;
    const DEFAULT_SIZE_OF_IRIS_DB: usize = 100;
    const DEFAULT_SIZE_OF_BATCH: usize = 10;
    const DEFAULT_COUNT_OF_BATCHES: usize = 10;

    // Returns a set of test resources.
    fn get_iris_stores() -> (BothEyes<SharedIrisesRef>, usize) {
        let mut rng = AesRng::seed_from_u64(DEFAULT_RNG_SEED);
        let iris_stores: BothEyes<SharedIrisesRef> = [StoreId::Left, StoreId::Right].map(|_| {
            let plaintext_store = PlaintextStore::new_random(&mut rng, DEFAULT_SIZE_OF_IRIS_DB);
            setup_aby3_shared_iris_stores_with_preloaded_db(&mut rng, &plaintext_store)
                .remove(DEFAULT_PARTY_ID)
        });

        (iris_stores, DEFAULT_SIZE_OF_IRIS_DB)
    }

    /// Test new.
    #[tokio::test]
    async fn test_new() -> Result<()> {
        let instance = BatchGenerator::new(
            1,
            DEFAULT_SIZE_OF_IRIS_DB as IrisSerialId,
            DEFAULT_SIZE_OF_BATCH,
            Vec::new(),
        );
        assert_eq!(*instance.range.end() as usize, DEFAULT_SIZE_OF_IRIS_DB);

        Ok(())
    }

    /// Test iteration.
    #[tokio::test]
    async fn test_iterator() -> Result<()> {
        // Set resources.
        let (iris_stores, db_size) = get_iris_stores();

        let mut instance = BatchGenerator::new(
            1,
            db_size as IrisSerialId,
            DEFAULT_SIZE_OF_BATCH,
            Vec::new(),
        );

        // Expecting M batches of N Iris's per batch.
        while let Some(batch) = instance.next_batch(&iris_stores).await? {
            assert_eq!(batch.size(), DEFAULT_SIZE_OF_BATCH);
        }
        assert_eq!(instance.batch_count, DEFAULT_COUNT_OF_BATCHES);

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
