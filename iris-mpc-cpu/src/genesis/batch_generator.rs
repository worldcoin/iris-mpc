use super::utils::{self, errors::IndexationError};
use crate::{
    execution::hawk_main::{BothEyes, LEFT, RIGHT},
    hawkers::aby3::aby3_store::{QueryRef, SharedIrisesRef},
};
use eyre::Result;
use iris_mpc_common::{vector_id::VectorId, IrisSerialId};
use std::{
    fmt,
    future::Future,
    iter::Peekable,
    ops::RangeInclusive,
    sync::{Once, OnceLock},
};

/// Component name for logging purposes.
const COMPONENT: &str = "Batch-Generator";

/// Synchronization mechanism for batch size initialization.
static BATCH_SIZE_INIT: Once = Once::new();
static BATCH_SIZE_VALUE: OnceLock<usize> = OnceLock::new();

/// A batch for upstream indexation.
#[derive(Debug)]
pub struct Batch {
    /// Ordinal batch identifier scoped by processing context.
    pub batch_id: usize,

    /// Array of left iris codes, in query format.
    pub left_queries: Vec<QueryRef>,

    /// Array of right iris codes, in query format.
    pub right_queries: Vec<QueryRef>,

    /// Array of vector ids of iris enrollments.
    pub vector_ids: Vec<VectorId>,
}

/// Constructor.
impl Batch {
    fn new(
        batch_id: usize,
        vector_ids: Vec<VectorId>,
        left_queries: Vec<QueryRef>,
        right_queries: Vec<QueryRef>,
    ) -> Self {
        Self {
            batch_id,
            vector_ids,
            left_queries,
            right_queries,
        }
    }
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

/// Generates batches of Iris identifiers for processing.
pub struct BatchGenerator {
    // Count of generated batches.
    batch_count: usize,

    // Policy to apply when calculating batch sizes.
    batch_size: BatchSize,

    // Set of Iris serial identifiers to exclude from indexing.
    exclusions: Vec<IrisSerialId>,

    // Range of Iris serial identifiers to be indexed.
    range: RangeInclusive<IrisSerialId>,

    // Iterator over range of Iris serial identifiers to be indexed.
    range_iter: Peekable<RangeInclusive<IrisSerialId>>,
}

/// Constructor.
impl BatchGenerator {
    pub fn new(
        start_id: IrisSerialId,
        end_id: IrisSerialId,
        batch_size: BatchSize,
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
            "current-batch-id={}, count-of-exclusions={}, range-of-iris-ids=({}..{})",
            self.batch_count,
            self.exclusions.len(),
            self.range.start(),
            self.range.end(),
        )
    }
}

/// Batch iterator.
pub trait BatchIterator {
    /// Count of generated batches.
    fn batch_count(&self) -> usize;

    /// Iterator over batches of Iris data to be indexed.
    ///
    /// # Arguments
    ///
    /// * `last_indexed_id` - Last Iris serial identifier indexed.
    /// * `iris_stores` - In memory cache of Iris shares data.
    ///
    /// # Returns
    ///
    /// Future that resolves to maybe a Batch or an IndexationError.
    ///
    fn next_batch(
        &mut self,
        last_indexed_id: IrisSerialId,
        iris_stores: &BothEyes<SharedIrisesRef>,
    ) -> impl Future<Output = Result<Option<Batch>, IndexationError>> + Send;
}

/// Policy over batch size calculations.
#[derive(Debug, Eq, PartialEq)]
pub enum BatchSize {
    /// Static batch size.
    Static(usize),
    /// Dynamic batch size with size error coefficient & hnsw-m param.
    Dynamic(usize, usize),
}

/// Trait: fmt::Display.
impl fmt::Display for BatchSize {
    #[allow(non_snake_case)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BatchSize::Static(size) => write!(f, "Static(size={})", size),
            BatchSize::Dynamic(error_correction, hnsw_M) => {
                write!(
                    f,
                    "Dynamic(error-correction={}, hnsw-M={})",
                    error_correction, hnsw_M
                )
            }
        }
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

    // Returns serial identifiers within batch.
    pub fn serial_ids(&self) -> Vec<IrisSerialId> {
        self.vector_ids.iter().map(|id| id.serial_id()).collect()
    }

    // Returns size of the batch.
    pub fn size(&self) -> usize {
        self.vector_ids.len()
    }
}

/// Methods.
impl BatchGenerator {
    /// Returns maximum size of next batch.
    /// TODO: currently wrapped in a once - verify when can be invoked each iteration.
    fn next_batch_size_max(&self, last_indexed_id: IrisSerialId) -> &'static usize {
        BATCH_SIZE_INIT.call_once(|| {
            let value = self.batch_size.next_max(last_indexed_id);
            BATCH_SIZE_VALUE.set(value).unwrap();
        });

        BATCH_SIZE_VALUE.get().unwrap()
    }

    /// Returns next batch of Iris serial identifiers to be indexed.
    fn next_identifiers(&mut self, last_indexed_id: IrisSerialId) -> Option<Vec<IrisSerialId>> {
        // Escape if exhausted.
        self.range_iter.peek()?;

        // Calculate batch size.
        let batch_size_max = self.next_batch_size_max(last_indexed_id);

        // Construct next batch.
        let mut identifiers = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && identifiers.len() < *batch_size_max {
            let next_id = self.range_iter.by_ref().next().unwrap();
            if !self.exclusions.contains(&next_id) {
                identifiers.push(next_id);
            } else {
                log_info(format!("Excluding deletion :: iris-serial-id={}", next_id));
            }
        }

        if identifiers.is_empty() {
            None
        } else {
            Some(identifiers)
        }
    }
}

/// Methods.
impl BatchIterator for BatchGenerator {
    // Count of generated batches.
    fn batch_count(&self) -> usize {
        self.batch_count
    }

    // Returns next batch of Iris data to be indexed or None if exhausted.
    async fn next_batch(
        &mut self,
        last_indexed_id: IrisSerialId,
        imem_iris_stores: &BothEyes<SharedIrisesRef>,
    ) -> Result<Option<Batch>, IndexationError> {
        if let Some(identifiers) = self.next_identifiers(last_indexed_id) {
            // Set vector identifiers - assumes left/right store equivalence.
            let vector_ids = imem_iris_stores[LEFT]
                .get_vector_ids(&identifiers)
                .await
                .into_iter()
                .zip(identifiers)
                .map(|(id_opt, serial_id)| {
                    id_opt.ok_or(IndexationError::MissingSerialId(serial_id))
                })
                .collect::<Result<Vec<_>, IndexationError>>()?;

            // Update internal state.
            self.batch_count += 1;

            Ok(Some(Batch::new(
                self.batch_count,
                vector_ids.clone(),
                imem_iris_stores[LEFT].get_queries(vector_ids.iter()).await,
                imem_iris_stores[RIGHT].get_queries(vector_ids.iter()).await,
            )))
        } else {
            Ok(None)
        }
    }
}

/// Methods.
impl BatchSize {
    /// Calculates maximum size of next batch to be indexed.
    #[allow(non_snake_case)]
    fn next_max(&self, last_indexed_id: IrisSerialId) -> usize {
        log_info(format!(
            "Calculating max batch size: last-indexed-id={} :: {}",
            last_indexed_id, self
        ));

        match last_indexed_id {
            // Empty graph therefore batch size defaults to 1.
            0 => {
                log_info(String::from(
                    "Using static max batch size of 1 as graph is empty",
                ));
                1
            }
            _ => match self {
                BatchSize::Dynamic(r, M) => {
                    // r: configurable parameter for error rate.
                    // M: HNSW parameter for nearest neighbors.
                    // n: current graph size (last_indexed_id).
                    let N = last_indexed_id;

                    // batch_size: floor(N/(Mr - 1) + 1)
                    let batch_size =
                        (N as f64 / (*M as f64 * *r as f64 - 1.0) + 1.0).floor() as usize;

                    log_info(format!(
                            "Dynamic max batch size calculated: {} (formula: N/(Mr-1)+1, where N={}, M={}, r={})",
                            batch_size, N, M, r
                        ));

                    batch_size
                }
                BatchSize::Static(size) => {
                    log_info(format!("Using static max batch size: {}", size));
                    *size
                }
            },
        }
    }
}

// Helper: component logging.
fn log_info(msg: String) {
    utils::log_info(COMPONENT, msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        execution::hawk_main::StoreId,
        hawkers::{
            aby3::test_utils::setup_aby3_shared_iris_stores_with_preloaded_db,
            plaintext_store::PlaintextStore,
        },
    };
    use aes_prng::AesRng;
    use eyre::Result;
    use rand::{Rng, SeedableRng};

    const BATCH_SIZE_ERROR_RATE: usize = 128;
    const EXCLUSIONS: [IrisSerialId; 9] = [3, 7, 12, 15, 22, 30, 70, 84, 92];
    const HNSW_PARAM_M: usize = 256;
    const INDEXATION_END_ID: IrisSerialId = 100;
    const INDEXATION_END_ID_MAX: IrisSerialId = 15_000_000;
    const INDEXATION_START_ID: IrisSerialId = 1;
    const LAST_INDEXED_IDS: [IrisSerialId; 10] = [
        0, 2312177, 6983790, 7110281, 7859739, 10174686, 10270291, 11961225, 12317574, 14277641,
    ];
    const PARTY_ID: usize = 0;
    const RNG_SEED: u64 = 0;
    const SIZE_OF_IRIS_DB: usize = 100;
    const STATIC_BATCH_SIZE_1: usize = 1;
    const STATIC_BATCH_SIZE_10: usize = 10;

    impl BatchSize {
        fn new_1() -> Self {
            Self::Static(STATIC_BATCH_SIZE_10)
        }
        fn new_2() -> Self {
            Self::Static(STATIC_BATCH_SIZE_1)
        }
        fn new_3() -> Self {
            Self::Dynamic(BATCH_SIZE_ERROR_RATE, HNSW_PARAM_M)
        }
    }

    impl BatchGenerator {
        fn new_0(batch_size: BatchSize, exclusions: Vec<IrisSerialId>) -> Self {
            Self::new(
                INDEXATION_START_ID,
                INDEXATION_END_ID,
                batch_size,
                exclusions,
            )
        }
        fn new_1() -> Self {
            Self::new_0(BatchSize::new_1(), Vec::new())
        }
        fn new_2() -> Self {
            Self::new_0(BatchSize::new_2(), Vec::new())
        }
        fn new_3() -> Self {
            Self::new_0(BatchSize::new_3(), Vec::new())
        }
        fn new_4() -> Self {
            Self::new_0(BatchSize::new_1(), Vec::from(EXCLUSIONS))
        }
    }

    // Returns a test imem Iris store.
    fn get_iris_imem_stores() -> (BothEyes<SharedIrisesRef>, usize) {
        let mut rng = AesRng::seed_from_u64(RNG_SEED);
        let iris_stores: BothEyes<SharedIrisesRef> = [StoreId::Left, StoreId::Right].map(|_| {
            let plaintext_store = PlaintextStore::new_random(&mut rng, SIZE_OF_IRIS_DB);
            setup_aby3_shared_iris_stores_with_preloaded_db(&mut rng, &plaintext_store)
                .remove(PARTY_ID)
        });

        (iris_stores, SIZE_OF_IRIS_DB)
    }

    // Returns a random ordered set of last indexed identifiers.
    fn get_last_indexed_identifiers() -> Vec<IrisSerialId> {
        let mut rng = rand::thread_rng();
        let mut random_vec: Vec<u32> = (0..10)
            .map(|_| rng.gen_range(2 as IrisSerialId..=INDEXATION_END_ID_MAX as IrisSerialId))
            .collect();
        random_vec.sort();
        random_vec.insert(0, 0);

        random_vec
    }

    #[test]
    fn test_new_generator() {
        for (generator, size, exclusions) in [
            (BatchGenerator::new_1(), BatchSize::new_1(), Vec::new()),
            (BatchGenerator::new_2(), BatchSize::new_2(), Vec::new()),
            (BatchGenerator::new_3(), BatchSize::new_3(), Vec::new()),
            (
                BatchGenerator::new_4(),
                BatchSize::new_1(),
                Vec::from(EXCLUSIONS),
            ),
        ] {
            assert_eq!(*generator.range.start(), INDEXATION_START_ID);
            assert_eq!(*generator.range.end(), INDEXATION_END_ID);
            assert_eq!(generator.batch_count, 0);
            assert_eq!(generator.exclusions, exclusions);
            assert_eq!(size, generator.batch_size);
        }
    }

    /// Test batch size iteration against sets of last indexed ids.
    #[test]
    fn test_batch_size_1() {
        for (instance, size) in [
            (BatchSize::new_1(), STATIC_BATCH_SIZE_10),
            (BatchSize::new_2(), STATIC_BATCH_SIZE_1),
        ] {
            for identifiers in [LAST_INDEXED_IDS.to_vec(), get_last_indexed_identifiers()] {
                for last_indexed_id in identifiers {
                    match last_indexed_id {
                        0 => {
                            // Graph is empty therefore batch size is 1.
                            assert_eq!(instance.next_max(last_indexed_id), 1);
                        }
                        _ => {
                            assert_eq!(instance.next_max(last_indexed_id), size);
                        }
                    }
                }
            }
        }
    }

    /// Test dynamic batch size iteration against a set of known last indexed ids.
    #[test]
    fn test_batch_size_2() {
        let instance = BatchSize::new_3();
        for last_indexed_id in LAST_INDEXED_IDS {
            match last_indexed_id {
                0 => {
                    // Graph is empty therefore batch size is 1.
                    assert_eq!(instance.next_max(last_indexed_id), 1);
                }
                _ => {
                    // TODO: how to correctly assert against precise expected value.
                    assert!(instance.next_max(last_indexed_id) >= STATIC_BATCH_SIZE_10);
                }
            }
        }
    }

    /// Test batch identifiers iteration against static set of last indexed ids.
    /// Expecting 11 batches with following sizes:
    ///   batch-01 -> 1;
    ///   batch-(02-10) -> 10;
    ///   batch-11 -> 9;
    #[test]
    fn test_next_identifiers_1() {
        let mut batch_id: usize = 0;
        let mut generator = BatchGenerator::new_1();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some(identifiers) = generator.next_identifiers(last_indexed_id) {
            batch_id += 1;
            assert!(!identifiers.is_empty());
            match batch_id {
                1 => {
                    assert_eq!(identifiers.len(), 1);
                }
                11 => {
                    assert_eq!(identifiers.len(), STATIC_BATCH_SIZE_10 - 1);
                }
                _ => {
                    assert_eq!(identifiers.len(), STATIC_BATCH_SIZE_10);
                }
            }
            last_indexed_id += identifiers.len() as IrisSerialId;
        }
        assert_eq!(batch_id, 11);
    }

    /// Test next identifiers: batch-size=1.
    #[test]
    fn test_next_identifiers_2() {
        let mut batch_id: usize = 0;
        let mut generator = BatchGenerator::new_2();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some(identifiers) = generator.next_identifiers(last_indexed_id) {
            batch_id += 1;
            assert_eq!(identifiers.len(), 1);
            last_indexed_id += 1;
        }
        assert_eq!(batch_id, 100);
    }

    /// Test batch generation: store length=100 :: batch-size=10.
    #[tokio::test]
    async fn test_next_batch_1() -> Result<()> {
        let mut generator = BatchGenerator::new_1();
        let (iris_stores, _) = get_iris_imem_stores();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some(batch) = generator.next_batch(last_indexed_id, &iris_stores).await? {
            assert!(batch.size() > 0);
            match batch.batch_id {
                1 => {
                    assert_eq!(batch.size(), 1);
                }
                11 => {
                    assert_eq!(batch.size(), 9);
                }
                _ => {
                    assert_eq!(batch.size(), 10);
                }
            }
            last_indexed_id += batch.size() as IrisSerialId;
        }

        Ok(())
    }

    /// Test batch generation against iris store with 100 Irises.
    /// Expecting 100 batches of size 1.
    #[tokio::test]
    async fn test_next_batch_2() -> Result<()> {
        let mut batch_id = 0;
        let mut generator = BatchGenerator::new_2();
        let (iris_stores, _) = get_iris_imem_stores();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some(batch) = generator.next_batch(last_indexed_id, &iris_stores).await? {
            assert_eq!(batch.size(), 1);
            batch_id = batch.batch_id;
            last_indexed_id = batch.id_end();
        }
        assert_eq!(batch_id, 100);

        Ok(())
    }

    #[tokio::test]
    async fn test_exclusions() -> Result<()> {
        let mut batches = Vec::new();
        let mut generator = BatchGenerator::new_4();
        let (iris_stores, _) = get_iris_imem_stores();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some(batch) = generator.next_batch(last_indexed_id, &iris_stores).await? {
            batches.push(batch.serial_ids());
            last_indexed_id = batch.id_end();
        }

        assert_eq!(
            batches,
            vec![
                vec![1],
                vec![2, 4, 5, 6, 8, 9, 10, 11, 13, 14],
                vec![16, 17, 18, 19, 20, 21, 23, 24, 25, 26],
                vec![27, 28, 29, 31, 32, 33, 34, 35, 36, 37],
                vec![38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                vec![48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
                vec![58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
                vec![68, 69, 71, 72, 73, 74, 75, 76, 77, 78],
                vec![79, 80, 81, 82, 83, 85, 86, 87, 88, 89],
                vec![90, 91, 93, 94, 95, 96, 97, 98, 99, 100],
            ]
        );

        Ok(())
    }
}
