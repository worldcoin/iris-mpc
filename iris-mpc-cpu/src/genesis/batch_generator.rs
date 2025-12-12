use super::utils::{self, errors::IndexationError};
use crate::{
    execution::hawk_main::{BothEyes, LEFT, RIGHT},
    hawkers::aby3::aby3_store::{Aby3Query, Aby3SharedIrisesRef},
};
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

    /// Array of left iris codes, in query format.
    pub left_queries: Vec<Aby3Query>,

    /// Array of right iris codes, in query format.
    pub right_queries: Vec<Aby3Query>,

    /// Array of vector ids of iris enrollments.
    pub vector_ids: Vec<VectorId>,

    /// Iris data for persistence.
    pub vector_ids_to_persist: Vec<VectorId>,
}

/// Constructor.
impl Batch {
    fn new(
        batch_id: usize,
        vector_ids: Vec<VectorId>,
        left_queries: Vec<Aby3Query>,
        right_queries: Vec<Aby3Query>,
        vector_ids_to_persist: Vec<VectorId>,
    ) -> Self {
        Self {
            batch_id,
            vector_ids,
            left_queries,
            right_queries,
            vector_ids_to_persist,
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
        iris_stores: &BothEyes<Aby3SharedIrisesRef>,
    ) -> impl Future<Output = Result<Option<Batch>, IndexationError>> + Send;
}

/// Policy over batch size calculations.
#[derive(Debug, Eq, PartialEq)]
pub enum BatchSize {
    /// Static batch size.
    Static(usize),
    /// Dynamic batch size with size error coefficient & hnsw-m param.
    #[allow(non_snake_case)]
    Dynamic {
        error_correction: usize,
        hnsw_M: usize,
    },
}

/// Constructor.
impl BatchSize {
    #[allow(non_snake_case)]
    pub fn new_dynamic(error_correction: usize, hnsw_M: usize) -> Self {
        // TODO: defensive guard by asserting reasonable threshold/floors for inputs.
        log_info(format!(
            "Creating dynamic batch size: error-correction={}, hnsw-M={}",
            error_correction, hnsw_M
        ));
        BatchSize::Dynamic {
            error_correction,
            hnsw_M,
        }
    }

    pub fn new_static(size: usize) -> Self {
        // TODO: defensive guard by asserting reasonable threshold/floors for inputs.
        log_info(format!("Creating static batch size: size={}", size));
        BatchSize::Static(size)
    }

    #[allow(non_snake_case)]
    pub fn new_static_from_dynamic_formula(
        last_indexed_id: IrisSerialId,
        error_correction: usize,
        hnsw_M: usize,
    ) -> Self {
        // TODO: defensive guard by asserting reasonable threshold/floors for inputs.
        log_info(format!(
            "Creating static batch size from dynamic formula: last-indexed-id={}, error-correction={}, hnsw-M={}",
            last_indexed_id, error_correction, hnsw_M
        ));
        Self::new_static(Self::get_dynamic_size(
            last_indexed_id,
            error_correction,
            hnsw_M,
        ))
    }
}

/// Trait: fmt::Display.
impl fmt::Display for BatchSize {
    #[allow(non_snake_case)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BatchSize::Static(size) => write!(f, "Static(size={})", size),
            BatchSize::Dynamic {
                error_correction,
                hnsw_M,
            } => {
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
        self.vector_ids_to_persist
            .last()
            .map(|id| id.serial_id())
            .unwrap_or_default()
    }

    // Returns Iris serial id of batch's first element.
    pub fn id_start(&self) -> IrisSerialId {
        self.vector_ids_to_persist
            .first()
            .map(|id| id.serial_id())
            .unwrap_or_default()
    }

    // Returns serial identifiers within batch.
    pub fn serial_ids(&self) -> Vec<IrisSerialId> {
        self.vector_ids.iter().map(|id| id.serial_id()).collect()
    }

    // Returns size of the batch.x
    pub fn size(&self) -> usize {
        self.vector_ids.len()
    }
}

/// Methods.
impl BatchGenerator {
    /// Returns next batch of Iris serial identifiers to be indexed.
    fn next_identifiers(
        &mut self,
        last_indexed_id: IrisSerialId,
    ) -> Option<(Vec<IrisSerialId>, Vec<IrisSerialId>)> {
        // Escape if exhausted.
        if self.range_iter.peek().is_none() {
            log_info(format!(
                "Exhausted range iterator: last-indexed-id={}",
                last_indexed_id
            ));
            return None;
        }

        // Calculate batch size.
        let batch_size_max = self.batch_size.next_max(last_indexed_id);

        // Construct next batch.
        let mut identifiers = Vec::<IrisSerialId>::new();
        let mut identifiers_for_copying = Vec::<IrisSerialId>::new();
        while self.range_iter.peek().is_some() && identifiers.len() < batch_size_max {
            let next_id = self.range_iter.by_ref().next().unwrap();
            identifiers_for_copying.push(next_id);
            if !self.exclusions.contains(&next_id) {
                identifiers.push(next_id);
            } else {
                log_info(format!("Excluding deletion :: iris-serial-id={}", next_id));
            }
        }

        if identifiers_for_copying.is_empty() {
            None
        } else {
            Some((identifiers, identifiers_for_copying))
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
        imem_iris_stores: &BothEyes<Aby3SharedIrisesRef>,
    ) -> Result<Option<Batch>, IndexationError> {
        let (identifiers, identifiers_for_indexation) = match self.next_identifiers(last_indexed_id)
        {
            Some(pair) => pair,
            None => {
                log_info(format!(
                    "Exhausted identifiers: last-indexed-id={}",
                    last_indexed_id
                ));
                return Ok(None);
            }
        };
        // Set vector identifiers - assumes left/right store equivalence.
        let vector_ids: Vec<VectorId> = imem_iris_stores[LEFT]
            .get_vector_ids(&identifiers)
            .await
            .into_iter()
            .zip(identifiers)
            .map(|(id_opt, serial_id)| id_opt.ok_or(IndexationError::MissingSerialId(serial_id)))
            .collect::<Result<Vec<_>, IndexationError>>()?;

        let vector_ids_for_persistence: Vec<VectorId> = imem_iris_stores[LEFT]
            .get_vector_ids(&identifiers_for_indexation)
            .await
            .into_iter()
            .zip(identifiers_for_indexation)
            .map(|(id_opt, serial_id)| id_opt.ok_or(IndexationError::MissingSerialId(serial_id)))
            .collect::<Result<Vec<_>, IndexationError>>()?;

        self.batch_count += 1;

        let left_queries = imem_iris_stores[LEFT]
            .get_vectors_or_empty(vector_ids.iter())
            .await
            .iter()
            .map(Aby3Query::new)
            .collect();

        let right_queries = imem_iris_stores[RIGHT]
            .get_vectors_or_empty(vector_ids.iter())
            .await
            .iter()
            .map(Aby3Query::new)
            .collect();

        Ok(Some(Batch::new(
            self.batch_count,
            vector_ids.clone(),
            left_queries,
            right_queries,
            vector_ids_for_persistence.clone(),
        )))
    }
}

/// Methods.
impl BatchSize {
    /// Dynamically computes size of next batch to be indexed.
    #[allow(non_snake_case)]
    pub fn get_dynamic_size(
        last_indexed_id: IrisSerialId,
        error_correction: usize,
        hnsw_M: usize,
    ) -> usize {
        // r: configurable parameter for error rate.
        let r = error_correction;

        // M: HNSW parameter for nearest neighbors.
        let M = hnsw_M;

        // n: current graph size (last_indexed_id).
        // TODO: Should be existing graph size rather than id of last indexed node. Typically deviation
        // will be minimal, but in the interests of precision we should use the existing graph size.
        let N = last_indexed_id;

        // batch_size: floor(N/(Mr - 1) + 1)
        (N as usize).div_euclid(M * r - 1) + 1
    }

    /// Calculates maximum size of next batch to be indexed.
    #[allow(non_snake_case)]
    fn next_max(&self, last_indexed_id: IrisSerialId) -> usize {
        match self {
            BatchSize::Static(size) => {
                log_info(format!(
                    "Calculated batch size: last-indexed-id={} :: size={}",
                    last_indexed_id, size
                ));
                *size
            }
            BatchSize::Dynamic {
                error_correction: r,
                hnsw_M: M,
            } => {
                let batch_size = Self::get_dynamic_size(last_indexed_id, *r, *M);
                log_info(format!(
                    "Calculated batch size: last-indexed-id={} :: size={} (formula: N/(Mr-1)+1, where N={}, M={}, r={})",
                    last_indexed_id, batch_size, last_indexed_id, M, r
                ));

                batch_size
            }
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
    const LAST_INDEXED_IDS: [(IrisSerialId, usize); 10] = [
        (0, 1),
        (2312177, 71),
        (6983790, 214),
        (7110281, 217),
        (7859739, 240),
        (10174686, 311),
        (10270291, 314),
        (11961225, 366),
        (12317574, 376),
        (14277641, 436),
    ];
    const PARTY_ID: usize = 0;
    const RNG_SEED: u64 = 0;
    const SIZE_OF_IRIS_DB: usize = 100;
    const STATIC_BATCH_SIZE_1: usize = 1;
    const STATIC_BATCH_SIZE_10: usize = 10;

    impl BatchSize {
        fn new_1() -> Self {
            Self::new_static(STATIC_BATCH_SIZE_10)
        }
        fn new_2() -> Self {
            Self::new_static(STATIC_BATCH_SIZE_1)
        }
        fn new_3() -> Self {
            Self::new_dynamic(BATCH_SIZE_ERROR_RATE, HNSW_PARAM_M)
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
    fn get_iris_imem_stores() -> (BothEyes<Aby3SharedIrisesRef>, usize) {
        let mut rng = AesRng::seed_from_u64(RNG_SEED);
        let iris_stores: BothEyes<Aby3SharedIrisesRef> =
            [StoreId::Left, StoreId::Right].map(|_| {
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
            for last_indexed_id in get_last_indexed_identifiers() {
                assert_eq!(instance.next_max(last_indexed_id), size);
            }
        }
    }

    /// Test dynamic batch size iteration against a set of known last indexed ids.
    #[test]
    fn test_batch_size_2() {
        let instance = BatchSize::new_3();
        for (last_indexed_id, batch_size) in LAST_INDEXED_IDS {
            assert_eq!(instance.next_max(last_indexed_id), batch_size);
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

        while let Some((identifiers, identifiers_for_indexation)) =
            generator.next_identifiers(last_indexed_id)
        {
            batch_id += 1;
            assert_eq!(identifiers.len(), STATIC_BATCH_SIZE_10);
            assert_eq!(identifiers_for_indexation.len(), STATIC_BATCH_SIZE_10);
            last_indexed_id += identifiers.len() as IrisSerialId;
        }
        assert_eq!(batch_id, 10);
    }

    /// Test next identifiers: batch-size=1.
    #[test]
    fn test_next_identifiers_2() {
        let mut batch_id: usize = 0;
        let mut generator = BatchGenerator::new_2();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some((identifiers, identifiers_for_indexation)) =
            generator.next_identifiers(last_indexed_id)
        {
            batch_id += 1;
            assert_eq!(identifiers.len(), 1);
            assert_eq!(identifiers_for_indexation.len(), 1);
            last_indexed_id += 1;
        }
        assert_eq!(batch_id, 100);
    }

    /// Test next identifiers: batch-size=1.
    #[test]
    fn test_next_identifiers_for_indexation_with_exclusions() {
        let mut batch_id: usize = 0;
        let mut generator = BatchGenerator::new_4();
        let mut last_indexed_id = 0 as IrisSerialId;

        let mut all_identifiers: Vec<IrisSerialId> = vec![];
        let mut all_identifiers_for_indexation: Vec<IrisSerialId> = vec![];

        while let Some((identifiers, identifiers_for_indexation)) =
            generator.next_identifiers(last_indexed_id)
        {
            batch_id += 1;
            last_indexed_id += 1;
            assert!(identifiers.len() <= identifiers_for_indexation.len());

            all_identifiers_for_indexation.extend(identifiers_for_indexation.clone());
            all_identifiers.extend(identifiers.clone());
        }
        assert_eq!(batch_id, 10);
        for id in EXCLUSIONS {
            assert!(!all_identifiers.contains(&id));
            assert!(all_identifiers_for_indexation.contains(&id));
        }
    }

    /// Test batch generation: store length=100 :: batch-size=10.
    #[tokio::test]
    async fn test_next_batch_1() -> Result<()> {
        let mut generator = BatchGenerator::new_1();
        let (iris_stores, _) = get_iris_imem_stores();
        let mut last_indexed_id = 0 as IrisSerialId;

        while let Some(batch) = generator.next_batch(last_indexed_id, &iris_stores).await? {
            assert_eq!(batch.size(), 10);
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
                vec![1, 2, 4, 5, 6, 8, 9, 10, 11, 13],
                vec![14, 16, 17, 18, 19, 20, 21, 23, 24, 25],
                vec![26, 27, 28, 29, 31, 32, 33, 34, 35, 36],
                vec![37, 38, 39, 40, 41, 42, 43, 44, 45, 46],
                vec![47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
                vec![57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                vec![67, 68, 69, 71, 72, 73, 74, 75, 76, 77],
                vec![78, 79, 80, 81, 82, 83, 85, 86, 87, 88],
                vec![89, 90, 91, 93, 94, 95, 96, 97, 98, 99],
                vec![100]
            ]
        );

        Ok(())
    }
}
