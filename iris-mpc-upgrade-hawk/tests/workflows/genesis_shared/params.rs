use iris_mpc_common::IrisSerialId;

/// Excapsulates data used to initialise test inputs.
#[derive(Debug, Clone, Copy)]
pub struct TestParams {
    // Initial batch size for indexing.
    batch_size: usize,

    // Error rate to be applied when calculating dynamic batch sizes.
    batch_size_error_rate: usize,

    // Serial identifier of maximum indexed Iris.
    max_indexation_id: IrisSerialId,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    perform_db_snapshot: bool,

    // Flag indicating whether a db backup will be used as initial data source.
    use_db_backup_as_source: bool,

    // State of an RNG being used to inject entropy to share creation.
    shares_generator_rng_state: u64,

    // Size of batches when generating Iris shares for testing purposes.
    shares_generator_batch_size: usize,

    // Batch size when persisting iris shares to pgres stores.
    shares_pgres_tx_batch_size: usize,
}

/// Constructor.
impl TestParams {
    pub fn new(
        batch_size: usize,
        batch_size_error_rate: usize,
        max_indexation_id: IrisSerialId,
        perform_db_snapshot: bool,
        use_db_backup_as_source: bool,
        shares_generator_batch_size: usize,
        shares_generator_rng_state: u64,
        shares_pgres_tx_batch_size: usize,
    ) -> Self {
        Self {
            batch_size,
            batch_size_error_rate,
            max_indexation_id,
            perform_db_snapshot,
            use_db_backup_as_source,
            shares_generator_batch_size,
            shares_generator_rng_state,
            shares_pgres_tx_batch_size,
        }
    }
}

/// Accessors.
impl TestParams {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn batch_size_error_rate(&self) -> usize {
        self.batch_size_error_rate
    }

    pub fn max_indexation_id(&self) -> IrisSerialId {
        self.max_indexation_id
    }

    pub fn perform_db_snapshot(&self) -> bool {
        self.perform_db_snapshot
    }

    pub fn use_db_backup_as_source(&self) -> bool {
        self.use_db_backup_as_source
    }

    pub fn shares_pgres_tx_batch_size(&self) -> usize {
        self.shares_pgres_tx_batch_size
    }

    pub fn shares_generator_batch_size(&self) -> usize {
        self.shares_generator_batch_size
    }

    pub fn shares_generator_rng_state(&self) -> u64 {
        self.shares_generator_rng_state
    }
}
