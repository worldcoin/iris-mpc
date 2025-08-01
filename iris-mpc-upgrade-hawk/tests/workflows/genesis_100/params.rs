use crate::utils::defaults;
use iris_mpc_common::IrisSerialId;

/// Excapsulates data used to initialise test inputs.
#[derive(Debug, Clone, Copy)]
pub struct TestParams {
    // Initial batch size for indexing.
    arg_batch_size: usize,

    // Error rate to be applied when calculating dynamic batch sizes.
    arg_batch_size_error_rate: usize,

    // Serial identifier of maximum indexed Iris.
    arg_max_indexation_id: IrisSerialId,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    arg_perform_db_snapshot: bool,

    // Flag indicating whether a db backup will be used as initial data source.
    arg_use_db_backup_as_source: bool,

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
        indexation_batch_size: usize,
        indexation_batch_size_error_rate: usize,
        arg_max_indexation_id: IrisSerialId,
        arg_perform_db_snapshot: bool,
        arg_use_db_backup_as_source: bool,
        shares_generator_batch_size: usize,
        shares_generator_rng_state: u64,
        shares_pgres_tx_batch_size: usize,
    ) -> Self {
        Self {
            arg_batch_size: indexation_batch_size,
            arg_batch_size_error_rate: indexation_batch_size_error_rate,
            arg_max_indexation_id,
            arg_perform_db_snapshot,
            arg_use_db_backup_as_source,
            shares_generator_batch_size,
            shares_generator_rng_state,
            shares_pgres_tx_batch_size,
        }
    }
}

/// Accessors.
impl TestParams {
    pub fn arg_batch_size(&self) -> usize {
        self.arg_batch_size
    }

    pub fn arg_batch_size_error_rate(&self) -> usize {
        self.arg_batch_size_error_rate
    }

    pub fn arg_max_indexation_id(&self) -> IrisSerialId {
        self.arg_max_indexation_id
    }

    pub fn arg_perform_db_snapshot(&self) -> bool {
        self.arg_perform_db_snapshot
    }

    pub fn arg_use_db_backup_as_source(&self) -> bool {
        self.arg_use_db_backup_as_source
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
