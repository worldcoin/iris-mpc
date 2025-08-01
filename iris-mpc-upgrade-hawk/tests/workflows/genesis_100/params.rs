use iris_mpc_common::IrisSerialId;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs;

/// Excapsulates data used to initialise test inputs.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    // Initial batch size for indexing.
    batch_size: usize,

    // Error rate to be applied when calculating dynamic batch sizes.
    batch_size_error_rate: usize,

    // Serial identifier of maximum indexed Iris.
    max_indexation_id: IrisSerialId,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    perform_db_snapshot: bool,

    // Batch size when persisting data to pgres stores.
    pgres_tx_batch_size: usize,

    // Flag indicating whether a db backup will be used as initial data source.
    use_db_backup_as_source: bool,

    // used for construction of HNSW graphs
    rng_state: u64,
}

impl Params {
    pub fn new(
        batch_size: usize,
        batch_size_error_rate: usize,
        max_indexation_id: IrisSerialId,
        perform_db_snapshot: bool,
        pgres_tx_batch_size: usize,
        use_db_backup_as_source: bool,
        rng_state: u64,
    ) -> Self {
        Self {
            batch_size,
            batch_size_error_rate,
            max_indexation_id,
            perform_db_snapshot,
            pgres_tx_batch_size,
            use_db_backup_as_source,
            rng_state,
        }
    }

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

    pub fn pgres_tx_batch_size(&self) -> usize {
        self.pgres_tx_batch_size
    }

    pub fn use_db_backup_as_source(&self) -> bool {
        self.use_db_backup_as_source
    }

    pub fn rng_state(&self) -> u64 {
        self.rng_state
    }

    pub fn to_genesis_execution_args(&self) -> ExecutionArgs {
        ExecutionArgs::new(
            self.batch_size(),
            self.batch_size_error_rate(),
            self.max_indexation_id(),
            self.perform_db_snapshot(),
            self.use_db_backup_as_source(),
        )
    }
}
