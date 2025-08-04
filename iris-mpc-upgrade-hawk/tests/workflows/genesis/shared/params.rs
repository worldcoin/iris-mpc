use crate::resources::NODE_CONFIG_KIND_GENESIS;
use iris_mpc_common::IrisSerialId;

/// Excapsulates data used to initialise test inputs.
#[derive(Debug, Clone, Copy)]
pub struct TestParams {
    // Initial batch size for indexing.
    batch_size: usize,

    // Error rate to be applied when calculating dynamic batch sizes.
    batch_size_error_rate: usize,

    // Maximum number of Ieis deletions to load into memory.
    max_deletions: usize,

    // Serial identifier of maximum indexed Iris.
    max_indexation_id: IrisSerialId,

    // Maximum number of Iris modifications to load into memory.
    max_modifications: usize,

    // Ordinal identifier of node config file to read from test resources.
    node_config_idx: usize,

    // Flag indicating whether a snapshot is to be taken when inner process completes.
    perform_db_snapshot: bool,

    // Flag indicating whether a db backup will be used as initial data source.
    use_db_backup_as_source: bool,

    // State of an RNG being used to inject entropy to share creation.
    shares_generator_rng_state: u64,

    // Size of batches when generating Iris shares for testing purposes.
    shares_generator_batch_size: usize,

    // Offset to apply when loading Iris shares.
    shares_generator_skip_offset: usize,

    // Batch size when persisting iris shares to pgres stores.
    shares_pgres_tx_batch_size: usize,
}

/// Accessors.
impl TestParams {
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn batch_size_error_rate(&self) -> usize {
        self.batch_size_error_rate
    }

    pub fn max_deletions(&self) -> usize {
        self.max_deletions
    }

    pub fn max_indexation_id(&self) -> IrisSerialId {
        self.max_indexation_id
    }

    pub fn max_modifications(&self) -> usize {
        self.max_modifications
    }

    pub fn node_config_idx(&self) -> usize {
        self.node_config_idx
    }

    pub fn node_config_kind(&self) -> &str {
        NODE_CONFIG_KIND_GENESIS
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

    pub fn shares_generator_skip_offset(&self) -> usize {
        self.shares_generator_skip_offset
    }
}

/// Builder for TestParams.
#[derive(Default)]
pub struct TestParamsBuilder {
    batch_size: usize,
    batch_size_error_rate: usize,
    max_deletions: usize,
    max_indexation_id: IrisSerialId,
    max_modifications: usize,
    node_config_idx: usize,
    perform_db_snapshot: bool,
    use_db_backup_as_source: bool,
    shares_generator_rng_state: u64,
    shares_generator_batch_size: usize,
    shares_generator_skip_offset: usize,
    shares_pgres_tx_batch_size: usize,
}

impl TestParamsBuilder {
    pub fn new() -> Self {
        Self {
            batch_size: 0,
            batch_size_error_rate: 256,
            max_deletions: 0,
            max_indexation_id: 1000,
            max_modifications: 0,
            node_config_idx: 0,
            perform_db_snapshot: false,
            use_db_backup_as_source: false,
            shares_generator_rng_state: 0,
            shares_generator_batch_size: 100,
            shares_generator_skip_offset: 0,
            shares_pgres_tx_batch_size: 100,
        }
    }
}

impl TestParamsBuilder {
    pub fn batch_size(mut self, value: usize) -> Self {
        self.batch_size = value;
        self
    }

    pub fn batch_size_error_rate(mut self, value: usize) -> Self {
        self.batch_size_error_rate = value;
        self
    }

    pub fn max_deletions(mut self, value: usize) -> Self {
        self.max_deletions = value;
        self
    }

    pub fn max_indexation_id(mut self, value: IrisSerialId) -> Self {
        self.max_indexation_id = value;
        self
    }

    pub fn max_modifications(mut self, value: usize) -> Self {
        self.max_modifications = value;
        self
    }

    pub fn perform_db_snapshot(mut self, value: bool) -> Self {
        self.perform_db_snapshot = value;
        self
    }

    pub fn use_db_backup_as_source(mut self, value: bool) -> Self {
        self.use_db_backup_as_source = value;
        self
    }

    pub fn node_config_idx(mut self, value: usize) -> Self {
        self.node_config_idx = value;
        self
    }

    pub fn shares_generator_batch_size(mut self, value: usize) -> Self {
        self.shares_generator_batch_size = value;
        self
    }

    pub fn shares_generator_rng_state(mut self, value: u64) -> Self {
        self.shares_generator_rng_state = value;
        self
    }

    pub fn shares_generator_skip_offset(mut self, value: usize) -> Self {
        self.shares_generator_skip_offset = value;
        self
    }

    pub fn shares_pgres_tx_batch_size(mut self, value: usize) -> Self {
        self.shares_pgres_tx_batch_size = value;
        self
    }
}

impl TestParamsBuilder {
    pub fn build(&self) -> TestParams {
        TestParams {
            batch_size: self.batch_size,
            batch_size_error_rate: self.batch_size_error_rate,
            max_deletions: self.max_deletions,
            max_indexation_id: self.max_indexation_id,
            max_modifications: self.max_modifications,
            node_config_idx: self.node_config_idx,
            perform_db_snapshot: self.perform_db_snapshot,
            use_db_backup_as_source: self.use_db_backup_as_source,
            shares_generator_batch_size: self.shares_generator_batch_size,
            shares_generator_rng_state: self.shares_generator_rng_state,
            shares_generator_skip_offset: self.shares_generator_skip_offset,
            shares_pgres_tx_batch_size: self.shares_pgres_tx_batch_size,
        }
    }
}
