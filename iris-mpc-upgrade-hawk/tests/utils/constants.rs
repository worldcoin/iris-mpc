use iris_mpc_common::IrisSerialId;

/// Number of participating MPC parties.
pub const COUNT_OF_PARTIES: usize = 3;

/// Default batch size.
pub const DEFAULT_BATCH_SIZE: usize = 0;

/// Default batch size error rate.
pub const DEFAULT_BATCH_SIZE_ERROR_RATE: usize = 256;

/// Default maximum indexation ID.
pub const DEFAULT_MAX_INDEXATION_ID: IrisSerialId = 100;

/// Default snapshot strategy.
pub const DEFAULT_SNAPSHOT_STRATEGY: bool = false;
