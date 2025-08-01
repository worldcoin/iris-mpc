/// Default size of Iris share batches used during a test.
pub const SHARES_GENERATOR_BATCH_SIZE: usize = 100;

/// Default state of an RNG being used to inject entropy to Iris shares creation.
pub const SHARES_GENERATOR_RNG_STATE: u64 = 93;

/// Default size of a PostgreSQL transaction when persisting Iris shares.
pub const SHARES_GENERATOR_PGRES_TX_BATCH_SIZE: usize = 100;
