use super::types::PartyIdx;

// Count of MPC protocol parties.
pub const PARTY_COUNT: usize = 3;

/// MPC party ordinal identifiers.
pub const PARTY_IDX_0: PartyIdx = 0;
pub const PARTY_IDX_1: PartyIdx = 1;
pub const PARTY_IDX_2: PartyIdx = 2;

/// MPC party ordinal identifier set.
pub const PARTY_IDX_SET: [PartyIdx; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];

/// Default size of Iris share batches used during a test.
pub const SHARES_GENERATOR_BATCH_SIZE: usize = 100;

/// Default state of an RNG being used to inject entropy to Iris shares creation.
pub const SHARES_GENERATOR_RNG_STATE: u64 = 93;

/// Default size of a PostgreSQL transaction when persisting Iris shares.
pub const SHARES_GENERATOR_PGRES_TX_BATCH_SIZE: usize = 100;
