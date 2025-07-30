/// Number of participating MPC parties.
pub const COUNT_OF_PARTIES: usize = 3;

/// Ordinal identifiers of MPC parties.
pub const PARTY_IDX_0: usize = 0;
pub const PARTY_IDX_1: usize = 1;
pub const PARTY_IDX_2: usize = 2;

/// Number of secret-shared iris code pairs to persist to Postgres per transaction.
pub const SECRET_SHARING_PG_TX_SIZE: usize = 100;
