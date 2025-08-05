use super::types::PartyIdx;

/// Count of MPC protocol parties.
#[cfg(test)]
pub const PARTY_COUNT: usize = 3;

/// MPC party ordinal identifiers.
pub const PARTY_IDX_0: PartyIdx = 0;
pub const PARTY_IDX_1: PartyIdx = 1;
pub const PARTY_IDX_2: PartyIdx = 2;

/// MPC party ordinal identifier set.
pub const PARTY_IDX_SET: [PartyIdx; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];
