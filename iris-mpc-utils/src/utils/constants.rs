/// Node config kinds.
#[allow(dead_code)]
pub const NODE_CONFIG_KIND_MAIN: &str = "main";
#[allow(dead_code)]
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";

/// MPC parties.
pub const PARTY_COUNT: usize = PARTY_IDX_SET.len();
pub const PARTY_IDX_0: usize = 0;
pub const PARTY_IDX_1: usize = 1;
pub const PARTY_IDX_2: usize = 2;
pub const PARTY_IDX_SET: [usize; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];
