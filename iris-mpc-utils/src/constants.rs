/// Node config kinds.
pub const NODE_CONFIG_KIND: [&str; 2] = [NODE_CONFIG_KIND_MAIN, NODE_CONFIG_KIND_GENESIS];
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";
pub const NODE_CONFIG_KIND_MAIN: &str = "main";

/// MPC parties.
pub const PARTY_IDX: [usize; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];
pub const PARTY_COUNT: usize = PARTY_IDX.len();
pub const PARTY_IDX_0: usize = 0;
pub const PARTY_IDX_1: usize = 1;
pub const PARTY_IDX_2: usize = 2;
