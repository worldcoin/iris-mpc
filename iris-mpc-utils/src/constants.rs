/// Test graph sizes.
pub const GRAPH_SIZE_RANGE: [usize; 8] = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000];

/// Node config kinds.
pub const NODE_CONFIG_KIND: [&str; 2] = [NODE_CONFIG_KIND_MAIN, NODE_CONFIG_KIND_GENESIS];
pub const NODE_CONFIG_KIND_GENESIS: &str = "genesis";
pub const NODE_CONFIG_KIND_MAIN: &str = "main";

/// MPC parties.
pub const N_PARTIES: usize = PARTY_INDICES.len();
pub const PARTY_INDICES: [usize; 3] = [PARTY_IDX_0, PARTY_IDX_1, PARTY_IDX_2];
pub const PARTY_IDX_0: usize = 0;
pub const PARTY_IDX_1: usize = 1;
pub const PARTY_IDX_2: usize = 2;
