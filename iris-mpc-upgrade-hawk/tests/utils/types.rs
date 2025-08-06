use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = (IrisCode, IrisCode);

// Network wide configuration set.
pub type NetConfig = [NodeConfig; 3];

//. Enum over set of node types.
#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
pub enum NodeType {
    CPU,
    GPU,
}

/// Type alias: Ordinal identifier of an MPC participant.
pub type PartyIdx = usize;
