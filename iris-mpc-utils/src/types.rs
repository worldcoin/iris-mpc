use super::constants::N_PARTIES;
use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::execution::hawk_main::BothEyes;
use std::path::Path;

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = BothEyes<IrisCode>;

// Network wide configuration set.
pub type NetConfig = [NodeConfig; N_PARTIES];

/// Set of node execution hosts.
#[derive(Debug, Clone, Copy)]
pub enum NodeExecutionHost {
    BareMetal,
    Docker,
}

impl Default for NodeExecutionHost {
    fn default() -> Self {
        match Path::new("/.dockerenv").exists() {
            true => NodeExecutionHost::Docker,
            _ => NodeExecutionHost::BareMetal,
        }
    }
}

/// Type alias: Ordinal identifier of an MPC participant.
pub type PartyIdx = usize;
