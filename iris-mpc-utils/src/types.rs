use std::path::Path;

use sodiumoxide::crypto::box_::PublicKey;

use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::execution::hawk_main::BothEyes;

use super::{
    constants::N_PARTIES,
    state::aws::{NodeAwsClient as NodeServiceClients, NodeAwsConfig},
};

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = BothEyes<IrisCode>;

// Network wide node configuration set.
pub type NetNodeConfig = [NodeConfig; N_PARTIES];

// Network wide configuration set.
pub type NetServiceConfig = [NodeAwsConfig; N_PARTIES];

// Network wide node AWS service clients.
pub type NetServiceClients = [NodeServiceClients; N_PARTIES];

// Network wide node encryption public keys.
pub type NetEncryptionPublicKeys = [PublicKey; N_PARTIES];

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
