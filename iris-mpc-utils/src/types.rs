use std::path::Path;

use sodiumoxide::crypto::box_::PublicKey;

use iris_mpc_common::{
    config::Config as NodeConfig, galois_engine::degree4::GaloisRingIrisCodeShare,
};

use super::constants::N_PARTIES;

// Iris code and mask galois shares.
pub type IrisCodeAndMaskShares = [[GaloisRingIrisCodeShare; N_PARTIES]; 2];

// Network wide node configuration set.
pub type NodeConfigSet = [NodeConfig; N_PARTIES];

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

// MPC party public keys (used for encryption).
pub type PublicKeyset = [PublicKey; N_PARTIES];
