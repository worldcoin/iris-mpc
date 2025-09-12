use super::constants::PARTY_COUNT;
use iris_mpc_common::{config::Config as NodeConfig, iris_db::iris::IrisCode};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;
use iris_mpc_store::StoredIrisRef;
use std::path::Path;

/// Set of node execution hosts.
#[derive(Debug, Clone, Copy)]
pub enum ExecutionHost {
    BareMetal,
    Docker,
}

impl Default for ExecutionHost {
    fn default() -> Self {
        match Path::new("/.dockerenv").exists() {
            true => ExecutionHost::Docker,
            _ => ExecutionHost::BareMetal,
        }
    }
}

// Pair of Iris shares associated with left/right eyes.
pub type GaloisRingSharedIrisPair = (GaloisRingSharedIris, GaloisRingSharedIris);

// Set of pairs of Iris shares associated with left/right eyes.
pub type GaloisRingSharedIrisPairSet = [GaloisRingSharedIrisPair; PARTY_COUNT];

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = (IrisCode, IrisCode);

// Network wide configuration set.
pub type NetConfig = [NodeConfig; PARTY_COUNT];

/// Type alias: Ordinal identifier of an MPC participant.
pub type PartyIdx = usize;

// Set of pairs of Iris shares associated with left/right eyes.
pub type StoredIrisRefSet<'a> = [StoredIrisRef<'a>; PARTY_COUNT];
