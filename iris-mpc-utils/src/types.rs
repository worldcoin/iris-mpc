use super::constants::N_PARTIES;
use iris_mpc_common::{
    config::Config as NodeConfig,
    iris_db::iris::{IrisCode, IrisCodeArray},
};
use iris_mpc_cpu::execution::hawk_main::BothEyes;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;
use serde::{Deserialize, Serialize};
use std::path::Path;

// Pair of Iris shares associated with left/right eyes.
pub type GaloisRingSharedIrisPair = BothEyes<GaloisRingSharedIris>;

// Set of pairs of Iris shares associated with left/right eyes.
pub type GaloisRingSharedIrisPairSet = [GaloisRingSharedIrisPair; N_PARTIES];

/// Iris code representation using base64 encoding compatible with Open IRIS
#[derive(Serialize, Deserialize)]
pub struct IrisCodeBase64 {
    pub iris_codes: String,
    pub mask_codes: String,
}

impl From<&IrisCode> for IrisCodeBase64 {
    fn from(value: &IrisCode) -> Self {
        Self {
            iris_codes: value.code.to_base64().unwrap(),
            mask_codes: value.mask.to_base64().unwrap(),
        }
    }
}

impl From<&IrisCodeBase64> for IrisCode {
    fn from(value: &IrisCodeBase64) -> Self {
        Self {
            code: IrisCodeArray::from_base64(&value.iris_codes).unwrap(),
            mask: IrisCodeArray::from_base64(&value.mask_codes).unwrap(),
        }
    }
}

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
