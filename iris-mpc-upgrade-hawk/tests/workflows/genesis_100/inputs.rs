use super::params::Params;
use crate::utils::{constants::COUNT_OF_PARTIES, GaloisRingSharedIrisPair, NetConfig};
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;
use itertools::{IntoChunks, Itertools};

// Network wide argument set.
pub type NetArgs = [NodeArgs; COUNT_OF_PARTIES];

/// Excapsulates data used to drive a test run.
#[derive(Debug, Clone)]
pub(super) struct Inputs {
    // Configuration for each node in network.
    args: NetArgs,

    // Configuration for each node in network.
    config: NetConfig,

    // Data used to initialise system state prior to a test run.
    #[allow(dead_code)]
    system_state: SystemStateInputs,
}

/// Constructor.
impl Inputs {
    pub fn new(args: NetArgs, config: NetConfig, system_state_inputs: SystemStateInputs) -> Self {
        Self {
            args,
            config,
            system_state: system_state_inputs,
        }
    }
}

/// Accessors.
impl Inputs {
    pub fn args(&self) -> &NetArgs {
        &self.args
    }

    pub fn args_of_node(&self, node_idx: usize) -> &NodeArgs {
        &self.args[node_idx]
    }

    pub fn config(&self) -> &NetConfig {
        &self.config
    }

    pub fn config_of_node(&self, node_idx: usize) -> &NodeConfig {
        &self.config[node_idx]
    }

    #[allow(dead_code)]
    pub fn system_state_inputs(&self) -> &SystemStateInputs {
        &self.system_state
    }
}

/// Inputs required to initialise system state prior to a test run.
#[derive(Debug, Clone)]
pub(super) struct SystemStateInputs {
    // Serial identifiers of deleted Iris's.
    #[allow(dead_code)]
    iris_deletions: Vec<IrisSerialId>,

    // Test parameters.
    params: Params,
}

/// Constructor.
impl SystemStateInputs {
    #[allow(dead_code)]
    pub fn new(params: Params, iris_deletions: Vec<IrisSerialId>) -> Self {
        Self {
            iris_deletions,
            params,
        }
    }
}

/// Accessors.
impl SystemStateInputs {
    #[allow(dead_code)]
    pub fn iris_deletions(&self) -> &Vec<IrisSerialId> {
        &self.iris_deletions
    }
}

/// Methods.
impl SystemStateInputs {
    pub fn iris_shares_stream(
        &self,
    ) -> IntoChunks<impl Iterator<Item = Box<[GaloisRingSharedIrisPair; COUNT_OF_PARTIES]>>> {
        std::iter::empty().chunks(self.params.batch_size())
    }
}
