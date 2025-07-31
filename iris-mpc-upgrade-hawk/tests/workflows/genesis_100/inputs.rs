use super::params::Params;
use crate::utils::{constants::COUNT_OF_PARTIES, GaloisRingSharedIrisPair, NetConfig};
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;
use itertools::{IntoChunks, Itertools};

/// Excapsulates data used to drive a test run.
#[derive(Debug, Clone)]
pub(super) struct Inputs {
    // Network configuration.
    config: NetConfig,

    // Data used to launch each node process during a test run.
    net_inputs: NetInputs,

    // Data used to initialise system state prior to a test run.
    #[allow(dead_code)]
    system_state_inputs: SystemStateInputs,
}

/// Constructor.
impl Inputs {
    pub fn new(
        net_config: NetConfig,
        net_inputs: NetInputs,
        system_state_inputs: SystemStateInputs,
    ) -> Self {
        Self {
            net_config,
            net_inputs,
            system_state_inputs,
        }
    }
}

/// Accessors.
impl Inputs {
    pub fn config(&self) -> &NetConfig {
        &self.config
    }

    pub fn net_inputs(&self) -> &NetInputs {
        &self.net_inputs
    }

    #[allow(dead_code)]
    pub fn system_state_inputs(&self) -> &SystemStateInputs {
        &self.system_state_inputs
    }
}

/// Inputs required to run a network.
#[derive(Debug, Clone)]
pub(super) struct NetInputs {
    /// Node input arguments.
    node_inputs: [NodeInputs; COUNT_OF_PARTIES],
}

/// Constructor.
impl NetInputs {
    pub fn new(node_inputs: [NodeInputs; COUNT_OF_PARTIES]) -> Self {
        Self { node_inputs }
    }
}

/// Accessors.
impl NetInputs {
    pub fn node_inputs(&self) -> &[NodeInputs; COUNT_OF_PARTIES] {
        &self.node_inputs
    }

    pub fn get_node_inputs(&self, node_idx: usize) -> &NodeInputs {
        &self.node_inputs[node_idx]
    }
}

/// Inputs required to run a node.
#[derive(Debug, Clone)]
pub(super) struct NodeInputs {
    /// Node input arguments.
    args: NodeArgs,

    /// Node input configuration.
    config: NodeConfig,
}

/// Constructor.
impl NodeInputs {
    pub fn new(args: NodeArgs, config: NodeConfig) -> Self {
        Self { args, config }
    }
}

/// Accessors.
impl NodeInputs {
    pub fn args(&self) -> &NodeArgs {
        &self.args
    }

    pub fn config(&self) -> &NodeConfig {
        &self.config
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
