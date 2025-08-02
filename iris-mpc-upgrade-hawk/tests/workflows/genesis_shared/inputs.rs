use super::{net::NetArgs, params::TestParams};
use iris_mpc_common::{
    config::{Config as NodeConfig, NetConfig},
    IrisSerialId, PartyIdx,
};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIrisPairSet;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;
use itertools::{IntoChunks, Itertools};

/// Excapsulates data used to drive a test run.
#[derive(Debug, Clone)]
pub struct TestInputs {
    // Arguments for each network node.
    net_args: NetArgs,

    // Configuration for each network node.
    net_config: NetConfig,

    // Data used to initialise system state prior to a test run.
    #[allow(dead_code)]
    system_state_inputs: SystemStateInputs,
}

/// Constructor.
impl TestInputs {
    pub fn new(
        net_args: NetArgs,
        net_config: NetConfig,
        system_state_inputs: SystemStateInputs,
    ) -> Self {
        Self {
            net_args,
            net_config,
            system_state_inputs,
        }
    }
}

/// Accessors.
impl TestInputs {
    pub fn net_args(&self) -> &NetArgs {
        &self.net_args
    }

    pub fn net_config(&self) -> &NetConfig {
        &self.net_config
    }

    pub fn node_args(&self, node_idx: PartyIdx) -> &NodeArgs {
        &self.net_args[node_idx]
    }

    pub fn node_config(&self, node_idx: PartyIdx) -> &NodeConfig {
        &self.net_config[node_idx]
    }

    #[allow(dead_code)]
    pub fn system_state_inputs(&self) -> &SystemStateInputs {
        &self.system_state_inputs
    }
}

/// Inputs required to initialise system state prior to a test run.
#[derive(Debug, Clone)]
pub struct SystemStateInputs {
    // Set of serial identifiers associated with deleted Iris's.
    #[allow(dead_code)]
    deletions_identifiers: Vec<IrisSerialId>,

    // Set of modifications identifiers associated with modified Iris's.
    #[allow(dead_code)]
    modification_identifiers: Vec<i64>,

    // Test parameters.
    params: TestParams,
}

/// Constructor.
impl SystemStateInputs {
    #[allow(dead_code)]
    pub fn new(params: TestParams, deletions: Vec<IrisSerialId>, modifications: Vec<i64>) -> Self {
        Self {
            deletions_identifiers: deletions,
            modification_identifiers: modifications,
            params,
        }
    }
}

/// Accessors.
impl SystemStateInputs {
    #[allow(dead_code)]
    pub fn iris_deletions(&self) -> &Vec<IrisSerialId> {
        &self.deletions_identifiers
    }
}

/// Methods.
impl SystemStateInputs {
    pub fn iris_shares_stream(
        &self,
    ) -> IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>> {
        std::iter::empty().chunks(self.params.batch_size())
    }
}
