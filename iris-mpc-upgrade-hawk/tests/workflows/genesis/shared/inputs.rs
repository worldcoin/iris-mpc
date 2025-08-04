use super::{net::NetArgs, params::TestParams};
use crate::resources;
use iris_mpc_common::{
    config::{Config as NodeConfig, NetConfig},
    IrisSerialId, PartyIdx,
};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIrisPairSet;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;
use itertools::IntoChunks;

/// Excapsulates data used to drive a test run.
#[derive(Debug, Clone)]
pub struct TestInputs {
    // Set of serial identifiers associated with deleted Iris's.
    iris_deletions: Vec<IrisSerialId>,

    // Set of modifications identifiers associated with modified Iris's.
    #[allow(dead_code)]
    iris_modifications: Vec<i64>,

    // Arguments for each network node.
    net_args: NetArgs,

    // Configuration for each network node.
    net_config: NetConfig,

    // Test parameters.
    #[allow(dead_code)]
    params: TestParams,
}

/// Constructor.
impl TestInputs {
    pub fn new(
        params: TestParams,
        net_args: NetArgs,
        net_config: NetConfig,
        iris_deletions: Vec<IrisSerialId>,
        iris_modifications: Vec<i64>,
    ) -> Self {
        Self {
            params,
            iris_deletions,
            iris_modifications,
            net_args,
            net_config,
        }
    }
}

/// Accessors.
impl TestInputs {
    pub fn iris_deletions(&self) -> &Vec<IrisSerialId> {
        &self.iris_deletions
    }

    pub fn iris_modifications(&self) -> &Vec<i64> {
        &self.iris_modifications
    }

    #[allow(dead_code)]
    pub fn iris_shares_stream(
        &self,
    ) -> IntoChunks<impl Iterator<Item = Box<GaloisRingSharedIrisPairSet>>> {
        resources::read_iris_shares_batch(
            self.params().shares_generator_batch_size(),
            self.params().max_indexation_id() as usize,
            self.params().shares_generator_rng_state(),
            self.params().shares_generator_skip_offset(),
        )
        .unwrap()
    }

    #[allow(dead_code)]
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

    pub fn params(&self) -> &TestParams {
        &self.params
    }
}
