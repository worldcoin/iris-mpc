use super::types::{NetInputs, NodeProcessInputs, TestInputs};
use crate::utils::{constants, factory::TestInputFactory, resources, TestRunContextInfo};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Default implementation of TestInputFactory.
pub struct DefaultTestInputFactory;

impl TestInputFactory for DefaultTestInputFactory {
    fn get_args(&self) -> NodeArgs {
        NodeArgs::new(
            constants::DEFAULT_BATCH_SIZE,
            constants::DEFAULT_BATCH_SIZE_ERROR_RATE,
            100,
            constants::DEFAULT_SNAPSHOT_STRATEGY,
            constants::DEFAULT_BACKUP_AS_SOURCE_STRATEGY,
        )
    }

    fn get_config(&self, ctx: &TestRunContextInfo, party_id: usize) -> NodeConfig {
        resources::read_node_config(ctx, format!("node-{}-genesis-0", party_id)).unwrap()
    }
}
