use crate::utils::{
    types::inputs::{NetInputs, TestInputs},
    TestRunContextInfo,
};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Trait for generating test inputs using the algorithm pattern.
pub trait TestInputFactory {
    /// Returns test run inputs.
    fn get_test_inputs(&self, ctx: &TestRunContextInfo) -> TestInputs {
        let net_inputs = self.get_net_inputs(ctx);
        // TODO: load system state inputs.
        TestInputs::new(net_inputs, None)
    }

    /// Returns network process inputs for usage during a test run.
    fn get_net_inputs(&self, ctx: &TestRunContextInfo) -> NetInputs {
        NetInputs::new([
            self.get_node_process_inputs(ctx, 0),
            self.get_node_process_inputs(ctx, 1),
            self.get_node_process_inputs(ctx, 2),
        ])
    }

    /// Returns node process inputs for usage during a test run.
    fn get_node_process_inputs(
        &self,
        ctx: &TestRunContextInfo,
        party_id: usize,
    ) -> NodeProcessInputs {
        let args = self.get_args();
        let config = self.get_config(ctx, party_id);
        NodeProcessInputs::new(args, config)
    }

    /// Returns node CLI args.
    fn get_args(&self) -> NodeArgs;

    /// Returns node configuration.
    fn get_config(&self, ctx: &TestRunContextInfo, party_id: usize) -> NodeConfig;
}
