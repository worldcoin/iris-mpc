use super::types::{NetInputs, NodeProcessInputs, TestInputs};
use crate::utils::{constants, resources, TestRunContextInfo};
use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

/// Returns test run inputs.
pub(super) fn get_test_inputs(ctx: &TestRunContextInfo) -> TestInputs {
    /// Returns network process inputs for usage during a test run.
    fn get_net_inputs(ctx: &TestRunContextInfo) -> NetInputs {
        /// Returns node process inputs for usage during a test run.
        fn get_node_process_inputs(ctx: &TestRunContextInfo, party_id: usize) -> NodeProcessInputs {
            /// Returns node CLI args.
            fn get_args() -> NodeArgs {
                NodeArgs::new(
                    constants::DEFAULT_BATCH_SIZE,
                    constants::DEFAULT_BATCH_SIZE_ERROR_RATE,
                    100,
                    constants::DEFAULT_SNAPSHOT_STRATEGY,
                    constants::DEFAULT_BACKUP_AS_SOURCE_STRATEGY,
                )
            }

            /// Returns node configuration.
            fn get_config(ctx: &TestRunContextInfo, party_id: usize) -> NodeConfig {
                resources::read_node_config(ctx, format!("node-{}-genesis-0", party_id)).unwrap()
            }

            NodeProcessInputs::new(get_args(), get_config(ctx, party_id))
        }

        NetInputs::new([
            get_node_process_inputs(ctx, 0),
            get_node_process_inputs(ctx, 1),
            get_node_process_inputs(ctx, 2),
        ])
    }

    // TODO: load system state inputs.

    TestInputs::new(get_net_inputs(ctx), None)
}
