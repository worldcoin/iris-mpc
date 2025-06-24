use super::types::NodeExecutionInputs;
use iris_mpc_common::config::Config;
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs;
use std::path::PathBuf;

/// Returns set of Iris shares for indexation during a test run.
pub fn get_iris_shares_from_plaintext_file(_fpath: PathBuf) -> Vec<u32> {
    unimplemented!()
}

/// Returns set of Iris serial identifiers to be excluded from indexation at genesis during a test run.
pub fn get_genesis_exclusions() -> Vec<u32> {
    vec![]
}

/// Returns set of Iris modification to be applied at genesis during a test run.
pub fn get_genesis_modifications() -> Vec<u32> {
    vec![]
}

/// Returns inputs for launching an MPC node during a test run.
pub fn get_net_inputs() -> Vec<(ExecutionArgs, Config)> {
    unimplemented!()
}

/// Returns inputs for launching an MPC node during a test run.
pub fn get_node_inputs(party_id: usize) -> NodeExecutionInputs {
    fn get_args() -> ExecutionArgs {
        ExecutionArgs::new(500, 0, 256, false)
    }

    pub fn get_config(_party_id: usize) -> Config {
        unimplemented!()
    }

    NodeExecutionInputs::new(get_args(), get_config(party_id))
}
