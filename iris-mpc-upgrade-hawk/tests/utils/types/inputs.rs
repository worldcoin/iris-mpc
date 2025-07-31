use crate::utils::constants::COUNT_OF_PARTIES;
use iris_mpc_common::{config::Config as NodeConfig, IrisSerialId};
use iris_mpc_upgrade_hawk::genesis::ExecutionArgs as NodeArgs;

#[macro_export]
macro_rules! make_node_configs {
    ( $func:expr) => {{
        use crate::utils::constants::COUNT_OF_PARTIES;
        use iris_mpc_common::config::Config as NodeConfig;
        let arr: [NodeConfig; COUNT_OF_PARTIES] = std::array::from_fn(|i| $func(i));
        arr
    }};
}

#[derive(Debug, Clone)]
pub struct TestInputs {
    // Data used to launch each node process during a test run.
    pub net_inputs: NetInputs,

    // Data used to initialise system state prior to a test run.
    #[allow(dead_code)]
    pub system_state_inputs: Option<SystemStateInputs>,
}

/// Constructor.
impl TestInputs {
    pub fn new(net_inputs: NetInputs) -> Self {
        Self {
            net_inputs,
            system_state_inputs: None,
        }
    }

    pub fn with_system_state(mut self, sys_inputs: SystemStateInputs) -> Self {
        self.system_state_inputs.replace(sys_inputs);
        self
    }
}

/// Accessors.
impl TestInputs {
    pub fn net_inputs(&self) -> &NetInputs {
        &self.net_inputs
    }

    #[allow(dead_code)]
    pub fn system_state_inputs(&self) -> &Option<SystemStateInputs> {
        &self.system_state_inputs
    }
}

/// Inputs required to run a network.
#[derive(Debug, Clone)]
pub struct NetInputs {
    /// Node input arguments.
    node_process_inputs: [NodeProcessInputs; COUNT_OF_PARTIES],
}

/// Constructor.
impl NetInputs {
    // args are always the same for all parties. the configs differ through.
    pub fn new(args: NodeArgs, configs: [NodeConfig; COUNT_OF_PARTIES]) -> Self {
        let node_process_inputs =
            configs.map(|config| NodeProcessInputs::new(args.clone(), config));
        Self {
            node_process_inputs,
        }
    }
}

/// Accessors.
impl NetInputs {
    pub fn node_process_inputs(&self) -> &[NodeProcessInputs; COUNT_OF_PARTIES] {
        &self.node_process_inputs
    }
}

/// Inputs required to run a node.
#[derive(Debug, Clone)]
pub struct NodeProcessInputs {
    /// Node input arguments.
    args: NodeArgs,

    /// Node input configuration.
    config: NodeConfig,
}

/// Constructor.
impl NodeProcessInputs {
    pub fn new(args: NodeArgs, config: NodeConfig) -> Self {
        Self { args, config }
    }
}

/// Accessors.
impl NodeProcessInputs {
    pub fn args(&self) -> &NodeArgs {
        &self.args
    }

    pub fn config(&self) -> &NodeConfig {
        &self.config
    }
}

/// Inputs required to initialise system state prior to a test run.
#[derive(Debug, Clone)]
pub struct SystemStateInputs {
    // Serial identifiers of deleted Iris's.
    #[allow(dead_code)]
    iris_deletions: Vec<IrisSerialId>,

    // Set of Iris shares to be processed.
    #[allow(dead_code)]
    iris_shares: Vec<IrisSerialId>,
}

/// Constructor.
impl SystemStateInputs {
    #[allow(dead_code)]
    pub fn new(iris_deletions: Vec<IrisSerialId>, iris_shares: Vec<IrisSerialId>) -> Self {
        Self {
            iris_deletions,
            iris_shares,
        }
    }
}

/// Accessors.
impl SystemStateInputs {
    #[allow(dead_code)]
    pub fn iris_deletions(&self) -> &Vec<IrisSerialId> {
        &self.iris_deletions
    }

    #[allow(dead_code)]
    pub fn iris_shares(&self) -> &Vec<IrisSerialId> {
        &self.iris_shares
    }
}
