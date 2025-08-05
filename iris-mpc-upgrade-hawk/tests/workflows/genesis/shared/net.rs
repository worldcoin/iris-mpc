use crate::utils::constants::PARTY_COUNT;
use eyre::{Report, Result};
use iris_mpc_upgrade_hawk::genesis::{
    ExecutionArgs as NodeArgs, ExecutionResult as NodeExecutionResult,
};

// Network wide argument set.
pub type NetArgs = [NodeArgs; PARTY_COUNT];

/// Helper type over set of node execution results.
/// TODO: convert to array -> [Result<NodeExecutionResult, Report>; 3].
pub type NetExecutionResult = Vec<Result<NodeExecutionResult, Report>>;
