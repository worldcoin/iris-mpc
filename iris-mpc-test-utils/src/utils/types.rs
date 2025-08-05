use iris_mpc_common::config::Config as NodeConfig;
use std::path::Path;

/// Enumeration over set of test run execution environments.
#[derive(Debug, Clone, Copy)]
pub enum ExecutionEnvironment {
    Local,
    Docker,
}

/// Constructor.
impl ExecutionEnvironment {
    pub fn new() -> Self {
        if Path::new("/.dockerenv").exists() {
            ExecutionEnvironment::Docker
        } else {
            ExecutionEnvironment::Local
        }
    }
}

// Network wide configuration set.
pub type NetConfig = [NodeConfig; 3];

/// Type alias: Ordinal identifier of an MPC participant.
pub type PartyIdx = usize;
