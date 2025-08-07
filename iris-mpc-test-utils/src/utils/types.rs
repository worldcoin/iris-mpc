use iris_mpc_common::config::Config as NodeConfig;
use iris_mpc_store::StoredIrisRef;
use std::path::Path;

/// Enumeration over set of test run execution environments.
#[derive(Debug, Clone, Copy)]
pub enum ExecutionEnvironment {
    Local,
    Docker,
}

/// Trait: Default.
impl Default for ExecutionEnvironment {
    fn default() -> Self {
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

// Set of pairs of Iris shares associated with left/right eyes.
pub type StoredIrisRefSet<'a> = [StoredIrisRef<'a>; 3];
