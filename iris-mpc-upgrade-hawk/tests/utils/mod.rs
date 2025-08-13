pub mod constants;
mod errors;
pub mod irises;
mod logger;
pub mod modifications;
pub mod mpc_node;
pub mod plaintext_genesis;
pub mod resources;
pub mod runner;
pub mod s3_deletions;

pub use errors::TestError;
use iris_mpc_common::{config::Config, iris_db::iris::IrisCode};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;
pub use runner::{TestRun, TestRunContextInfo, TestRunEnvironment};

use crate::utils::constants::COUNT_OF_PARTIES;

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = (IrisCode, IrisCode);

// Pair of Iris shares aassociated with left/right eyes.
pub type GaloisRingSharedIrisPair = (GaloisRingSharedIris, GaloisRingSharedIris);

// Network wide configuration set.
pub type HawkConfigs = [Config; COUNT_OF_PARTIES];
