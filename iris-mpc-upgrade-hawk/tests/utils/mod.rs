pub mod constants;
mod errors;
pub mod irises;
mod logger;
pub mod mpc_node;
pub mod resources;
pub mod runner;
pub mod s3_client;
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

// copied from genesis because genesis requires using an Aby3Store while the tests use a PlainTextStore

/// Domain for persistent state store entry for last indexed id
pub const STATE_DOMAIN: &str = "genesis";

/// Key for persistent state store entry for last indexed iris id
pub const STATE_KEY_LAST_INDEXED_IRIS_ID: &str = "last_indexed_iris_id";

/// Key for persistent state store entry for last indexed modification id
pub const STATE_KEY_LAST_INDEXED_MODIFICATION_ID: &str = "last_indexed_modification_id";
