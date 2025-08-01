use super::constants::COUNT_OF_PARTIES;
use iris_mpc_common::{config::Config, iris_db::iris::IrisCode};
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = (IrisCode, IrisCode);

// Pair of Iris shares aassociated with left/right eyes.
pub type GaloisRingSharedIrisPair = (GaloisRingSharedIris, GaloisRingSharedIris);

// Network wide configuration set.
pub type HawkConfigs = [Config; COUNT_OF_PARTIES];
