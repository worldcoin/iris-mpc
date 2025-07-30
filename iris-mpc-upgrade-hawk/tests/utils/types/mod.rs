pub mod inputs;
pub mod resources;
pub mod test;

use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

// Pair of Iris codes aassociated with left/right eyes.
pub type IrisCodePair = (IrisCode, IrisCode);

// Pair of Iris shares aassociated with left/right eyes.
pub type GaloisRingSharedIrisPair = (GaloisRingSharedIris, GaloisRingSharedIris);
