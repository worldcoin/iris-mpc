use rand::{CryptoRng, Rng};

use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

use crate::constants::N_PARTIES;

/// Converts a plaintext format Iris code pair to a boxed array of Galois Ring Iris shares.
pub fn to_galois_ring_shares<R: Rng + CryptoRng>(
    rng: &mut R,
    iris_code: IrisCode,
) -> [GaloisRingSharedIris; N_PARTIES] {
    GaloisRingSharedIris::generate_shares_locally(rng, iris_code)
    // let shares = GaloisRingSharedIris::generate_shares_locally(rng, iris_code.to_owned());

    // Box::new([
    //     shares[0].to_owned(),
    //     shares[1].to_owned(),
    //     shares[2].to_owned(),
    // ])
}

#[cfg(test)]
mod tests {
    // use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

    // use super::{to_galois_ring_share_pair_set, IrisCode};
    // use crate::constants::N_PARTIES;

    // const DEFAULT_RNG_STATE: u64 = 42;

    // fn get_iris_shares() -> Box<[(GaloisRingSharedIris, GaloisRingSharedIris); N_PARTIES]> {
    //     to_galois_ring_share_pair_set(
    //         DEFAULT_RNG_STATE,
    //         &(IrisCode::default(), IrisCode::default()),
    //     )
    // }

    // #[test]
    // fn test_to_galois_ring_shares() {
    //     assert!(get_iris_shares().len() == N_PARTIES)
    // }
}
