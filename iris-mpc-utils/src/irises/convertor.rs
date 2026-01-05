use rand::{prelude::StdRng, SeedableRng};

use iris_mpc_common::iris_db::iris::IrisCode;
use iris_mpc_cpu::protocol::shared_iris::GaloisRingSharedIris;

use crate::constants::N_PARTIES;

/// Converts a plaintext format Iris code pair to a boxed array of Galois Ring Iris shares.
pub fn to_galois_ring_shares(
    rng_state: u64,
    iris_code: &IrisCode,
) -> Box<[GaloisRingSharedIris; N_PARTIES]> {
    // Set RNG for each pair to match shares_encoding.rs behavior
    let mut shares_seed = StdRng::seed_from_u64(rng_state);

    // Set MPC party specific Iris shares from Iris code + entropy.
    let shares =
        GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, iris_code.to_owned());

    Box::new([
        shares[0].to_owned(),
        shares[1].to_owned(),
        shares[2].to_owned(),
    ])
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
