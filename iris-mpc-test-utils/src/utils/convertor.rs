use iris_mpc_common::iris_db::iris::IrisCodePair;
use iris_mpc_cpu::protocol::shared_iris::{GaloisRingSharedIris, GaloisRingSharedIrisPairSet};
use rand::{prelude::StdRng, SeedableRng};

/// Converts an RNG state plus a plaintext format Iris code pair to a 3 element vector of Galois Ring Iris shares.
///
/// # Arguments
///
/// * `rng_state` - State of an RNG being used to inject entropy to sahre creation.
/// * `code_pair` - Pair of Iris codes deserialized from a plaintext source.
///
/// # Returns
///
/// A 3 element array of Galois Ring Iris shares.
///
pub fn to_galois_ring_shares(
    rng_state: u64,
    code_pair: &IrisCodePair,
) -> GaloisRingSharedIrisPairSet {
    // Set RNG for each pair to match shares_encoding.rs behavior
    let mut shares_seed = StdRng::seed_from_u64(rng_state);

    // Set MPC participant specific Iris shares from Iris code + entropy.
    let (code_l, code_r) = code_pair;
    let shares_l =
        GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_l.to_owned());
    let shares_r =
        GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_r.to_owned());

    [
        (shares_l[0].to_owned(), shares_r[0].to_owned()),
        (shares_l[1].to_owned(), shares_r[1].to_owned()),
        (shares_l[2].to_owned(), shares_r[2].to_owned()),
    ]
}

#[cfg(test)]
mod tests {
    use super::super::constants::PARTY_COUNT;
    use super::{to_galois_ring_shares, IrisCodePair};

    const DEFAULT_RNG_STATE: u64 = 93;

    #[test]
    fn test_to_galois_ring_shares() {
        let converted = to_galois_ring_shares(DEFAULT_RNG_STATE, &IrisCodePair::default());
        assert!(converted.len() == PARTY_COUNT)
    }
}
