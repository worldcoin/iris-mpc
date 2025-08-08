use crate::utils::types::StoredIrisRefSet;
use iris_mpc_common::{iris_db::iris::IrisCodePair, IrisSerialId};
use iris_mpc_cpu::protocol::shared_iris::{
    GaloisRingSharedIris, GaloisRingSharedIrisPair, GaloisRingSharedIrisPairSet,
};
use iris_mpc_store::StoredIrisRef;
use rand::{prelude::StdRng, SeedableRng};

/// Converts an RNG state plus a plaintext format Iris code pair to a boxed 3 element array of Galois Ring Iris shares.
///
/// # Arguments
///
/// * `rng_state` - State of an RNG being used to inject entropy to sahre creation.
/// * `code_pair` - Pair of Iris codes deserialized from a plaintext source.
///
/// # Returns
///
/// A boxed 3 element array array of Galois Ring Iris shares.
///
pub fn to_galois_ring_share_pair_set(
    rng_state: u64,
    code_pair: &IrisCodePair,
) -> Box<GaloisRingSharedIrisPairSet> {
    // Set RNG for each pair to match shares_encoding.rs behavior
    let mut shares_seed = StdRng::seed_from_u64(rng_state);

    // Set MPC participant specific Iris shares from Iris code + entropy.
    let (code_l, code_r) = code_pair;
    let shares_l =
        GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_l.to_owned());
    let shares_r =
        GaloisRingSharedIris::generate_shares_locally(&mut shares_seed, code_r.to_owned());

    Box::new([
        (shares_l[0].to_owned(), shares_r[0].to_owned()),
        (shares_l[1].to_owned(), shares_r[1].to_owned()),
        (shares_l[2].to_owned(), shares_r[2].to_owned()),
    ])
}

/// Converts a pair of Iris shares to a stored Iris reference (in readiness for upload to a dB).
///
/// # Arguments
///
/// * `serial_id` - Iris stored reference serial ID.
/// * `share_pair` - Pair of Iris codes deserialized from a plaintext source.
///
/// # Returns
///
/// A stored Iris reference (in readiness for upload to a dB)
///
#[allow(dead_code)]
pub fn to_stored_iris_ref(
    serial_id: IrisSerialId,
    share_pair: &GaloisRingSharedIrisPair,
) -> StoredIrisRef<'_> {
    let (iris_l, iris_r) = share_pair;

    StoredIrisRef {
        id: serial_id as i64,
        left_code: &iris_l.code.coefs,
        left_mask: &iris_l.mask.coefs,
        right_code: &iris_r.code.coefs,
        right_mask: &iris_r.mask.coefs,
    }
}

/// Converts a set of Iris share pairs to a set of Iris stored references in readiness for upload to a dB.
///
/// # Arguments
///
/// * `serial_id` - Iris stored reference serial ID.
/// * `share_pair_set` - Iris code pair deserialized from a plaintext source.
///
/// # Returns
///
/// A 3 element array of stored Iris references.
///
#[allow(dead_code)]
pub fn to_stored_iris_ref_set(
    serial_id: IrisSerialId,
    share_pair_set: &GaloisRingSharedIrisPairSet,
) -> Box<StoredIrisRefSet<'_>> {
    Box::new([
        to_stored_iris_ref(serial_id, &share_pair_set[0]),
        to_stored_iris_ref(serial_id, &share_pair_set[1]),
        to_stored_iris_ref(serial_id, &share_pair_set[2]),
    ])
}

#[cfg(test)]
mod tests {
    use super::super::constants::PARTY_COUNT;
    use super::{
        to_galois_ring_share_pair_set, to_stored_iris_ref, to_stored_iris_ref_set,
        GaloisRingSharedIrisPairSet, IrisCodePair,
    };

    const DEFAULT_RNG_STATE: u64 = 93;

    fn get_iris_shares() -> Box<GaloisRingSharedIrisPairSet> {
        to_galois_ring_share_pair_set(DEFAULT_RNG_STATE, &IrisCodePair::default())
    }

    #[test]
    fn test_to_galois_ring_shares() {
        assert!(get_iris_shares().len() == PARTY_COUNT)
    }

    #[test]
    fn test_to_store_iris_ref() {
        let shares = get_iris_shares();
        let shar_pair = &shares[0];
        to_stored_iris_ref(1, shar_pair);
    }

    #[test]
    fn test_to_store_iris_ref_set() {
        let shares = get_iris_shares();
        assert!(to_stored_iris_ref_set(1, &shares).len() == PARTY_COUNT)
    }
}
