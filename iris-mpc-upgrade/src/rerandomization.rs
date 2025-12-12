use std::io::Read;

use iris_mpc_common::{
    galois::degree4::{basis::Monomial, GaloisRingElement},
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
};
use iris_mpc_store::DbStoredIris;

pub fn randomize_iris(
    iris: DbStoredIris,
    master_seed: &[u8],
    party_id: usize,
) -> (
    i64,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
) {
    let (left_code, left_mask, right_code, right_mask) = (
        GaloisRingIrisCodeShare {
            id: party_id + 1,
            coefs: iris.left_code().try_into().unwrap(),
        },
        GaloisRingTrimmedMaskCodeShare {
            id: party_id + 1,
            coefs: iris.left_mask().try_into().unwrap(),
        },
        GaloisRingIrisCodeShare {
            id: party_id + 1,
            coefs: iris.right_code().try_into().unwrap(),
        },
        GaloisRingTrimmedMaskCodeShare {
            id: party_id + 1,
            coefs: iris.right_mask().try_into().unwrap(),
        },
    );

    randomize_iris_inner(
        iris.id(),
        left_code,
        left_mask,
        right_code,
        right_mask,
        master_seed,
        party_id,
    )
}

fn randomize_iris_inner(
    iris_id: i64,
    mut left_code: GaloisRingIrisCodeShare,
    mut left_mask: GaloisRingTrimmedMaskCodeShare,
    mut right_code: GaloisRingIrisCodeShare,
    mut right_mask: GaloisRingTrimmedMaskCodeShare,
    master_seed: &[u8],
    party_id: usize,
) -> (
    i64,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
    GaloisRingIrisCodeShare,
    GaloisRingTrimmedMaskCodeShare,
) {
    let mut hasher = blake3::Hasher::new();
    hasher.update(master_seed);
    hasher.update(&iris_id.to_le_bytes());
    let mut xof = hasher.finalize_xof();
    randomize_galois_ring_coefs(&mut left_code.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut left_mask.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut right_code.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut right_mask.coefs, &mut xof, party_id);
    (iris_id, left_code, left_mask, right_code, right_mask)
}

fn randomize_galois_ring_coefs(coefs: &mut [u16], xof: &mut blake3::OutputReader, party_id: usize) {
    for coefs in coefs.chunks_mut(4) {
        assert!(coefs.len() == 4, "Expected 4 coefficients per chunk");
        let mut gr = GaloisRingElement::<Monomial>::from_coefs(coefs.try_into().unwrap());
        let mut r = [0u16; 4];
        xof.read_exact(bytemuck::cast_slice_mut(&mut r[..]))
            .expect("can read from xof");
        let mut r = GaloisRingElement::<Monomial>::from_coefs(r);
        r = r * GaloisRingElement::<Monomial>::EXCEPTIONAL_SEQUENCE[party_id + 1];
        gr = gr + r;
        coefs.copy_from_slice(&gr.coefs[..]);
    }
}

#[cfg(test)]
mod tests {
    use iris_mpc_common::{
        galois_engine::degree4::FullGaloisRingIrisCodeShare, iris_db::iris::IrisCode,
    };

    use super::*;

    #[test]
    fn test_rerandomization() {
        let rng = &mut rand::thread_rng();
        let iris_code_l = IrisCode::random_rng(rng);
        let iris_code_r = IrisCode::random_rng(rng);
        let mask_expected = (iris_code_l.mask & iris_code_r.mask).count_ones();

        let [mut left_0, mut left_1, mut left_2] =
            FullGaloisRingIrisCodeShare::encode_iris_code(&iris_code_l, rng);
        let [right_0, right_1, right_2] =
            FullGaloisRingIrisCodeShare::encode_iris_code(&iris_code_r, rng);

        let master_seed = [42u8; 32];
        let iris_id = 123;

        let (_, left_0_code, mut left_0_mask, right_0_code, right_0_mask) = randomize_iris_inner(
            iris_id,
            left_0.code.clone(),
            left_0.mask.clone(),
            right_0.code.clone(),
            right_0.mask.clone(),
            &master_seed,
            0,
        );
        let (_, left_1_code, mut left_1_mask, right_1_code, right_1_mask) = randomize_iris_inner(
            iris_id,
            left_1.code.clone(),
            left_1.mask.clone(),
            right_1.code.clone(),
            right_1.mask.clone(),
            &master_seed,
            1,
        );
        let (_, left_2_code, mut left_2_mask, right_2_code, right_2_mask) = randomize_iris_inner(
            iris_id,
            left_2.code.clone(),
            left_2.mask.clone(),
            right_2.code.clone(),
            right_2.mask.clone(),
            &master_seed,
            2,
        );

        let dot0 = left_0_code.full_dot(&right_0_code);
        let dot1 = left_1_code.full_dot(&right_1_code);
        let dot2 = left_2_code.full_dot(&right_2_code);
        let dot_code = dot0.wrapping_add(dot1).wrapping_add(dot2);

        left_0_mask.preprocess_mask_code_query_share();
        left_1_mask.preprocess_mask_code_query_share();
        left_2_mask.preprocess_mask_code_query_share();

        let dot0 = left_0_mask.trick_dot(&right_0_mask);
        let dot1 = left_1_mask.trick_dot(&right_1_mask);
        let dot2 = left_2_mask.trick_dot(&right_2_mask);
        let dot_mask = dot0.wrapping_add(dot1).wrapping_add(dot2);

        // original dot product
        let dot0 = left_0.code.full_dot(&right_0.code);
        let dot1 = left_1.code.full_dot(&right_1.code);
        let dot2 = left_2.code.full_dot(&right_2.code);
        let original_code_dot = dot0.wrapping_add(dot1).wrapping_add(dot2);

        assert_eq!(dot_code, original_code_dot);

        left_0.mask.preprocess_mask_code_query_share();
        left_1.mask.preprocess_mask_code_query_share();
        left_2.mask.preprocess_mask_code_query_share();

        let dot0 = left_0.mask.trick_dot(&right_0.mask);
        let dot1 = left_1.mask.trick_dot(&right_1.mask);
        let dot2 = left_2.mask.trick_dot(&right_2.mask);
        let original_mask_dot = dot0.wrapping_add(dot1).wrapping_add(dot2);
        assert_eq!(mask_expected as u16, original_mask_dot.wrapping_mul(2));
        assert_eq!(dot_mask, original_mask_dot);
    }
}
