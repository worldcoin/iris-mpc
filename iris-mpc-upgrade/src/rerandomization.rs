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
    let mut hasher = blake3::Hasher::new();
    hasher.update(master_seed);
    hasher.update(&iris.id().to_le_bytes());
    let mut xof = hasher.finalize_xof();

    let (mut left_code, mut left_mask, mut right_code, mut right_mask) = (
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

    randomize_galois_ring_coefs(&mut left_code.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut left_mask.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut right_code.coefs, &mut xof, party_id);
    randomize_galois_ring_coefs(&mut right_mask.coefs, &mut xof, party_id);
    (iris.id(), left_code, left_mask, right_code, right_mask)
}

fn randomize_galois_ring_coefs(coefs: &mut [u16], xof: &mut blake3::OutputReader, party_id: usize) {
    for coefs in coefs.chunks_mut(4) {
        assert!(coefs.len() == 4, "Expected 4 coefficients per chunk");
        let mut gr = GaloisRingElement::<Monomial>::from_coefs(coefs.try_into().unwrap());
        let mut r = [0u16; 4];
        xof.read_exact(bytemuck::cast_slice_mut(&mut r[..]))
            .expect("can read from xof");
        let mut r = GaloisRingElement::<Monomial>::from_coefs(r);
        r = r * GaloisRingElement::<Monomial>::EXCEPTIONAL_SEQUENCE[party_id];
        gr = gr + r;
        coefs.copy_from_slice(&gr.coefs[..]);
    }
}
