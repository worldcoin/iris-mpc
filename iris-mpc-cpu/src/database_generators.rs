use crate::shares::{ring_impl::RingElement, share::Share};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::IrisCode,
};
use rand::{CryptoRng, Rng, RngCore};

type ShareRing = u16;
type ShareRingPlain = RingElement<ShareRing>;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct GaloisRingSharedIris {
    pub code: GaloisRingIrisCodeShare,
    pub mask: GaloisRingTrimmedMaskCodeShare,
}

pub fn create_random_sharing<R: RngCore>(rng: &mut R, input: u16) -> Vec<Share<u16>> {
    let val = RingElement(input);
    let a = rng.gen::<ShareRingPlain>();
    let b = rng.gen::<ShareRingPlain>();
    let c = val - a - b;

    let share1 = Share::new(a, c);
    let share2 = Share::new(b, a);
    let share3 = Share::new(c, b);

    vec![share1, share2, share3]
}

pub fn generate_galois_iris_shares<R: Rng + CryptoRng>(
    rng: &mut R,
    iris: IrisCode,
) -> Vec<GaloisRingSharedIris> {
    let code_shares = GaloisRingIrisCodeShare::encode_iris_code(&iris.code, &iris.mask, rng);
    let mask_shares = GaloisRingIrisCodeShare::encode_mask_code(&iris.mask, rng);
    vec![
        GaloisRingSharedIris {
            code: code_shares[0].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[0]),
        },
        GaloisRingSharedIris {
            code: code_shares[1].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[1]),
        },
        GaloisRingSharedIris {
            code: code_shares[2].clone(),
            mask: GaloisRingTrimmedMaskCodeShare::from(&mask_shares[2]),
        },
    ]
}
