use crate::shares::{ring_impl::RingElement, share::Share, vecshare::VecShare};
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    iris_db::iris::{IrisCode, IrisCodeArray},
};
use rand::{CryptoRng, Rng, RngCore};
use std::sync::Arc;

type ShareRing = u16;
type ShareType = Share<ShareRing>;
type VecShareType = VecShare<u16>;
type ShareRingPlain = RingElement<ShareRing>;
// type ShareType = Share<u16>;

#[derive(PartialEq, Eq, Debug, Default, Clone)]
pub struct SharedIris {
    pub shares: VecShareType,
    pub mask:   IrisCodeArray,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct GaloisRingSharedIris {
    pub code: GaloisRingIrisCodeShare,
    pub mask: GaloisRingTrimmedMaskCodeShare,
}

#[derive(Clone)]
pub struct SharedDB {
    pub shares: Arc<Vec<VecShareType>>,
    pub masks:  Arc<Vec<IrisCodeArray>>,
}

pub struct RawSharedDatabase {
    pub player0_shares: Vec<SharedIris>,
    pub player1_shares: Vec<SharedIris>,
    pub player2_shares: Vec<SharedIris>,
}

/// This one is taken from iris-mpc-semi/iris.rs
pub struct IrisShare {}
impl IrisShare {
    pub fn get_shares<R: RngCore>(input: bool, mask: bool, rng: &mut R) -> Vec<ShareType> {
        let val = RingElement((input & mask) as ShareRing);
        let to_share = RingElement(mask as ShareRing) - val - val;

        let a = rng.gen::<ShareRingPlain>();
        let b = rng.gen::<ShareRingPlain>();
        let c = to_share - a - b;

        let share1 = Share::new(a, c);
        let share2 = Share::new(b, a);
        let share3 = Share::new(c, b);

        vec![share1, share2, share3]
    }
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

pub fn generate_iris_shares<R: Rng>(rng: &mut R, iris: IrisCode) -> Vec<SharedIris> {
    let mut res = vec![SharedIris::default(); 3];
    for res_i in res.iter_mut() {
        res_i
            .mask
            .as_raw_mut_slice()
            .copy_from_slice(iris.mask.as_raw_slice());
    }

    for i in 0..IrisCode::IRIS_CODE_SIZE {
        // We simulate the parties already knowing the shares of the code.
        let shares = IrisShare::get_shares(iris.code.get_bit(i), iris.mask.get_bit(i), rng);
        for party_id in 0..3 {
            res[party_id].shares.push(shares[party_id].to_owned());
        }
    }
    res
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
