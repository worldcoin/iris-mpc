use crate::shares::{ring_impl::RingElement, share::Share, vecshare::VecShare};
use eyre::eyre;
use iris_mpc_common::iris_db::iris::{IrisCode, IrisCodeArray};
use rand::{Rng, RngCore};
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

pub(crate) fn create_shared_database_raw<R: RngCore>(
    rng: &mut R,
    in_mem: &[IrisCode],
) -> eyre::Result<RawSharedDatabase> {
    let mut shared_irises = (0..3)
        .map(|_| Vec::with_capacity(in_mem.len()))
        .collect::<Vec<_>>();
    for code in in_mem.iter() {
        let shared_code: Vec<_> = (0..IrisCode::IRIS_CODE_SIZE)
            .map(|i| IrisShare::get_shares(code.code.get_bit(i), code.mask.get_bit(i), rng))
            .collect();

        let shared_3_n: Vec<_> = (0..3)
            .map(|p_id| {
                let shared_n: Vec<Share<u16>> = (0..IrisCode::IRIS_CODE_SIZE)
                    .map(|iris_index| shared_code[iris_index][p_id].clone())
                    .collect();
                shared_n
            })
            .collect();
        // We simulate the parties already knowing the shares of the code.
        for party_id in 0..3 {
            shared_irises[party_id].push(SharedIris {
                shares: VecShareType::new_vec(shared_3_n[party_id].clone()),
                mask:   code.mask,
            });
        }
    }
    let player2_shares = shared_irises
        .pop()
        .ok_or(eyre!("error popping shared iris"))?;
    let player1_shares = shared_irises
        .pop()
        .ok_or(eyre!("error popping a shared iris"))?;
    let player0_shares = shared_irises
        .pop()
        .ok_or(eyre! {"error popping a shared iris"})?;
    Ok(RawSharedDatabase {
        player0_shares,
        player1_shares,
        player2_shares,
    })
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

pub fn create_ground_truth_database<R: RngCore>(
    rng: &mut R,
    num_items: usize,
) -> eyre::Result<Vec<IrisCode>> {
    let mut codes = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        let code = IrisCode::random_rng(rng);
        codes.push(code);
    }
    Ok(codes)
}